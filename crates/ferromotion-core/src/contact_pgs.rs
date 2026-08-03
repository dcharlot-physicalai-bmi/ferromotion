//! **Projected Gauss-Seidel contact solver** — the robust alternative to the interior-point solve for
//! many simultaneous frictional contacts.
//!
//! The interior-point core ([`solve_frictional_ipm`](crate::solve_frictional_ipm)) is differentiable and
//! excellent for a small contact set, but a floating body standing on several sticking feet is an
//! over-constrained, degenerate problem: more constraints than degrees of freedom, with impulses that
//! are not uniquely determined. A damped Newton method on that system can fail to converge and still
//! return, which downstream looks like a body gaining energy from nothing.
//!
//! This solver takes the opposite trade. It sweeps the contacts one at a time, solving each contact's
//! own 3×3 problem exactly against the current velocities and projecting the result back onto the
//! friction cone, then repeats. Each projection is non-expansive, so a sweep can never amplify the
//! impulses: the iteration degrades to *slow* on a hard problem rather than *wrong*. It is what
//! production engines use for exactly this reason. Two further gains over the pyramid formulation: the
//! cone here is the true circular Coulomb cone rather than a facet approximation, and the previous
//! step's impulses can be handed back in as a warm start.
//!
//! The trade given up is the closed-form derivative; for gradients through contact, keep the
//! interior-point path.

use nalgebra::{DMatrix, DVector, Vector3};

/// A contact for the Gauss-Seidel solver: a `3 × nv` Jacobian mapping generalized velocity to the
/// contact point's world velocity (rows are x, y, z, with **z the surface normal**), the signed gap
/// `phi` (negative when penetrating), and the friction coefficient.
#[derive(Clone, Debug)]
pub struct PgsContact {
    pub j: DMatrix<f64>,
    pub phi: f64,
    pub mu: f64,
}

/// **Gap stabilisation for a resting contact.** Feeding the whole gap error back as a velocity demand in
/// one step makes a resting contact chatter: a foot a micron clear of the floor is told to keep
/// separating, so its impulse drops to zero, gravity puts it a micron under, and the correction
/// over-pushes it back out. The impulse then toggles every timestep. That is the integrator talking, not
/// the contact, and it destroys any linearisation of the motion because the mode sequence flips
/// arbitrarily fast.
///
/// The remedy is the one production engines settle on. Allow a small penetration `slop` inside which no
/// gap feedback is applied at all, recover only a fraction `erp` of any excess per step, and bound the
/// correction speed so a deep overlap cannot launch the body. The resulting bias is continuous, and it is
/// exactly zero across the whole resting band, which is what removes the toggle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PgsStabilization {
    /// Penetration allowed without any correction, in metres. Inside this band a contact is treated as
    /// resting and its impulse is set by the velocity condition alone.
    pub slop: f64,
    /// Fraction of the excess penetration corrected per step. Below 1 so the correction cannot overshoot.
    pub erp: f64,
    /// Upper bound on the correction speed, in m/s, so recovering a deep overlap stays bounded.
    pub max_correction: f64,
}

impl Default for PgsStabilization {
    fn default() -> Self {
        PgsStabilization { slop: 1e-4, erp: 0.2, max_correction: 0.5 }
    }
}

impl PgsStabilization {
    /// The exact gap feedback, correcting the full error every step. This is the behaviour that chatters;
    /// it is available because it is the right choice when a contact is genuinely being made or broken and
    /// no penetration is acceptable.
    pub fn exact() -> Self {
        PgsStabilization { slop: 0.0, erp: 1.0, max_correction: f64::INFINITY }
    }

    /// The velocity bias added to the normal condition for a gap of `phi`. Continuous in `phi`, and zero
    /// throughout the resting band `|phi| <= slop`.
    pub fn normal_bias(&self, phi: f64, dt: f64) -> f64 {
        if phi > self.slop {
            (phi - self.slop) / dt // separating: let the contact deactivate
        } else if phi < -self.slop {
            (-self.erp * (-phi - self.slop) / dt).max(-self.max_correction) // push out, gently and boundedly
        } else {
            0.0 // resting: no gap feedback, so nothing to toggle against
        }
    }
}

/// The outcome of a Gauss-Seidel contact solve.
#[derive(Clone, Debug)]
pub struct PgsResult {
    pub v_next: DVector<f64>,
    /// Per-contact impulse `[tx, ty, n]`, suitable for warm-starting the next step.
    pub lambda: Vec<Vector3<f64>>,
    /// Largest impulse change in the final sweep: how converged the answer is.
    pub residual: f64,
    /// Worst violation of the cone and complementarity conditions at the returned point.
    pub violation: f64,
    pub iters: usize,
}

/// Solve the frictional contact problem by projected Gauss-Seidel.
///
/// `m` is the joint-space inertia (`nv × nv`), `v_free` the velocity the body would have with no
/// contact, and each contact contributes its Jacobian and gap. `warm` optionally supplies the previous
/// step's impulses. Returns the post-contact velocity along with how well it converged.
///
/// Gap feedback is stabilised by [`PgsStabilization::default`], which is what keeps a resting contact
/// from chattering. Use [`solve_contacts_pgs_with`] to choose the stabilisation.
pub fn solve_contacts_pgs(
    m: &DMatrix<f64>,
    v_free: &DVector<f64>,
    contacts: &[PgsContact],
    dt: f64,
    max_iters: usize,
    warm: Option<&[Vector3<f64>]>,
) -> PgsResult {
    solve_contacts_pgs_with(m, v_free, contacts, dt, max_iters, warm, PgsStabilization::default())
}

/// [`solve_contacts_pgs`] with the gap stabilisation named explicitly. Pass
/// [`PgsStabilization::exact`] for the unstabilised condition that corrects the full gap every step.
#[allow(clippy::too_many_arguments)]
pub fn solve_contacts_pgs_with(
    m: &DMatrix<f64>,
    v_free: &DVector<f64>,
    contacts: &[PgsContact],
    dt: f64,
    max_iters: usize,
    warm: Option<&[Vector3<f64>]>,
    stab: PgsStabilization,
) -> PgsResult {
    let nc = contacts.len();
    if nc == 0 {
        return PgsResult { v_next: v_free.clone(), lambda: Vec::new(), residual: 0.0, violation: 0.0, iters: 0 };
    }
    // A singular mass matrix is a caller error, not a reason to abort someone's simulation: report it
    // through `violation` and leave the velocity untouched rather than panicking.
    let Some(minv) = m.clone().try_inverse() else {
        return PgsResult { v_next: v_free.clone(), lambda: vec![Vector3::zeros(); nc], residual: f64::INFINITY, violation: f64::INFINITY, iters: 0 };
    };

    // Per contact: M⁻¹Jᵀ (nv×3) to apply an impulse, and the 3×3 Delassus block J M⁻¹ Jᵀ that says how
    // this contact's own impulse moves its own contact point.
    let mut mjt: Vec<DMatrix<f64>> = Vec::with_capacity(nc);
    let mut wdiag: Vec<Vector3<f64>> = Vec::with_capacity(nc);
    for c in contacts {
        let mj = &minv * c.j.transpose();
        let wd = &c.j * &mj; // the Delassus block: how this contact's impulse moves its own point
        // Only the diagonal is needed, because each direction is relaxed in turn. Redundant contacts
        // make the full block singular; the diagonal never is, which is why the sweep tolerates the
        // degenerate case that defeats a direct solve.
        let floor_w = 1e-12;
        wdiag.push(Vector3::new(wd[(0, 0)].max(floor_w), wd[(1, 1)].max(floor_w), wd[(2, 2)].max(floor_w)));
        mjt.push(mj);
    }

    let mut lam: Vec<Vector3<f64>> = match warm {
        Some(w) if w.len() == nc => w.to_vec(),
        _ => vec![Vector3::zeros(); nc],
    };
    // start from the velocity implied by the warm impulses
    let mut v = v_free.clone();
    for i in 0..nc {
        if lam[i] != Vector3::zeros() {
            v += &mjt[i] * DVector::from_row_slice(&[lam[i].x, lam[i].y, lam[i].z]);
        }
    }

    let mut residual = 0.0;
    let mut iters = 0;
    for it in 0..max_iters {
        let mut delta = 0.0f64;
        for i in 0..nc {
            let (c, wd) = (&contacts[i], &wdiag[i]);
            // 1. Normal, by complementarity: push just hard enough to stop the gap closing, never pull.
            let un = (c.j.row(2) * &v)[0] + stab.normal_bias(c.phi, dt);
            let ln_new = (lam[i].z - un / wd.z).max(0.0);
            let dn = ln_new - lam[i].z;
            if dn != 0.0 {
                v += &mjt[i] * DVector::from_row_slice(&[0.0, 0.0, dn]);
                lam[i].z = ln_new;
            }
            // 2. Friction, clamped to the normal we just found. Solving the normal first is what keeps
            // a large tangential demand from inflating the contact force, which a joint projection onto
            // the cone would do.
            let ut = Vector3::new((c.j.row(0) * &v)[0], (c.j.row(1) * &v)[0], 0.0);
            let mut lt = Vector3::new(lam[i].x - ut.x / wd.x, lam[i].y - ut.y / wd.y, 0.0);
            let tn = (lt.x * lt.x + lt.y * lt.y).sqrt();
            let cap = c.mu * lam[i].z;
            if tn > cap {
                if tn > 0.0 {
                    lt *= cap / tn; // slide: friction saturates on the cone, opposing the motion
                } else {
                    lt = Vector3::zeros();
                }
            }
            let dt_imp = Vector3::new(lt.x - lam[i].x, lt.y - lam[i].y, 0.0);
            if dt_imp.norm() > 0.0 {
                v += &mjt[i] * DVector::from_row_slice(&[dt_imp.x, dt_imp.y, 0.0]);
                lam[i].x = lt.x;
                lam[i].y = lt.y;
            }
            delta = delta.max(dn.abs().max(dt_imp.norm()));
        }
        iters = it + 1;
        residual = delta;
        if delta < 1e-12 {
            break;
        }
    }

    // How well the returned point satisfies the contact conditions. Three physical checks: a contact
    // may only push, the friction impulse must lie in the cone, and a pressing contact must not still
    // be closing. Scaled by the impulse size so the number means the same thing at any mass.
    let mut violation = 0.0f64;
    let scale = lam.iter().fold(1e-9f64, |m, l| m.max(l.norm()));
    for i in 0..nc {
        let u = &contacts[i].j * &v;
        let un = u[2] + stab.normal_bias(contacts[i].phi, dt);
        let l = lam[i];
        let tn = (l.x * l.x + l.y * l.y).sqrt();
        violation = violation.max((-l.z).max(0.0) / scale); // must not pull
        violation = violation.max((tn - contacts[i].mu * l.z).max(0.0) / scale); // must stay in the cone
        if l.z > 1e-9 * scale {
            violation = violation.max((-un).max(0.0)); // a loaded contact must not still be closing
        }
    }

    PgsResult { v_next: v, lambda: lam, residual, violation, iters }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 3-DoF unit mass on the ground, sliding. Friction must decelerate it and never reverse it, and
    /// the impulse must stay inside the cone. Mirrors the interior-point solver's own sliding test.
    #[test]
    fn friction_decelerates_a_sliding_block() {
        let m = DMatrix::identity(3, 3);
        let mut j = DMatrix::zeros(3, 3);
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0; // contact point velocity == body velocity
        let c = PgsContact { j, phi: 0.0, mu: 0.5 };
        let (g, dt, v0) = (9.81, 0.01, 2.0);
        let (mut vx, mut vz, mut min_vx) = (v0, 0.0, v0);
        let mut stop_t = None;
        for k in 0..80 {
            let vf = DVector::from_row_slice(&[vx, 0.0, vz - g * dt]);
            let r = solve_contacts_pgs(&m, &vf, std::slice::from_ref(&c), dt, 200, None);
            vx = r.v_next[0];
            vz = r.v_next[2];
            min_vx = min_vx.min(vx);
            assert!(r.violation < 1e-6, "cone or complementarity violated: {}", r.violation);
            if stop_t.is_none() && vx.abs() < 1e-9 {
                stop_t = Some((k + 1) as f64 * dt);
            }
        }
        // The sharp oracle: sliding Coulomb friction decelerates at exactly mu*g, so a block starting
        // at v0 stops at v0/(mu*g). This is what separates a correct solve from one that inflates the
        // contact force to satisfy a large tangential demand.
        let expected = v0 / (c.mu * g);
        let got = stop_t.expect("block never came to rest");
        eprintln!("PGS sliding block: stopped at {got:.3} s, analytic v0/(mu*g) = {expected:.3} s, min vx {min_vx:.4}");
        assert!((got - expected).abs() < 2.0 * dt, "deceleration is not mu*g: stopped at {got} s, expected {expected} s");
        assert!(min_vx > -1e-9, "friction reversed the block: {min_vx}");
    }

    /// The stabilised bias is continuous, including at both edges of the resting band, and vanishes
    /// across the band. A jump there would just move the chatter to the boundary.
    #[test]
    fn the_gap_bias_is_continuous_and_zero_while_resting() {
        let s = PgsStabilization::default();
        let dt = 5e-4;
        assert_eq!(s.normal_bias(0.0, dt), 0.0);
        assert_eq!(s.normal_bias(s.slop * 0.5, dt), 0.0, "resting band must carry no gap feedback");
        assert_eq!(s.normal_bias(-s.slop * 0.5, dt), 0.0);
        // continuous at both edges
        let eps = s.slop * 1e-6;
        assert!(s.normal_bias(s.slop + eps, dt).abs() < 1e-6, "jump at the separating edge");
        assert!(s.normal_bias(-s.slop - eps, dt).abs() < 1e-6, "jump at the penetrating edge");
        // separation is still permitted, and the push-out speed stays bounded however deep the overlap
        assert!(s.normal_bias(0.01, dt) > 0.0, "a clearly separated contact must be allowed to deactivate");
        assert!(s.normal_bias(-1e6, dt) >= -s.max_correction, "correction speed must be capped");
    }

    /// A contact never adds energy: whatever the incoming velocity, the post-contact kinetic energy of
    /// a unit mass cannot exceed what it had. The property that makes the sweep safe.
    #[test]
    fn contact_never_increases_kinetic_energy() {
        let m = DMatrix::identity(3, 3);
        let mut j = DMatrix::zeros(3, 3);
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0;
        for &mu in &[0.0, 0.5, 1.0, 2.0, 5.0] {
            let c = PgsContact { j: j.clone(), phi: 0.0, mu };
            for &(vx, vy, vz) in &[(1.0, 0.0, -1.0), (3.0, -2.0, -0.5), (0.1, 0.1, -9.0), (-4.0, 1.0, -2.0)] {
                let vf = DVector::from_row_slice(&[vx, vy, vz]);
                let r = solve_contacts_pgs(&m, &vf, std::slice::from_ref(&c), 0.01, 300, None);
                let (e0, e1) = (vf.norm_squared(), r.v_next.norm_squared());
                assert!(e1 <= e0 + 1e-9, "mu {mu}: contact created energy, {e0} -> {e1}");
                assert!(r.violation < 1e-6, "mu {mu}: violation {}", r.violation);
            }
        }
    }

    /// Redundant contacts are the degenerate case: four coincident constraints on one body have no
    /// unique impulse split. The sweep must still return a sane, converged velocity.
    #[test]
    fn redundant_contacts_stay_bounded() {
        let m = DMatrix::identity(3, 3);
        let mut j = DMatrix::zeros(3, 3);
        j[(0, 0)] = 1.0;
        j[(1, 1)] = 1.0;
        j[(2, 2)] = 1.0;
        let contacts: Vec<PgsContact> = (0..4).map(|_| PgsContact { j: j.clone(), phi: -0.001, mu: 1.5 }).collect();
        let vf = DVector::from_row_slice(&[2.0, 1.0, -3.0]);
        let r = solve_contacts_pgs(&m, &vf, &contacts, 0.002, 500, None);
        eprintln!("PGS redundant: v {:?}, residual {:.2e}, violation {:.2e}, iters {}", r.v_next.as_slice(), r.residual, r.violation, r.iters);
        assert!(r.v_next.iter().all(|x| x.is_finite()), "diverged");
        assert!(r.v_next.norm() <= vf.norm() + 1e-9, "gained speed from redundant contacts");
        assert!(r.violation < 1e-6, "violation {}", r.violation);
    }
}
