//! **Hybrid dynamics: impacts, guards, and orbital stability** — the machinery a walking certificate
//! needs and a continuous-phase certificate cannot supply.
//!
//! A legged gait is not a flow, it is a *hybrid* system: the robot flows until a foot reaches the
//! ground (a **guard** surface), the velocity jumps (a **reset**, the impact), and the flow resumes.
//! Stability of the gait is a property of the composition, and it is decided at the reset. A Lyapunov
//! condition verified along the continuous phase alone can be satisfied for every parameter value
//! while the gait diverges, which makes it unfalsifiable and therefore vacuous as a gait certificate.
//!
//! This module supplies the three objects that make the hybrid statement checkable:
//!
//! * [`plastic_impact`] — the physical reset. An inelastic impact is the impulse that drives the
//!   contacting point's velocity to zero; it is a projection, so it always dissipates energy.
//! * [`saltation_matrix`] — the correct Jacobian *across* a guard. Differentiating the flow and the
//!   reset separately gets this wrong, because neighbouring trajectories cross the guard at different
//!   times; the saltation term is exactly that timing correction.
//! * [`poincare_stability`] and [`hybrid_certificate`] — orbital stability from the return map's
//!   spectral radius, and the impact-aware certificate with its design rule.
//!
//! The certificate follows the three-condition protocol: continuous descent, impact expansion
//! `μ² = λ_max(P^{-1/2} Δᵀ P Δ P^{-1/2})`, and cycle contraction `μ² e^{-c₃T/ε} < 1`. The last gives a
//! largest admissible rate parameter `ε̄ = c₃T / (2 ln μ)`, which turns the classical "for ε
//! sufficiently small" into a number.

use nalgebra::{DMatrix, DVector};

/// The velocity jump of a **plastic (perfectly inelastic) impact**: the impulse that brings the
/// contacting point to rest. With mass matrix `m`, pre-impact generalized velocity `v_minus`, and
/// contact Jacobian `jc` (`k × nv`, mapping generalized velocity to the contact point's velocity),
/// the post-impact velocity is
///
/// `v⁺ = v⁻ − M⁻¹Jᵀ (J M⁻¹ Jᵀ)⁻¹ J v⁻`,
///
/// which is the `M`-orthogonal projection onto `{v : J v = 0}`. Being a projection in the kinetic-energy
/// metric, it can only remove energy — the impact is dissipative by construction, never generative.
pub fn plastic_impact(m: &DMatrix<f64>, v_minus: &DVector<f64>, jc: &DMatrix<f64>) -> DVector<f64> {
    let Some(minv) = m.clone().try_inverse() else {
        return v_minus.clone();
    };
    let mjt = &minv * jc.transpose();
    let w = jc * &mjt; // the Delassus operator of the impacting set
    let ju = jc * v_minus;
    // a redundant contact set makes W singular; the pseudo-inverse picks the minimum-norm impulse,
    // which is the physically meaningful choice when the impulse split is not determined
    let lambda = match w.clone().try_inverse() {
        Some(winv) => winv * ju,
        None => match w.pseudo_inverse(1e-12) {
            Ok(wp) => wp * ju,
            Err(_) => return v_minus.clone(),
        },
    };
    v_minus - mjt * lambda
}

/// The reset Jacobian of a plastic impact: `Δ = I − M⁻¹Jᵀ(J M⁻¹ Jᵀ)⁻¹J`. Constant in the velocity, so
/// it *is* the linearisation, which is what the impact-expansion test needs.
pub fn plastic_impact_jacobian(m: &DMatrix<f64>, jc: &DMatrix<f64>) -> DMatrix<f64> {
    let nv = m.nrows();
    let Some(minv) = m.clone().try_inverse() else {
        return DMatrix::identity(nv, nv);
    };
    let mjt = &minv * jc.transpose();
    let w = jc * &mjt;
    let winv = match w.clone().try_inverse() {
        Some(x) => x,
        None => match w.pseudo_inverse(1e-12) {
            Ok(x) => x,
            Err(_) => return DMatrix::identity(nv, nv),
        },
    };
    DMatrix::identity(nv, nv) - mjt * winv * jc
}

/// The **saltation matrix**: the Jacobian of the state transition across a guard surface.
///
/// `Ξ = DΔ + (f⁺ − DΔ·f⁻) gᵀ / (gᵀ f⁻)`
///
/// where `DΔ` is the reset Jacobian, `g = ∂h/∂x` the guard normal, and `f⁻`, `f⁺` the vector fields
/// just before and just after the event. The second term is the correction nobody gets for free:
/// a neighbouring trajectory hits the guard at a slightly different *time*, and ignoring that gives the
/// wrong linearisation and therefore the wrong stability verdict. Returns `None` when the trajectory
/// does not cross the guard transversally (`gᵀf⁻ ≈ 0`), where the linearisation does not exist.
pub fn saltation_matrix(reset_jacobian: &DMatrix<f64>, guard_normal: &DVector<f64>, f_minus: &DVector<f64>, f_plus: &DVector<f64>) -> Option<DMatrix<f64>> {
    let denom = guard_normal.dot(f_minus);
    if denom.abs() < 1e-12 {
        return None; // grazing contact: no transversal crossing, no linearisation
    }
    let corr = (f_plus - reset_jacobian * f_minus) * guard_normal.transpose() / denom;
    Some(reset_jacobian + corr)
}

/// Orbital stability of a periodic hybrid orbit from its **monodromy matrix** (the linearisation of the
/// Poincaré return map). Returns `(spectral_radius, stable)`; a hybrid limit cycle is orbitally stable
/// when every eigenvalue except the trivial one along the flow lies inside the unit circle.
pub fn poincare_stability(monodromy: &DMatrix<f64>) -> (f64, bool) {
    let eig = monodromy.complex_eigenvalues();
    let rho = eig.iter().fold(0.0f64, |m, e| m.max(e.norm()));
    (rho, rho < 1.0)
}

/// The outcome of the three-condition hybrid certificate.
#[derive(Clone, Copy, Debug)]
pub struct HybridCertificate {
    /// `μ²`, the impact expansion in the Lyapunov metric: `λ_max(P^{-1/2} Δᵀ P Δ P^{-1/2})`.
    pub mu_sq: f64,
    /// `χ = μ² e^{-c₃T/ε}`; the cycle contracts when this is below one.
    pub chi: f64,
    /// Whether the certificate is granted.
    pub certified: bool,
    /// The largest admissible rate parameter, `ε̄ = c₃T / (2 ln μ)`. `None` when the impact is
    /// non-expansive to within round-off (`μ ≤ 1`), in which case every `ε` is admissible.
    pub eps_bar: Option<f64>,
}

/// The **impact expansion** `μ² = λ_max(P^{-1/2} Δᵀ P Δ P^{-1/2})`: how much the reset can inflate the
/// Lyapunov function `V(x) = xᵀPx`. One symmetric eigenvalue problem, not a search. `p` must be
/// symmetric positive-definite.
pub fn impact_expansion(p: &DMatrix<f64>, reset_jacobian: &DMatrix<f64>) -> Option<f64> {
    // P^{1/2} by symmetric eigendecomposition, so the similarity is exact rather than a Cholesky proxy
    let se = p.clone().symmetric_eigen();
    if se.eigenvalues.iter().any(|&l| l <= 0.0) {
        return None; // not positive-definite: V is not a Lyapunov function
    }
    let sqrt_d = DMatrix::from_diagonal(&se.eigenvalues.map(|l| l.sqrt()));
    let inv_sqrt_d = DMatrix::from_diagonal(&se.eigenvalues.map(|l| 1.0 / l.sqrt()));
    let p_half = &se.eigenvectors * sqrt_d * se.eigenvectors.transpose();
    let p_mhalf = &se.eigenvectors * inv_sqrt_d * se.eigenvectors.transpose();
    let s = &p_mhalf * reset_jacobian.transpose() * &p_half * &p_half * reset_jacobian * &p_mhalf;
    let sym = (&s + s.transpose()) * 0.5; // symmetrize against round-off
    Some(sym.symmetric_eigen().eigenvalues.iter().fold(0.0f64, |m, &l| m.max(l)))
}

/// The hybrid certificate. `p` is the Lyapunov matrix, `reset_jacobian` the linearised impact, `c3` the
/// continuous-phase decay coefficient (`V̇ ≤ −(c₃/ε)V`), `t_step` the step duration and `eps` the rate
/// parameter. Grants the certificate when `μ² e^{-c₃T/ε} < 1`, and reports the largest admissible `ε`.
///
/// This is sufficient, not necessary: it is a quadratic-form bound, so it is sound but conservative
/// against the exact condition `ρ(Δ e^{A T}) < 1`. Where the return map is available, prefer
/// [`poincare_stability`] on the monodromy, which is tighter.
pub fn hybrid_certificate(p: &DMatrix<f64>, reset_jacobian: &DMatrix<f64>, c3: f64, t_step: f64, eps: f64) -> Option<HybridCertificate> {
    let mu_sq = impact_expansion(p, reset_jacobian)?;
    let chi = mu_sq * (-c3 * t_step / eps).exp();
    let mu = mu_sq.sqrt();
    // `mu` must clear one by more than round-off before the rule means anything: at mu = 1 + 1e-13 the
    // formula divides by ln(mu) ≈ 1e-13 and reports an astronomically large "bound", which is not a
    // bound at all. A numerically non-expansive impact admits every rate, and should say so.
    let eps_bar = if mu > 1.0 + 1e-9 { Some(c3 * t_step / (2.0 * mu.ln())) } else { None };
    Some(HybridCertificate { mu_sq, chi, certified: chi < 1.0, eps_bar })
}

/// The Jacobian of a return map at a point, by central differences: the **monodromy matrix** of the
/// orbit through that point. `map` returns `None` where the trajectory fails to return (a fall, a
/// missed guard), in which case there is no Jacobian to report.
pub fn return_map_jacobian(map: &dyn Fn(&DVector<f64>) -> Option<DVector<f64>>, x: &DVector<f64>, eps: f64) -> Option<DMatrix<f64>> {
    let n = x.len();
    let base = map(x)?;
    let mut j = DMatrix::zeros(base.len(), n);
    for c in 0..n {
        let (mut xp, mut xm) = (x.clone(), x.clone());
        xp[c] += eps;
        xm[c] -= eps;
        let col = (map(&xp)? - map(&xm)?) / (2.0 * eps);
        j.set_column(c, &col);
    }
    Some(j)
}

/// Find a **fixed point of a return map** — that is, a periodic orbit — by Newton on `P(x) − x = 0`.
///
/// Uses a least-squares (pseudo-inverse) Newton step on purpose. A conservative system's return map
/// preserves a quantity such as energy, which makes `J − I` exactly singular along the level set, so a
/// plain Newton solve fails on precisely the systems this is most wanted for. The least-squares step
/// converges to the fixed point on the level set the initial guess sits on.
pub fn find_limit_cycle(map: &dyn Fn(&DVector<f64>) -> Option<DVector<f64>>, x0: &DVector<f64>, tol: f64, max_iter: usize) -> Option<DVector<f64>> {
    let n = x0.len();
    let mut x = x0.clone();
    for _ in 0..max_iter {
        let fx = map(&x)?;
        let g = &fx - &x;
        if g.norm() < tol {
            return Some(x);
        }
        let j = return_map_jacobian(map, &x, 1e-6)?;
        let jg = j - DMatrix::identity(n, n);
        let step = jg.pseudo_inverse(1e-10).ok()? * &g;
        // damp so a bad linearisation cannot throw the iterate out of the map's domain
        let mut alpha = 1.0;
        let mut moved = false;
        for _ in 0..20 {
            let trial = &x - alpha * &step;
            if map(&trial).is_some_and(|ft| (ft - &trial).norm() < g.norm()) {
                x = trial;
                moved = true;
                break;
            }
            alpha *= 0.5;
        }
        if !moved {
            return None;
        }
    }
    let fx = map(&x)?;
    if (fx - &x).norm() < tol {
        Some(x)
    } else {
        None
    }
}

/// One event along a hybrid trajectory: where it happened, and the pieces needed to linearise across it.
#[derive(Clone, Debug)]
pub struct HybridEvent {
    /// Smooth flow Jacobian accumulated from the previous event (or the start) up to this one.
    pub flow_jacobian: DMatrix<f64>,
    /// Reset Jacobian of the impact itself.
    pub reset_jacobian: DMatrix<f64>,
    /// Guard normal `∂h/∂x` at the crossing.
    pub guard_normal: DVector<f64>,
    /// Vector field immediately before and after the reset.
    pub f_minus: DVector<f64>,
    pub f_plus: DVector<f64>,
}

/// **Compose a sound monodromy for a contact-rich return map.**
///
/// Finite-differencing a return map that crosses contact events does not give its linearisation. A
/// small perturbation changes *which* foot is on the ground at a given instant, so the difference
/// quotient measures a jump divided by the step, and the answer diverges as the step shrinks. Confirmed
/// on the quadruped: the apparent spectral radius ran 11 → 291 → 769 → 21088 at probe steps 1e-3 down
/// to 1e-6, while the robot walks perfectly well.
///
/// The linearisation instead alternates smooth pieces and jumps:
///
/// `Π = Φ_final · Ξ_n · Φ_n · … · Ξ_1 · Φ_1`
///
/// where each `Φ` is the flow Jacobian over a smooth stretch and each `Ξ` the
/// [`saltation_matrix`] at an event. Returns `None` if any crossing is non-transversal, where no
/// linearisation exists.
pub fn compose_monodromy(events: &[HybridEvent], flow_after_last: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = flow_after_last.nrows();
    let mut acc = DMatrix::identity(n, n);
    for e in events {
        let xi = saltation_matrix(&e.reset_jacobian, &e.guard_normal, &e.f_minus, &e.f_plus)?;
        acc = xi * &e.flow_jacobian * acc;
    }
    Some(flow_after_last * acc)
}

/// The flow Jacobian of a smooth stretch, `Φ = exp(A·t)`, from a constant linearisation `A`. For a
/// stretch short enough that `A` is near-constant this is the right object to hand
/// [`compose_monodromy`]; over a long stretch, integrate the variational equation instead and pass that.
pub fn flow_jacobian(a: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    (a * t).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diag(v: &[f64]) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_row_slice(v))
    }

    /// A plastic impact brings the contact point to rest and can only remove kinetic energy. Both are
    /// definitional, and both fail if the projection is built wrong.
    #[test]
    fn plastic_impact_stops_the_contact_and_dissipates() {
        let m = diag(&[2.0, 3.0, 1.5]);
        // the contact sees the third coordinate plus a bit of the first
        let jc = DMatrix::from_row_slice(1, 3, &[0.4, 0.0, 1.0]);
        for v in [vec![1.0, -2.0, -3.0], vec![0.0, 0.5, -1.0], vec![-4.0, 1.0, -0.2]] {
            let vm = DVector::from_row_slice(&v);
            let vp = plastic_impact(&m, &vm, &jc);
            let after = (&jc * &vp)[0];
            let ke = |x: &DVector<f64>| 0.5 * (x.transpose() * &m * x)[(0, 0)];
            assert!(after.abs() < 1e-12, "contact still moving after impact: {after}");
            assert!(ke(&vp) <= ke(&vm) + 1e-12, "impact created energy: {} -> {}", ke(&vm), ke(&vp));
        }
    }

    /// The reset Jacobian is the linearisation of the impact, and the impact is idempotent: applying it
    /// twice changes nothing, because the state is already on the constraint.
    #[test]
    fn reset_jacobian_matches_the_impact_and_is_idempotent() {
        let m = diag(&[1.0, 2.0, 4.0, 0.5]);
        let jc = DMatrix::from_row_slice(2, 4, &[1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, -0.3]);
        let d = plastic_impact_jacobian(&m, &jc);
        let vm = DVector::from_row_slice(&[0.7, -1.1, 2.0, 0.4]);
        let direct = plastic_impact(&m, &vm, &jc);
        let viaj = &d * &vm;
        assert!((&direct - &viaj).amax() < 1e-12, "reset Jacobian disagrees with the impact");
        let twice = plastic_impact(&m, &direct, &jc);
        assert!((&twice - &direct).amax() < 1e-12, "impact is not idempotent");
    }

    /// The saltation matrix is the Jacobian across a guard, and the check that it is right is the
    /// composition identity `J_event = Φ(τ)·Ξ·Φ(t_hit)`: the event map's true Jacobian factors into
    /// flow-to-guard, jump, flow-onward. A bouncing point mass has all three in closed form. Using the
    /// reset Jacobian alone in place of `Ξ` gets a measurably different answer, which is the whole
    /// reason the object exists.
    #[test]
    fn saltation_matrix_satisfies_the_event_map_composition() {
        // state (h, v); flow ḣ = v, v̇ = 0; guard h = 0; reset v ↦ −e·v
        let e = 0.6;
        let (h0, v0, total) = (0.5, -1.0, 1.2);
        let t_hit = -h0 / v0;
        let tau = total - t_hit;
        assert!(tau > 0.0, "the event must happen inside the window");

        let f_minus = DVector::from_row_slice(&[v0, 0.0]);
        let dreset = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -e]);
        let v_plus = -e * v0;
        let f_plus = DVector::from_row_slice(&[v_plus, 0.0]);
        let g = DVector::from_row_slice(&[1.0, 0.0]); // ∂h/∂x
        let xi = saltation_matrix(&dreset, &g, &f_minus, &f_plus).expect("transversal crossing");

        // the true event map, in closed form: flow to the guard, reset, flow the remaining time
        let event_map = |h: f64, v: f64| -> (f64, f64) {
            let th = -h / v;
            let vp = -e * v;
            (vp * (total - th), vp)
        };
        let (bh, bv) = event_map(h0, v0);
        let eps = 1e-7;
        let (ph, pv) = event_map(h0 + eps, v0);
        let (qh, qv) = event_map(h0, v0 + eps);
        let fd = DMatrix::from_row_slice(2, 2, &[(ph - bh) / eps, (qh - bh) / eps, (pv - bv) / eps, (qv - bv) / eps]);

        // Φ(t) for this flow is [[1, t], [0, 1]]
        let flow = |t: f64| DMatrix::from_row_slice(2, 2, &[1.0, t, 0.0, 1.0]);
        let composed = flow(tau) * &xi * flow(t_hit);
        let naive = flow(tau) * &dreset * flow(t_hit); // the reset alone, no timing correction

        let salt_err = (&composed - &fd).amax();
        let naive_err = (&naive - &fd).amax();
        eprintln!("saltation: composed-vs-FD {salt_err:.2e}, reset-only-vs-FD {naive_err:.2e}");
        assert!(salt_err < 1e-5, "saltation composition disagrees with the event map: {salt_err}");
        assert!(naive_err > 1e-2, "test is not discriminating: reset-only should be clearly wrong, got {naive_err}");
    }

    /// Orbital stability from the monodromy spectral radius, on a case with a known answer.
    #[test]
    fn poincare_stability_reads_the_spectral_radius() {
        let stable = DMatrix::from_row_slice(2, 2, &[0.5, 0.1, 0.0, -0.3]);
        let (rho, ok) = poincare_stability(&stable);
        assert!(ok && (rho - 0.5).abs() < 1e-9, "rho {rho}");
        let unstable = DMatrix::from_row_slice(2, 2, &[1.2, 0.0, 0.0, 0.4]);
        let (rho2, ok2) = poincare_stability(&unstable);
        assert!(!ok2 && (rho2 - 1.2).abs() < 1e-9, "rho {rho2}");
    }

    /// The certificate reproduces the published design rule. With an expansive reset the admissible
    /// rate is bounded, and evaluating the certificate exactly at `ε̄` puts `χ` at one — which is what
    /// makes `ε̄` the threshold rather than a heuristic.
    #[test]
    fn certificate_design_rule_is_the_threshold() {
        let p = DMatrix::identity(2, 2);
        let mu_target = 2.5f64; // an expansive impact
        let delta = DMatrix::from_row_slice(2, 2, &[mu_target, 0.0, 0.0, 0.4]);
        let (c3, t) = (1.0, 0.4);

        let cert = hybrid_certificate(&p, &delta, c3, t, 0.2).expect("P is positive definite");
        assert!((cert.mu_sq.sqrt() - mu_target).abs() < 1e-9, "mu {}", cert.mu_sq.sqrt());
        let eb = cert.eps_bar.expect("expansive reset has a finite eps_bar");
        let formula = c3 * t / (2.0 * mu_target.ln());
        assert!((eb - formula).abs() < 1e-12, "design rule wrong: {eb} vs {formula}");

        // at the boundary chi == 1 exactly; inside it certifies, outside it does not
        let at = hybrid_certificate(&p, &delta, c3, t, eb).unwrap();
        assert!((at.chi - 1.0).abs() < 1e-9, "chi at eps_bar should be 1, got {}", at.chi);
        assert!(hybrid_certificate(&p, &delta, c3, t, eb * 0.5).unwrap().certified, "should certify below eps_bar");
        assert!(!hybrid_certificate(&p, &delta, c3, t, eb * 2.0).unwrap().certified, "should not certify above eps_bar");
        eprintln!("hybrid certificate: mu {:.3}, eps_bar {:.4}, chi(eps_bar) {:.6}", cert.mu_sq.sqrt(), eb, at.chi);
    }

    /// The failure the continuous-phase condition cannot see: a reset that is non-expansive in the
    /// Lyapunov metric needs no bound on the rate, while an expansive one does. The certificate
    /// distinguishes them; a continuous-only check cannot, because it never looks at the reset.
    #[test]
    fn a_contracting_reset_needs_no_rate_bound() {
        let p = DMatrix::identity(2, 2);
        let contracting = DMatrix::from_row_slice(2, 2, &[0.6, 0.0, 0.0, 0.3]);
        let cert = hybrid_certificate(&p, &contracting, 1.0, 0.4, 50.0).unwrap();
        assert!(cert.eps_bar.is_none(), "a non-expansive impact admits every rate");
        assert!(cert.certified, "a contracting reset with any decay must certify");
    }
}

#[cfg(test)]
mod robot_tests {
    use super::*;
    use crate::{quadruped, tree_floating_mass_matrix, whole_body_contact_jacobian, whole_body_forward_kinematics, LinkInertia};
    use nalgebra::{Isometry3, Matrix3, Vector3};

    /// The impact machinery on the real quadruped rather than a toy matrix: a foot strikes the ground
    /// with the body descending, and the plastic impact must stop that foot dead while removing energy
    /// from the whole floating body. This is the reset a gait certificate is actually about.
    #[test]
    fn foot_strike_on_the_quadruped_stops_the_foot_and_dissipates() {
        let (joints, inertia, parent, feet) = quadruped();
        let n = joints.len();
        let bi = LinkInertia { mass: 8.0, com: Vector3::zeros(), inertia: Matrix3::from_diagonal(&Vector3::new(0.08, 0.08, 0.12)) };
        let q = vec![0.0; n];
        let base = Isometry3::translation(0.0, 0.0, 0.60);
        let world = whole_body_forward_kinematics(&joints, &parent, base, &q);
        let m = tree_floating_mass_matrix(&joints, &inertia, &parent, &bi, &q);

        // one foot reaches the ground; its Jacobian is the guard's constraint
        let (body, off, _mu) = feet[0];
        let jc = whole_body_contact_jacobian(&joints, &parent, &world, base, Some(body), off);

        // the body is falling and drifting forward when the foot lands
        let mut v_minus = DVector::zeros(6 + n);
        v_minus[3] = 0.4; // forward
        v_minus[5] = -1.2; // descending
        let v_plus = plastic_impact(&m, &v_minus, &jc);

        let foot_before = (&jc * &v_minus).norm();
        let foot_after = (&jc * &v_plus).norm();
        let ke = |v: &DVector<f64>| 0.5 * (v.transpose() * &m * v)[(0, 0)];
        eprintln!("quadruped foot strike: foot speed {foot_before:.4} -> {foot_after:.2e} m/s, kinetic energy {:.4} -> {:.4} J", ke(&v_minus), ke(&v_plus));
        assert!(foot_after < 1e-10, "the struck foot is still moving: {foot_after}");
        assert!(ke(&v_plus) < ke(&v_minus), "impact must remove energy");
        assert!(ke(&v_plus) >= 0.0, "kinetic energy went negative");

        // and the reset Jacobian is the linearisation of exactly that map
        let delta = plastic_impact_jacobian(&m, &jc);
        assert!((&delta * &v_minus - &v_plus).amax() < 1e-10, "reset Jacobian disagrees on the real body");

        // the impact expansion in the kinetic-energy metric: a projection can never expand it, so the
        // quadruped's own foot strike is non-expansive and imposes no bound on the rate parameter
        let mu_sq = impact_expansion(&m, &delta).expect("mass matrix is positive definite");
        let cert = hybrid_certificate(&m, &delta, 1.0, 0.4, 5.0).expect("certificate computable");
        eprintln!("quadruped impact expansion mu^2 = {mu_sq:.6} (kinetic-energy metric), eps_bar = {:?}", cert.eps_bar);
        assert!(mu_sq <= 1.0 + 1e-9, "a plastic impact cannot expand the kinetic-energy metric: {mu_sq}");
        assert!(cert.eps_bar.is_none() && cert.certified, "a non-expansive impact should certify at any rate");
    }
}

// ------------------------------------------------------------------------------------------------
// Transverse coordinates — where a gait certificate actually lives.
//
// A periodic orbit is never asymptotically stable as a point set in the usual sense: perturb along
// the direction of travel and the trajectory stays on the orbit, merely shifted in phase. The
// monodromy matrix therefore always carries a trivial unit eigenvalue along the flow, and any
// Lyapunov function on the full state must be flat in that direction. Orbital stability is a
// statement about the *transverse* directions only, so the metric that decides it has to live there.
//
// This matters concretely for an impact. Measured in the kinetic-energy metric a plastic impact is a
// projection and so can never expand: mu^2 = 1 exactly, and the certificate reports no bound at all.
// Measured transverse to the orbit the same impact can and does expand, which is where a real bound
// on the rate parameter comes from.
// ------------------------------------------------------------------------------------------------

/// An orthonormal basis for the subspace transverse to `f`, as an `n × (n−1)` matrix whose columns
/// span `{v : fᵀv = 0}`. Built by Householder reflection, so it is well-conditioned for any non-zero
/// `f`. Returns `None` if `f` is (numerically) zero, where "transverse" has no meaning.
pub fn transverse_basis(f: &DVector<f64>) -> Option<DMatrix<f64>> {
    let n = f.len();
    let nf = f.norm();
    if nf < 1e-12 || n < 2 {
        return None;
    }
    // Householder H = I − 2wwᵀ maps f/|f| onto ∓e₁, so columns 1.. of H span f's orthogonal
    // complement. Take v = u + sign(u₀)e₁, never u − e₁: the latter cancels catastrophically when f
    // already points along e₁, which is precisely the common case.
    let mut u = f / nf;
    let sign = if u[0] >= 0.0 { 1.0 } else { -1.0 };
    u[0] += sign;
    let un = u.norm();
    let h = if un < 1e-12 {
        DMatrix::identity(n, n)
    } else {
        let w = u / un;
        DMatrix::identity(n, n) - 2.0 * &w * w.transpose()
    };
    Some(h.columns(1, n - 1).into_owned())
}

/// Restrict a full-state map to the transverse subspace: `Π_⊥ = Zᵀ M Z`, with `Z` from
/// [`transverse_basis`]. For a monodromy matrix this removes the trivial unit eigenvalue along the
/// flow and leaves exactly the eigenvalues that decide orbital stability.
pub fn transverse_restriction(m: &DMatrix<f64>, z: &DMatrix<f64>) -> DMatrix<f64> {
    z.transpose() * m * z
}

/// Build a **transverse Lyapunov metric** for a periodic orbit: the `P ≻ 0` solving
/// `Π_⊥ᵀ P Π_⊥ − P = −Q` for the transverse monodromy. Exists exactly when the orbit is orbitally
/// stable, so a `Some` here is itself a stability certificate, and the `P` it returns is the metric
/// in which an impact's expansion should be measured.
pub fn transverse_metric(monodromy: &DMatrix<f64>, f_on_orbit: &DVector<f64>) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
    let z = transverse_basis(f_on_orbit)?;
    let pi = transverse_restriction(monodromy, &z);
    let k = pi.nrows();
    let p = crate::solve_lyapunov_discrete(&pi, &DMatrix::identity(k, k))?;
    // the Stein solution is only a metric if it is positive definite
    let sym = (&p + p.transpose()) * 0.5;
    if sym.clone().symmetric_eigen().eigenvalues.iter().any(|&l| l <= 1e-12) {
        return None;
    }
    Some((sym, z))
}

#[cfg(test)]
mod saltation_composition_tests {
    use super::*;

    /// The bouncing ball has an analytic return map, so the composed monodromy can be checked against
    /// a closed form rather than against another numerical scheme. Apex-to-apex with restitution `e`:
    /// the next apex height is `e²` times the last, so the true map derivative is exactly `e²`.
    #[test]
    fn composed_monodromy_matches_the_analytic_bouncing_ball() {
        let (g, e, h0) = (9.81_f64, 0.8_f64, 1.0_f64);
        // fall from apex h0: impact speed v = sqrt(2 g h0), time t1 = v/g
        let v_impact = (2.0 * g * h0).sqrt();
        let t1 = v_impact / g;
        let v_after = e * v_impact;
        let t2 = v_after / g; // rise back to the new apex

        // state (h, v); flow ḣ = v, v̇ = −g, so A = [[0,1],[0,0]] and Φ(t) = [[1,t],[0,1]]
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let phi1 = flow_jacobian(&a, t1);
        let phi2 = flow_jacobian(&a, t2);
        let reset = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -e]);

        let ev = HybridEvent {
            flow_jacobian: phi1,
            reset_jacobian: reset.clone(),
            guard_normal: DVector::from_row_slice(&[1.0, 0.0]), // h = 0
            f_minus: DVector::from_row_slice(&[-v_impact, -g]),
            f_plus: DVector::from_row_slice(&[v_after, -g]),
        };
        let mono = compose_monodromy(std::slice::from_ref(&ev), &phi2).expect("transversal impact");

        // The full monodromy of an autonomous system carries the along-flow direction, so restrict it
        // to the Poincaré section first — the two pieces of this module composing. At the apex the flow
        // is (0, −g), whose orthogonal complement is exactly the section's tangent {(dh, 0)}.
        let f_apex = DVector::from_row_slice(&[0.0, -g]);
        let z = transverse_basis(&f_apex).expect("the apex section is transverse to the flow");
        let (rho, stable) = poincare_stability(&transverse_restriction(&mono, &z));
        eprintln!("bouncing ball: section monodromy rho = {rho:.6}, analytic e^2 = {:.6}, stable = {stable}", e * e);
        assert!((rho - e * e).abs() < 1e-6, "section monodromy {rho} should be e^2 = {}", e * e);
        assert!(stable, "a lossy bounce must contract");

        // the same composition with the reset alone, no saltation term, gets it wrong
        let naive = &phi2 * &reset * flow_jacobian(&a, t1);
        let (rho_naive, _) = poincare_stability(&transverse_restriction(&naive, &z));
        eprintln!("  without the saltation term: rho = {rho_naive:.6}  (out by {:.2}x)", rho_naive / (e * e));
        assert!((rho_naive - e * e).abs() > 1e-3, "the naive composition should be visibly wrong, got {rho_naive}");
    }

    /// A perfectly elastic bounce neither grows nor decays: the composed monodromy must land exactly on
    /// the unit circle. This is the boundary case where an error of any size would show.
    #[test]
    fn an_elastic_bounce_is_exactly_marginal() {
        let (g, h0) = (9.81_f64, 0.5_f64);
        let v = (2.0 * g * h0).sqrt();
        let t = v / g;
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 0.0, 0.0]);
        let ev = HybridEvent {
            flow_jacobian: flow_jacobian(&a, t),
            reset_jacobian: DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -1.0]),
            guard_normal: DVector::from_row_slice(&[1.0, 0.0]),
            f_minus: DVector::from_row_slice(&[-v, -g]),
            f_plus: DVector::from_row_slice(&[v, -g]),
        };
        let mono = compose_monodromy(std::slice::from_ref(&ev), &flow_jacobian(&a, t)).unwrap();
        let z = transverse_basis(&DVector::from_row_slice(&[0.0, -g])).unwrap();
        let (rho, _) = poincare_stability(&transverse_restriction(&mono, &z));
        eprintln!("elastic bounce: section rho = {rho:.9} (must be 1)");
        assert!((rho - 1.0).abs() < 1e-9, "an elastic bounce must be exactly marginal, got {rho}");
    }

    /// A non-transversal (grazing) crossing has no linearisation, and the composition says so instead of
    /// returning a number.
    #[test]
    fn grazing_contact_has_no_monodromy() {
        let a = DMatrix::identity(2, 2);
        let ev = HybridEvent {
            flow_jacobian: DMatrix::identity(2, 2),
            reset_jacobian: DMatrix::identity(2, 2),
            guard_normal: DVector::from_row_slice(&[1.0, 0.0]),
            f_minus: DVector::from_row_slice(&[0.0, 1.0]), // moves along the guard, not through it
            f_plus: DVector::from_row_slice(&[0.0, 1.0]),
        };
        assert!(compose_monodromy(std::slice::from_ref(&ev), &a).is_none(), "grazing must refuse");
    }
}

#[cfg(test)]
mod transverse_tests {
    use super::*;

    /// The basis really is orthonormal and really is transverse.
    #[test]
    fn transverse_basis_is_orthonormal_and_orthogonal_to_the_flow() {
        for f in [vec![1.0, 0.0, 0.0], vec![0.3, -1.2, 0.7, 2.0], vec![-1.0, -1.0]] {
            let fv = DVector::from_row_slice(&f);
            let z = transverse_basis(&fv).expect("non-zero flow");
            assert_eq!(z.ncols(), fv.len() - 1);
            let gram = z.transpose() * &z;
            assert!((gram - DMatrix::identity(z.ncols(), z.ncols())).amax() < 1e-12, "not orthonormal");
            assert!((z.transpose() * &fv).amax() < 1e-12, "not orthogonal to the flow");
        }
    }

    /// The trivial eigenvalue along the flow is exactly what the restriction removes: a monodromy with
    /// a unit eigenvalue along `f` and contracting transverse directions is unstable by the full
    /// spectral radius (which reads 1) and stable by the transverse one.
    #[test]
    fn transverse_restriction_removes_the_trivial_eigenvalue() {
        // eigenvector f with eigenvalue 1; the orthogonal complement contracts at 0.5 and 0.25
        let f = DVector::from_row_slice(&[1.0, 0.0, 0.0]);
        let mono = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25]);
        let (rho_full, stable_full) = poincare_stability(&mono);
        let z = transverse_basis(&f).unwrap();
        let pi = transverse_restriction(&mono, &z);
        let (rho_t, stable_t) = poincare_stability(&pi);
        eprintln!("full monodromy rho = {rho_full:.4} (stable {stable_full}); transverse rho = {rho_t:.4} (stable {stable_t})");
        assert!((rho_full - 1.0).abs() < 1e-9 && !stable_full, "the flow direction must show as marginal");
        assert!(rho_t < 0.6 && stable_t, "the transverse dynamics are the ones that decide: {rho_t}");
    }

    /// The transverse metric exists exactly when the orbit is orbitally stable, and in that metric an
    /// expansive reset produces a real bound on the rate parameter — which is the whole point, and
    /// what the kinetic-energy metric cannot show.
    #[test]
    fn transverse_metric_certifies_and_bounds_the_rate() {
        let f = DVector::from_row_slice(&[1.0, 0.0, 0.0]);
        let mono = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.25]);
        let (p, z) = transverse_metric(&mono, &f).expect("orbitally stable orbit has a transverse metric");
        assert_eq!(p.nrows(), 2, "the metric lives on the transverse subspace");

        // a reset that expands one transverse direction
        let reset_full = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.8, 0.0, 0.0, 0.0, 0.9]);
        let reset_t = transverse_restriction(&reset_full, &z);
        let cert = hybrid_certificate(&p, &reset_t, 1.0, 0.4, 0.15).expect("computable");
        eprintln!("transverse certificate: mu = {:.4}, eps_bar = {:?}, chi = {:.4}", cert.mu_sq.sqrt(), cert.eps_bar, cert.chi);
        assert!(cert.mu_sq > 1.0, "an expansive reset must show mu > 1 in the transverse metric");
        let eb = cert.eps_bar.expect("expansive reset gives a finite rate bound");
        assert!(eb.is_finite() && eb > 0.0, "eps_bar {eb}");
        assert!(hybrid_certificate(&p, &reset_t, 1.0, 0.4, eb * 0.5).unwrap().certified, "must certify below the bound");
        assert!(!hybrid_certificate(&p, &reset_t, 1.0, 0.4, eb * 2.0).unwrap().certified, "must not certify above it");
    }

    /// An orbitally *unstable* orbit has no transverse metric: the Stein equation has no positive
    /// definite solution. The certificate refuses rather than returning something meaningless.
    #[test]
    fn an_unstable_orbit_has_no_transverse_metric() {
        let f = DVector::from_row_slice(&[1.0, 0.0, 0.0]);
        let mono = DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.3, 0.0, 0.0, 0.0, 0.5]);
        assert!(transverse_metric(&mono, &f).is_none(), "an expanding transverse direction admits no metric");
    }
}
