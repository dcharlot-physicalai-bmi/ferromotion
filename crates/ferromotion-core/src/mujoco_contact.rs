//! **MuJoCo's soft-constraint contact, as MuJoCo computes it** — the compliant rung of the contact-fidelity
//! ladder in the form 3.6 million installs a month actually run, measured against `mj_forward`.
//!
//! MuJoCo does not apply a spring-damper force. It solves, each step, for the constraint force `f` that
//! minimises `½ fᵀ(A + R)f + fᵀ(J·a₀ − a_ref)` over the friction cone, where `A = J M⁻¹ Jᵀ` is the inverse
//! inertia seen by the constraints, `R` a diagonal **regulariser** that makes the constraint soft, `a₀` the
//! unconstrained acceleration and `a_ref` a **reference acceleration** the constraint would like to have
//! (Todorov, *Convex and analytically-invertible dynamics with contacts and constraints*, ICRA 2014). The
//! acceleration is then `a = a₀ + M⁻¹Jᵀf`. Everything soft about it lives in `R` and `a_ref`, and both are
//! set by the two five- and two-number vectors every MJCF file carries, `solimp` and `solref`:
//!
//! * **impedance** `d(r) ∈ (0, 1)` from `solimp = (d₀, d_width, width, midpoint, power)`: a sigmoid of
//!   `x = |pos − margin| / width` — `y = a·xᵖ` below the midpoint, `1 − b·(1−x)ᵖ` above it — scaled to
//!   `d₀ + y·(d_width − d₀)`, saturating at `d_width` past `width` ([`mujoco_impedance`]);
//! * **stiffness and damping** from `solref = (timeconst, dampratio)`: `B = 2/(d_width·timeconst)`,
//!   `K = 1/(d_width²·timeconst²·dampratio²)`; or, when both are negative, directly `K = −solref₀/d_width²`,
//!   `B = −solref₁/d_width` ([`mujoco_kbip`]);
//! * **reference** `a_ref = −B·v − K·d·(pos − margin)` per row;
//! * **regulariser** `R = (1 − d)/d · Ā` where `Ā` is not the true diagonal of `A` but MuJoCo's
//!   *approximation* of it from per-body inverse weights computed once at `qpos0`
//!   ([`mujoco_diag_approx`]); for a pyramidal cone every facet row then takes the common
//!   `R = 2·μ²·(1 − d)/d · (Ā_n + μ²·Ā_n)` so that its friction impedance matches the elliptic model.
//!
//! All of that was read from `engine_core_constraint.c` (`getimpedance`, `mj_makeImpedance`,
//! `mj_diagApprox`, `mj_referenceConstraint`) and `engine_setconst.c` (`body_invweight0`), and every number
//! in the tests below is MuJoCo 3.13.0's own `efc_R`, `efc_aref`, `efc_force` and `qacc` on a sphere
//! pressed into a plane — at rest, closing, opening, with custom and with negative `solref`, frictionless and
//! with the default pyramidal cone at rest and sliding. The Euler integrator's semantics are what is
//! reproduced; MuJoCo's implicit integrators additionally divide `R` and `a_ref` by an implicit row factor
//! (`mj_isMetric`), which is not modelled here.
//!
//! What this is for: [`crate::contact_law_residuals`] can now be asked of MuJoCo's own model — a solver
//! residual is not a law residual, and this is the solver the field's results are computed with.

use nalgebra::{DMatrix, DVector};

/// MuJoCo's `solimp` — `(d₀, d_width, width, midpoint, power)`. MJCF default `0.9 0.95 0.001 0.5 2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolImp {
    pub d0: f64,
    pub d_width: f64,
    pub width: f64,
    pub midpoint: f64,
    pub power: f64,
}

impl Default for SolImp {
    fn default() -> Self {
        Self { d0: 0.9, d_width: 0.95, width: 0.001, midpoint: 0.5, power: 2.0 }
    }
}

/// MuJoCo's `solref` — `(timeconst, dampratio)`, or `(−stiffness, −damping)` when both are negative. MJCF
/// default `0.02 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolRef(pub f64, pub f64);

impl Default for SolRef {
    fn default() -> Self {
        Self(0.02, 1.0)
    }
}

const MINVAL: f64 = 1e-15;

/// `getimpedance`: the impedance `d` and its derivative `d'` at constraint position `pos` (negative =
/// penetrating for a contact) and `margin`.
pub fn mujoco_impedance(s: &SolImp, pos: f64, margin: f64) -> (f64, f64) {
    if s.d0 == s.d_width || s.width <= MINVAL {
        return (0.5 * (s.d0 + s.d_width), 0.0);
    }
    let mut x = (pos - margin) / s.width;
    let mut sgn = 1.0;
    if x < 0.0 {
        x = -x;
        sgn = -1.0;
    }
    if x >= 1.0 || x <= 0.0 {
        return (if x >= 1.0 { s.d_width } else { s.d0 }, 0.0);
    }
    let (y, yp) = if s.power == 1.0 {
        (x, 1.0)
    } else if x <= s.midpoint {
        let a = 1.0 / s.midpoint.powf(s.power - 1.0);
        (a * x.powf(s.power), s.power * a * x.powf(s.power - 1.0))
    } else {
        let b = 1.0 / (1.0 - s.midpoint).powf(s.power - 1.0);
        (1.0 - b * (1.0 - x).powf(s.power), s.power * b * (1.0 - x).powf(s.power - 1.0))
    };
    (s.d0 + y * (s.d_width - s.d0), yp * sgn * (s.d_width - s.d0) / s.width)
}

/// MuJoCo's `efc_KBIP` for one row: `[K, B, impedance, impedance']`.
pub fn mujoco_kbip(r: &SolRef, s: &SolImp, pos: f64, margin: f64) -> [f64; 4] {
    let (imp, imp_p) = mujoco_impedance(s, pos, margin);
    let k = if r.0 > 0.0 { 1.0 / (s.d_width * s.d_width * r.0 * r.0 * r.1 * r.1).max(MINVAL) } else { -r.0 / (s.d_width * s.d_width).max(MINVAL) };
    let b = if r.1 > 0.0 { 2.0 / (s.d_width * r.0).max(MINVAL) } else { -r.1 / s.d_width.max(MINVAL) };
    [k, b, imp, imp_p]
}

/// A body's `body_invweight0` — the mean diagonal of its inverse spatial inertia `J M⁻¹ Jᵀ` at the body's
/// centre of mass, translation and rotation, evaluated at `qpos0`. For a free rigid body that is simply
/// `(1/m, mean(1/I_principal))`; for a link in a tree, compute `J_com M⁻¹ J_comᵀ` at `qpos0` and average the
/// two 3×3 diagonals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InvWeight {
    pub tran: f64,
    pub rot: f64,
}

impl InvWeight {
    /// A free rigid body: `1/m` and the mean of the three principal inverse inertias.
    pub fn free_body(mass: f64, principal_inertia: [f64; 3]) -> Self {
        let rot = (1.0 / principal_inertia[0] + 1.0 / principal_inertia[1] + 1.0 / principal_inertia[2]) / 3.0;
        Self { tran: 1.0 / mass.max(MINVAL), rot }
    }
    /// A body welded to the world (static or mocap): MuJoCo gives it zero inverse weight.
    pub const STATIC: Self = Self { tran: 0.0, rot: 0.0 };
}

/// `mj_diagApprox` for one contact: the approximate inverse inertia of each of its rows, from the two
/// bodies' inverse weights. Frictionless: one value; pyramidal: `2·(condim−1)` values, one pair per friction
/// direction `k`, each `tran + μ_k²·(tran | rot)`.
pub fn mujoco_diag_approx(a: InvWeight, b: InvWeight, condim: usize, friction: &[f64], pyramidal: bool) -> Vec<f64> {
    let (tran, rot) = (a.tran + b.tran, a.rot + b.rot);
    if condim == 1 {
        return vec![tran];
    }
    if pyramidal {
        let mut out = Vec::with_capacity(2 * (condim - 1));
        for j in 0..condim - 1 {
            let fri = friction[j];
            let v = tran + fri * fri * if j < 2 { tran } else { rot };
            out.push(v);
            out.push(v);
        }
        out
    } else {
        (0..condim).map(|j| if j < 3 { tran } else { rot }).collect()
    }
}

/// One contact between two bodies, as MuJoCo's `mjContact` carries it into the constraint solver.
#[derive(Clone, Debug)]
pub struct MjContact {
    /// Rows `normal, t1, t2` of the contact Jacobian (relative velocity of body 2 with respect to body 1, in
    /// the contact frame, per generalised velocity), `3 × nv`; only the first row is used when `condim == 1`.
    pub jac: DMatrix<f64>,
    /// Signed distance, negative when penetrating (`mjContact.dist`).
    pub dist: f64,
    /// `mjContact.includemargin`: the margin at which the contact is active (`pos − margin` is what the
    /// impedance and reference see).
    pub margin: f64,
    /// `1` (frictionless) or `3` (sliding friction). Torsional and rolling rows (`4`, `6`) are not carried.
    pub condim: usize,
    /// `friction[0..2]` — the two tangential coefficients (MuJoCo copies `geom friction[0]` into both).
    pub friction: [f64; 2],
    pub solref: SolRef,
    pub solimp: SolImp,
    /// The two bodies' inverse weights, geom 1 then geom 2.
    pub invweight: [InvWeight; 2],
}

/// Everything MuJoCo derives per row and the force it solves for.
#[derive(Clone, Debug)]
pub struct MjContactSolve {
    /// Constraint Jacobian, `nefc × nv`.
    pub jac: DMatrix<f64>,
    /// `efc_R`.
    pub r: Vec<f64>,
    /// `efc_aref`.
    pub aref: Vec<f64>,
    /// `efc_KBIP` per row.
    pub kbip: Vec<[f64; 4]>,
    /// `efc_force` — the solved constraint forces (each row `≥ 0`).
    pub force: Vec<f64>,
    /// `qacc = a₀ + M⁻¹Jᵀf`.
    pub qacc: DVector<f64>,
    /// Largest complementarity residual of the projected Gauss–Seidel solve at exit.
    pub residual: f64,
}

/// **Assemble the rows and solve MuJoCo's dual problem** for a set of contacts on a system with mass matrix
/// `m` (`nv × nv`), unconstrained acceleration `a0` and `qvel`. `pyramidal` selects the cone (`true` is
/// MuJoCo's default `cone="pyramidal"`; the elliptic cone is not carried here) and `impratio` is
/// `<option impratio>` (default 1).
///
/// The solve is projected Gauss–Seidel on `min ½ fᵀ(A + R)f + fᵀ(J·a₀ − a_ref)`, `f ≥ 0`, which for
/// frictionless and pyramidal rows is exactly the box-constrained convex problem MuJoCo's own solvers converge
/// to; it runs until the complementarity residual is below `tol` or `max_iter` sweeps.
pub fn solve_contacts_mujoco(m: &DMatrix<f64>, a0: &DVector<f64>, qvel: &DVector<f64>, contacts: &[MjContact], pyramidal: bool, impratio: f64, tol: f64, max_iter: usize) -> Result<MjContactSolve, String> {
    let nv = m.nrows();
    if m.ncols() != nv || a0.len() != nv || qvel.len() != nv {
        return Err("mass matrix, a0 and qvel disagree on nv".into());
    }
    // --- rows: J, pos, margin, diagA, solver params
    let mut rows: Vec<DVector<f64>> = Vec::new();
    let mut pos = Vec::new();
    let mut margin = Vec::new();
    let mut diag_a = Vec::new();
    let mut params: Vec<(SolRef, SolImp)> = Vec::new();
    let mut groups: Vec<(usize, usize, bool, f64)> = Vec::new(); // (first row, nrows, pyramidal, mu)
    for c in contacts {
        if c.jac.ncols() != nv || c.jac.nrows() < if c.condim == 1 { 1 } else { 3 } {
            return Err("contact Jacobian has the wrong shape".into());
        }
        if c.condim != 1 && c.condim != 3 {
            return Err(format!("condim {} is not carried (1 or 3)", c.condim));
        }
        let jn = c.jac.row(0).transpose();
        let first = rows.len();
        let da = mujoco_diag_approx(c.invweight[0], c.invweight[1], c.condim, &c.friction, pyramidal);
        if c.condim == 1 {
            rows.push(jn);
            pos.push(c.dist);
            margin.push(c.margin);
            diag_a.push(da[0]);
            params.push((c.solref, c.solimp));
            groups.push((first, 1, false, 0.0));
        } else if pyramidal {
            // one pair of opposing pyramid edges per friction direction: J_n ± μ_k J_tk
            for k in 0..2 {
                let jt = c.jac.row(1 + k).transpose();
                rows.push(&jn + c.friction[k] * &jt);
                rows.push(&jn - c.friction[k] * &jt);
                for _ in 0..2 {
                    pos.push(c.dist);
                    margin.push(c.margin);
                    params.push((c.solref, c.solimp));
                }
                diag_a.push(da[2 * k]);
                diag_a.push(da[2 * k + 1]);
            }
            groups.push((first, 4, true, c.friction[0]));
        } else {
            return Err("the elliptic cone is not carried here; use pyramidal".into());
        }
    }
    let nefc = rows.len();
    // --- impedance, K, B, R (mj_makeImpedance), reference (mj_referenceConstraint)
    let mut kb = Vec::with_capacity(nefc);
    let mut r = Vec::with_capacity(nefc);
    for i in 0..nefc {
        let (solref, solimp) = params[i];
        let k = mujoco_kbip(&solref, &solimp, pos[i], margin[i]);
        r.push(((1.0 - k[2]) * diag_a[i] / k[2]).max(MINVAL));
        kb.push(k);
    }
    for &(first, n, pyr, mu) in &groups {
        if pyr {
            // R[1] = R[0]/impratio; mu of the regularised cone = mu·sqrt(R[1]/R[0]); Rpy = 2·mu²·R[0] for all
            let r1 = r[first] / impratio.max(MINVAL);
            let mu_reg = mu * (r1 / r[first]).sqrt();
            let rpy = 2.0 * mu_reg * mu_reg * r[first];
            for i in first..first + n {
                r[i] = rpy;
            }
        }
    }
    let mut aref = Vec::with_capacity(nefc);
    let mut jac = DMatrix::zeros(nefc, nv);
    for i in 0..nefc {
        jac.row_mut(i).copy_from(&rows[i].transpose());
        let vel = rows[i].dot(qvel);
        aref.push(-kb[i][1] * vel - kb[i][0] * kb[i][2] * (pos[i] - margin[i]));
    }
    // --- dual problem: (A + R) f = a_ref − J a₀ with f ≥ 0, by projected Gauss–Seidel
    let minv = m.clone().cholesky().ok_or("mass matrix is not positive definite")?;
    let minv_jt = minv.solve(&jac.transpose()); // nv × nefc
    let a = &jac * &minv_jt; // nefc × nefc
    let rhs: Vec<f64> = (0..nefc).map(|i| aref[i] - jac.row(i).dot(&a0.transpose())).collect();
    let mut f = vec![0.0; nefc];
    let mut residual = f64::INFINITY;
    for _ in 0..max_iter {
        residual = 0.0f64;
        for i in 0..nefc {
            let mut s = rhs[i];
            for j in 0..nefc {
                if j != i {
                    s -= a[(i, j)] * f[j];
                }
            }
            let fi = (s / (a[(i, i)] + r[i])).max(0.0);
            residual = residual.max((fi - f[i]).abs());
            f[i] = fi;
        }
        if residual < tol {
            break;
        }
    }
    let force = DVector::from_vec(f.clone());
    let qacc = a0 + &minv_jt * &force;
    Ok(MjContactSolve { jac, r, aref, kbip: kb, force: f, qacc, residual })
}

#[cfg(test)]
mod tests {
    //! Every expected number is MuJoCo 3.13.0's (`mj_forward` on a unit-mass, 0.1 m sphere on a plane, Euler,
    //! warmstart off, Newton solver at tolerance 1e-15) — `efc_R`, `efc_aref`, `efc_force`, `qacc`.
    use super::*;

    const R_BALL: f64 = 0.1;
    const M_BALL: f64 = 1.0;
    const I_BALL: f64 = 0.4 * M_BALL * R_BALL * R_BALL; // solid sphere

    /// The free-body system MuJoCo's free joint sees at identity orientation: qvel = [v_world; ω_body].
    fn ball() -> (DMatrix<f64>, DVector<f64>) {
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![M_BALL, M_BALL, M_BALL, I_BALL, I_BALL, I_BALL]));
        let a0 = DVector::from_vec(vec![0.0, 0.0, -9.81, 0.0, 0.0, 0.0]);
        (m, a0)
    }

    /// Contact of the ball (geom 2) with the world plane (geom 1): normal +z, tangents +y and −x as MuJoCo's
    /// frame `[0 0 1, 0 1 0, -1 0 0]` gives them. **MuJoCo puts the contact point halfway between the two
    /// surfaces** (`mjContact.pos`), so the lever arm is `R + dist/2`, not `R` — with `R` the sliding facet
    /// force came out 13.2086 against MuJoCo's 13.2429, a 0.26% that was entirely this half-penetration.
    fn contact(z: f64, condim: usize, solref: SolRef, solimp: SolImp) -> MjContact {
        let dist = z - R_BALL;
        let arm = R_BALL + dist / 2.0;
        let mut jac = DMatrix::zeros(3, 6);
        // velocity of the contact point on the ball: v + ω × r with r = (0, 0, −arm)
        // normal (z): v_z
        jac[(0, 2)] = 1.0;
        // t1 = +y: v_y + (ω × r)_y, (ω × r) = (ω_y·r_z − ω_z·r_y, ω_z·r_x − ω_x·r_z, ω_x·r_y − ω_y·r_x)
        jac[(1, 1)] = 1.0;
        jac[(1, 3)] = arm; // ω_z·0 − ω_x·(−arm)
        // t2 = −x: −(v_x + ω_y·r_z) = −v_x + arm·ω_y
        jac[(2, 0)] = -1.0;
        jac[(2, 4)] = arm;
        MjContact {
            jac,
            dist,
            margin: 0.0,
            condim,
            friction: [1.0, 1.0],
            solref,
            solimp,
            invweight: [InvWeight::STATIC, InvWeight::free_body(M_BALL, [I_BALL; 3])],
        }
    }

    fn solve(z: f64, vz: f64, vx: f64, condim: usize, solref: SolRef, solimp: SolImp) -> MjContactSolve {
        let (m, a0) = ball();
        let qvel = DVector::from_vec(vec![vx, 0.0, vz, 0.0, 0.0, 0.0]);
        solve_contacts_mujoco(&m, &a0, &qvel, &[contact(z, condim, solref, solimp)], true, 1.0, 1e-16, 10_000).unwrap()
    }

    fn close(a: f64, b: f64, tol: f64, what: &str) {
        assert!((a - b).abs() <= tol * b.abs().max(1.0), "{what}: {a} vs MuJoCo {b}");
    }

    #[test]
    fn impedance_and_kbip_are_mujocos() {
        let (d, _) = mujoco_impedance(&SolImp::default(), -1e-4, 0.0);
        close(d, 0.901, 1e-12, "impedance at 0.1 width");
        let (d, _) = mujoco_impedance(&SolImp::default(), -5e-4, 0.0);
        close(d, 0.925, 1e-12, "impedance at the midpoint");
        let (d, _) = mujoco_impedance(&SolImp::default(), -5e-3, 0.0);
        close(d, 0.95, 1e-12, "impedance saturated");
        let k = mujoco_kbip(&SolRef::default(), &SolImp::default(), -5e-4, 0.0);
        close(k[0], 2770.083102493075, 1e-12, "K");
        close(k[1], 105.26315789473685, 1e-12, "B");
        close(k[3], -99.99999999999979, 1e-9, "impedance derivative");
        let custom = mujoco_kbip(&SolRef(0.05, 0.5), &SolImp { d0: 0.8, d_width: 0.99, width: 0.01, midpoint: 0.3, power: 3.0 }, -5e-4, 0.0);
        close(custom[0], 1632.4864809713292, 1e-12, "custom K");
        close(custom[1], 40.4040404040404, 1e-12, "custom B");
        close(custom[2], 0.8002638888888889, 1e-12, "custom impedance");
        let direct = mujoco_kbip(&SolRef(-1000.0, -50.0), &SolImp::default(), -5e-4, 0.0);
        close(direct[0], 1108.03324099723, 1e-12, "direct K");
        close(direct[1], 52.631578947368425, 1e-12, "direct B");
    }

    #[test]
    fn frictionless_sphere_on_plane_matches_mujoco_across_a_penetration_ladder() {
        // (z, R, aref, force, qacc_z)
        let cases = [
            (0.0999, 1.098779134295e-01, 2.495844875346e-01, 9.063685623269e+00, -0.746314376731295),
            (0.0995, 8.108108108108e-02, 1.281163434903e+00, 1.025932617729e+01, 0.44932617728531987),
            (0.099, 5.263157894737e-02, 2.631578947368e+00, 1.181950000000e+01, 2.009500000000002),
            (0.095, 5.263157894737e-02, 1.315789473684e+01, 2.181950000000e+01, 12.009500000000013),
        ];
        for (z, r, aref, force, qacc) in cases {
            let s = solve(z, 0.0, 0.0, 1, SolRef::default(), SolImp::default());
            close(s.r[0], r, 1e-11, "R");
            close(s.aref[0], aref, 1e-11, "aref");
            close(s.force[0], force, 1e-11, "force");
            close(s.qacc[2], qacc, 1e-11, "qacc_z");
        }
        // MuJoCo activates a contact only when dist < margin, so a ball exactly touching (dist = 0) has no
        // constraint row at all and falls at g; passed here it would push (rhs = 9.81 > 0). The activation
        // test is the caller's, upstream of this solver, as it is in MuJoCo.
    }

    #[test]
    fn closing_and_opening_velocities_enter_through_the_reference() {
        let s = solve(0.0995, -0.3, 0.0, 1, SolRef::default(), SolImp::default());
        close(s.aref[0], 3.286011080332e+01, 1e-11, "closing aref");
        close(s.force[0], 3.946985249307e+01, 1e-11, "closing force");
        close(s.qacc[2], 29.6598524930748, 1e-11, "closing qacc");
        let s = solve(0.0995, 0.3, 0.0, 1, SolRef::default(), SolImp::default());
        close(s.aref[0], -3.029778393352e+01, 1e-11, "opening aref");
        assert_eq!(s.force[0], 0.0, "an opening contact carries no force");
        close(s.qacc[2], -9.81, 1e-12, "opening qacc");
    }

    #[test]
    fn custom_and_direct_solref_match_mujoco() {
        let s = solve(0.0995, -0.1, 0.0, 1, SolRef(0.05, 0.5), SolImp { d0: 0.8, d_width: 0.99, width: 0.01, midpoint: 0.3, power: 3.0 });
        close(s.r[0], 2.495878095767e-01, 1e-11, "custom R");
        close(s.aref[0], 4.693614030314e+00, 1e-11, "custom aref");
        close(s.force[0], 1.160671856684e+01, 1e-11, "custom force");
        close(s.qacc[2], 1.7967185668428245, 1e-11, "custom qacc");
        let s = solve(0.0995, -0.1, 0.0, 1, SolRef(-1000.0, -50.0), SolImp::default());
        close(s.aref[0], 5.775623268698e+00, 1e-11, "direct aref");
        close(s.force[0], 1.441670152355e+01, 1e-11, "direct force");
        close(s.qacc[2], 4.606701523545706, 1e-11, "direct qacc");
    }

    #[test]
    fn pyramidal_cone_at_rest_and_sliding_matches_mujoco() {
        // at rest: four equal facet forces whose normal components sum to the frictionless answer
        let s = solve(0.0995, 0.0, 0.0, 3, SolRef::default(), SolImp::default());
        assert_eq!(s.r.len(), 4);
        for i in 0..4 {
            close(s.r[i], 3.243243243243e-01, 1e-11, "pyramidal R");
            close(s.aref[i], 1.281163434903e+00, 1e-11, "pyramidal aref");
            close(s.force[i], 2.564831544321e+00, 1e-10, "pyramidal facet force at rest");
        }
        close(s.qacc[2], 0.44932617728532215, 1e-10, "resting qacc_z");
        assert!(s.qacc[0].abs() < 1e-12 && s.qacc[4].abs() < 1e-10);
        // sliding at +0.5 m/s in x: only the facet opposing the slide carries force; friction = μ f
        let s = solve(0.0995, 0.0, 0.5, 3, SolRef::default(), SolImp::default());
        close(s.force[2], 1.324290563555e+01, 1e-10, "active facet force");
        assert_eq!(s.force[0], 0.0);
        assert_eq!(s.force[1], 0.0);
        assert_eq!(s.force[3], 0.0);
        close(s.qacc[0], -13.24290563554995, 1e-10, "sliding qacc_x");
        close(s.qacc[2], 3.4329056355499494, 1e-10, "sliding qacc_z");
        close(s.qacc[4], 330.2449592865268, 1e-9, "sliding spin-up ω_y");
    }

    #[test]
    fn diag_approx_is_mujocos_two_side_sum() {
        let ball = InvWeight::free_body(1.0, [I_BALL; 3]);
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 1, &[1.0, 1.0], true), vec![1.0]);
        // pyramidal: tran + μ²·tran for the two sliding directions
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 3, &[1.0, 1.0], true), vec![2.0, 2.0, 2.0, 2.0]);
        // elliptic: tran for the first three rows
        assert_eq!(mujoco_diag_approx(InvWeight::STATIC, ball, 3, &[1.0, 1.0], false), vec![1.0, 1.0, 1.0]);
    }
}
