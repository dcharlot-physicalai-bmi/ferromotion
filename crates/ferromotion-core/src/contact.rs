//! Contact-implicit dynamics: unilateral contact resolved as a convex complementarity problem
//! (Anitescu / Stewart-Trinkle style). Each step solves for the non-negative normal impulses that
//! prevent penetration — `0 ≤ λ ⟂ (post-contact separation velocity) ≥ 0` — as a QP via `clarabel`.
//! This is the core primitive that contact-implicit trajectory optimization / MPC differentiate
//! through. Pure Rust → WASM-clean.

use clarabel::algebra::CscMatrix;
use clarabel::solver::{DefaultSettingsBuilder, DefaultSolver, IPSolver, SupportedConeT};
use nalgebra::{DMatrix, DVector};

/// One unilateral normal contact: `jn` maps generalized velocity → separation velocity along the
/// contact normal (a row of the contact Jacobian, length = dof), and `phi` is the current signed
/// gap (`≥ 0` = separated, `0` = touching).
#[derive(Clone, Debug)]
pub struct Contact {
    pub jn: DVector<f64>,
    pub phi: f64,
}

/// Result of a contact solve.
#[derive(Clone, Debug)]
pub struct ContactSolve {
    /// Non-negative normal impulses, one per contact.
    pub impulses: Vec<f64>,
    /// Post-contact generalized velocity `v⁺ = v_free + M⁻¹ Jᵀ λ`.
    pub v_next: DVector<f64>,
}

/// Symmetric matrix → upper-triangular CSC (clarabel wants `P` upper-triangular).
fn csc_upper(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.ncols();
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        for i in 0..=j {
            rowval.push(i);
            nzval.push(p[(i, j)]);
        }
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// `-I` (n×n) in CSC — the constraint matrix for `x ≥ 0` written as `-x ≤ 0`.
fn csc_neg_identity(n: usize) -> CscMatrix<f64> {
    let (mut colptr, mut rowval, mut nzval) = (vec![0usize], Vec::new(), Vec::new());
    for j in 0..n {
        rowval.push(j);
        nzval.push(-1.0);
        colptr.push(rowval.len());
    }
    CscMatrix::new(n, n, colptr, rowval, nzval)
}

/// `min ½ xᵀA x + bᵀx  s.t.  x ≥ 0` (A symmetric PSD). The KKT system of this QP is exactly the
/// contact LCP `0 ≤ x ⟂ (Ax + b) ≥ 0`.
fn solve_nonneg_qp(a: &DMatrix<f64>, b: &[f64]) -> Vec<f64> {
    let n = a.ncols();
    let p = csc_upper(a);
    let a_c = csc_neg_identity(n);
    let b_c = vec![0.0; n];
    let cones = [SupportedConeT::NonnegativeConeT(n)];
    let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
    let mut solver = DefaultSolver::new(&p, b, &a_c, &b_c, &cones, settings).unwrap();
    solver.solve();
    solver.solution.x.clone()
}

/// Resolve unilateral contacts for one step. `m` is the mass matrix (SPD), `v_free` the generalized
/// velocity after external forces but before contact, `dt` the timestep (used for gap stabilization).
/// Returns impulses `λ ≥ 0` and `v⁺` with `0 ≤ λ ⟂ (J v⁺ + φ/dt) ≥ 0` — no penetration, and impulses
/// only where a contact is closing or touching.
pub fn solve_contacts(m: &DMatrix<f64>, v_free: &DVector<f64>, contacts: &[Contact], dt: f64) -> ContactSolve {
    let k = contacts.len();
    if k == 0 {
        return ContactSolve { impulses: vec![], v_next: v_free.clone() };
    }
    let n = v_free.len();
    let minv = m.clone().try_inverse().expect("mass matrix invertible");
    let mut j = DMatrix::zeros(k, n);
    for (i, c) in contacts.iter().enumerate() {
        j.row_mut(i).copy_from(&c.jn.transpose());
    }
    // Delassus operator A = J M⁻¹ Jᵀ (PSD); b = J v_free + φ/dt (Baumgarte gap stabilization).
    let a = &j * &minv * j.transpose();
    let mut b = &j * v_free;
    for (i, c) in contacts.iter().enumerate() {
        b[i] += c.phi / dt;
    }
    let impulses = solve_nonneg_qp(&a, b.as_slice());
    let lam = DVector::from_row_slice(&impulses);
    let v_next = v_free + &minv * j.transpose() * lam;
    ContactSolve { impulses, v_next }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_mass_drops_and_rests_without_penetrating() {
        // 1-DoF vertical point mass (dof = height). Upward normal maps v → ż.
        let mass = 2.0;
        let m = DMatrix::from_element(1, 1, mass);
        let jn = DVector::from_row_slice(&[1.0]);
        let (g, dt) = (9.81, 0.005);
        let (mut z, mut v) = (1.0, 0.0);
        let (mut min_z, mut worst_comp) = (z, 0.0f64);

        for _ in 0..800 {
            let v_free = DVector::from_row_slice(&[v - g * dt]); // gravity impulse
            let contacts = [Contact { jn: jn.clone(), phi: z }];
            let sol = solve_contacts(&m, &v_free, &contacts, dt);
            v = sol.v_next[0];
            z += v * dt;
            min_z = min_z.min(z);
            // Complementarity: λ · (J·v⁺ + φ/dt) ≈ 0.
            let comp = sol.impulses[0] * (v + (z - v * dt) / dt);
            worst_comp = worst_comp.max(comp.abs());
        }

        assert!(min_z > -1e-3, "penetrated the floor: min z = {min_z}");
        assert!(z.abs() < 5e-3 && v.abs() < 0.1, "did not rest on the floor: z={z}, v={v}");
        assert!(worst_comp < 1e-2, "complementarity violated: {worst_comp}");
    }

    #[test]
    fn no_impulse_while_separated() {
        // Well above the floor and falling: the contact must stay inactive (λ = 0).
        let m = DMatrix::from_element(1, 1, 1.0);
        let contacts = [Contact { jn: DVector::from_row_slice(&[1.0]), phi: 1.0 }];
        let v_free = DVector::from_row_slice(&[-0.5]);
        let sol = solve_contacts(&m, &v_free, &contacts, 0.01);
        assert!(sol.impulses[0] < 1e-9, "spurious impulse while separated: {}", sol.impulses[0]);
        assert!((sol.v_next[0] - (-0.5)).abs() < 1e-9);
    }

    #[test]
    fn two_masses_on_a_shared_floor() {
        // Two independent vertical masses, both resting: each supports its own weight.
        let (m1, m2, g, dt) = (1.0, 3.0, 9.81, 0.005);
        let m = DMatrix::from_diagonal(&DVector::from_row_slice(&[m1, m2]));
        let contacts = [
            Contact { jn: DVector::from_row_slice(&[1.0, 0.0]), phi: 0.0 },
            Contact { jn: DVector::from_row_slice(&[0.0, 1.0]), phi: 0.0 },
        ];
        let v_free = DVector::from_row_slice(&[-g * dt, -g * dt]);
        let sol = solve_contacts(&m, &v_free, &contacts, dt);
        // Impulse balances weight: λ ≈ mᵢ·g·dt; post-contact velocity ≈ 0.
        assert!((sol.impulses[0] - m1 * g * dt).abs() < 1e-6, "λ1 = {}", sol.impulses[0]);
        assert!((sol.impulses[1] - m2 * g * dt).abs() < 1e-6, "λ2 = {}", sol.impulses[1]);
        assert!(sol.v_next.norm() < 1e-6, "masses should be at rest, v = {:?}", sol.v_next);
    }
}
