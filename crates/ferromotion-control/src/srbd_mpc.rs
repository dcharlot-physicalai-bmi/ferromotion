//! Single-Rigid-Body-Dynamics convex MPC — the modern legged/quadruped body controller (MIT-Cheetah
//! style). The robot is treated as one rigid body; the decision variables are the foot contact forces
//! over a horizon. State (13): `[θ(3), p(3), ω(3), v(3), g]` (gravity augmented so the dynamics are
//! affine→linear). Costs track a reference body state; **friction-cone (pyramid) constraints** keep each
//! stance foot's force physical, and swing feet are pinned to zero force. Condensed QP via `clarabel`
//! (pure Rust → WASM-clean). A real step beyond the point-mass LIPM `centroidal_mpc`.

use crate::qp::solve_qp;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

/// Number of state variables in the SRBD model.
pub const NX: usize = 13;

/// SRBD convex MPC over foot contact forces.
#[derive(Clone, Debug)]
pub struct SrbdMpc {
    pub mass: f64,
    pub inertia: Matrix3<f64>, // body/world inertia about the CoM (assumed ≈ world-aligned)
    pub feet: Vec<Vector3<f64>>, // foot positions relative to the CoM
    pub mu: f64,               // friction coefficient
    pub fz_max: f64,           // max normal force per stance foot
    pub horizon: usize,
    pub dt: f64,
    pub q: [f64; NX], // state tracking weights
    pub r_force: f64, // force regularization weight
}

impl SrbdMpc {
    fn skew(r: &Vector3<f64>) -> Matrix3<f64> {
        Matrix3::new(0.0, -r.z, r.y, r.z, 0.0, -r.x, -r.y, r.x, 0.0)
    }

    /// Discrete state-space (`Ad`, `Bd`) of the SRBD model, constant over the horizon (R ≈ world).
    fn dynamics(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let nf = self.feet.len();
        let u = 3 * nf;
        let mut a = DMatrix::<f64>::zeros(NX, NX);
        // θ̇ = ω
        a[(0, 6)] = 1.0; a[(1, 7)] = 1.0; a[(2, 8)] = 1.0;
        // ṗ = v
        a[(3, 9)] = 1.0; a[(4, 10)] = 1.0; a[(5, 11)] = 1.0;
        // v̇_z += g  (state[12] holds gravity accel, e.g. −9.81)
        a[(11, 12)] = 1.0;

        let mut b = DMatrix::<f64>::zeros(NX, u);
        let ig_inv = self.inertia.try_inverse().expect("inertia invertible");
        for (j, r) in self.feet.iter().enumerate() {
            let bw = ig_inv * Self::skew(r); // ω̇ += I⁻¹ (r × f)
            for rr in 0..3 {
                for cc in 0..3 {
                    b[(6 + rr, 3 * j + cc)] = bw[(rr, cc)];
                }
                b[(9 + rr, 3 * j + rr)] = 1.0 / self.mass; // v̇ += f/m
            }
        }

        let ad = DMatrix::identity(NX, NX) + &a * self.dt;
        let bd = &b * self.dt;
        (ad, bd)
    }

    /// Optimal contact forces for the current state. `stance[k][j]` = foot j in contact at step k
    /// (swing ⇒ that foot's force is constrained to zero). Returns the first step's forces (`3·nfeet`).
    pub fn control(&self, x0: &[f64], x_ref: &[f64], stance: &[Vec<bool>]) -> Vec<f64> {
        let nf = self.feet.len();
        let u = 3 * nf;
        let n = self.horizon;
        let (ad, bd) = self.dynamics();

        // Condensed prediction X = Sx·x0 + Su·U over x₁…x_N.
        let mut apow = vec![DMatrix::<f64>::identity(NX, NX)];
        for _ in 0..n {
            let nxt = &ad * apow.last().unwrap();
            apow.push(nxt);
        }
        let mut sx = DMatrix::zeros(n * NX, NX);
        let mut su = DMatrix::zeros(n * NX, n * u);
        for k in 0..n {
            sx.view_mut((k * NX, 0), (NX, NX)).copy_from(&apow[k + 1]);
            for j in 0..=k {
                let blk = &apow[k - j] * &bd;
                su.view_mut((k * NX, j * u), (NX, u)).copy_from(&blk);
            }
        }

        // Cost: Σ (xₖ−xref)ᵀ Q (xₖ−xref) + uₖᵀ R uₖ.
        let mut qbar = DMatrix::zeros(n * NX, n * NX);
        let mut rbar = DMatrix::zeros(n * u, n * u);
        for k in 0..n {
            for i in 0..NX {
                qbar[(k * NX + i, k * NX + i)] = self.q[i];
            }
            for i in 0..u {
                rbar[(k * u + i, k * u + i)] = self.r_force;
            }
        }
        let xref = DVector::from_fn(n * NX, |i, _| x_ref[i % NX]);
        let x0v = DVector::from_row_slice(x0);
        let sut_q = su.transpose() * &qbar;
        let h = &sut_q * &su + &rbar;
        let h = 0.5 * (&h + &h.transpose());
        let f = &sut_q * (&sx * &x0v - &xref);

        // Friction pyramid + normal-force bounds, per foot per step (6 rows each).
        let rows = 6 * nf * n;
        let mut a_c = DMatrix::<f64>::zeros(rows, n * u);
        let mut b_c = DVector::<f64>::zeros(rows);
        let mut row = 0;
        for k in 0..n {
            for j in 0..nf {
                let (fx, fy, fz) = (k * u + 3 * j, k * u + 3 * j + 1, k * u + 3 * j + 2);
                let in_stance = stance.get(k).and_then(|s| s.get(j)).copied().unwrap_or(true);
                let fzmax = if in_stance { self.fz_max } else { 0.0 };
                // |fx| ≤ μ fz  →  fx − μ fz ≤ 0 ,  −fx − μ fz ≤ 0   (same for fy)
                a_c[(row, fx)] = 1.0; a_c[(row, fz)] = -self.mu; row += 1;
                a_c[(row, fx)] = -1.0; a_c[(row, fz)] = -self.mu; row += 1;
                a_c[(row, fy)] = 1.0; a_c[(row, fz)] = -self.mu; row += 1;
                a_c[(row, fy)] = -1.0; a_c[(row, fz)] = -self.mu; row += 1;
                // 0 ≤ fz ≤ fzmax
                a_c[(row, fz)] = 1.0; b_c[row] = fzmax; row += 1;
                a_c[(row, fz)] = -1.0; b_c[row] = 0.0; row += 1;
            }
        }

        let f_slice: Vec<f64> = f.iter().cloned().collect();
        let b_slice: Vec<f64> = b_c.iter().cloned().collect();
        let sol = solve_qp(&h, &f_slice, &a_c, &b_slice);
        sol[0..u].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadruped() -> SrbdMpc {
        SrbdMpc {
            mass: 12.0,
            inertia: Matrix3::from_diagonal(&Vector3::new(0.4, 0.5, 0.6)),
            feet: vec![
                Vector3::new(0.2, 0.15, -0.3),
                Vector3::new(0.2, -0.15, -0.3),
                Vector3::new(-0.2, 0.15, -0.3),
                Vector3::new(-0.2, -0.15, -0.3),
            ],
            mu: 0.6,
            fz_max: 250.0,
            horizon: 10,
            dt: 0.03,
            q: [200.0, 200.0, 200.0, 200.0, 200.0, 400.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            r_force: 1e-4,
        }
    }

    #[test]
    fn stabilizes_standing_balance_with_physical_forces() {
        let mpc = quadruped();
        let nf = mpc.feet.len();
        let g = -9.81;
        let mut x_ref = [0.0; NX];
        x_ref[12] = g;
        let stance = vec![vec![true; nf]; mpc.horizon];
        let (ad, bd) = mpc.dynamics();

        // Perturbed initial body state: tilted + displaced.
        let mut x = [0.15, -0.1, 0.05, 0.03, -0.02, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, g];

        let mut last_forces = vec![0.0; 3 * nf];
        for _ in 0..120 {
            let forces = mpc.control(&x, &x_ref, &stance);
            // Integrate the SRBD model one step with the applied first-step forces.
            let xv = DVector::from_row_slice(&x);
            let uv = DVector::from_row_slice(&forces);
            let xn = &ad * xv + &bd * uv;
            for i in 0..NX {
                x[i] = xn[i];
            }
            last_forces = forces;
        }

        // Converged to the reference body state (attitude + position + velocities → 0).
        let state_err: f64 = (0..12).map(|i| (x[i] - x_ref[i]).powi(2)).sum::<f64>().sqrt();
        assert!(state_err < 1e-2, "did not stabilize: state error {state_err}, x = {x:?}");

        // Forces support the body weight and lie inside the friction cone.
        let total_fz: f64 = (0..nf).map(|j| last_forces[3 * j + 2]).sum();
        assert!((total_fz - mpc.mass * 9.81).abs() < 1.0, "Σfz {total_fz} != m·g {}", mpc.mass * 9.81);
        for j in 0..nf {
            let (fx, fy, fz) = (last_forces[3 * j], last_forces[3 * j + 1], last_forces[3 * j + 2]);
            assert!(fz >= -1e-6 && fz <= mpc.fz_max + 1e-6, "fz out of range: {fz}");
            assert!(fx.abs() <= mpc.mu * fz + 1e-6, "friction x violated: |{fx}| > μ·{fz}");
            assert!(fy.abs() <= mpc.mu * fz + 1e-6, "friction y violated: |{fy}| > μ·{fz}");
        }
    }

    #[test]
    fn swing_feet_carry_no_force() {
        let mpc = quadruped();
        let g = -9.81;
        let mut x_ref = [0.0; NX];
        x_ref[12] = g;
        // Feet 2 and 3 swing (no contact) the whole horizon.
        let stance = vec![vec![true, true, false, false]; mpc.horizon];
        let mut x = [0.0; NX];
        x[12] = g;
        x[3] = 0.02; // small displacement so the controller acts

        let forces = mpc.control(&x, &x_ref, &stance);
        for j in [2usize, 3] {
            let fmag = (forces[3 * j].powi(2) + forces[3 * j + 1].powi(2) + forces[3 * j + 2].powi(2)).sqrt();
            assert!(fmag < 1e-6, "swing foot {j} carries force {fmag}");
        }
        // The two stance feet still support the full weight.
        let total_fz: f64 = [0usize, 1].iter().map(|&j| forces[3 * j + 2]).sum();
        assert!(total_fz > 0.0, "stance feet should push up, Σfz = {total_fz}");
    }
}
