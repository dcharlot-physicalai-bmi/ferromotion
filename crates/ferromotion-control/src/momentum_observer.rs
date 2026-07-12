//! Generalized-momentum observer (De Luca) — external-torque / contact estimation without `q̈`.
//!
//! The generalized momentum is `p = M(q)·q̇`. Differentiating along the rigid-body dynamics
//! `M q̈ + C q̇ + G = τ + τ_ext` and using the skew property `Ṁ = C + Cᵀ` gives the key identity
//!
//! ```text
//!   ṗ = τ + τ_ext + Cᵀ(q,q̇)·q̇ − G(q).
//! ```
//!
//! The observer forms the residual
//!
//! ```text
//!   r(t) = K_I · ( p(t) − p(0) − ∫₀ᵗ ( τ + Cᵀ(q,q̇)q̇ − G(q) + r ) dτ ),
//! ```
//!
//! whose dynamics reduce to the first-order lag `ṙ = K_I·(τ_ext − r)`, so `r → τ_ext` with
//! bandwidth `K_I` (no acceleration and no differentiation of `p` are ever needed).
//!
//! ## Bias convention (documented explicitly)
//! We integrate the model term `β(q,q̇) = Cᵀ(q,q̇)·q̇ − G(q)` alongside the *applied* torque
//! `τ_applied` and the current residual `r`. `G(q)` is [`gravity_vector`] with the same `gravity`
//! the plant uses (`τ = M q̈ + C q̇ + G`, i.e. `G` is on the left-hand side). The transpose-Coriolis
//! term is obtained from the clean Christoffel result
//!
//! ```text
//!   (Cᵀ(q,q̇)·q̇)_i = ½ q̇ᵀ (∂M/∂q_i) q̇ = ∂/∂q_i [ ½ q̇ᵀ M(q) q̇ ],
//! ```
//!
//! i.e. the gradient of the kinetic energy w.r.t. `q` at fixed `q̇`, computed by central differences
//! on [`mass_matrix`]. This avoids assembling the full Coriolis matrix while staying WASM-clean.
//!
//! At a resting steady state (`q̇ = 0`, `q̈ = 0`) the integrand's fixed point is `τ_applied + β + r = 0`,
//! which with the equilibrium `τ_applied + τ_ext = G` yields `r = G − τ_applied = τ_ext` exactly:
//! the observer has zero steady-state bias.

use nalgebra::DVector;
use ferromotion_core::{gravity_vector, mass_matrix, LinkInertia, Robot};

/// First-order generalized-momentum observer. `ki` is the observer gain `K_I` (per-joint bandwidth,
/// rad/s); larger tracks faster but must satisfy `ki·dt < 2` for the explicit update to stay stable.
#[derive(Clone, Debug)]
pub struct MomentumObserver {
    pub ki: f64,
    /// Running integral of `(τ_applied + β + r)` (the observer's internal accumulator).
    integ: Vec<f64>,
    /// Momentum captured at the first step / after `reset`, so that `r(0) = 0`.
    p0: Option<Vec<f64>>,
}

impl MomentumObserver {
    pub fn new(ki: f64) -> Self {
        Self { ki, integ: Vec::new(), p0: None }
    }

    /// Forget all accumulated state; the next [`step`](Self::step) re-anchors `p(0)` and returns `0`.
    pub fn reset(&mut self) {
        self.integ.clear();
        self.p0 = None;
    }

    /// One observer step. Sees only the *applied* joint torque `tau_applied` (never the external
    /// torque), the current measured `q`/`qd`, and `dt`. Returns the estimated external joint torque
    /// `τ_ext` (length `robot.dof()`).
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        robot: &Robot,
        inertia: &[LinkInertia],
        q: &[f64],
        qd: &[f64],
        tau_applied: &[f64],
        dt: f64,
        gravity: nalgebra::Vector3<f64>,
    ) -> Vec<f64> {
        let n = robot.dof();
        let qdv = DVector::from_row_slice(qd);

        // Current generalized momentum p = M(q)·q̇.
        let m = mass_matrix(robot, inertia, q);
        let p: Vec<f64> = (&m * &qdv).iter().copied().collect();

        // Anchor p(0) and clear the integral on the first call (or after reset / DoF change).
        if self.p0.as_ref().map(|p0| p0.len() != n).unwrap_or(true) {
            self.p0 = Some(p.clone());
            self.integ = vec![0.0; n];
        }
        let p0 = self.p0.as_ref().unwrap();

        // Model bias β = Cᵀ(q,q̇)·q̇ − G(q).
        let g = gravity_vector(robot, inertia, q, gravity);
        let ct = coriolis_transpose_qd(robot, inertia, q, &qdv);

        // Residual from the integral form: r = K_I·(p − p0 − ∫(...)). Uses the integral accumulated
        // through the *previous* step, so at k = 0 (integ = 0, p = p0) we get r = 0.
        let r: Vec<f64> = (0..n).map(|i| self.ki * (p[i] - p0[i] - self.integ[i])).collect();

        // Advance the integral with the just-computed residual (semi-implicit / forward Euler of
        // ṙ = K_I(τ_ext − r)).
        for i in 0..n {
            self.integ[i] += (tau_applied[i] + ct[i] - g[i] + r[i]) * dt;
        }
        r
    }
}

/// Transpose-Coriolis term `Cᵀ(q,q̇)·q̇`, via the Christoffel identity
/// `(Cᵀq̇)_i = ∂/∂q_i[½ q̇ᵀ M(q) q̇]`, evaluated by central differences on the mass matrix.
fn coriolis_transpose_qd(robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &DVector<f64>) -> Vec<f64> {
    let n = robot.dof();
    let eps = 1e-6;
    // Kinetic energy T(q) = ½ q̇ᵀ M(q) q̇ at fixed q̇.
    let ke = |qq: &[f64]| -> f64 { 0.5 * qd.dot(&(&mass_matrix(robot, inertia, qq) * qd)) };
    let mut out = vec![0.0; n];
    let mut qq = q.to_vec();
    for i in 0..n {
        let q_i = q[i];
        qq[i] = q_i + eps;
        let tp = ke(&qq);
        qq[i] = q_i - eps;
        let tm = ke(&qq);
        qq[i] = q_i;
        out[i] = (tp - tm) / (2.0 * eps);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use ferromotion_core::{forward_dynamics, from_urdf_full};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    /// Core property: with a KNOWN constant external joint torque injected into the plant only, the
    /// observer (which sees only the applied torque) converges to that external torque.
    #[test]
    fn observer_recovers_constant_external_torque() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let tau_ext_true = [0.7, -0.4];

        // Applied torque = gravity compensation + light PD to q = 0 (keeps the arm bounded); the
        // observer is fed exactly this. A gentle regulator is enough — the observer does not depend
        // on the arm being still, only on seeing the true applied torque.
        let mut obs = MomentumObserver::new(80.0);
        let (kp, kd) = (30.0, 8.0);
        let (mut q, mut qd, dt) = (vec![0.0, 0.0], vec![0.0, 0.0], 1e-3);
        let mut r = vec![0.0; 2];
        for _ in 0..8000 {
            let gcomp = ferromotion_core::gravity_vector(&robot, &inertia, &q, g);
            let tau_applied: Vec<f64> =
                (0..2).map(|i| gcomp[i] + kp * (0.0 - q[i]) + kd * (0.0 - qd[i])).collect();
            r = obs.step(&robot, &inertia, &q, &qd, &tau_applied, dt, g);
            // Plant feels applied + true external torque.
            let tau_plant: Vec<f64> = (0..2).map(|i| tau_applied[i] + tau_ext_true[i]).collect();
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau_plant, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        for i in 0..2 {
            assert!((r[i] - tau_ext_true[i]).abs() < 2e-2, "τ_ext[{i}] est {} vs {}", r[i], tau_ext_true[i]);
        }
    }

    /// With no external torque the residual must stay near zero throughout free (gravity-driven)
    /// swinging motion — the observer must not manufacture phantom contact torques.
    #[test]
    fn observer_stays_near_zero_without_contact() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut obs = MomentumObserver::new(80.0);
        let (mut q, mut qd, dt) = (vec![0.5, -0.3], vec![0.0, 0.0], 1e-3);
        let mut max_r: f64 = 0.0;
        for k in 0..4000 {
            // Free swing: zero applied torque, and the plant also gets zero external torque.
            let tau_applied = [0.0, 0.0];
            let r = obs.step(&robot, &inertia, &q, &qd, &tau_applied, dt, g);
            // Skip the very first transient steps while the integral anchors.
            if k > 200 {
                max_r = max_r.max(r.iter().fold(0.0_f64, |a, &v| a.max(v.abs())));
            }
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau_applied, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        assert!(max_r < 5e-3, "phantom residual during free swing: {max_r}");
    }

    /// The Christoffel/kinetic-energy Coriolis-transpose term must satisfy the skew identity
    /// `Ṁ q̇ = C q̇ + Cᵀ q̇`, checked against the exact RNEA Coriolis vector `C q̇`.
    #[test]
    fn coriolis_transpose_matches_skew_identity() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let (q, qd) = ([0.3, -0.7], [1.1, -0.6]);
        let qdv = DVector::from_row_slice(&qd);
        let ct = coriolis_transpose_qd(&robot, &inertia, &q, &qdv);

        // Exact centrifugal/Coriolis vector C q̇ = RNEA(q, q̇, 0, g = 0).
        let cqd = ferromotion_core::inverse_dynamics(&robot, &inertia, &q, &qd, &[0.0, 0.0], Vector3::zeros());

        // Ṁ q̇ via central difference of M along the trajectory q̇ direction: Ṁ = Σ_k ∂M/∂q_k q̇_k.
        let eps = 1e-6;
        let qp: Vec<f64> = (0..2).map(|i| q[i] + eps * qd[i]).collect();
        let qm: Vec<f64> = (0..2).map(|i| q[i] - eps * qd[i]).collect();
        let mdot = (mass_matrix(&robot, &inertia, &qp) - mass_matrix(&robot, &inertia, &qm)) / (2.0 * eps);
        let mdot_qd = &mdot * &qdv;

        for i in 0..2 {
            assert!((mdot_qd[i] - (cqd[i] + ct[i])).abs() < 1e-4, "skew identity violated at {i}");
        }
    }

    /// `reset` re-anchors the observer: after resetting, a fresh convergence run reproduces τ_ext.
    #[test]
    fn reset_reanchors_the_observer() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut obs = MomentumObserver::new(80.0);
        // Dirty the state with some steps.
        let mut q = vec![0.2, 0.1];
        let mut qd = vec![0.0, 0.0];
        for _ in 0..50 {
            obs.step(&robot, &inertia, &q, &qd, &[0.0, 0.0], 1e-3, g);
        }
        obs.reset();
        assert!(obs.p0.is_none() && obs.integ.is_empty());

        // Fresh run converges to a new τ_ext.
        let tau_ext_true = [-0.5, 0.3];
        let (kp, kd, dt) = (30.0, 8.0, 1e-3);
        q = vec![0.0, 0.0];
        qd = vec![0.0, 0.0];
        let mut r = vec![0.0; 2];
        for _ in 0..8000 {
            let gcomp = ferromotion_core::gravity_vector(&robot, &inertia, &q, g);
            let tau_applied: Vec<f64> =
                (0..2).map(|i| gcomp[i] + kp * (0.0 - q[i]) + kd * (0.0 - qd[i])).collect();
            r = obs.step(&robot, &inertia, &q, &qd, &tau_applied, dt, g);
            let tau_plant: Vec<f64> = (0..2).map(|i| tau_applied[i] + tau_ext_true[i]).collect();
            let qdd = forward_dynamics(&robot, &inertia, &q, &qd, &tau_plant, g);
            for i in 0..2 {
                qd[i] += qdd[i] * dt;
                q[i] += qd[i] * dt;
            }
        }
        for i in 0..2 {
            assert!((r[i] - tau_ext_true[i]).abs() < 2e-2, "post-reset τ_ext[{i}] {} vs {}", r[i], tau_ext_true[i]);
        }
    }
}
