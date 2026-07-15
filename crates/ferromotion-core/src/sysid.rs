//! **Inertial parameter identification** — system identification of the robot's own dynamics.
//!
//! The Recursive Newton-Euler algorithm is **linear in the inertial parameters** when each link is
//! described by `φ_i = [m, h = m·c, I_o]` (mass, first moment, inertia about the *link origin*),
//! because the spatial inertia `I = [[I_o, ĥ], [ĥᵀ, m·I₃]]` is linear in them. So the inverse
//! dynamics factor as `τ = Y(q, q̇, q̈) · φ` with the **regressor** `Y`, and the parameters can be
//! fit from motion data by least squares — the classical result behind robot system ID (Atkeson;
//! Khalil & Dombre).
//!
//! We build `Y` column-by-column by evaluating a spatial RNEA with unit parameter vectors — exact,
//! and derivation-free. Note `Y` is **rank-deficient** for a fixed base: only the *base parameters*
//! (an identifiable subspace) are recoverable, so identification recovers the dynamics *predictively*
//! rather than a unique `φ`. Physical consistency is checked via the **pseudo-inertia** `J(φ) ≻ 0`
//! (Traversaro et al.), the condition an LMI-constrained identification would enforce.
//! Pure `nalgebra` → WASM-clean.

use crate::aba::{block6, crf, crm, motion_subspace, motion_transform, skew};
use crate::{LinkInertia, Robot};
use nalgebra::{DMatrix, DVector, Matrix3, Matrix4, Matrix6, Vector3, Vector6};

/// Inertial parameters per link: `[m, hx, hy, hz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]` (`I` about the link
/// origin, `h = m·c`).
pub const PARAMS_PER_LINK: usize = 10;

/// Spatial inertia built directly from the 10 linear parameters of one link.
fn spatial_from_params(p: &[f64]) -> Matrix6<f64> {
    let m = p[0];
    let h = Vector3::new(p[1], p[2], p[3]);
    let i_o = Matrix3::new(p[4], p[5], p[6], p[5], p[7], p[8], p[6], p[8], p[9]);
    let hx = skew(h);
    block6(i_o, hx, -hx, m * Matrix3::identity())
}

/// Pack a robot's [`LinkInertia`] list into the linear parameter vector (`10·nlinks`).
pub fn params_from_inertia(inertia: &[LinkInertia]) -> DVector<f64> {
    let mut p = DVector::zeros(PARAMS_PER_LINK * inertia.len());
    for (i, li) in inertia.iter().enumerate() {
        let (m, c) = (li.mass, li.com);
        let cx = skew(c);
        let i_o = li.inertia - m * cx * cx; // parallel-axis: COM inertia → origin inertia
        let h = m * c;
        let b = i * PARAMS_PER_LINK;
        p[b] = m;
        p[b + 1] = h.x;
        p[b + 2] = h.y;
        p[b + 3] = h.z;
        p[b + 4] = i_o[(0, 0)];
        p[b + 5] = i_o[(0, 1)];
        p[b + 6] = i_o[(0, 2)];
        p[b + 7] = i_o[(1, 1)];
        p[b + 8] = i_o[(1, 2)];
        p[b + 9] = i_o[(2, 2)];
    }
    p
}

/// Spatial-form inverse dynamics from per-link spatial inertias (linear in them).
fn spatial_rnea(robot: &Robot, si: &[Matrix6<f64>], q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>) -> Vec<f64> {
    let n = robot.dof();
    let mut a0 = Vector6::zeros();
    a0.fixed_rows_mut::<3>(3).copy_from(&(-gravity)); // gravity as base acceleration
    let (mut x, mut s) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut v, mut a, mut f) = (vec![Vector6::zeros(); n], vec![Vector6::zeros(); n], vec![Vector6::zeros(); n]);
    for i in 0..n {
        let tr = robot.joints[i].transform(q[i]);
        let xi = motion_transform(*tr.rotation.to_rotation_matrix().matrix(), tr.translation.vector);
        let sf = motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner());
        let (vp, ap) = if i == 0 { (Vector6::zeros(), a0) } else { (v[i - 1], a[i - 1]) };
        v[i] = xi * vp + sf * qd[i];
        a[i] = xi * ap + sf * qdd[i] + crm(v[i]) * (sf * qd[i]);
        f[i] = si[i] * a[i] + crf(v[i]) * (si[i] * v[i]);
        x.push(xi);
        s.push(sf);
    }
    let mut tau = vec![0.0; n];
    for i in (0..n).rev() {
        tau[i] = s[i].dot(&f[i]);
        if i > 0 {
            let ft = x[i].transpose() * f[i];
            f[i - 1] += ft;
        }
    }
    tau
}

/// The **inertial regressor** `Y(q, q̇, q̈)` (`dof × 10·nlinks`) with `τ = Y·φ`. Built by evaluating
/// the (parameter-linear) spatial RNEA on unit parameter vectors.
pub fn inertial_regressor(robot: &Robot, q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>) -> DMatrix<f64> {
    let (n, nl) = (robot.dof(), robot.dof());
    let ncols = PARAMS_PER_LINK * nl;
    let mut y = DMatrix::zeros(n, ncols);
    for j in 0..ncols {
        let mut phi = vec![0.0; ncols];
        phi[j] = 1.0;
        let si: Vec<Matrix6<f64>> = (0..nl).map(|k| spatial_from_params(&phi[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK])).collect();
        let col = spatial_rnea(robot, &si, q, qd, qdd, gravity);
        for i in 0..n {
            y[(i, j)] = col[i];
        }
    }
    y
}

/// One measurement: a configuration/velocity/acceleration and the measured joint torques.
pub struct IdSample {
    pub q: Vec<f64>,
    pub qd: Vec<f64>,
    pub qdd: Vec<f64>,
    pub tau: Vec<f64>,
}

/// Least-squares identification of the inertial parameters from motion data. Because `Y` is
/// rank-deficient for a fixed base, this returns the **minimum-norm** solution within the
/// identifiable (base-parameter) subspace — it reproduces the dynamics, but is not a unique `φ`.
pub fn identify(robot: &Robot, samples: &[IdSample], gravity: Vector3<f64>) -> DVector<f64> {
    let n = robot.dof();
    let ncols = PARAMS_PER_LINK * n;
    let mut big_y = DMatrix::zeros(n * samples.len(), ncols);
    let mut big_t = DVector::zeros(n * samples.len());
    for (k, s) in samples.iter().enumerate() {
        let y = inertial_regressor(robot, &s.q, &s.qd, &s.qdd, gravity);
        big_y.view_mut((k * n, 0), (n, ncols)).copy_from(&y);
        for i in 0..n {
            big_t[k * n + i] = s.tau[i];
        }
    }
    big_y.pseudo_inverse(1e-9).expect("regressor pseudo-inverse") * big_t
}

/// The **pseudo-inertia** matrix `J(φ)` of one link: `[[½tr(I_o)I₃ − I_o, h], [hᵀ, m]]`. A parameter
/// set is *physically consistent* iff `J ≻ 0` (Traversaro et al.) — the condition LMI-constrained
/// identification enforces.
pub fn pseudo_inertia(p: &[f64]) -> Matrix4<f64> {
    let m = p[0];
    let h = Vector3::new(p[1], p[2], p[3]);
    let i_o = Matrix3::new(p[4], p[5], p[6], p[5], p[7], p[8], p[6], p[8], p[9]);
    let sigma = 0.5 * i_o.trace() * Matrix3::identity() - i_o;
    let mut j = Matrix4::zeros();
    j.fixed_view_mut::<3, 3>(0, 0).copy_from(&sigma);
    j.fixed_view_mut::<3, 1>(0, 3).copy_from(&h);
    j.fixed_view_mut::<1, 3>(3, 0).copy_from(&h.transpose());
    j[(3, 3)] = m;
    j
}

/// Whether a link's parameters are physically consistent (`pseudo-inertia ≻ 0`).
pub fn is_physically_consistent(p: &[f64]) -> bool {
    pseudo_inertia(p).symmetric_eigenvalues().iter().all(|&e| e > 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_full, inverse_dynamics};

    const ARM3: &str = r#"<robot name="a3">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0.1 0.05" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0.002" iyy="0.03" iyz="0.0015" izz="0.025"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0.05" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.012" iyz="0" izz="0.011"/></inertial></link>
      <link name="l3"><inertial><origin xyz="0.15 0.02 0" rpy="0 0 0"/><mass value="0.6"/>
        <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.006" iyz="0.0005" izz="0.005"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1"/><axis xyz="0 0 1"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.5 0 0"/><axis xyz="0 1 0"/></joint>
      <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.4 0 0"/><axis xyz="0 1 0"/></joint>
      <joint name="jt" type="fixed"><parent link="l3"/><child link="tool"/><origin xyz="0.3 0 0"/></joint>
    </robot>"#;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64) * 2.0 - 1.0
    }

    #[test]
    fn regressor_reproduces_the_rnea_torques() {
        // τ = Y(q,q̇,q̈)·φ must equal the RNEA exactly — this validates both the regressor and its
        // linearity in the inertial parameters.
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let phi = params_from_inertia(&inertia);
        let mut seed = 42u64;
        for _ in 0..5 {
            let q: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
            let qd: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
            let qdd: Vec<f64> = (0..3).map(|_| lcg(&mut seed)).collect();
            let y = inertial_regressor(&robot, &q, &qd, &qdd, g);
            let tau_reg = &y * &phi;
            let tau_ref = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            for i in 0..3 {
                assert!((tau_reg[i] - tau_ref[i]).abs() < 1e-9, "τ[{i}]: regressor {} vs RNEA {}", tau_reg[i], tau_ref[i]);
            }
        }
    }

    #[test]
    fn identification_recovers_the_dynamics_from_data() {
        let (robot, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let mut seed = 7u64;
        let mut make_sample = |seed: &mut u64| {
            let q: Vec<f64> = (0..3).map(|_| lcg(seed)).collect();
            let qd: Vec<f64> = (0..3).map(|_| lcg(seed)).collect();
            let qdd: Vec<f64> = (0..3).map(|_| lcg(seed)).collect();
            let tau = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
            IdSample { q, qd, qdd, tau }
        };
        let train: Vec<IdSample> = (0..60).map(|_| make_sample(&mut seed)).collect();
        let phi_hat = identify(&robot, &train, g);

        // The regressor is rank-deficient (only base parameters are identifiable) …
        let y0 = inertial_regressor(&robot, &train[0].q, &train[0].qd, &train[0].qdd, g);
        assert!(y0.ncols() == 30, "expected 10 params × 3 links");

        // … but the identified model reproduces the true torques on *held-out* data.
        for _ in 0..10 {
            let s = make_sample(&mut seed);
            let y = inertial_regressor(&robot, &s.q, &s.qd, &s.qdd, g);
            let pred = &y * &phi_hat;
            for i in 0..3 {
                assert!((pred[i] - s.tau[i]).abs() < 1e-6, "held-out τ[{i}]: predicted {} vs true {}", pred[i], s.tau[i]);
            }
        }
    }

    #[test]
    fn pseudo_inertia_detects_physical_consistency() {
        let (_, inertia) = from_urdf_full(ARM3, "base", "tool").unwrap();
        let phi = params_from_inertia(&inertia);
        // A real robot's links are physically consistent.
        for k in 0..3 {
            let p = phi.as_slice()[k * PARAMS_PER_LINK..(k + 1) * PARAMS_PER_LINK].to_vec();
            assert!(is_physically_consistent(&p), "link {k} should be physically consistent");
        }
        // Negative mass is not.
        let mut bad = phi.as_slice()[0..PARAMS_PER_LINK].to_vec();
        bad[0] = -1.0;
        assert!(!is_physically_consistent(&bad), "negative mass must be rejected");
    }
}
