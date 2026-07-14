//! Analytical derivatives of the rigid-body dynamics — a clean-room implementation of Carpentier &
//! Mansard, *Analytical Derivatives of Rigid Body Dynamics Algorithms* (RSS 2018, the method behind
//! Pinocchio's `computeRNEADerivatives`/`computeABADerivatives`).
//!
//! We differentiate the Recursive Newton-Euler pass exactly by **forward-mode sensitivity**: seeding
//! a direction in `q` or `q̇` and propagating the tangents of the link velocities, accelerations, and
//! forces through the same two recursions gives an exact column of `∂τ_ID/∂q` or `∂τ_ID/∂q̇` in O(n),
//! so the full matrices cost O(n²) — no finite differences, no AD framework. The forward-dynamics
//! partials then follow from the identity `ID(q, q̇, a(q,q̇,τ)) ≡ τ`:
//! `∂a/∂τ = M⁻¹`, `∂a/∂q = −M⁻¹ ∂ID/∂q|_{q̈=a}`, `∂a/∂q̇ = −M⁻¹ ∂ID/∂q̇|_{q̈=a}`. These exact gradients
//! replace the finite-difference linearization in the optimal-control stack. Pure `nalgebra` → WASM-clean.

use crate::{forward_dynamics, mass_matrix, JointKind, LinkInertia, Robot};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};

fn skew(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Cached RNEA state (both passes), enabling exact directional derivatives of inverse dynamics.
struct IdTangent<'a> {
    robot: &'a Robot,
    inertia: &'a [LinkInertia],
    n: usize,
    gravity: Vector3<f64>,
    rr: Vec<Matrix3<f64>>, // frame i → i-1 rotation
    pp: Vec<Vector3<f64>>, // origin of frame i in i-1
    zz: Vec<Vector3<f64>>, // joint axis in frame i (constant in q)
    ap: Vec<Vector3<f64>>, // joint axis in the parent (i-1) frame
    omega: Vec<Vector3<f64>>,
    omegad: Vec<Vector3<f64>>,
    vd: Vec<Vector3<f64>>,
    f_tot: Vec<Vector3<f64>>, // total transmitted force (frame i)
    n_tot: Vec<Vector3<f64>>, // total transmitted moment (frame i)
    qd: Vec<f64>,
}

impl<'a> IdTangent<'a> {
    fn new(robot: &'a Robot, inertia: &'a [LinkInertia], q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>) -> Self {
        let n = robot.dof();
        let (mut rr, mut pp, mut zz, mut ap) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let a = robot.joints[i].transform(q[i]);
            rr.push(*a.rotation.to_rotation_matrix().matrix());
            pp.push(a.translation.vector);
            zz.push(robot.joints[i].axis.into_inner());
            ap.push(robot.joints[i].origin.rotation * robot.joints[i].axis.into_inner());
        }
        // Forward pass.
        let (mut omega, mut omegad, mut vd, mut ff) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for i in 0..n {
            let rt = rr[i].transpose();
            let z = zz[i];
            let (pw, pwd, pvd) = if i == 0 { (Vector3::zeros(), Vector3::zeros(), -gravity) } else { (omega[i - 1], omegad[i - 1], vd[i - 1]) };
            let base = rt * (pvd + pwd.cross(&pp[i]) + pw.cross(&pw.cross(&pp[i])));
            match robot.joints[i].kind {
                JointKind::Revolute => {
                    omega[i] = rt * pw + qd[i] * z;
                    omegad[i] = rt * pwd + (rt * pw).cross(&(qd[i] * z)) + qdd[i] * z;
                    vd[i] = base;
                }
                JointKind::Prismatic => {
                    omega[i] = rt * pw;
                    omegad[i] = rt * pwd;
                    vd[i] = base + 2.0 * (rt * pw).cross(&(qd[i] * z)) + qdd[i] * z;
                }
            }
            let li = &inertia[i];
            let vdc = vd[i] + omegad[i].cross(&li.com) + omega[i].cross(&omega[i].cross(&li.com));
            ff[i] = li.mass * vdc;
        }
        // Backward pass (cache total transmitted force/moment per link).
        let (mut f_tot, mut n_tot) = (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for i in (0..n).rev() {
            let li = &inertia[i];
            let nn = li.inertia * omegad[i] + omega[i].cross(&(li.inertia * omega[i]));
            let (rr_next, p_next, f_child, n_child) = if i + 1 < n { (rr[i + 1], pp[i + 1], f_tot[i + 1], n_tot[i + 1]) } else { (Matrix3::identity(), Vector3::zeros(), Vector3::zeros(), Vector3::zeros()) };
            f_tot[i] = rr_next * f_child + ff[i];
            n_tot[i] = nn + rr_next * n_child + li.com.cross(&ff[i]) + p_next.cross(&(rr_next * f_child));
        }
        Self { robot, inertia, n, gravity, rr, pp, zz, ap, omega, omegad, vd, f_tot, n_tot, qd: qd.to_vec() }
    }

    /// Exact directional derivative `∂τ_ID` in the direction `(dq, dqd)` (with `dq̈ = 0`).
    fn directional(&self, dq: &[f64], dqd: &[f64]) -> DVector<f64> {
        let n = self.n;
        // Transform tangents (only joint i's transform depends on q_i).
        let (mut drt, mut drr, mut dpp) = (vec![Matrix3::zeros(); n], vec![Matrix3::zeros(); n], vec![Vector3::zeros(); n]);
        for i in 0..n {
            match self.robot.joints[i].kind {
                JointKind::Revolute => {
                    drr[i] = dq[i] * (skew(self.ap[i]) * self.rr[i]); // d(R) = [ap]× R
                    drt[i] = -dq[i] * (self.rr[i].transpose() * skew(self.ap[i])); // d(Rᵀ) = −Rᵀ [ap]×
                }
                JointKind::Prismatic => dpp[i] = dq[i] * self.ap[i], // d(p) = ap
            }
        }

        // Tangent forward pass.
        let (mut d_omega, mut d_omegad, mut d_vd, mut d_ff, mut d_nn) =
            (vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n], vec![Vector3::zeros(); n]);
        for i in 0..n {
            let rt = self.rr[i].transpose();
            let z = self.zz[i];
            let (p, dp) = (self.pp[i], dpp[i]);
            let (pw, pwd, pvd) = if i == 0 { (Vector3::zeros(), Vector3::zeros(), -self.gravity) } else { (self.omega[i - 1], self.omegad[i - 1], self.vd[i - 1]) };
            let (dpw, dpwd, dpvd) = if i == 0 { (Vector3::zeros(), Vector3::zeros(), Vector3::zeros()) } else { (d_omega[i - 1], d_omegad[i - 1], d_vd[i - 1]) };
            let inner = pvd + pwd.cross(&p) + pw.cross(&pw.cross(&p));
            let d_inner = dpvd + dpwd.cross(&p) + pwd.cross(&dp) + dpw.cross(&pw.cross(&p)) + pw.cross(&(dpw.cross(&p) + pw.cross(&dp)));
            let d_base = drt[i] * inner + rt * d_inner;
            match self.robot.joints[i].kind {
                JointKind::Revolute => {
                    d_omega[i] = drt[i] * pw + rt * dpw + dqd[i] * z;
                    let (rtpw, d_rtpw) = (rt * pw, drt[i] * pw + rt * dpw);
                    d_omegad[i] = drt[i] * pwd + rt * dpwd + d_rtpw.cross(&(self.qd[i] * z)) + rtpw.cross(&(dqd[i] * z));
                    d_vd[i] = d_base;
                }
                JointKind::Prismatic => {
                    d_omega[i] = drt[i] * pw + rt * dpw;
                    d_omegad[i] = drt[i] * pwd + rt * dpwd;
                    let (rtpw, d_rtpw) = (rt * pw, drt[i] * pw + rt * dpw);
                    d_vd[i] = d_base + 2.0 * (d_rtpw.cross(&(self.qd[i] * z)) + rtpw.cross(&(dqd[i] * z)));
                }
            }
            let li = &self.inertia[i];
            let (w, _wd) = (self.omega[i], self.omegad[i]);
            let d_vdc = d_vd[i] + d_omegad[i].cross(&li.com) + d_omega[i].cross(&w.cross(&li.com)) + w.cross(&(d_omega[i].cross(&li.com)));
            d_ff[i] = li.mass * d_vdc;
            d_nn[i] = li.inertia * d_omegad[i] + d_omega[i].cross(&(li.inertia * w)) + w.cross(&(li.inertia * d_omega[i]));
        }

        // Tangent backward pass.
        let mut dtau = DVector::zeros(n);
        let mut d_f = vec![Vector3::zeros(); n];
        let mut d_n = vec![Vector3::zeros(); n];
        for i in (0..n).rev() {
            let li = &self.inertia[i];
            let (rr_next, p_next, f_child, n_child) = if i + 1 < n { (self.rr[i + 1], self.pp[i + 1], self.f_tot[i + 1], self.n_tot[i + 1]) } else { (Matrix3::identity(), Vector3::zeros(), Vector3::zeros(), Vector3::zeros()) };
            let (drr_next, dp_next, df_child, dn_child) = if i + 1 < n { (drr[i + 1], dpp[i + 1], d_f[i + 1], d_n[i + 1]) } else { (Matrix3::zeros(), Vector3::zeros(), Vector3::zeros(), Vector3::zeros()) };
            let rnf = rr_next * f_child;
            let d_rnf = drr_next * f_child + rr_next * df_child;
            d_f[i] = d_rnf + d_ff[i];
            d_n[i] = d_nn[i] + drr_next * n_child + rr_next * dn_child + li.com.cross(&d_ff[i]) + dp_next.cross(&rnf) + p_next.cross(&d_rnf);
            dtau[i] = match self.robot.joints[i].kind {
                JointKind::Revolute => d_n[i].dot(&self.zz[i]),
                JointKind::Prismatic => d_f[i].dot(&self.zz[i]),
            };
        }
        dtau
    }

    /// Full derivative matrices `(∂ID/∂q, ∂ID/∂q̇)`, each n×n.
    fn matrices(&self) -> (DMatrix<f64>, DMatrix<f64>) {
        let n = self.n;
        let (mut dq_mat, mut dqd_mat) = (DMatrix::zeros(n, n), DMatrix::zeros(n, n));
        let z = vec![0.0; n];
        for j in 0..n {
            let mut e = z.clone();
            e[j] = 1.0;
            dq_mat.set_column(j, &self.directional(&e, &z));
            dqd_mat.set_column(j, &self.directional(&z, &e));
        }
        (dq_mat, dqd_mat)
    }
}

/// Analytical derivatives of inverse dynamics: `(∂τ/∂q, ∂τ/∂q̇)` at `(q, q̇, q̈)`, each n×n.
pub fn id_derivatives(robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], qdd: &[f64], gravity: Vector3<f64>) -> (DMatrix<f64>, DMatrix<f64>) {
    IdTangent::new(robot, inertia, q, qd, qdd, gravity).matrices()
}

/// Analytical forward-dynamics partials `(∂a/∂q, ∂a/∂q̇, ∂a/∂τ)` at `(q, q̇, τ)`, each n×n.
pub fn forward_dynamics_derivatives(robot: &Robot, inertia: &[LinkInertia], q: &[f64], qd: &[f64], tau: &[f64], gravity: Vector3<f64>) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>) {
    let a = forward_dynamics(robot, inertia, q, qd, tau, gravity);
    let m = mass_matrix(robot, inertia, q);
    let minv = m.try_inverse().expect("mass matrix invertible");
    let (did_dq, did_dqd) = id_derivatives(robot, inertia, q, qd, &a, gravity);
    (-&minv * &did_dq, -&minv * &did_dqd, minv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{from_urdf_full, inverse_dynamics};

    const ARM2: &str = r#"<robot name="a2">
      <link name="base"/>
      <link name="l1"><inertial><origin xyz="0.3 0.1 0" rpy="0 0 0"/><mass value="1.5"/>
        <inertia ixx="0.02" ixy="0.001" ixz="0" iyy="0.02" iyz="0" izz="0.03"/></inertial></link>
      <link name="l2"><inertial><origin xyz="0.25 0 0.05" rpy="0 0 0"/><mass value="1.0"/>
        <inertia ixx="0.01" ixy="0" ixz="0.001" iyy="0.01" iyz="0" izz="0.015"/></inertial></link>
      <link name="tool"/>
      <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0.6 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
      <joint name="jt" type="fixed"><parent link="l2"/><child link="tool"/><origin xyz="0.5 0 0" rpy="0 0 0"/></joint>
    </robot>"#;

    #[test]
    fn id_derivatives_match_finite_difference() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (q, qd, qdd) = (vec![0.3, -0.7], vec![0.5, -0.2], vec![0.4, 0.1]);
        let (dq, dqd) = id_derivatives(&robot, &inertia, &q, &qd, &qdd, g);
        let (eps, n) = (1e-6, robot.dof());
        for j in 0..n {
            let (mut qp, mut qm) = (q.clone(), q.clone());
            qp[j] += eps;
            qm[j] -= eps;
            let fd = (DVector::from_vec(inverse_dynamics(&robot, &inertia, &qp, &qd, &qdd, g)) - DVector::from_vec(inverse_dynamics(&robot, &inertia, &qm, &qd, &qdd, g))) / (2.0 * eps);
            let (mut vp, mut vm) = (qd.clone(), qd.clone());
            vp[j] += eps;
            vm[j] -= eps;
            let fdv = (DVector::from_vec(inverse_dynamics(&robot, &inertia, &q, &vp, &qdd, g)) - DVector::from_vec(inverse_dynamics(&robot, &inertia, &q, &vm, &qdd, g))) / (2.0 * eps);
            for i in 0..n {
                assert!((dq[(i, j)] - fd[i]).abs() < 1e-4, "∂τ/∂q[{i},{j}]: analytic {} vs fd {}", dq[(i, j)], fd[i]);
                assert!((dqd[(i, j)] - fdv[i]).abs() < 1e-4, "∂τ/∂q̇[{i},{j}]: analytic {} vs fd {}", dqd[(i, j)], fdv[i]);
            }
        }
    }

    #[test]
    fn forward_dynamics_derivatives_match_finite_difference() {
        let (robot, inertia) = from_urdf_full(ARM2, "base", "tool").unwrap();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let (q, qd, tau) = (vec![0.2, 0.5], vec![-0.3, 0.4], vec![0.6, -0.2]);
        let (da_dq, da_dqd, da_dtau) = forward_dynamics_derivatives(&robot, &inertia, &q, &qd, &tau, g);
        let (eps, n) = (1e-6, robot.dof());
        for j in 0..n {
            let col = |base: &[f64], which: u8| -> DVector<f64> {
                let run = |e: f64| {
                    let (mut qq, mut vv, mut tt) = (q.clone(), qd.clone(), tau.clone());
                    match which {
                        0 => qq[j] += e,
                        1 => vv[j] += e,
                        _ => tt[j] += e,
                    }
                    let _ = base;
                    DVector::from_vec(forward_dynamics(&robot, &inertia, &qq, &vv, &tt, g))
                };
                (run(eps) - run(-eps)) / (2.0 * eps)
            };
            let (fq, fv, ft) = (col(&q, 0), col(&qd, 1), col(&tau, 2));
            for i in 0..n {
                assert!((da_dq[(i, j)] - fq[i]).abs() < 1e-3, "∂a/∂q[{i},{j}]: {} vs {}", da_dq[(i, j)], fq[i]);
                assert!((da_dqd[(i, j)] - fv[i]).abs() < 1e-3, "∂a/∂q̇[{i},{j}]: {} vs {}", da_dqd[(i, j)], fv[i]);
                assert!((da_dtau[(i, j)] - ft[i]).abs() < 1e-3, "∂a/∂τ[{i},{j}]: {} vs {}", da_dtau[(i, j)], ft[i]);
            }
        }
    }
}
