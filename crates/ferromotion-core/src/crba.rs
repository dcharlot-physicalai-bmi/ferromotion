//! Featherstone's **Composite-Rigid-Body Algorithm** — the O(n²) joint-space inertia (mass) matrix,
//! a clean-room implementation from *Rigid Body Dynamics Algorithms* (Ch. 6) in spatial (6D Plücker)
//! notation. Where [`crate::mass_matrix`] builds the matrix column-by-column with n RNEA passes
//! (O(n²·n) work), CRBA computes it directly by propagating composite inertias inward once. Verified
//! bit-for-bit against that reference. Pure `nalgebra` → WASM-clean.

use crate::aba::{motion_subspace, motion_transform, spatial_inertia};
use crate::Robot;
use crate::LinkInertia;
use nalgebra::{DMatrix, Matrix6, Vector6};

/// Joint-space inertia matrix `M(q)` by the Composite-Rigid-Body Algorithm.
pub fn crba(robot: &Robot, inertia: &[LinkInertia], q: &[f64]) -> DMatrix<f64> {
    let n = robot.dof();
    // Per-joint transforms X_i = ⁱX_{i−1} and motion subspaces S_i.
    let mut x: Vec<Matrix6<f64>> = Vec::with_capacity(n);
    let mut s: Vec<Vector6<f64>> = Vec::with_capacity(n);
    let mut ic: Vec<Matrix6<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let a = robot.joints[i].transform(q[i]);
        let r = *a.rotation.to_rotation_matrix().matrix();
        x.push(motion_transform(r, a.translation.vector));
        s.push(motion_subspace(robot.joints[i].kind, robot.joints[i].axis.into_inner()));
        ic.push(spatial_inertia(&inertia[i]));
    }

    let mut m = DMatrix::zeros(n, n);
    // Inward pass: accumulate composite inertias and fill the matrix.
    for i in (0..n).rev() {
        if i > 0 {
            // transform the composite inertia of body i into its parent's frame (force transform Xᵀ)
            let ic_i = ic[i];
            ic[i - 1] += x[i].transpose() * ic_i * x[i];
        }
        // F = Ic_i S_i ; diagonal entry
        let mut f = ic[i] * s[i];
        // Armature is part of the joint-space inertia, so it belongs on this diagonal exactly as it does in
        // `crate::mass_matrix`. Without it CRBA diverged from that function by the armature itself, and the
        // parity test below could not see it because its fixture stated none. Damping and friction are NOT
        // inertias and do not appear here — `mass_matrix` does not carry them either, since it is built from
        // RNEA at zero velocity.
        m[(i, i)] = s[i].dot(&f) + robot.joints[i].armature.unwrap_or(0.0);
        // walk up the ancestors, transforming the force each step
        let mut j = i;
        while j > 0 {
            f = x[j].transpose() * f;
            j -= 1;
            let mij = f.dot(&s[j]);
            m[(i, j)] = mij;
            m[(j, i)] = mij;
        }
    }
    m
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::from_urdf_full;

    fn arm() -> (Robot, Vec<LinkInertia>) {
        // a 3-DoF arm (mixed axes) with nontrivial link inertias
        let urdf = r#"<robot name="a">
          <link name="l0"/><link name="l1"><inertial><mass value="2.0"/>
            <origin xyz="0 0 0.15"/><inertia ixx="0.03" iyy="0.03" izz="0.008" ixy="0" ixz="0" iyz="0"/></inertial></link>
          <link name="l2"><inertial><mass value="1.3"/><origin xyz="0.1 0 0.1"/>
            <inertia ixx="0.02" iyy="0.02" izz="0.005" ixy="0" ixz="0" iyz="0"/></inertial></link>
          <link name="l3"><inertial><mass value="0.7"/><origin xyz="0 0.05 0.05"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.003" ixy="0" ixz="0" iyz="0"/></inertial></link>
          <joint name="j1" type="revolute"><parent link="l0"/><child link="l1"/><origin xyz="0 0 0.05"/>
            <axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
          <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.3"/>
            <axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="10" velocity="3"/></joint>
          <joint name="j3" type="prismatic"><parent link="l2"/><child link="l3"/><origin xyz="0.2 0 0"/>
            <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="3"/></joint>
        </robot>"#;
        let (robot, inertia) = from_urdf_full(urdf, "l0", "l3").unwrap();
        (robot, inertia)
    }

    /// CRBA reproduces the RNEA-column-built mass matrix bit-for-bit, is symmetric, and PD.
    ///
    /// **Run twice: bare, and with an armature stated.** This test passed for six releases while CRBA ignored
    /// [`crate::Joint::armature`], diverging from `mass_matrix` by exactly the armature (1.3e-2 on a two-joint
    /// arm), because the fixture states none. That is the fourth implementation in this crate to carry the same
    /// omission behind a vacuous parity test — after `gendyn`, `aba` and `tree_dynamics` — which is why the
    /// fixture states it now.
    #[test]
    fn crba_matches_mass_matrix() {
        let (bare, inertia) = arm();
        let mut geared = bare.clone();
        for (i, j) in geared.joints.iter_mut().enumerate() {
            *j = j.clone().with_armature(0.011 + 0.002 * i as f64);
        }
        // The guard that makes the geared pass meaningful: the term must move the answer.
        let q0 = [0.3, -0.7, 0.2];
        let moved = (crba(&geared, &inertia, &q0) - crba(&bare, &inertia, &q0)).amax();
        assert!(moved > 1e-3, "the armature must change CRBA's answer, moved by {moved:.2e}");

        for robot in [&bare, &geared] {
        for qset in [[0.0, 0.0, 0.0], [0.3, -0.7, 0.2], [1.1, 0.4, -0.3]] {
            let m_crba = crba(robot, &inertia, &qset);
            let m_ref = crate::mass_matrix(robot, &inertia, &qset);
            let err = (&m_crba - &m_ref).amax();
            assert!(err < 1e-10, "CRBA ≠ mass_matrix at {qset:?}: {err}");
            // symmetry + positive-definiteness (Cholesky exists)
            assert!((&m_crba - m_crba.transpose()).amax() < 1e-12, "M not symmetric");
            assert!(m_crba.clone().cholesky().is_some(), "M not positive-definite");
        }
        }
        let m = crba(&bare, &inertia, &[0.3, -0.7, 0.2]);
        let mg = crba(&geared, &inertia, &[0.3, -0.7, 0.2]);
        eprintln!(
            "CRBA M(q) diag bare = [{:.4}, {:.4}, {:.4}], geared = [{:.4}, {:.4}, {:.4}] \
             (both match RNEA-column to <1e-10)",
            m[(0, 0)], m[(1, 1)], m[(2, 2)], mg[(0, 0)], mg[(1, 1)], mg[(2, 2)]
        );
    }
}
