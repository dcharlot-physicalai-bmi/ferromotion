//! **Point and body Jacobians for a branched kinematic tree**, and the per-body inverse weights MuJoCo's
//! constraint regularisation reads.
//!
//! The contact model in [`crate::mujoco_contact`] takes a contact's Jacobian as given, because it was written
//! against hand-built contacts. To use it on a model that was actually loaded — an MJCF through
//! [`crate::tree_from_mjcf`], whose free joint is six tree joints so the whole thing is one fixed-base tree —
//! something has to turn "these two geoms touch, here, along this normal" into constraint rows. That is what
//! this is: the world velocity of a point on a body per generalised velocity, the angular velocity of a body
//! per generalised velocity, and `body_invweight0`.

use crate::dynamics::LinkInertia;
use crate::kinematic_tree::KinematicTree;
use crate::{Iso, Joint, JointKind};
use nalgebra::{DMatrix, Matrix3, Point3, Vector3};

/// **The `3 × nv` Jacobian of a world point's linear velocity.** `frames` are the joint world frames from
/// [`KinematicTree::frames`], `joint` the tree joint the point rides on (`None` for a point fixed to the
/// world), and `point` its current world position.
///
/// Only the joint's ancestors move it, so the column count is the tree's but the work is the depth.
pub fn tree_point_jacobian(joints: &[Joint], parent: &[isize], frames: &[Iso], joint: Option<usize>, point: Vector3<f64>) -> DMatrix<f64> {
    let mut j = DMatrix::zeros(3, joints.len());
    let mut k = match joint {
        Some(b) => b as isize,
        None => -1,
    };
    while k >= 0 {
        let idx = k as usize;
        let f = frames[idx];
        let axis_w = f.rotation.to_rotation_matrix() * joints[idx].axis.into_inner();
        let col = match joints[idx].kind {
            JointKind::Revolute => axis_w.cross(&(point - f.translation.vector)),
            JointKind::Prismatic => axis_w,
        };
        j.fixed_view_mut::<3, 1>(0, idx).copy_from(&col);
        k = parent[idx];
    }
    j
}

/// **The `3 × nv` Jacobian of a body's angular velocity.** A prismatic joint contributes nothing.
pub fn tree_angular_jacobian(joints: &[Joint], parent: &[isize], frames: &[Iso], joint: Option<usize>) -> DMatrix<f64> {
    let mut j = DMatrix::zeros(3, joints.len());
    let mut k = match joint {
        Some(b) => b as isize,
        None => -1,
    };
    while k >= 0 {
        let idx = k as usize;
        if joints[idx].kind == JointKind::Revolute {
            let axis_w = frames[idx].rotation.to_rotation_matrix() * joints[idx].axis.into_inner();
            j.fixed_view_mut::<3, 1>(0, idx).copy_from(&axis_w);
        }
        k = parent[idx];
    }
    j
}

/// **A contact's constraint rows**, `dim × nv`: the relative velocity of geom 2 with respect to geom 1 at the
/// contact point, resolved in the contact frame.
///
/// Row 0 is therefore the rate at which the two surfaces separate, which is what MuJoCo's `efc_pos` (the
/// signed distance) differentiates to, and a positive force along it pushes them apart. Rows 1 and 2 are the
/// two tangents; rows 3..6, for condim 4 and 6, are the torsional and rolling rates, which are differences of
/// ANGULAR velocity rather than of point velocity.
pub fn contact_jacobian(joints: &[Joint], parent: &[isize], frames: &[Iso], body1: Option<usize>, body2: Option<usize>, point: Vector3<f64>, frame: &[Vector3<f64>; 3], condim: usize) -> DMatrix<f64> {
    let nv = joints.len();
    let dif = tree_point_jacobian(joints, parent, frames, body2, point) - tree_point_jacobian(joints, parent, frames, body1, point);
    let mut j = DMatrix::zeros(condim.min(3), nv);
    for r in 0..condim.min(3) {
        j.row_mut(r).copy_from(&(frame[r].transpose() * &dif));
    }
    if condim <= 3 {
        return j;
    }
    // torsional (about the normal) and, at condim 6, rolling (about the two tangents)
    let adif = tree_angular_jacobian(joints, parent, frames, body2) - tree_angular_jacobian(joints, parent, frames, body1);
    let mut full = DMatrix::zeros(condim, nv);
    full.view_mut((0, 0), (3, nv)).copy_from(&j);
    let extra: &[usize] = if condim == 4 { &[0] } else { &[0, 1, 2] };
    for (r, &axis) in extra.iter().enumerate() {
        full.row_mut(3 + r).copy_from(&(frame[axis].transpose() * &adif));
    }
    full
}

/// **`body_invweight0` for one body**: the mean diagonal of `J M⁻¹ Jᵀ` at the body's centre of mass, taken
/// separately over the translational and rotational blocks.
///
/// MuJoCo evaluates this once at `qpos0` and reuses it for the whole simulation — it is the approximation
/// that makes `mj_diagApprox` cheap, and it is why the regularisation `R` does not follow the true diagonal
/// of `A` as the model moves. Reproducing MuJoCo's `R` therefore means reproducing this, at `qpos0`,
/// including its staleness.
pub fn body_invweight(m_inv: &DMatrix<f64>, joints: &[Joint], parent: &[isize], frames: &[Iso], joint: Option<usize>, com: Vector3<f64>) -> crate::mujoco_contact::InvWeight {
    let jt = tree_point_jacobian(joints, parent, frames, joint, com);
    let ja = tree_angular_jacobian(joints, parent, frames, joint);
    let at = &jt * m_inv * jt.transpose();
    let ar = &ja * m_inv * ja.transpose();
    let mean = |a: &DMatrix<f64>| (0..3).map(|i| a[(i, i)]).sum::<f64>() / 3.0;
    crate::mujoco_contact::InvWeight { tran: mean(&at), rot: mean(&ar) }
}

/// The world centre of mass of the link riding `joint`, from its [`LinkInertia`].
pub fn body_com(frames: &[Iso], inertia: &[LinkInertia], joint: usize) -> Vector3<f64> {
    (frames[joint] * Point3::from(inertia[joint].com)).coords
}

/// The three-column contact frame as a matrix, for callers that hold `ContactRecord::frame`.
pub fn frame_matrix(frame: &[Vector3<f64>; 3]) -> Matrix3<f64> {
    Matrix3::from_columns(&[frame[0], frame[1], frame[2]])
}

/// Every tree joint's world frame at `q` — the input the Jacobians here take.
pub fn tree_frames(tree: &KinematicTree, q: &[f64]) -> Vec<Iso> {
    tree.frames(Iso::identity(), q)
}

#[cfg(test)]
mod tests {
    //! The Jacobians are checked against a central difference of the forward kinematics they differentiate,
    //! which is the only check that does not restate the derivation being tested.
    use super::*;
    use crate::{Iso, Joint};
    use nalgebra::{Translation3, UnitQuaternion};

    fn chain() -> (Vec<Joint>, Vec<isize>) {
        // a 4-DoF tree with a branch: 0 -> 1 -> 2, and 0 -> 3
        let o = |x: f64, y: f64, z: f64| Iso::from_parts(Translation3::new(x, y, z), UnitQuaternion::identity());
        let joints = vec![
            Joint::revolute(o(0.0, 0.0, 0.1), Vector3::z()),
            Joint::revolute(o(0.2, 0.0, 0.0), Vector3::y()),
            Joint::prismatic(o(0.15, 0.0, 0.05), Vector3::x()),
            Joint::revolute(o(0.0, 0.25, 0.0), Vector3::x()),
        ];
        (joints, vec![-1, 0, 1, 0])
    }

    fn fk(joints: &[Joint], parent: &[isize], q: &[f64]) -> Vec<Iso> {
        let mut w = vec![Iso::identity(); joints.len()];
        for i in 0..joints.len() {
            let local = joints[i].origin * if joints[i].kind == JointKind::Revolute { Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&joints[i].axis, q[i])) } else { Iso::from_parts(Translation3::from(joints[i].axis.into_inner() * q[i]), UnitQuaternion::identity()) };
            w[i] = if parent[i] < 0 { local } else { w[parent[i] as usize] * local };
        }
        w
    }

    #[test]
    fn the_point_jacobian_differentiates_the_forward_kinematics() {
        let (joints, parent) = chain();
        let q = [0.3, -0.4, 0.12, 0.9];
        let offset = Vector3::new(0.07, -0.02, 0.11);
        for body in [0usize, 1, 2, 3] {
            let frames = fk(&joints, &parent, &q);
            let point = (frames[body] * Point3::from(offset)).coords;
            let j = tree_point_jacobian(&joints, &parent, &frames, Some(body), point);
            for k in 0..4 {
                let h = 1e-7;
                let (mut lo, mut hi) = (q, q);
                lo[k] -= h;
                hi[k] += h;
                let pl = (fk(&joints, &parent, &lo)[body] * Point3::from(offset)).coords;
                let ph = (fk(&joints, &parent, &hi)[body] * Point3::from(offset)).coords;
                let fd = (ph - pl) / (2.0 * h);
                let col = j.column(k);
                assert!((fd - col).norm() < 1e-6, "body {body} column {k}: {:?} vs finite difference {:?}", col.as_slice(), fd.as_slice());
            }
        }
    }

    #[test]
    fn the_angular_jacobian_differentiates_the_orientation() {
        let (joints, parent) = chain();
        let q = [0.3, -0.4, 0.12, 0.9];
        for body in [0usize, 1, 2, 3] {
            let frames = fk(&joints, &parent, &q);
            let j = tree_angular_jacobian(&joints, &parent, &frames, Some(body));
            for k in 0..4 {
                let h = 1e-7;
                let (mut lo, mut hi) = (q, q);
                lo[k] -= h;
                hi[k] += h;
                let rl = fk(&joints, &parent, &lo)[body].rotation;
                let rh = fk(&joints, &parent, &hi)[body].rotation;
                // ω = (Ṙ Rᵀ)ᵛ, taken as the axis-angle of the small relative rotation over the step
                let d = rh * rl.inverse();
                let fd = d.scaled_axis() / (2.0 * h);
                let col = j.column(k);
                assert!((fd - col).norm() < 1e-6, "body {body} column {k}: {:?} vs finite difference {:?}", col.as_slice(), fd.as_slice());
            }
        }
    }

    /// **Against MuJoCo itself, on a model the MJCF loader read.** A slide + two hinges with a sphere on the
    /// floor; MuJoCo's `efc_J` for the elliptic contact and its `body_invweight0` are from
    /// `scripts/mujoco_efcj_probe.py`. This is the check that the glue orders its rows, signs its normal and
    /// picks its bodies the way MuJoCo does — none of which the finite-difference tests above can see.
    #[test]
    fn the_contact_rows_and_invweights_match_mujoco_on_a_loaded_model() {
        const XML: &str = r#"<mujoco>
  <option cone="elliptic" jacobian="dense"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="cart" pos="0.1 -0.05 0.36">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0.05 0.02 -0.3" size="0.03"/>
      <body name="tip" pos="0.05 0.02 -0.3">
        <joint name="wrist" type="hinge" axis="1 0 0"/>
        <geom name="ball" type="sphere" size="0.08"/>
      </body>
    </body>
  </worldbody>
</mujoco>"#;
        let t = crate::tree_from_mjcf_str(XML).expect("loads");
        let q = [0.0, 0.15, 0.2];
        assert_eq!(t.tree.dof(), 3);
        let frames = tree_frames(&t.tree, &q);

        // MuJoCo's contact: floor (world) against the ball on body `tip`, normal +z, at this point
        let point = Vector3::new(0.10460711415472237, -0.03, -0.012051615002246358);
        let frame = [Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)];
        let tip = t.body_frames.get("tip").expect("tip body").0;
        let j = contact_jacobian(&t.tree.joints, &t.tree.parent, &frames, None, Some(tip), point, &frame, 3);
        let want = [
            [0.0, -0.004607114154722369, 0.0],
            [0.0, 0.0, 0.0671853978782421],
            [-1.0, 0.37205161500224637, 0.0],
        ];
        for (r, row) in want.iter().enumerate() {
            for (k, w) in row.iter().enumerate() {
                assert!((j[(r, k)] - w).abs() < 1e-12, "efc_J[{r}][{k}] = {} vs MuJoCo {w}", j[(r, k)]);
            }
        }

        // body_invweight0 is evaluated at qpos0, not at the current state
        let zero = vec![0.0; t.tree.dof()];
        let f0 = tree_frames(&t.tree, &zero);
        let m0 = crate::tree_mass_matrix(&t.tree.joints, &t.tree.inertia, &t.tree.parent, &zero);
        let m_inv = m0.clone().try_inverse().expect("mass matrix is invertible");
        for (body, tran, rot) in [("cart", 0.2099019541981764, 9.152164669297447), ("tip", 0.14984443260001193, 69.86495729163299)] {
            let idx = t.body_frames.get(body).expect("body").0;
            // the centre of mass must be a WORLD point: `LinkInertia::com` is in the joint's own frame
            let w = body_invweight(&m_inv, &t.tree.joints, &t.tree.parent, &f0, Some(idx), body_com(&f0, &t.tree.inertia, idx));
            assert!((w.tran - tran).abs() < 1e-9 * tran, "{body} invweight tran {} vs MuJoCo {tran}", w.tran);
            assert!((w.rot - rot).abs() < 1e-9 * rot, "{body} invweight rot {} vs MuJoCo {rot}", w.rot);
        }
    }

    #[test]
    fn the_contact_rows_are_the_relative_velocity_in_the_contact_frame() {
        let (joints, parent) = chain();
        let q = [0.3, -0.4, 0.12, 0.9];
        let frames = fk(&joints, &parent, &q);
        let point = Vector3::new(0.31, 0.02, 0.2);
        let n = Vector3::new(0.0, 0.0, 1.0);
        let frame = crate::make_frame(n, Vector3::zeros());
        let j = contact_jacobian(&joints, &parent, &frames, Some(3), Some(2), point, &frame, 3);
        let dif = tree_point_jacobian(&joints, &parent, &frames, Some(2), point) - tree_point_jacobian(&joints, &parent, &frames, Some(3), point);
        for r in 0..3 {
            for k in 0..4 {
                assert!((j[(r, k)] - frame[r].dot(&dif.column(k))).abs() < 1e-15);
            }
        }
        // a contact against the world moves only with the one body
        let against_world = contact_jacobian(&joints, &parent, &frames, None, Some(2), point, &frame, 3);
        let own = tree_point_jacobian(&joints, &parent, &frames, Some(2), point);
        for r in 0..3 {
            for k in 0..4 {
                assert!((against_world[(r, k)] - frame[r].dot(&own.column(k))).abs() < 1e-15);
            }
        }
        // condim 6 adds the torsional row about the normal and two rolling rows about the tangents
        let six = contact_jacobian(&joints, &parent, &frames, Some(3), Some(2), point, &frame, 6);
        assert_eq!(six.nrows(), 6);
        let adif = tree_angular_jacobian(&joints, &parent, &frames, Some(2)) - tree_angular_jacobian(&joints, &parent, &frames, Some(3));
        for (r, axis) in [0usize, 1, 2].into_iter().enumerate() {
            for k in 0..4 {
                assert!((six[(3 + r, k)] - frame[axis].dot(&adif.column(k))).abs() < 1e-15);
            }
        }
        assert_eq!(contact_jacobian(&joints, &parent, &frames, Some(3), Some(2), point, &frame, 4).nrows(), 4);
        assert_eq!(contact_jacobian(&joints, &parent, &frames, Some(3), Some(2), point, &frame, 1).nrows(), 1);
    }
}
