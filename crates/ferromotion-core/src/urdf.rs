//! URDF loading. Parses from a string (not a file path) so it works identically native and
//! in the browser: fetch the URDF over HTTP, hand us the text. Fixed joints are folded into
//! neighbouring origins so the resulting [`Robot`] is a clean actuated serial chain.

use crate::dynamics::{combine_inertia, transform_inertia, LinkInertia};
use crate::{Iso, Joint, Robot};
use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};
use std::collections::HashMap;

fn pose_to_iso(p: &urdf_rs::Pose) -> Iso {
    let t = Translation3::new(p.xyz[0], p.xyz[1], p.xyz[2]);
    // URDF rpy is fixed-axis roll(x)/pitch(y)/yaw(z) — nalgebra's from_euler_angles matches.
    let r = UnitQuaternion::from_euler_angles(p.rpy[0], p.rpy[1], p.rpy[2]);
    Iso::from_parts(t, r)
}

/// Inertial parameters of a named link, in the link frame (zero if the URDF omits `<inertial>`).
fn link_inertia(robot: &urdf_rs::Robot, name: &str) -> LinkInertia {
    let Some(link) = robot.links.iter().find(|l| l.name == name) else {
        return LinkInertia::zero();
    };
    let ine = &link.inertial;
    let com = Vector3::new(ine.origin.xyz[0], ine.origin.xyz[1], ine.origin.xyz[2]);
    let rm = *UnitQuaternion::from_euler_angles(ine.origin.rpy[0], ine.origin.rpy[1], ine.origin.rpy[2])
        .to_rotation_matrix()
        .matrix();
    let i = &ine.inertia;
    // URDF inertia is about the COM in the inertial-origin orientation; rotate into the link frame.
    let ic = Matrix3::new(i.ixx, i.ixy, i.ixz, i.ixy, i.iyy, i.iyz, i.ixz, i.iyz, i.izz);
    LinkInertia { mass: ine.mass.value, com, inertia: rm * ic * rm.transpose() }
}

/// Build an actuated serial [`Robot`] from URDF text (tree path `base` → `tip`). Revolute/
/// continuous/prismatic joints become DoF; fixed joints are absorbed into the adjacent transform.
pub fn from_urdf_str(xml: &str, base: &str, tip: &str) -> Result<Robot, String> {
    Ok(from_urdf_full(xml, base, tip)?.0)
}

/// Like [`from_urdf_str`], but also returns each actuated link's [`LinkInertia`] (for dynamics).
/// A fixed joint's child link is welded onto the preceding actuated link (composite inertia).
pub fn from_urdf_full(xml: &str, base: &str, tip: &str) -> Result<(Robot, Vec<LinkInertia>), String> {
    let robot = urdf_rs::read_from_string(xml).map_err(|e| format!("URDF parse error: {e}"))?;

    let by_child: HashMap<&str, &urdf_rs::Joint> =
        robot.joints.iter().map(|j| (j.child.link.as_str(), j)).collect();

    // Walk from tip up to base, collecting the joints on the path.
    let mut chain: Vec<&urdf_rs::Joint> = Vec::new();
    let mut link = tip;
    let mut guard = 0;
    while link != base {
        let j = *by_child
            .get(link)
            .ok_or_else(|| format!("no joint produces link '{link}' on the path to base '{base}'"))?;
        chain.push(j);
        link = j.parent.link.as_str();
        guard += 1;
        if guard > robot.joints.len() + 1 {
            return Err("cycle or broken URDF tree".into());
        }
    }
    chain.reverse();

    let mut joints = Vec::new();
    let mut inertias: Vec<LinkInertia> = Vec::new();
    let mut pre = Iso::identity(); // fixed transforms since the last actuated joint
    for j in chain {
        let origin = pose_to_iso(&j.origin);
        let child = link_inertia(&robot, &j.child.link);
        let mut axis = Vector3::new(j.axis.xyz[0], j.axis.xyz[1], j.axis.xyz[2]);
        if axis.norm() < 1e-9 {
            axis = Vector3::x(); // URDF default axis
        }
        use urdf_rs::JointType::*;
        match j.joint_type {
            Fixed => {
                pre *= origin;
                // Weld this fixed link onto the last actuated link (transform into its frame).
                if let Some(last) = inertias.last_mut() {
                    *last = combine_inertia(last, &transform_inertia(&child, &pre));
                }
            }
            Revolute => {
                joints.push(Joint::revolute(pre * origin, axis).with_limits(j.limit.lower, j.limit.upper));
                inertias.push(child);
                pre = Iso::identity();
            }
            Continuous => {
                joints.push(Joint::revolute(pre * origin, axis));
                inertias.push(child);
                pre = Iso::identity();
            }
            Prismatic => {
                joints.push(Joint::prismatic(pre * origin, axis).with_limits(j.limit.lower, j.limit.upper));
                inertias.push(child);
                pre = Iso::identity();
            }
            Floating | Planar | Spherical => {
                return Err(format!("unsupported joint type (floating/planar/spherical) on '{}'", j.name))
            }
        }
    }

    Ok((Robot { joints, ee_offset: pre }, inertias))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pose_error, solve_ik, IkOptions};

    /// A real 6-DoF spatial arm: fixed base, revolute z/y joints, fixed tool.
    const ARM_6DOF: &str = r#"
<robot name="arm6">
  <link name="world"/><link name="base"/>
  <link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="l6"/><link name="tool"/>
  <joint name="j0" type="fixed">
    <parent link="world"/><child link="base"/><origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="l1"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="l2"/><origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="l2"/><child link="l3"/><origin xyz="0 0 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j4" type="revolute">
    <parent link="l3"/><child link="l4"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j5" type="revolute">
    <parent link="l4"/><child link="l5"/><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="j6" type="revolute">
    <parent link="l5"/><child link="l6"/><origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-3.14" upper="3.14" effort="10" velocity="3"/>
  </joint>
  <joint name="jtool" type="fixed">
    <parent link="l6"/><child link="tool"/><origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
</robot>"#;

    #[test]
    fn loads_arm_and_folds_fixed_joints() {
        let r = from_urdf_str(ARM_6DOF, "world", "tool").unwrap();
        assert_eq!(r.dof(), 6, "6 actuated joints (2 fixed folded away)");
        // At q=0 every rotation is identity, so the tool sits at the summed z-offsets.
        let p = r.fk(&[0.0; 6]).translation.vector;
        assert!((p - Vector3::new(0.0, 0.0, 0.85)).norm() < 1e-9, "tool at {p:?}");
    }

    #[test]
    fn rpy_origin_is_parsed() {
        // A single fixed joint rotated +90° about x; loader folds it into the tool offset.
        let xml = r#"<robot name="r"><link name="base"/><link name="tool"/>
          <joint name="j" type="fixed"><parent link="base"/><child link="tool"/>
          <origin xyz="0 0 1" rpy="1.5707963 0 0"/></joint></robot>"#;
        let r = from_urdf_str(xml, "base", "tool").unwrap();
        let ee = r.fk(&[]);
        assert!((ee.translation.vector - Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-9);
        // +90° about x maps +y → +z.
        let mapped = ee.rotation * Vector3::y();
        assert!((mapped - Vector3::z()).norm() < 1e-6, "rotated y = {mapped:?}");
    }

    #[test]
    fn ik_on_loaded_arm() {
        let r = from_urdf_str(ARM_6DOF, "world", "tool").unwrap();
        let q_true = [0.2f64, -0.6, 0.9, 0.4, 0.7, -0.3];
        let target = r.fk(&q_true);
        let seed: Vec<f64> = q_true.iter().map(|q| q + 0.15).collect();
        let res = solve_ik(&r, &target, &seed, &IkOptions::default());
        assert!(res.converged, "did not converge: err={} iters={}", res.error, res.iters);
        assert!(pose_error(&r.fk(&res.q), &target).norm() < 1e-4);
    }
}
