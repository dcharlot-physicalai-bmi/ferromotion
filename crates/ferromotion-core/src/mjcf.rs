//! MJCF (MuJoCo XML) loading — the model format of the MuJoCo ecosystem, parsed from a string so
//! it works identically native and in the browser. Produces the same `(Robot, Vec<LinkInertia>)`
//! as the URDF loader, so every downstream consumer (dynamics, gendyn, calib, control) is
//! format-agnostic.
//!
//! **Honest subset** (asserted, not silently wrong): a single serial chain of nested `<body>`
//! elements under `<worldbody>`, at most one `<joint>` per body (`hinge` — the MJCF default — or
//! `slide`), explicit `<inertial>` blocks (we do not infer inertia from geoms), `pos` +
//! `quat`/`euler`/`axisangle` orientations, joint `pos` offsets and `axis`. MuJoCo *conventions*
//! are honored where they differ from URDF: **angles default to degrees** (`<compiler
//! angle="radian">` switches), quaternions are `(w, x, y, z)`, `euler` is the intrinsic x-y-z
//! sequence, and a joint's `pos` places its axis *through a point of the body frame* — folded here
//! into the joint origin with the residual carried to the next body. Jointless bodies fold into
//! the adjacent transform like URDF fixed joints. Zero-dependency mini-XML parser (no namespaces,
//! comments skipped) — wasm-clean.

use crate::dynamics::LinkInertia;
use crate::{Iso, Joint, JointKind, Robot};
use nalgebra::{Matrix3, Translation3, Unit, UnitQuaternion, Vector3};

// ---------------------------------------------------------------------------------------------
// Mini XML: permissive, non-validating, enough for MJCF.
// ---------------------------------------------------------------------------------------------

#[derive(Debug)]
struct El {
    name: String,
    attrs: Vec<(String, String)>,
    children: Vec<El>,
}

impl El {
    fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
    fn child(&self, name: &str) -> Option<&El> {
        self.children.iter().find(|c| c.name == name)
    }
    fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a El> {
        self.children.iter().filter(move |c| c.name == name)
    }
}

fn parse_xml(s: &str) -> Result<El, String> {
    let b = s.as_bytes();
    let mut i = 0usize;
    let mut stack: Vec<El> = vec![El { name: String::new(), attrs: vec![], children: vec![] }];
    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        if s[i..].starts_with("<!--") {
            i = s[i..].find("-->").map(|j| i + j + 3).ok_or("unterminated comment")?;
            continue;
        }
        if s[i..].starts_with("<?") || s[i..].starts_with("<!") {
            i = s[i..].find('>').map(|j| i + j + 1).ok_or("unterminated declaration")?;
            continue;
        }
        let close = s[i..].find('>').map(|j| i + j).ok_or("unterminated tag")?;
        let inner = &s[i + 1..close];
        i = close + 1;
        if let Some(name) = inner.strip_prefix('/') {
            let done = stack.pop().ok_or("unbalanced close tag")?;
            if done.name != name.trim() {
                return Err(format!("mismatched </{}> for <{}>", name.trim(), done.name));
            }
            stack.last_mut().ok_or("close past root")?.children.push(done);
            continue;
        }
        let self_close = inner.ends_with('/');
        let inner = inner.strip_suffix('/').unwrap_or(inner);
        let mut parts = inner.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").to_string();
        let mut attrs = Vec::new();
        if let Some(rest) = parts.next() {
            let rb = rest.as_bytes();
            let mut k = 0usize;
            while k < rb.len() {
                while k < rb.len() && (rb[k] as char).is_whitespace() {
                    k += 1;
                }
                let ks = k;
                while k < rb.len() && rb[k] != b'=' && !(rb[k] as char).is_whitespace() {
                    k += 1;
                }
                if ks == k {
                    break;
                }
                let key = rest[ks..k].to_string();
                while k < rb.len() && (rb[k] == b'=' || (rb[k] as char).is_whitespace()) {
                    k += 1;
                }
                if k < rb.len() && (rb[k] == b'"' || rb[k] == b'\'') {
                    let q = rb[k];
                    k += 1;
                    let vs = k;
                    while k < rb.len() && rb[k] != q {
                        k += 1;
                    }
                    attrs.push((key, rest[vs..k].to_string()));
                    k += 1;
                }
            }
        }
        let el = El { name, attrs, children: vec![] };
        if self_close {
            stack.last_mut().ok_or("element past root")?.children.push(el);
        } else {
            stack.push(el);
        }
    }
    let mut root = stack.pop().ok_or("empty document")?;
    if !stack.is_empty() {
        return Err(format!("unclosed <{}>", root.name));
    }
    root.children.pop().ok_or_else(|| "no root element".to_string())
}

// ---------------------------------------------------------------------------------------------
// MJCF semantics.
// ---------------------------------------------------------------------------------------------

fn floats(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace().map(|t| t.parse::<f64>().map_err(|e| format!("bad number '{t}': {e}"))).collect()
}

fn vec3(s: &str) -> Result<Vector3<f64>, String> {
    let v = floats(s)?;
    if v.len() != 3 {
        return Err(format!("expected 3 numbers, got '{s}'"));
    }
    Ok(Vector3::new(v[0], v[1], v[2]))
}

/// Orientation of an element from MJCF's `quat` / `euler` / `axisangle` (first present wins;
/// identity otherwise). `deg` converts euler/axisangle angles when the compiler is in degree mode.
fn orientation(el: &El, deg: f64) -> Result<UnitQuaternion<f64>, String> {
    if let Some(q) = el.attr("quat") {
        let v = floats(q)?;
        if v.len() != 4 {
            return Err("quat needs 4 numbers (w x y z)".into());
        }
        // MJCF order is (w, x, y, z)
        return Ok(UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[0], v[1], v[2], v[3])));
    }
    if let Some(e) = el.attr("euler") {
        let v = floats(e)?;
        if v.len() != 3 {
            return Err("euler needs 3 numbers".into());
        }
        // MJCF default eulerseq "xyz": intrinsic rotations → R = Rx·Ry·Rz
        let rx = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), v[0] * deg);
        let ry = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), v[1] * deg);
        let rz = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), v[2] * deg);
        return Ok(rx * ry * rz);
    }
    if let Some(aa) = el.attr("axisangle") {
        let v = floats(aa)?;
        if v.len() != 4 {
            return Err("axisangle needs 4 numbers".into());
        }
        let axis = Unit::new_normalize(Vector3::new(v[0], v[1], v[2]));
        return Ok(UnitQuaternion::from_axis_angle(&axis, v[3] * deg));
    }
    Ok(UnitQuaternion::identity())
}

fn pose(el: &El, deg: f64) -> Result<Iso, String> {
    let p = el.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
    Ok(Iso::from_parts(Translation3::from(p), orientation(el, deg)?))
}

fn inertial(el: Option<&El>, deg: f64) -> Result<LinkInertia, String> {
    let Some(el) = el else { return Ok(LinkInertia::zero()) };
    let mass = el.attr("mass").ok_or("inertial needs mass")?.parse::<f64>().map_err(|e| e.to_string())?;
    let com = el.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
    let rot = orientation(el, deg)?;
    let rm = *rot.to_rotation_matrix().matrix();
    let ic = if let Some(d) = el.attr("diaginertia") {
        let v = floats(d)?;
        if v.len() != 3 {
            return Err("diaginertia needs 3 numbers".into());
        }
        Matrix3::from_diagonal(&Vector3::new(v[0], v[1], v[2]))
    } else if let Some(f) = el.attr("fullinertia") {
        let v = floats(f)?;
        if v.len() != 6 {
            return Err("fullinertia needs 6 numbers (xx yy zz xy xz yz)".into());
        }
        Matrix3::new(v[0], v[3], v[4], v[3], v[1], v[5], v[4], v[5], v[2])
    } else {
        return Err("inertial needs diaginertia or fullinertia".into());
    };
    // MJCF inertia is about the COM in the inertial frame's orientation; rotate into the body frame.
    Ok(LinkInertia { mass, com, inertia: rm * ic * rm.transpose() })
}

/// Build an actuated serial [`Robot`] + link inertias from MJCF text. See the module doc for the
/// honored subset and conventions.
pub fn from_mjcf_full(xml: &str) -> Result<(Robot, Vec<LinkInertia>), String> {
    let (robot, inertias, _cs) = from_mjcf_constrained(xml)?;
    Ok((robot, inertias))
}

/// [`from_mjcf_full`] plus the model's `<equality>` constraints as a ready
/// [`crate::constraint::ConstraintSet`]: `<connect body1 anchor>` welds the anchor point (in the
/// named body's frame) to its world position at `qpos = 0` — the MuJoCo semantics — and
/// `<joint joint1 joint2 polycoef="a0 a1 …">` becomes a mimic coupling `q₁ = a0 + a1·q₂`
/// (higher polynomial coefficients are outside the subset and fail loudly).
pub fn from_mjcf_constrained(xml: &str) -> Result<(Robot, Vec<LinkInertia>, crate::constraint::ConstraintSet), String> {
    let root = parse_xml(xml)?;
    if root.name != "mujoco" {
        return Err(format!("root element is <{}>, expected <mujoco>", root.name));
    }
    // MJCF defaults to DEGREES; <compiler angle="radian"> switches.
    let deg = match root.child("compiler").and_then(|c| c.attr("angle")) {
        Some("radian") => 1.0,
        Some("degree") | None => std::f64::consts::PI / 180.0,
        Some(other) => return Err(format!("unknown compiler angle '{other}'")),
    };
    let world = root.child("worldbody").ok_or("no <worldbody>")?;

    let mut joints = Vec::new();
    let mut inertias = Vec::new();
    let mut body_names: Vec<(String, usize, Vector3<f64>)> = Vec::new(); // (name, upto, joint pos shift)
    let mut joint_names: Vec<(String, usize)> = Vec::new();
    // carry: transform from the last joint's frame to the current body's parent frame
    let mut carry = Iso::identity();
    let mut body = world.child("body");
    while let Some(b) = body {
        let bpose = pose(b, deg)?;
        let joints_here: Vec<&El> = b.children_named("joint").collect();
        if joints_here.len() > 1 {
            return Err("composite joints (multiple <joint> per body) are outside the supported subset".into());
        }
        if b.children_named("body").count() > 1 {
            return Err("branching trees are outside the supported subset (serial chains only)".into());
        }
        if let Some(j) = joints_here.first() {
            let kind = match j.attr("type").unwrap_or("hinge") {
                "hinge" => JointKind::Revolute,
                "slide" => JointKind::Prismatic,
                other => return Err(format!("unsupported joint type '{other}' (hinge/slide only)")),
            };
            let axis = j.attr("axis").map(vec3).transpose()?.unwrap_or_else(|| Vector3::z()); // MJCF default axis
            let jpos = j.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
            // joint frame = body frame translated to the joint's anchor point
            let origin = carry * bpose * Iso::from_parts(Translation3::from(jpos), UnitQuaternion::identity());
            let mut joint = Joint { origin, axis: Unit::new_normalize(axis), kind, limits: None };
            if let (Some("true") | None, Some(r)) = (j.attr("limited"), j.attr("range")) {
                let v = floats(r)?;
                if v.len() == 2 {
                    let s = if kind == JointKind::Revolute { deg } else { 1.0 };
                    joint = joint.with_limits(v[0] * s, v[1] * s);
                }
            }
            joints.push(joint);
            if let Some(nm) = j.attr("name") {
                joint_names.push((nm.to_string(), joints.len() - 1));
            }
            if let Some(nm) = b.attr("name") {
                body_names.push((nm.to_string(), joints.len(), jpos));
            }
            // link frame = joint frame; body-frame quantities shift by −jpos
            let mut li = inertial(b.child("inertial"), deg)?;
            li.com -= jpos;
            inertias.push(li);
            // next body's pos is in THIS body frame = joint frame shifted back by jpos
            carry = Iso::from_parts(Translation3::from(-jpos), UnitQuaternion::identity());
        } else {
            // jointless body: fold its transform (and require it carry no inertia we would lose)
            let li = inertial(b.child("inertial"), deg)?;
            if li.mass > 0.0 {
                if let Some(last) = inertias.last_mut() {
                    // fold the static body's inertia into the previous link, expressed in its frame
                    let tf = carry * bpose;
                    let folded = crate::dynamics::transform_inertia(&li, &tf);
                    *last = crate::dynamics::combine_inertia(last, &folded);
                } else {
                    return Err("inertial mass on a jointless base body is outside the subset".into());
                }
            }
            carry = carry * bpose;
        }
        body = b.child("body");
    }
    if joints.is_empty() {
        return Err("no actuated joints found".into());
    }
    let robot = Robot { joints, ee_offset: carry };

    let mut cs = crate::constraint::ConstraintSet::new();
    if let Some(eq) = root.child("equality") {
        for c in eq.children_named("connect") {
            let b1 = c.attr("body1").ok_or("connect needs body1")?;
            if c.attr("body2").is_some() {
                return Err("connect body2 (body-body welds) is outside the subset — world welds only".into());
            }
            let anchor = c.attr("anchor").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
            let &(_, upto, jpos) = body_names
                .iter()
                .find(|(n, _, _)| n == b1)
                .ok_or_else(|| format!("connect body1 '{b1}' not found among jointed bodies"))?;
            // anchor is in the BODY frame; the link frame is shifted by the joint pos
            let local = anchor - jpos;
            // MuJoCo welds at the model configuration (qpos = 0): target = world point there
            let q0 = vec![0.0; robot.dof()];
            let target = (robot.frame_pose(&q0, upto) * nalgebra::Point3::from(local)).coords;
            cs.anchor_point(upto, local, target);
        }
        for jq in eq.children_named("joint") {
            let j1 = jq.attr("joint1").ok_or("equality joint needs joint1")?;
            let j2 = jq.attr("joint2").ok_or("only two-joint couplings are in the subset (joint2 required)")?;
            let poly = jq.attr("polycoef").map(floats).transpose()?.unwrap_or_else(|| vec![0.0, 1.0]);
            if poly.iter().skip(2).any(|&c| c != 0.0) {
                return Err("nonlinear polycoef couplings are outside the subset".into());
            }
            let (a0, a1) = (poly.first().copied().unwrap_or(0.0), poly.get(1).copied().unwrap_or(1.0));
            let find = |name: &str| {
                joint_names
                    .iter()
                    .find(|(n, _)| n == name)
                    .map(|&(_, i)| i)
                    .ok_or_else(|| format!("equality joint '{name}' not found"))
            };
            cs.mimic(find(j1)?, find(j2)?, a1, a0);
        }
    }
    Ok((robot, inertias, cs))
}

/// Kinematics-only variant of [`from_mjcf_full`].
pub fn from_mjcf_str(xml: &str) -> Result<Robot, String> {
    from_mjcf_full(xml).map(|(r, _)| r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inverse_dynamics;

    const ARM: &str = r#"
<mujoco model="arm3">
  <compiler angle="radian"/>
  <worldbody>
    <body pos="0 0 0.3" euler="0 0 0.4">
      <joint type="hinge" axis="0 0 1"/>
      <inertial pos="0.02 -0.01 0.05" mass="1.5" diaginertia="0.02 0.03 0.025"/>
      <body pos="0.1 0 0.2" quat="0.9887711 0.1494381 0 0">
        <joint type="hinge" axis="0 1 0" pos="0 0 0.05"/>
        <inertial pos="0 0 0.1" mass="1.8" fullinertia="0.025 0.032 0.025 0.001 0.002 0.0015"/>
        <body pos="0 0.05 0.25">
          <joint type="slide" axis="1 0 0"/>
          <inertial pos="0.04 0 0.07" mass="2.1" diaginertia="0.03 0.034 0.025"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"#;

    /// The MJCF chain must load and produce sane, consistent kinematics + dynamics: correct DoF
    /// kinds/axes, joint-pos offsets honored (FK differs from the offsetless chain), and RNEA runs.
    #[test]
    fn arm_loads_with_joint_offsets_and_inertias() {
        let (robot, inertia) = from_mjcf_full(ARM).expect("parse");
        assert_eq!(robot.dof(), 3);
        assert_eq!(robot.joints[0].kind, JointKind::Revolute);
        assert_eq!(robot.joints[2].kind, JointKind::Prismatic);
        assert_eq!(inertia.len(), 3);
        assert!((inertia[1].mass - 1.8).abs() < 1e-12);
        // joint-pos shift: link 1's COM is expressed in the JOINT frame (body com 0.1z − jpos 0.05z)
        assert!((inertia[1].com[2] - 0.05).abs() < 1e-12, "com z {}", inertia[1].com[2]);
        // FK is finite and moves with q
        let p0 = robot.fk(&[0.0, 0.0, 0.0]).translation.vector;
        let p1 = robot.fk(&[0.5, -0.3, 0.1]).translation.vector;
        assert!((p0 - p1).norm() > 1e-3);
        // dynamics runs and gravity loads the arm
        let tau = inverse_dynamics(&robot, &inertia, &[0.3, -0.5, 0.1], &[0.0; 3], &[0.0; 3], nalgebra::Vector3::new(0.0, 0.0, -9.81));
        assert!(tau[1].abs() > 0.1, "shoulder must feel gravity: {tau:?}");
    }

    /// MJCF quat order is (w, x, y, z) and a jointless body folds into the chain end.
    #[test]
    fn conventions_quat_and_fixed_body_folding() {
        let xml = r#"
<mujoco><compiler angle="radian"/><worldbody>
  <body pos="0 0 0.1">
    <joint/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
    <body pos="0 0 0.2" quat="0.7071068 0 0.7071068 0"/>
  </body>
</worldbody></mujoco>"#;
        let (robot, _) = from_mjcf_full(xml).expect("parse");
        assert_eq!(robot.dof(), 1);
        // ee offset: +0.2 z then a 90° rotation about y (w=x cos45 pattern in (w,x,y,z) order)
        let ee = robot.fk(&[0.0]);
        assert!((ee.translation.vector - nalgebra::Vector3::new(0.0, 0.0, 0.3)).norm() < 1e-6);
        let rot_y = ee.rotation.to_rotation_matrix();
        assert!((rot_y.matrix()[(0, 2)] - 1.0).abs() < 1e-6, "90° about y maps z→x");
    }

    /// MJCF defaults to DEGREES: with no <compiler>, euler="0 0 90" is a quarter turn.
    #[test]
    fn degrees_are_the_default() {
        let xml = r#"
<mujoco><worldbody>
  <body euler="0 0 90">
    <joint axis="1 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
  </body>
</worldbody></mujoco>"#;
        let robot = from_mjcf_str(xml).expect("parse");
        // the joint axis (1,0,0) in a frame yawed 90° points along world +y
        let jac = robot.jacobian(&[0.0]);
        assert!((jac[(4, 0)] - 1.0).abs() < 1e-6, "axis should be world +y, jac col {:?}", jac.column(0));
    }

    /// `<equality>` ingestion: a connect weld becomes a working anchor (the four-bar pattern) and
    /// a joint polycoef coupling becomes a working mimic — verified DYNAMICALLY through
    /// constrained_step, not just structurally.
    #[test]
    fn equality_connect_and_joint_become_working_constraints() {
        let xml = r#"
<mujoco><compiler angle="radian"/>
  <worldbody>
    <body name="l1" pos="0 0 0">
      <joint name="j1" axis="0 0 1"/>
      <inertial pos="0.5 0 0" mass="1" diaginertia="0.02 0.02 0.02"/>
      <body name="l2" pos="1 0 0">
        <joint name="j2" axis="0 0 1"/>
        <inertial pos="0.75 0 0" mass="1.5" diaginertia="0.04 0.04 0.04"/>
        <body name="l3" pos="1.5 0 0">
          <joint name="j3" axis="0 0 1"/>
          <inertial pos="0.5 0 0" mass="1" diaginertia="0.02 0.02 0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <connect body1="l3" anchor="1 0 0"/>
    <joint joint1="j2" joint2="j1" polycoef="0 -1 0 0 0"/>
  </equality>
</mujoco>"#;
        let (robot, inertia, cs) = from_mjcf_constrained(xml).expect("parse");
        assert!(!cs.is_empty());
        // dynamics under the loaded constraints: the anchored point must not move, and the mimic
        // must couple the first two joints, from the weld configuration q = 0
        let q = [0.0; 3];
        let v = [0.3, -0.3, 0.3]; // consistent with the mimic (v2 = −v1) at q0
        let res = crate::constraint::constrained_step(
            &robot,
            &inertia,
            &q,
            &v,
            &[0.5, 0.1, -0.2],
            1e-3,
            Vector3::new(0.0, -9.81, 0.0),
            &cs,
        );
        // mimic holds on the next velocities
        assert!((res.v_next[1] + res.v_next[0]).abs() < 1e-8, "mimic v2 = −v1: {:?}", res.v_next);
        // the welded point's velocity vanishes
        let jq: Vec<f64> = res.v_next.clone();
        let p_dot = {
            // finite-difference the anchor point's position along v_next
            let eps = 1e-7;
            let q2: Vec<f64> = q.iter().zip(&jq).map(|(a, b)| a + eps * b).collect();
            let p0 = (robot.frame_pose(&q, 3) * nalgebra::Point3::new(1.0, 0.0, 0.0)).coords;
            let p1 = (robot.frame_pose(&q2, 3) * nalgebra::Point3::new(1.0, 0.0, 0.0)).coords;
            (p1 - p0) / eps
        };
        assert!(p_dot.norm() < 1e-6, "welded point must be stationary: |ṗ| = {}", p_dot.norm());
    }

    #[test]
    fn equality_out_of_subset_fails_loudly() {
        let nonlinear = r#"
<mujoco><worldbody><body name="a"><joint name="j1"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
  <body name="b"><joint name="j2"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
</body></worldbody>
<equality><joint joint1="j1" joint2="j2" polycoef="0 1 0.5 0 0"/></equality></mujoco>"#;
        assert!(from_mjcf_constrained(nonlinear).unwrap_err().contains("nonlinear"));
    }

    /// Out-of-subset constructs fail loudly, not silently wrong.
    #[test]
    fn out_of_subset_is_a_loud_error() {
        let branching = r#"
<mujoco><worldbody><body><joint/>
  <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
  <body pos="1 0 0"><joint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
  <body pos="0 1 0"><joint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
</body></worldbody></mujoco>"#;
        assert!(from_mjcf_full(branching).unwrap_err().contains("serial"));
        let ball = r#"<mujoco><worldbody><body><joint type="ball"/></body></worldbody></mujoco>"#;
        assert!(from_mjcf_full(ball).unwrap_err().contains("hinge/slide"));
    }
}
