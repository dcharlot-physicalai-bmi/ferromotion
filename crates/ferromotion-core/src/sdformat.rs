//! **SDFormat (Gazebo SDF) import** — the third model format alongside URDF and MJCF. SDF is
//! element-text XML (`<pose>x y z r p y</pose>`, `<mass>2.0</mass>`) rather than URDF's attributes,
//! so it carries a small self-contained text-capturing parser. Loads a serial kinematic chain from
//! base to tip: link inertials (`mass`, COM `pose`, the six inertia products) and joints
//! (`revolute`/`prismatic`/`fixed`, parent/child, `pose`, `axis`). Joint `<pose>` is taken as the
//! parent→joint transform (the common `relative_to` = parent case). Pure `nalgebra` → WASM-clean.

use crate::{Iso, Joint, JointKind, LinkInertia, Robot};
use nalgebra::{Matrix3, Translation3, Unit, UnitQuaternion, Vector3};
use std::collections::HashMap;

// ---- minimal text-capturing XML ----
#[derive(Debug, Default)]
struct Node {
    name: String,
    attrs: Vec<(String, String)>,
    text: String,
    children: Vec<Node>,
}
impl Node {
    fn child(&self, n: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == n)
    }
    fn children_named<'a>(&'a self, n: &'a str) -> impl Iterator<Item = &'a Node> {
        self.children.iter().filter(move |c| c.name == n)
    }
    fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
    fn text_f64(&self, n: &str, default: f64) -> f64 {
        self.child(n).and_then(|c| c.text.trim().parse().ok()).unwrap_or(default)
    }
}

fn parse(s: &str) -> Result<Node, String> {
    let b = s.as_bytes();
    let mut i = 0;
    let mut stack: Vec<Node> = vec![Node::default()];
    while i < b.len() {
        if b[i] != b'<' {
            let j = s[i..].find('<').map(|k| i + k).unwrap_or(b.len());
            let t = s[i..j].trim();
            if !t.is_empty() {
                stack.last_mut().unwrap().text.push_str(t);
            }
            i = j;
            continue;
        }
        if s[i..].starts_with("<!--") {
            i = s[i..].find("-->").map(|k| i + k + 3).ok_or("unterminated comment")?;
            continue;
        }
        if s[i..].starts_with("<?") || s[i..].starts_with("<!") {
            i = s[i..].find('>').map(|k| i + k + 1).ok_or("bad decl")?;
            continue;
        }
        let close = s[i..].find('>').map(|k| i + k).ok_or("unterminated tag")?;
        let inner = &s[i + 1..close];
        i = close + 1;
        if let Some(name) = inner.strip_prefix('/') {
            let done = stack.pop().ok_or("unbalanced close")?;
            if done.name != name.trim() {
                return Err(format!("mismatched </{}> vs <{}>", name.trim(), done.name));
            }
            stack.last_mut().unwrap().children.push(done);
        } else {
            let self_close = inner.ends_with('/');
            let inner = inner.trim_end_matches('/').trim();
            let mut parts = inner.split_whitespace();
            let name = parts.next().unwrap_or("").to_string();
            // attributes  key="value"
            let mut attrs = Vec::new();
            let rest = &inner[name.len()..];
            for cap in rest.split('"').collect::<Vec<_>>().chunks(2) {
                if let (2, Some(k)) = (cap.len(), cap.first().and_then(|c| c.trim().strip_suffix('='))) {
                    attrs.push((k.trim().to_string(), cap[1].to_string()));
                }
            }
            let node = Node { name, attrs, text: String::new(), children: vec![] };
            if self_close {
                stack.last_mut().unwrap().children.push(node);
            } else {
                stack.push(node);
            }
        }
    }
    stack.pop().ok_or("empty document".into())
}

fn triple(s: &str) -> Vector3<f64> {
    let v: Vec<f64> = s.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    Vector3::new(*v.first().unwrap_or(&0.0), *v.get(1).unwrap_or(&0.0), *v.get(2).unwrap_or(&0.0))
}

/// Parse an SDF `<pose>x y z roll pitch yaw</pose>` into an isometry (extrinsic XYZ Euler).
fn pose_iso(node: Option<&Node>) -> Iso {
    let txt = node.map(|n| n.text.trim()).unwrap_or("");
    let v: Vec<f64> = txt.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    let g = |i: usize| *v.get(i).unwrap_or(&0.0);
    Iso::from_parts(Translation3::new(g(0), g(1), g(2)), UnitQuaternion::from_euler_angles(g(3), g(4), g(5)))
}

/// Load a serial chain from `base` link to `tip` link. Returns the `Robot` and per-actuated-link
/// inertias (fixed joints fold their transform into the following actuated joint).
pub fn from_sdf(xml: &str, base: &str, tip: &str) -> Result<(Robot, Vec<LinkInertia>), String> {
    let root = parse(xml)?;
    // find the model element anywhere near the top
    let model = root.child("sdf").and_then(|s| s.child("model")).or_else(|| root.child("model")).ok_or("no <model>")?;

    // index links and joints
    let mut link_inertial: HashMap<String, LinkInertia> = HashMap::new();
    for link in model.children_named("link") {
        let name = link.attr("name").ok_or("link without name")?.to_string();
        if let Some(inr) = link.child("inertial") {
            let mass = inr.text_f64("mass", 0.0);
            let com = pose_iso(inr.child("pose")).translation.vector;
            let ii = inr.child("inertia");
            let g = |k: &str| ii.map(|n| n.text_f64(k, 0.0)).unwrap_or(0.0);
            let (ixx, iyy, izz, ixy, ixz, iyz) = (g("ixx"), g("iyy"), g("izz"), g("ixy"), g("ixz"), g("iyz"));
            let inertia = Matrix3::new(ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz);
            link_inertial.insert(name, LinkInertia { mass, com, inertia });
        } else {
            link_inertial.insert(name, LinkInertia { mass: 0.0, com: Vector3::zeros(), inertia: Matrix3::zeros() });
        }
    }
    let by_child: HashMap<String, &Node> = model
        .children_named("joint")
        .filter_map(|j| j.child("child").map(|c| (c.text.trim().to_string(), j)))
        .collect();

    // walk tip → base collecting joints
    let mut chain: Vec<&Node> = Vec::new();
    let mut link = tip.to_string();
    let mut guard = 0;
    while link != base {
        let j = *by_child.get(&link).ok_or_else(|| format!("no joint produces link '{link}'"))?;
        chain.push(j);
        link = j.child("parent").map(|p| p.text.trim().to_string()).ok_or("joint without parent")?;
        guard += 1;
        if guard > model.children.len() + 2 {
            return Err("cycle or broken SDF tree".into());
        }
    }
    chain.reverse();

    let mut joints = Vec::new();
    let mut inertias = Vec::new();
    let mut pre = Iso::identity();
    for j in chain {
        let origin = pre * pose_iso(j.child("pose"));
        let jtype = j.attr("type").unwrap_or("fixed");
        let child_link = j.child("child").map(|c| c.text.trim().to_string()).unwrap_or_default();
        if jtype == "fixed" {
            pre = origin; // fold into the next actuated joint
            continue;
        }
        let ax = j.child("axis");
        let axis = ax.and_then(|a| a.child("xyz")).map(|x| triple(&x.text)).unwrap_or(Vector3::z());
        let kind = if jtype == "prismatic" { JointKind::Prismatic } else { JointKind::Revolute };
        // SDF keeps all of this under <axis>: position and capability bounds in <limit>, passive damping in
        // <dynamics>. It was reachable the whole time (the accessor above already holds <axis>) and simply
        // not read, which is the same defect `Joint::effort` records for URDF. SDF has no armature field.
        let num = |parent: Option<&Node>, tag: &str| -> Option<f64> {
            parent.and_then(|p| p.child(tag)).and_then(|c| c.text.trim().parse::<f64>().ok())
        };
        let limit = ax.and_then(|a| a.child("limit"));
        let mut joint = Joint {
            origin,
            axis: Unit::new_normalize(axis),
            kind,
            limits: None,
            effort: None,
            max_velocity: None,
            armature: None,
            damping: None,
            friction: None,
        };
        if let (Some(lo), Some(hi)) = (num(limit, "lower"), num(limit, "upper")) {
            joint = joint.with_limits(lo, hi);
        }
        if let Some(e) = num(limit, "effort") {
            joint = joint.with_effort(e);
        }
        if let Some(v) = num(limit, "velocity") {
            joint = joint.with_max_velocity(v);
        }
        let dyn_el = ax.and_then(|a| a.child("dynamics"));
        if let Some(d) = num(dyn_el, "damping") {
            joint = joint.with_damping(d);
        }
        if let Some(f) = num(dyn_el, "friction") {
            joint = joint.with_friction(f);
        }
        joints.push(joint);
        inertias.push(link_inertial.get(&child_link).cloned().unwrap_or(LinkInertia { mass: 0.0, com: Vector3::zeros(), inertia: Matrix3::zeros() }));
        pre = Iso::identity();
    }
    Ok((Robot { joints, ee_offset: pre }, inertias))
}

#[cfg(test)]
mod verification {
    use super::*;
    use crate::{gravity_vector, mass_matrix};

    /// The same pendulum the URDF/MJCF loaders are checked on, in SDF: a 2 kg link with COM 0.5 m
    /// out about a y-axis joint. Loaded dynamics reproduce the analytic gravity torque `m·g·d` and
    /// axis inertia `I_yy + m·d²`.
    #[test]
    fn sdf_pendulum_dynamics() {
        let sdf = r#"<sdf version="1.7"><model name="pend">
          <link name="base"/>
          <link name="l1">
            <inertial><mass>2.0</mass><pose>0.5 0 0 0 0 0</pose>
              <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
            </inertial>
          </link>
          <joint name="j1" type="revolute">
            <parent>base</parent><child>l1</child><pose>0 0 0 0 0 0</pose>
            <axis><xyz>0 1 0</xyz></axis>
          </joint>
        </model></sdf>"#;
        let (robot, inertia) = from_sdf(sdf, "base", "l1").unwrap();
        assert_eq!(robot.dof(), 1);
        let g = gravity_vector(&robot, &inertia, &[0.0], Vector3::new(0.0, 0.0, -9.81));
        let m = mass_matrix(&robot, &inertia, &[0.0]);
        eprintln!("SDF pendulum: gravity torque {:.4} (expect 9.81), M[0,0] {:.4} (expect 0.51)", g[0], m[(0, 0)]);
        assert!((g[0].abs() - 9.81).abs() < 1e-4, "gravity torque {}", g[0]);
        assert!((m[(0, 0)] - 0.51).abs() < 1e-6, "M[0,0] = {}", m[(0, 0)]);
    }

    /// A 2-DoF SDF chain parses to the right structure (2 actuated joints, masses recovered).
    #[test]
    fn sdf_two_dof_chain() {
        let sdf = r#"<sdf><model name="a2">
          <link name="l0"/>
          <link name="l1"><inertial><mass>1.5</mass><pose>0 0 0.1 0 0 0</pose>
            <inertia><ixx>0.02</ixx><iyy>0.02</iyy><izz>0.005</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial></link>
          <link name="l2"><inertial><mass>0.8</mass><pose>0.1 0 0 0 0 0</pose>
            <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.003</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial></link>
          <joint name="j1" type="revolute"><parent>l0</parent><child>l1</child><pose>0 0 0.05 0 0 0</pose><axis><xyz>0 0 1</xyz></axis></joint>
          <joint name="j2" type="prismatic"><parent>l1</parent><child>l2</child><pose>0 0 0.3 0 0 0</pose><axis><xyz>1 0 0</xyz></axis></joint>
        </model></sdf>"#;
        let (robot, inertia) = from_sdf(sdf, "l0", "l2").unwrap();
        assert_eq!(robot.dof(), 2);
        assert!((inertia[0].mass - 1.5).abs() < 1e-9 && (inertia[1].mass - 0.8).abs() < 1e-9, "masses wrong");
        assert_eq!(robot.joints[1].kind, JointKind::Prismatic);
        eprintln!("SDF 2-DoF chain: dof {}, masses {:.2}/{:.2}", robot.dof(), inertia[0].mass, inertia[1].mass);
    }

    /// **SDF states limits, effort, velocity and damping under `<axis>`, and none of it was read.**
    ///
    /// The parser already held the `<axis>` element to get the joint direction, so every one of these was a
    /// single accessor away. This is the same defect `Joint::effort` records for URDF, found by fixing that one
    /// and then checking whether the other importers had it too — a fix in one loader is a hypothesis about the
    /// rest. SDF has no armature concept, so that stays `None`.
    #[test]
    fn sdf_limits_effort_velocity_and_damping_survive_the_loader() {
        let sdf = r#"<sdf version="1.7"><model name="arm">
          <link name="base"/>
          <link name="l1">
            <inertial><mass>2.0</mass><pose>0.5 0 0 0 0 0</pose>
              <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
            </inertial>
          </link>
          <joint name="j1" type="revolute">
            <parent>base</parent><child>l1</child><pose>0 0 0 0 0 0</pose>
            <axis><xyz>0 1 0</xyz>
              <limit><lower>-1.2</lower><upper>2.4</upper><effort>7.5</effort><velocity>3.25</velocity></limit>
              <dynamics><damping>0.42</damping><friction>0.09</friction></dynamics>
            </axis>
          </joint>
        </model></sdf>"#;
        let (robot, _) = from_sdf(sdf, "base", "l1").unwrap();
        let j = &robot.joints[0];
        assert_eq!(j.limits, Some((-1.2, 2.4)));
        assert_eq!(j.effort, Some(7.5));
        assert_eq!(j.max_velocity, Some(3.25));
        assert_eq!(j.damping, Some(0.42));
        assert_eq!(j.friction, Some(0.09), "SDF states Coulomb friction under <axis><dynamics>");
        assert_eq!(j.armature, None, "SDF has no armature field to read");
    }

    /// A joint that states none of it must come back unstated, not defaulted. Without this the test above
    /// would pass on a loader that filled in constants regardless of the file.
    #[test]
    fn sdf_says_nothing_when_the_file_says_nothing() {
        let sdf = r#"<sdf version="1.7"><model name="arm">
          <link name="base"/>
          <link name="l1">
            <inertial><mass>2.0</mass><pose>0.5 0 0 0 0 0</pose>
              <inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
            </inertial>
          </link>
          <joint name="j1" type="revolute">
            <parent>base</parent><child>l1</child><pose>0 0 0 0 0 0</pose>
            <axis><xyz>0 1 0</xyz></axis>
          </joint>
        </model></sdf>"#;
        let (robot, _) = from_sdf(sdf, "base", "l1").unwrap();
        let j = &robot.joints[0];
        assert_eq!(
            (j.limits, j.effort, j.max_velocity, j.damping, j.friction),
            (None, None, None, None, None)
        );
    }
}
