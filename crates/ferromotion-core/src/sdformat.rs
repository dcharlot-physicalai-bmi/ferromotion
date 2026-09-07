//! **SDFormat (Gazebo SDF) import** — the third model format alongside URDF and MJCF. SDF is
//! element-text XML (`<pose>x y z r p y</pose>`, `<mass>2.0</mass>`) rather than URDF's attributes,
//! so it carries a small self-contained text-capturing parser. Loads a serial kinematic chain from
//! base to tip: link inertials (`mass`, COM `pose`, the six inertia products) and joints
//! (`revolute`/`prismatic`/`fixed`, parent/child, `pose`, `axis`). Joint `<pose>` is taken as the
//! parent→joint transform (the common `relative_to` = parent case). Pure `nalgebra` → WASM-clean.
//!
//! ⛔ **TWO SILENT PATHS TO A MASSLESS LINK.** A link with no `<inertial>` gets a zero inertia, and so
//! does one whose `<inertial>` omits `<mass>` — the second through this file's own
//! `text_f64("mass", 0.0)` default, which a check for the element's presence alone would miss. Both
//! turn a link into a body that simulates and is wrong. [`geometry_from_sdf`] reports the names by
//! **both** routes and returns the geometry needed to compute the inertia with
//! [`primitive_link_inertia`](crate::primitive_link_inertia).

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

/// **The geometry an SDFormat model declares, and which links state no mass.**
///
/// ⛔ **A link with no `<inertial>` gets a ZERO inertia from [`from_sdf`], and so does one whose
/// `<inertial>` omits `<mass>`** — the latter through a `text_f64("mass", 0.0)` default. Both are
/// silent, and both turn a link into a massless body that simulates and is wrong. This function is how
/// a caller finds out, and how it gets the geometry to compute the inertia itself with
/// [`primitive_link_inertia`](crate::primitive_link_inertia).
///
/// SDF states sizes as **full extents**, like URDF and unlike MJCF, so nothing is halved or doubled
/// here. Its `<pose>` is `x y z roll pitch yaw`, six numbers of element text.
///
/// Supported: `<box><size>`, `<sphere><radius>`, `<cylinder><radius><length>`,
/// `<capsule><radius><length>` and `<mesh><uri>` with an optional `<scale>`. `<plane>`, `<heightmap>`
/// and `<polyline>` are reported as unsupported rather than approximated. A `model://` URI resolves
/// through [`resolve_uri`](crate::resolve_uri), the same table URDF's `package://` uses.
///
/// Returns the shapes paired with the names of links that carry no usable mass, in model order.
pub fn geometry_from_sdf(xml: &str) -> Result<(Vec<crate::GeometryRef>, Vec<String>), String> {
    let root = parse(xml)?;
    let model = root
        .child("sdf")
        .and_then(|s| s.child("model"))
        .or_else(|| root.child("model"))
        .ok_or("no <model>")?;

    let mut out = Vec::new();
    let mut massless = Vec::new();
    for link in model.children_named("link") {
        let name = link.attr("name").ok_or("link without name")?.to_string();
        // ⛔ BOTH ways a link ends up massless, not just the missing-<inertial> one: an `<inertial>`
        // that omits `<mass>` reads 0.0 through the loader's own default, and that is the case a check
        // for the element's presence alone would miss.
        let mass = link.child("inertial").map(|i| i.text_f64("mass", 0.0)).unwrap_or(0.0);
        if !(mass.is_finite() && mass > 0.0) {
            massless.push(name.clone());
        }
        for (role, tag) in [(crate::GeomRole::Collision, "collision"), (crate::GeomRole::Visual, "visual")] {
            for c in link.children_named(tag) {
                let geo = c.child("geometry").ok_or_else(|| format!("<{tag}> in link '{name}' has no <geometry>"))?;
                let geometry = sdf_geom(geo, &name)?;
                out.push(crate::GeometryRef { link: name.clone(), role, origin: pose_iso(c.child("pose")), geometry });
            }
        }
    }
    Ok((out, massless))
}

/// One `<geometry>` as a [`crate::LinkGeometry`]. SDF's extents are already full, so nothing is scaled.
fn sdf_geom(geo: &Node, link: &str) -> Result<crate::LinkGeometry, String> {
    use crate::LinkGeometry as LG;
    if let Some(b) = geo.child("box") {
        let txt = b.child("size").map(|n| n.text.trim().to_string()).unwrap_or_default();
        let v: Vec<f64> = txt.split_whitespace().filter_map(|x| x.parse().ok()).collect();
        if v.len() != 3 {
            return Err(format!("<box><size> in link '{link}' needs 3 numbers, got '{txt}'"));
        }
        return Ok(LG::Box { size: Vector3::new(v[0], v[1], v[2]) });
    }
    if let Some(sp) = geo.child("sphere") {
        return Ok(LG::Sphere { radius: sp.text_f64("radius", f64::NAN) });
    }
    if let Some(c) = geo.child("cylinder") {
        return Ok(LG::Cylinder { radius: c.text_f64("radius", f64::NAN), length: c.text_f64("length", f64::NAN) });
    }
    if let Some(c) = geo.child("capsule") {
        return Ok(LG::Capsule { radius: c.text_f64("radius", f64::NAN), length: c.text_f64("length", f64::NAN) });
    }
    if let Some(m) = geo.child("mesh") {
        let uri = m.child("uri").map(|n| n.text.trim().to_string()).unwrap_or_default();
        if uri.is_empty() {
            return Err(format!("<mesh> in link '{link}' has no <uri>"));
        }
        // an absent <scale> is unity, and a partial one is an error rather than a padded guess
        let scale = match m.child("scale").map(|n| n.text.trim().to_string()) {
            None => Vector3::new(1.0, 1.0, 1.0),
            Some(t) => {
                let v: Vec<f64> = t.split_whitespace().filter_map(|x| x.parse().ok()).collect();
                if v.len() != 3 {
                    return Err(format!("<mesh><scale> in link '{link}' needs 3 numbers, got '{t}'"));
                }
                Vector3::new(v[0], v[1], v[2])
            }
        };
        return Ok(LG::Mesh { uri, scale });
    }
    let named: Vec<&str> = geo.children_named("plane").map(|_| "plane")
        .chain(geo.children_named("heightmap").map(|_| "heightmap"))
        .chain(geo.children_named("polyline").map(|_| "polyline"))
        .collect();
    Err(format!(
        "<geometry> in link '{link}' declares {}, which is outside this subset: a plane is not a solid, and a heightmap or polyline is not a primitive this crate generates",
        if named.is_empty() { "no shape this crate reads".to_string() } else { named.join(", ") }
    ))
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
    /// SDF states FULL extents, like URDF and unlike MJCF — so nothing is halved, and this asserts
    /// that against the closed form so the convention cannot drift.
    #[test]
    fn sdf_sizes_are_full_extents_and_every_shape_converts() {
        let xml = r#"<sdf version="1.9"><model name="g">
          <link name="base">
            <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
            <collision name="c"><pose>0 0 0.05 0 0 0</pose>
              <geometry><box><size>0.2 0.4 0.1</size></box></geometry></collision>
          </link>
          <link name="l1">
            <inertial><mass>2.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
            <collision name="c"><geometry><cylinder><radius>0.03</radius><length>0.30</length></cylinder></geometry></collision>
            <collision name="c2"><geometry><capsule><radius>0.02</radius><length>0.2</length></capsule></geometry></collision>
            <visual name="v"><geometry><sphere><radius>0.05</radius></sphere></geometry></visual>
          </link>
          <joint name="j1" type="revolute"><parent>base</parent><child>l1</child>
            <pose>0 0 0.1 0 0 0</pose><axis><xyz>0 0 1</xyz></axis></joint>
        </model></sdf>"#;
        let (geo, massless) = geometry_from_sdf(xml).expect("parses");
        assert!(massless.is_empty(), "both links state a positive mass, got {massless:?}");
        assert_eq!(geo.len(), 4);

        // a FULL extent: 0.2 × 0.4 × 0.1 stays exactly that
        assert_eq!(geo[0].geometry, crate::LinkGeometry::Box { size: Vector3::new(0.2, 0.4, 0.1) });
        let m = crate::primitive_mesh(&geo[0].geometry, 8).expect("generates");
        assert!((m.volume() / (0.2 * 0.4 * 0.1) - 1.0).abs() < 1e-14, "volume {}", m.volume());
        assert!((geo[0].origin.translation.vector - Vector3::new(0.0, 0.0, 0.05)).norm() < 1e-15, "the pose must be carried");

        assert_eq!(geo[1].geometry, crate::LinkGeometry::Cylinder { radius: 0.03, length: 0.30 }, "length is a length, not a half-length");
        assert_eq!(geo[2].geometry, crate::LinkGeometry::Capsule { radius: 0.02, length: 0.2 });
        assert_eq!(geo[3].geometry, crate::LinkGeometry::Sphere { radius: 0.05 });
        // both collisions precede the visual for the same link, so a role filter is meaningful
        assert_eq!(geo[1].role, crate::GeomRole::Collision);
        assert_eq!(geo[3].role, crate::GeomRole::Visual);

        // and the same box read as a HALF extent, the MJCF convention, would be 8x smaller — asserted
        // so a future change that applies MJCF's rule here is caught
        let as_half = crate::primitive_mesh(&crate::LinkGeometry::Box { size: Vector3::new(0.4, 0.8, 0.2) }, 8).expect("generates");
        assert!((as_half.volume() / m.volume() - 8.0).abs() < 1e-12, "SDF is full-extent; doubling would be 8x");
        eprintln!("  SDF box <size>0.2 0.4 0.1</size> -> {:.6} m³ (MJCF's half-extent rule would give {:.6})", m.volume(), as_half.volume());
    }

    /// **The two ways an SDF link ends up massless, both silent.** A missing `<inertial>`, and an
    /// `<inertial>` that omits `<mass>` — the second reads 0.0 through the loader's own `text_f64`
    /// default, and a check for the element's presence alone would miss it.
    #[test]
    fn both_silent_paths_to_a_massless_link_are_reported() {
        let xml = r#"<sdf version="1.9"><model name="g">
          <link name="base">
            <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
            <collision name="c"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision>
          </link>
          <link name="no_inertial">
            <collision name="c"><geometry><box><size>0.16 0.16 0.08</size></box></geometry></collision>
          </link>
          <link name="no_mass">
            <inertial><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
            <collision name="c"><geometry><sphere><radius>0.05</radius></sphere></geometry></collision>
          </link>
          <joint name="j1" type="revolute"><parent>base</parent><child>no_inertial</child>
            <pose>0 0 0.1 0 0 0</pose><axis><xyz>0 0 1</xyz></axis></joint>
          <joint name="j2" type="revolute"><parent>no_inertial</parent><child>no_mass</child>
            <pose>0 0 0.1 0 0 0</pose><axis><xyz>0 1 0</xyz></axis></joint>
        </model></sdf>"#;

        // what the loader produces: two massless links, and nothing said so
        let (_, inertias) = from_sdf(xml, "base", "no_mass").expect("loads");
        assert_eq!(inertias.len(), 2, "two actuated joints");
        assert!(inertias.iter().all(|li| li.mass == 0.0), "both actuated links are massless: {:?}", inertias.iter().map(|l| l.mass).collect::<Vec<_>>());

        // and what this adds: both names, by both routes
        let (geo, massless) = geometry_from_sdf(xml).expect("parses");
        assert_eq!(massless, vec!["no_inertial".to_string(), "no_mass".to_string()], "BOTH silent paths, not just the missing element");
        assert!(!massless.contains(&"base".to_string()), "and not the link that states a mass");

        // the geometry is enough to compute what the loader could not
        let (li, skipped) = crate::primitive_link_inertia(&geo, "no_inertial", 2700.0, 32);
        let li = li.expect("its box is enough");
        assert_eq!(skipped, 0);
        let want = 2700.0 * 0.16 * 0.16 * 0.08;
        assert!((li.mass - want).abs() < 1e-9, "inferred {} vs {want}", li.mass);
        eprintln!("  no_inertial: loader 0.0000 kg, inferred {:.4} kg; no_mass caught by the <mass> default, not the element check", li.mass);
    }

    /// Mesh URIs, the scale default, and `model://` resolving through the same table URDF uses.
    #[test]
    fn sdf_mesh_uris_carry_their_scale_and_resolve() {
        let xml = r#"<sdf version="1.9"><model name="g">
          <link name="base">
            <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
            <collision name="c"><geometry><mesh><uri>model://arm/meshes/l1.stl</uri>
              <scale>0.001 0.001 0.001</scale></mesh></geometry></collision>
            <visual name="v"><geometry><mesh><uri>meshes/plain.obj</uri></mesh></geometry></visual>
          </link>
        </model></sdf>"#;
        let (geo, _) = geometry_from_sdf(xml).expect("parses");
        assert_eq!(geo.len(), 2);
        assert_eq!(
            geo[0].geometry,
            crate::LinkGeometry::Mesh { uri: "model://arm/meshes/l1.stl".into(), scale: Vector3::new(0.001, 0.001, 0.001) }
        );
        assert_eq!(
            geo[1].geometry,
            crate::LinkGeometry::Mesh { uri: "meshes/plain.obj".into(), scale: Vector3::new(1.0, 1.0, 1.0) },
            "an absent scale is unity, not zero"
        );
        // and the URI resolves through the same table URDF's package:// uses
        match &geo[0].geometry {
            crate::LinkGeometry::Mesh { uri, .. } => assert_eq!(
                crate::resolve_uri(uri, &[("arm", "/opt/models/arm")]).as_deref(),
                Some("/opt/models/arm/meshes/l1.stl")
            ),
            other => panic!("expected a mesh, got {other:?}"),
        }
        eprintln!("  model://arm/meshes/l1.stl -> {:?}", crate::resolve_uri("model://arm/meshes/l1.stl", &[("arm", "/opt/models/arm")]));
    }

    /// Refusals. Each is a real authoring case, and each must fail loudly rather than yield a shape
    /// that is quietly the wrong size or a NaN that propagates into a mass.
    #[test]
    fn sdf_geometry_refusals() {
        let ok = r#"<sdf version="1.9"><model name="g"><link name="base">
          <inertial><mass>1.0</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz></inertia></inertial>
          <collision name="c"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision>
        </link></model></sdf>"#;
        assert!(geometry_from_sdf(ok).is_ok(), "control");

        for (label, frag) in [
            ("a box with two numbers", "<geometry><box><size>0.1 0.2</size></box></geometry>"),
            ("a box with no size", "<geometry><box/></geometry>"),
            ("a plane", "<geometry><plane><normal>0 0 1</normal></plane></geometry>"),
            ("a heightmap", "<geometry><heightmap><uri>h.png</uri></heightmap></geometry>"),
            ("a polyline", "<geometry><polyline><height>1</height></polyline></geometry>"),
            ("an empty geometry", "<geometry/>"),
            ("a mesh with no uri", "<geometry><mesh><scale>1 1 1</scale></mesh></geometry>"),
            ("a mesh scale of two numbers", "<geometry><mesh><uri>a.stl</uri><scale>1 1</scale></mesh></geometry>"),
        ] {
            let bad = ok.replace("<geometry><box><size>0.1 0.1 0.1</size></box></geometry>", frag);
            assert!(geometry_from_sdf(&bad).is_err(), "{label} must be refused, got Ok");
        }
        // a collision with no <geometry> at all
        let no_geo = ok.replace("<geometry><box><size>0.1 0.1 0.1</size></box></geometry>", "");
        assert!(geometry_from_sdf(&no_geo).is_err(), "a <collision> with no <geometry> must be refused");

        // ⛔ a shape whose dimension is MISSING reads NaN, and `primitive_mesh` must refuse it rather
        // than generate a body with a NaN mass. This is the path that would otherwise reach the
        // dynamics as a plausible-looking robot.
        let nan_cyl = ok.replace(
            "<geometry><box><size>0.1 0.1 0.1</size></box></geometry>",
            "<geometry><cylinder><radius>0.03</radius></cylinder></geometry>",
        );
        let (geo, _) = geometry_from_sdf(&nan_cyl).expect("a cylinder with no <length> still parses");
        assert!(
            crate::primitive_mesh(&geo[0].geometry, 16).is_none(),
            "a cylinder with no <length> reads NaN and must be refused downstream, not meshed"
        );
        let (li, skipped) = crate::primitive_link_inertia(&geo, "base", 1000.0, 16);
        assert!(li.is_none() && skipped == 1, "and it must count as an unusable shape, got {li:?} / {skipped}");
        eprintln!("  a cylinder with no <length>: parses, then refused by primitive_mesh rather than meshed with NaN");

        assert!(geometry_from_sdf("<notsdf/>").is_err(), "no <model>");
        assert!(geometry_from_sdf(r#"<sdf><model name="e"/></sdf>"#).is_ok_and(|(g, m)| g.is_empty() && m.is_empty()), "an empty model is empty, not an error");
    }

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
