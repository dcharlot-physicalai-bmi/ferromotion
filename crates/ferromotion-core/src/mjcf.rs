//! MJCF (MuJoCo XML) loading — the model format of the MuJoCo ecosystem, parsed from a string so
//! it works identically native and in the browser. Produces the same `(Robot, Vec<LinkInertia>)`
//! as the URDF loader, so every downstream consumer (dynamics, gendyn, calib, control) is
//! format-agnostic.
//!
//! ⛔ **A body with no `<inertial>` loads MASSLESS, and MuJoCo's own compiler would have inferred it
//! from the geoms.** Many published models therefore carry geoms and no `<inertial>` at all, and this
//! loader gives them zero mass with no error. [`geometry_from_mjcf`] is how a caller finds out which
//! bodies those are and gets the geometry to do the inference with
//! [`primitive_link_inertia`](crate::primitive_link_inertia). Measured on a box-geom body: the loader
//! reports 0.0000 kg where the geometry implies 5.5296 kg.
//!
//! **Honest subset** (asserted, not silently wrong): a single serial chain of nested `<body>`
//! elements under `<worldbody>`, at most one `<joint>` per body (`hinge` — the MJCF default — or
//! `slide`), explicit `<inertial>` blocks (inertia is not inferred from geoms *by the loader*; see
//! above for how to infer it), `pos` +
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
pub(crate) struct El {
    pub(crate) name: String,
    pub(crate) attrs: Vec<(String, String)>,
    pub(crate) children: Vec<El>,
}

impl El {
    pub(crate) fn attr(&self, k: &str) -> Option<&str> {
        self.attrs.iter().find(|(a, _)| a == k).map(|(_, v)| v.as_str())
    }
    pub(crate) fn child(&self, name: &str) -> Option<&El> {
        self.children.iter().find(|c| c.name == name)
    }
    pub(crate) fn children_named<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a El> {
        self.children.iter().filter(move |c| c.name == name)
    }
}

pub(crate) fn parse_xml(s: &str) -> Result<El, String> {
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

pub(crate) fn floats(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace().map(|t| t.parse::<f64>().map_err(|e| format!("bad number '{t}': {e}"))).collect()
}

pub(crate) fn vec3(s: &str) -> Result<Vector3<f64>, String> {
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
/// **The geometry an MJCF declares, and which bodies state no inertia at all.**
///
/// ⛔ **A body with no `<inertial>` silently gets a ZERO inertia from [`from_mjcf_full`]**, which means
/// a model that relies on MuJoCo's own inference loads here as a massless robot with no error. That is
/// not a rare case: MuJoCo's compiler infers mass and inertia from a body's geoms by default, so a
/// great many published models carry geoms and no `<inertial>` at all. This function is how a caller
/// finds out, and how it gets the geometry needed to do the inference itself with
/// [`primitive_link_inertia`](crate::primitive_link_inertia).
///
/// # MJCF's sizes are HALF-extents, and that is the trap
///
/// A `<geom type="box" size="a b c"/>` is `2a × 2b × 2c`, and a `<geom type="cylinder" size="r h"/>`
/// has length `2h`. URDF states the same shapes as full extents. Every size is converted here, so the
/// returned [`crate::LinkGeometry`] is in URDF's convention and [`primitive_mesh`](crate::primitive_mesh)
/// consumes it unchanged. `mjcf_box_size_is_a_half_extent` pins the factor of two, because a silent
/// 8x volume error is exactly the kind that looks like a plausible robot.
///
/// Supported types: `box`, `sphere`, `cylinder`, `capsule`, `mesh`, and the `fromto` form of a
/// cylinder or capsule (two endpoints instead of a position and a half-length). `plane` and
/// `ellipsoid` are reported as unsupported rather than approximated — a plane is not a solid and an
/// ellipsoid is not any of the primitives this crate generates.
///
/// A `mesh` geom's `mesh="name"` is resolved through `<asset><mesh name file scale>` and prefixed with
/// `<compiler meshdir>`, which is MJCF's equivalent of URDF's `package://`. The returned URI is that
/// path; nothing is fetched.
///
/// Returns the shapes in body order paired with the names of the bodies that declared no `<inertial>`.
pub fn geometry_from_mjcf(xml: &str) -> Result<(Vec<crate::GeometryRef>, Vec<String>), String> {
    let root = parse_xml(xml)?;
    if root.name != "mujoco" {
        return Err(format!("root element is <{}>, expected <mujoco>", root.name));
    }
    let deg = match root.child("compiler").and_then(|c| c.attr("angle")) {
        Some("radian") => 1.0,
        Some("degree") | None => std::f64::consts::PI / 180.0,
        Some(other) => return Err(format!("unknown compiler angle '{other}'")),
    };
    let meshdir = root.child("compiler").and_then(|c| c.attr("meshdir")).unwrap_or("").to_string();
    // <asset><mesh name="x" file="x.stl" scale="s s s"/>
    let mut assets: Vec<(String, String, Vector3<f64>)> = Vec::new();
    if let Some(asset) = root.child("asset") {
        for m in asset.children_named("mesh") {
            let name = m.attr("name").unwrap_or("").to_string();
            let file = m.attr("file").unwrap_or("").to_string();
            let scale = m.attr("scale").map(vec3).transpose()?.unwrap_or_else(|| Vector3::new(1.0, 1.0, 1.0));
            assets.push((name, file, scale));
        }
    }

    let world = root.child("worldbody").ok_or("no <worldbody>")?;
    let mut out = Vec::new();
    let mut no_inertial = Vec::new();
    let mut body = world.child("body");
    let mut idx = 0usize;
    while let Some(b) = body {
        let link = b.attr("name").map(|s| s.to_string()).unwrap_or_else(|| format!("body{idx}"));
        if b.child("inertial").is_none() {
            no_inertial.push(link.clone());
        }
        for g in b.children_named("geom") {
            let (geometry, origin) = mjcf_geom(g, deg, &assets, &meshdir)?;
            // MJCF has no visual/collision split: a geom is both unless `contype`/`conaffinity` are
            // zeroed, which marks it visual-only. Reported as the role, so a caller can prefer the
            // collision set exactly as it does for URDF.
            let visual_only = g.attr("contype") == Some("0") && g.attr("conaffinity") == Some("0");
            out.push(crate::GeometryRef {
                link: link.clone(),
                role: if visual_only { crate::GeomRole::Visual } else { crate::GeomRole::Collision },
                origin,
                geometry,
            });
        }
        body = b.child("body");
        idx += 1;
    }
    Ok((out, no_inertial))
}

/// One `<geom>` as a [`LinkGeometry`] in URDF's full-extent convention, plus its pose.
fn mjcf_geom(
    g: &El,
    deg: f64,
    assets: &[(String, String, Vector3<f64>)],
    meshdir: &str,
) -> Result<(crate::LinkGeometry, Iso), String> {
    use crate::LinkGeometry as LG;
    // MJCF's default geom type is `sphere`, which is easy to forget and gives a wrong shape silently.
    let ty = g.attr("type").unwrap_or("sphere");
    let size = g.attr("size").map(floats).transpose()?.unwrap_or_default();
    let need = |k: usize| -> Result<&Vec<f64>, String> {
        if size.len() < k {
            Err(format!("geom type '{ty}' needs {k} size value(s), got {}", size.len()))
        } else {
            Ok(&size)
        }
    };

    // The `fromto` form gives two endpoints; the shape's axis is the segment and its length is the
    // segment's length, so the pose is computed rather than read.
    if let Some(ft) = g.attr("fromto") {
        let v = floats(ft)?;
        if v.len() != 6 {
            return Err("geom fromto needs 6 numbers".into());
        }
        let (p1, p2) = (Vector3::new(v[0], v[1], v[2]), Vector3::new(v[3], v[4], v[5]));
        let axis = p2 - p1;
        let len = axis.norm();
        // NaN-safe: a non-finite endpoint gives a NaN length, which must be refused rather than
        // compared. `is_finite() && > 0` says that; `!(len > 0.0)` says it too but reads as a trap.
        if !(len.is_finite() && len > 0.0) {
            return Err("geom fromto endpoints coincide".into());
        }
        let r = *need(1)?.first().ok_or("fromto geom needs a radius in size")?;
        let rot = UnitQuaternion::rotation_between(&Vector3::z(), &axis).unwrap_or_else(|| {
            // exactly antiparallel to +z: `rotation_between` has no unique answer, so pick one
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
        });
        let iso = Iso::from_parts(Translation3::from((p1 + p2) / 2.0), rot);
        return match ty {
            "cylinder" => Ok((LG::Cylinder { radius: r, length: len }, iso)),
            // MJCF's capsule fromto gives the CYLINDER segment; the caps add `r` beyond each end.
            "capsule" => Ok((LG::Capsule { radius: r, length: len }, iso)),
            other => Err(format!("geom type '{other}' does not take fromto")),
        };
    }

    let iso = pose(g, deg)?;
    match ty {
        // ⛔ every one of these is a HALF-extent in MJCF and a FULL extent in `LinkGeometry`
        "box" => {
            let s = need(3)?;
            Ok((LG::Box { size: Vector3::new(2.0 * s[0], 2.0 * s[1], 2.0 * s[2]) }, iso))
        }
        "sphere" => Ok((LG::Sphere { radius: need(1)?[0] }, iso)),
        "cylinder" => {
            let s = need(2)?;
            Ok((LG::Cylinder { radius: s[0], length: 2.0 * s[1] }, iso))
        }
        "capsule" => {
            let s = need(2)?;
            Ok((LG::Capsule { radius: s[0], length: 2.0 * s[1] }, iso))
        }
        "mesh" => {
            let name = g.attr("mesh").ok_or("a mesh geom needs a `mesh` attribute")?;
            let (_, file, scale) = assets
                .iter()
                .find(|(n, _, _)| n == name)
                .ok_or_else(|| format!("geom references mesh '{name}', which no <asset><mesh> declares"))?;
            let uri = if meshdir.is_empty() || meshdir.ends_with('/') {
                format!("{meshdir}{file}")
            } else {
                format!("{meshdir}/{file}")
            };
            Ok((LG::Mesh { uri, scale: *scale }, iso))
        }
        other => Err(format!(
            "geom type '{other}' is outside this subset: a plane is not a solid and an ellipsoid is not a primitive this crate generates"
        )),
    }
}

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
            let axis = j.attr("axis").map(vec3).transpose()?.unwrap_or_else(Vector3::z); // MJCF default axis
            let jpos = j.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
            // joint frame = body frame translated to the joint's anchor point
            let origin = carry * bpose * Iso::from_parts(Translation3::from(jpos), UnitQuaternion::identity());
            // MJCF states actuator force limits on the <actuator> element rather than the joint, so they are not
            // available here; None rather than a guessed default. See `Joint::effort`.
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
            // MJCF is the one format here that states the actuator's own inertia and damping on the joint.
            // MuJoCo carries `armature` because a URDF-derived model without it is ill-conditioned: the term
            // is N^2 J_rotor, which on a small distal link dominates the link itself. See `Joint::armature`.
            if let Some(a) = j.attr("armature").and_then(|v| v.trim().parse::<f64>().ok()) {
                joint = joint.with_armature(a);
            }
            if let Some(d) = j.attr("damping").and_then(|v| v.trim().parse::<f64>().ok()) {
                joint = joint.with_damping(d);
            }
            // MJCF calls Coulomb friction `frictionloss`, not `friction` — `<geom friction>` is the contact
            // coefficient and an entirely different quantity. Reading the wrong one would be silent.
            if let Some(f) = j.attr("frictionloss").and_then(|v| v.trim().parse::<f64>().ok()) {
                joint = joint.with_friction(f);
            }
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
            carry *= bpose;
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

/// **Write a `Robot` and its link inertias as MJCF.** The counterpart to [`from_mjcf_full`], and the reason it
/// exists: **URDF has no field for reflected rotor inertia**, so a plant loaded from URDF and then given the
/// servo terms it needs — [`crate::Joint::armature`], [`crate::Joint::damping`] — had no format to be saved in.
/// MJCF states both per joint. This closes that gap: load a URDF, attach the actuator model, write MJCF, and
/// the terms survive the trip.
///
/// The output targets exactly the subset [`from_mjcf_full`] parses, so it **round-trips**: each joint becomes a
/// nested `<body>` whose pose is that joint's origin, with the `<joint>` anchored at the body frame and an
/// `<inertial>` carrying mass, centre of mass and the full inertia tensor. `ee_offset` becomes a trailing
/// jointless body. There is a test that exports, re-imports, and compares the mass matrix and gravity vector
/// rather than the text, because matching bytes is not the property that matters.
///
/// Angles are radians (`<compiler angle="radian"/>`), so nothing depends on the reader's default.
///
/// What it does not write, because the parser does not read it back and a silent round-trip loss is worse than
/// an absent field: geometry, materials, actuators, sensors and equality constraints. [`crate::Joint::effort`]
/// and [`crate::Joint::max_velocity`] are among the casualties — MJCF states those on `<actuator>`, which is
/// outside the subset, so they are dropped and the doc comment says so rather than the file pretending.
pub fn to_mjcf(robot: &Robot, inertia: &[LinkInertia], model: &str) -> Result<String, String> {
    if inertia.len() != robot.dof() {
        return Err(format!("{} inertias for {} joints", inertia.len(), robot.dof()));
    }
    // Rust's `Display` for f64 emits the SHORTEST string that parses back to the identical bit pattern, so
    // this is lossless. A first version used `{:.17}` and trimmed trailing zeros, which silently rounded
    // 0.012900000000000002 to 0.0129 and broke the round trip by one ulp — a serialisation format that loses
    // precision quietly is exactly the defect this file exists to document elsewhere.
    let num = |v: f64| -> String {
        if v == 0.0 { "0".to_string() } else { format!("{v}") } // `v == 0.0` also catches -0.0
    };
    let v3 = |v: &Vector3<f64>| format!("{} {} {}", num(v.x), num(v.y), num(v.z));

    let mut out = String::new();
    out.push_str(&format!("<mujoco model=\"{}\">\n", xml_escape(model)));
    out.push_str("  <compiler angle=\"radian\"/>\n  <worldbody>\n");

    let mut indent = 4;
    for (i, j) in robot.joints.iter().enumerate() {
        let pad = " ".repeat(indent);
        let t = j.origin.translation.vector;
        let q = j.origin.rotation;
        // nalgebra stores (i,j,k,w); MJCF quat is (w,x,y,z).
        out.push_str(&format!(
            "{pad}<body name=\"link{i}\" pos=\"{}\" quat=\"{} {} {} {}\">\n",
            v3(&t),
            num(q.w),
            num(q.i),
            num(q.j),
            num(q.k)
        ));
        let kind = match j.kind {
            JointKind::Revolute => "hinge",
            JointKind::Prismatic => "slide",
        };
        let mut attrs = format!("name=\"joint{i}\" type=\"{kind}\" axis=\"{}\"", v3(&j.axis.into_inner()));
        if let Some((lo, hi)) = j.limits {
            attrs.push_str(&format!(" limited=\"true\" range=\"{} {}\"", num(lo), num(hi)));
        }
        // The two the URDF could not state. Written unconditionally when present, since carrying them is the
        // entire point of this function.
        if let Some(a) = j.armature {
            attrs.push_str(&format!(" armature=\"{}\"", num(a)));
        }
        if let Some(d) = j.damping {
            attrs.push_str(&format!(" damping=\"{}\"", num(d)));
        }
        // MJCF spells Coulomb friction `frictionloss`; `friction` on a <geom> is the contact coefficient.
        if let Some(f) = j.friction {
            attrs.push_str(&format!(" frictionloss=\"{}\"", num(f)));
        }
        out.push_str(&format!("{pad}  <joint {attrs}/>\n"));

        let li = &inertia[i];
        let m = &li.inertia;
        out.push_str(&format!(
            "{pad}  <inertial pos=\"{}\" mass=\"{}\" fullinertia=\"{} {} {} {} {} {}\"/>\n",
            v3(&li.com),
            num(li.mass),
            num(m[(0, 0)]),
            num(m[(1, 1)]),
            num(m[(2, 2)]),
            num(m[(0, 1)]),
            num(m[(0, 2)]),
            num(m[(1, 2)])
        ));
        indent += 2;
    }

    // The tool frame as a trailing jointless body, which the parser folds into `ee_offset`.
    let eo = robot.ee_offset;
    let pad = " ".repeat(indent);
    let q = eo.rotation;
    out.push_str(&format!(
        "{pad}<body name=\"tool\" pos=\"{}\" quat=\"{} {} {} {}\"/>\n",
        v3(&eo.translation.vector),
        num(q.w),
        num(q.i),
        num(q.j),
        num(q.k)
    ));

    for _ in 0..robot.dof() {
        indent -= 2;
        out.push_str(&format!("{}</body>\n", " ".repeat(indent)));
    }
    out.push_str("  </worldbody>\n</mujoco>\n");
    Ok(out)
}

/// Escape the five XML metacharacters. A robot name is caller-supplied text and a bare `&` would make the
/// output unparseable by the very function meant to read it back.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    /// **MJCF sizes are HALF-extents and URDF's are full.** A silent factor of two on every axis is an
    /// 8x volume error that looks like a plausible robot, so the conversion is pinned against the
    /// closed form rather than against another reader.
    #[test]
    fn mjcf_box_size_is_a_half_extent() {
        let xml = r#"<mujoco model="g"><compiler angle="radian"/><worldbody>
          <body name="l1"><joint type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size="0.1 0.2 0.05"/>
          </body></worldbody></mujoco>"#;
        let (geo, no_inertial) = geometry_from_mjcf(xml).expect("parses");
        assert!(no_inertial.is_empty(), "this body states an inertial");
        assert_eq!(geo.len(), 1);
        // MJCF "0.1 0.2 0.05" is a 0.2 × 0.4 × 0.1 box
        assert_eq!(geo[0].geometry, crate::LinkGeometry::Box { size: Vector3::new(0.2, 0.4, 0.1) });
        let m = crate::primitive_mesh(&geo[0].geometry, 8).expect("generates");
        // 1e-15 relative, not 1e-18 absolute: f64 resolution at 0.008 is already 1.8e-18, and my
        // first version asserted below it (0.008000000000000004 vs 0.008000000000000002).
        assert!((m.volume() / (0.2 * 0.4 * 0.1) - 1.0).abs() < 1e-14, "volume {} vs {}", m.volume(), 0.2 * 0.4 * 0.1);
        // and the factor is exactly 8 against the wrong reading, which is the error this pins
        let wrong = crate::primitive_mesh(&crate::LinkGeometry::Box { size: Vector3::new(0.1, 0.2, 0.05) }, 8).expect("generates");
        assert!((m.volume() / wrong.volume() - 8.0).abs() < 1e-12, "half-vs-full is a factor of 8 in volume, got {}", m.volume() / wrong.volume());
        eprintln!("  MJCF box size=\"0.1 0.2 0.05\" -> {:.6} m³; read as a full extent it would be {:.6} (8x out)", m.volume(), wrong.volume());
    }

    /// Cylinder and capsule half-lengths, the default geom type, and the `fromto` form, which computes
    /// its own pose from two endpoints rather than reading one.
    #[test]
    fn mjcf_geom_forms_convert_to_urdf_conventions() {
        let xml = r#"<mujoco model="g"><compiler angle="radian"/><worldbody>
          <body name="l1"><joint type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="cylinder" size="0.03 0.15"/>
            <geom type="capsule" size="0.02 0.10"/>
            <geom size="0.04"/>
            <geom type="cylinder" fromto="0 0 0  0 0 0.5" size="0.01"/>
            <geom type="capsule" fromto="0 0 0  0.3 0.4 0.0" size="0.02"/>
            <geom type="cylinder" fromto="0.1 0.1 0.1  0.1 0.1 0.6" size="0.015"/>
          </body></worldbody></mujoco>"#;
        let (geo, _) = geometry_from_mjcf(xml).expect("parses");
        assert_eq!(geo.len(), 6);
        // half-length 0.15 is a length of 0.30
        assert_eq!(geo[0].geometry, crate::LinkGeometry::Cylinder { radius: 0.03, length: 0.30 });
        assert_eq!(geo[1].geometry, crate::LinkGeometry::Capsule { radius: 0.02, length: 0.20 });
        // ⛔ MJCF's DEFAULT geom type is `sphere`, not box — easy to forget, and silently a wrong shape
        assert_eq!(geo[2].geometry, crate::LinkGeometry::Sphere { radius: 0.04 }, "an untyped geom is a sphere in MJCF");

        // fromto along +z: length 0.5, centred at its midpoint, axis already +z so identity rotation
        assert_eq!(geo[3].geometry, crate::LinkGeometry::Cylinder { radius: 0.01, length: 0.5 });
        assert!((geo[3].origin.translation.vector - Vector3::new(0.0, 0.0, 0.25)).norm() < 1e-15, "midpoint, got {:?}", geo[3].origin.translation.vector);
        assert!(geo[3].origin.rotation.angle() < 1e-12, "already along +z, so no rotation");

        // fromto in the xy plane: length 0.5 (3-4-5), and the rotation must carry +z onto the segment
        match geo[4].geometry {
            crate::LinkGeometry::Capsule { radius, length } => {
                assert!((radius - 0.02).abs() < 1e-15);
                assert!((length - 0.5).abs() < 1e-12, "|(0.3,0.4,0)| = 0.5, got {length}");
            }
            ref other => panic!("expected a capsule, got {other:?}"),
        }
        let mapped = geo[4].origin.rotation * Vector3::z();
        let want = Vector3::new(0.3, 0.4, 0.0).normalize();
        assert!((mapped - want).norm() < 1e-12, "the rotation must send +z along the segment: {mapped:?} vs {want:?}");
        assert!((geo[4].origin.translation.vector - Vector3::new(0.15, 0.20, 0.0)).norm() < 1e-15, "midpoint");
        eprintln!("  fromto (0,0,0)->(0.3,0.4,0): length 0.5, midpoint (0.15, 0.20, 0), +z mapped to {mapped:?}");

        // ⛔ A fromto that does NOT start at the origin. Both fixtures above do, so |p2| happens to
        // equal |p2 − p1| and reading the length off the second endpoint alone passed. This one has
        // |p2 − p1| = 0.5 while |p2| = 0.6164, so the two readings differ.
        match geo[5].geometry {
            crate::LinkGeometry::Cylinder { radius, length } => {
                assert!((radius - 0.015).abs() < 1e-15);
                assert!((length - 0.5).abs() < 1e-12, "the length is |p2 − p1| = 0.5, not |p2| = 0.6164, got {length}");
            }
            ref other => panic!("expected a cylinder, got {other:?}"),
        }
        assert!(
            (geo[5].origin.translation.vector - Vector3::new(0.1, 0.1, 0.35)).norm() < 1e-15,
            "midpoint of (0.1,0.1,0.1)-(0.1,0.1,0.6), got {:?}", geo[5].origin.translation.vector
        );
        eprintln!("  fromto (0.1,0.1,0.1)->(0.1,0.1,0.6): length 0.5 (|p2| would be {:.4}), midpoint (0.1, 0.1, 0.35)", Vector3::new(0.1, 0.1, 0.6).norm());
    }

    /// **The silent-zero-inertia case, made visible.** A body with geoms and no `<inertial>` loads as
    /// MASSLESS from `from_mjcf_full`, with no error — MuJoCo's own compiler would have inferred it
    /// from the geoms. `geometry_from_mjcf` reports which bodies those are, and the geometry is enough
    /// to do the inference.
    #[test]
    fn a_geom_only_body_is_massless_and_now_says_so() {
        let xml = r#"<mujoco model="g"><compiler angle="radian"/><worldbody>
          <body name="l1"><joint type="hinge" axis="0 0 1"/>
            <geom type="box" size="0.08 0.08 0.04"/>
            <body name="l2" pos="0 0 0.1"><joint type="hinge" axis="0 1 0"/>
              <inertial pos="0 0 0" mass="2.0" diaginertia="0.01 0.01 0.01"/>
              <geom type="cylinder" size="0.02 0.10"/>
            </body>
          </body></worldbody></mujoco>"#;

        // what the loader produces today: l1 is massless and nothing said so
        let (_, inertias) = from_mjcf_full(xml).expect("loads");
        assert_eq!(inertias.len(), 2);
        assert_eq!(inertias[0].mass, 0.0, "l1 states no inertial, so the loader gives it ZERO mass");
        assert!((inertias[1].mass - 2.0).abs() < 1e-15, "l2 states one");

        // and what this function adds: the names, and the geometry to fix it
        let (geo, no_inertial) = geometry_from_mjcf(xml).expect("parses");
        assert_eq!(no_inertial, vec!["l1".to_string()], "exactly the body that needs inference");
        // ⛔ The two walks must agree on WHICH bodies exist. The loader's subset is a serial chain of
        // NESTED <body> elements, and `geometry_from_mjcf` mirrors that walk — my first fixture used
        // sibling bodies, so both saw only one and the disagreement was with my expectation, not
        // between them. Asserted now so a future divergence between the two walks is visible.
        let links: std::collections::BTreeSet<&str> = geo.iter().map(|g| g.link.as_str()).collect();
        assert_eq!(links.len(), inertias.len(), "the geometry walk must visit the same bodies the loader does: {links:?} vs {} inertias", inertias.len());
        let (li, skipped) = crate::primitive_link_inertia(&geo, "l1", 2700.0, 64);
        let li = li.expect("its box is enough to infer an inertia");
        assert_eq!(skipped, 0, "no asset is needed for a box");
        let want = 2700.0 * 0.16 * 0.16 * 0.08;
        assert!((li.mass - want).abs() < 1e-9, "inferred mass {} vs {want}", li.mass);
        assert!(li.mass > 0.0 && inertias[0].mass == 0.0, "the inference gives a body where the loader gave none");
        eprintln!("  l1 declares geoms and no <inertial>: loader mass {:.4} kg, inferred from geometry {:.4} kg", inertias[0].mass, li.mass);
    }

    /// Mesh references resolve through `<asset>` and `<compiler meshdir>` — MJCF's equivalent of
    /// URDF's `package://` — carrying the scale, and refuse a name no asset declares.
    #[test]
    fn mjcf_mesh_geoms_resolve_through_the_asset_table() {
        let xml = r#"<mujoco model="g"><compiler angle="radian" meshdir="assets/meshes"/>
          <asset><mesh name="link2" file="link2.stl" scale="0.001 0.001 0.001"/>
                 <mesh name="plain" file="plain.obj"/></asset>
          <worldbody><body name="l1"><joint type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="mesh" mesh="link2"/>
            <geom type="mesh" mesh="plain"/>
          </body></worldbody></mujoco>"#;
        let (geo, _) = geometry_from_mjcf(xml).expect("parses");
        assert_eq!(geo.len(), 2);
        assert_eq!(
            geo[0].geometry,
            crate::LinkGeometry::Mesh { uri: "assets/meshes/link2.stl".into(), scale: Vector3::new(0.001, 0.001, 0.001) },
            "meshdir must be prefixed and the scale carried"
        );
        assert_eq!(
            geo[1].geometry,
            crate::LinkGeometry::Mesh { uri: "assets/meshes/plain.obj".into(), scale: Vector3::new(1.0, 1.0, 1.0) },
            "an absent scale is unity, not zero"
        );
        // a mesh geom that names nothing must fail loudly rather than resolve to an empty path
        let bad = xml.replace(r#"mesh="link2""#, r#"mesh="missing""#);
        assert!(geometry_from_mjcf(&bad).is_err(), "a mesh name no asset declares must be an error");
        eprintln!("  meshdir=\"assets/meshes\" + file=\"link2.stl\" -> {:?}", geo[0].geometry);
    }

    /// The visual/collision distinction MJCF expresses differently, and the types this subset refuses
    /// rather than approximating.
    #[test]
    fn mjcf_roles_and_refusals() {
        let xml = r#"<mujoco model="g"><compiler angle="radian"/><worldbody>
          <body name="l1"><joint type="hinge" axis="0 0 1"/>
            <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
            <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
            <geom type="box" size="0.05 0.05 0.05"/>
          </body></worldbody></mujoco>"#;
        let (geo, _) = geometry_from_mjcf(xml).expect("parses");
        // MJCF has no visual/collision split; contype=0 conaffinity=0 is how a geom is marked
        // visual-only, and that maps onto the role a caller already filters on for URDF.
        assert_eq!(geo[0].role, crate::GeomRole::Visual, "a non-colliding geom is visual");
        assert_eq!(geo[1].role, crate::GeomRole::Collision, "a colliding geom is collision");
        let (li, _) = crate::primitive_link_inertia(&geo, "l1", 1000.0, 16);
        let li = li.expect("the collision geom describes it");
        assert!((li.mass - 1000.0 * 0.1 * 0.1 * 0.1).abs() < 1e-9, "collision must win: {} kg", li.mass);

        for (label, frag) in [
            ("plane", r#"<geom type="plane" size="1 1 0.1"/>"#),
            ("ellipsoid", r#"<geom type="ellipsoid" size="0.1 0.2 0.3"/>"#),
            ("box with two sizes", r#"<geom type="box" size="0.1 0.2"/>"#),
            ("cylinder with one size", r#"<geom type="cylinder" size="0.1"/>"#),
            ("fromto with coincident ends", r#"<geom type="capsule" fromto="0 0 0  0 0 0" size="0.01"/>"#),
            ("fromto on a box", r#"<geom type="box" fromto="0 0 0  0 0 1" size="0.01"/>"#),
        ] {
            let bad = xml.replace(r#"<geom type="box" size="0.05 0.05 0.05"/>"#, frag);
            assert!(geometry_from_mjcf(&bad).is_err(), "{label} must be refused, not approximated");
        }
        assert!(geometry_from_mjcf("<notmujoco/>").is_err(), "the wrong root element");
        assert!(geometry_from_mjcf(r#"<mujoco/>"#).is_err(), "no <worldbody>");
        assert!(geometry_from_mjcf(r#"<mujoco><worldbody/></mujoco>"#).is_ok_and(|(g, n)| g.is_empty() && n.is_empty()), "an empty world is empty, not an error");
    }

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

    /// **MJCF is the one format here that states the actuator's own inertia**, and it was being dropped.
    ///
    /// MuJoCo carries `armature` because a model built from link geometry alone is ill-conditioned wherever a
    /// geared servo drives a light link. Reading it is not a convenience: it is the difference between a
    /// simulable plant and one that is not. The absent case must stay absent rather than becoming zero, so a
    /// caller can tell "this model says nothing" from "this model says the rotor is weightless".
    #[test]
    fn mjcf_armature_and_damping_survive_the_loader() {
        const GEARED: &str = r#"
<mujoco model="geared">
  <compiler angle="radian"/>
  <worldbody>
    <body pos="0 0 0.3">
      <joint type="hinge" axis="0 0 1" armature="0.0119" damping="0.64"/>
      <inertial pos="0 0 0.1" mass="1.5" diaginertia="0.02 0.03 0.025"/>
      <body pos="0.1 0 0.2">
        <joint type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0.1" mass="0.8" diaginertia="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>"#;
        let (robot, inertia) = from_mjcf_full(GEARED).expect("parse");
        assert_eq!(robot.joints[0].armature, Some(0.0119));
        assert_eq!(robot.joints[0].damping, Some(0.64));
        assert_eq!(
            robot.joints[1].armature, None,
            "a joint that states nothing must stay unstated"
        );
        assert_eq!(robot.joints[1].damping, None);

        // It must reach the dynamics, not merely the struct: armature raises joint 0's diagonal and nothing else.
        let m = crate::mass_matrix(&robot, &inertia, &[0.0, 0.0]);
        let mut bare = robot.clone();
        bare.joints[0] = bare.joints[0]
            .clone()
            .with_armature(-1.0)
            .with_damping(-1.0); // clears both
        let m0 = crate::mass_matrix(&bare, &inertia, &[0.0, 0.0]);
        assert!(
            (m[(0, 0)] - m0[(0, 0)] - 0.0119).abs() < 1e-14,
            "armature must reach the mass matrix"
        );
        assert_eq!(m[(1, 1)], m0[(1, 1)], "and must not touch the other joint");
    }

    /// **The round trip is the test, and the plant is the comparison.** Export a robot that carries armature
    /// and damping, read it back, and check the *dynamics* agree — mass matrix, gravity vector and forward
    /// kinematics — rather than comparing text. Matching bytes is not the property that matters, and a text
    /// comparison would pass on two files that describe different robots written the same way.
    #[test]
    fn mjcf_round_trips_through_the_dynamics() {
        let so101 = include_str!("../examples/so101.urdf");
        let (mut robot, inertia) = crate::from_urdf_full(so101, "base_link", "gripper_link").unwrap();
        let n = robot.dof();
        // Attach exactly what URDF cannot state. If these do not survive, the export is pointless.
        for (i, j) in robot.joints.iter_mut().enumerate() {
            *j = j
                .clone()
                .with_armature(1.19e-2 + i as f64 * 1e-3)
                .with_damping(0.64 - i as f64 * 0.05)
                .with_friction(0.08 + i as f64 * 0.01);
        }

        let xml = to_mjcf(&robot, &inertia, "so101").expect("export");
        let (back, back_inertia) = from_mjcf_full(&xml).expect("re-import the file we just wrote");
        assert_eq!(back.dof(), n, "degrees of freedom");

        for i in 0..n {
            let (a, b) = (&robot.joints[i], &back.joints[i]);
            assert_eq!(b.armature, a.armature, "armature on joint {i} — the whole reason this exists");
            assert_eq!(b.damping, a.damping, "damping on joint {i}");
            assert_eq!(b.friction, a.friction, "friction on joint {i} — MJCF calls it frictionloss");
            assert_eq!(b.kind, a.kind, "kind on joint {i}");
            let (al, bl) = (a.limits.unwrap(), b.limits.unwrap());
            assert!((al.0 - bl.0).abs() < 1e-12 && (al.1 - bl.1).abs() < 1e-12, "limits on joint {i}");
        }

        // The dynamics are the real check: they fold origin, axis, inertia and armature together, so an error
        // in any one of them shows up here even if the per-field checks above were satisfied.
        let g = Vector3::new(0.0, 0.0, -9.81);
        for q in [vec![0.0; n], vec![0.3, -0.7, 0.5, -0.2, 1.1], vec![-1.2, 0.9, -0.4, 1.3, -0.8]] {
            let (m0, m1) = (crate::mass_matrix(&robot, &inertia, &q), crate::mass_matrix(&back, &back_inertia, &q));
            for i in 0..n {
                for j in 0..n {
                    assert!((m0[(i, j)] - m1[(i, j)]).abs() < 1e-9, "M[{i},{j}] at {q:?}: {m0:?} vs {m1:?}");
                }
            }
            let g0 = crate::gravity_vector(&robot, &inertia, &q, g);
            let g1 = crate::gravity_vector(&back, &back_inertia, &q, g);
            for i in 0..n {
                assert!((g0[i] - g1[i]).abs() < 1e-9, "gravity[{i}] at {q:?}: {g0:?} vs {g1:?}");
            }
            let (p0, p1) = (robot.fk(&q).translation.vector, back.fk(&q).translation.vector);
            assert!((p0 - p1).norm() < 1e-9, "tool position at {q:?}: {p0:?} vs {p1:?}");
        }
    }

    /// **The control the round-trip test needs.** If `to_mjcf` dropped armature entirely, the test above would
    /// still pass whenever the importer also defaulted it to the same thing. This proves the exported text
    /// actually carries the term, and that a robot without one exports without one.
    #[test]
    fn the_exported_text_carries_armature_only_when_the_model_states_it() {
        let so101 = include_str!("../examples/so101.urdf");
        let (robot, inertia) = crate::from_urdf_full(so101, "base_link", "gripper_link").unwrap();

        let bare = to_mjcf(&robot, &inertia, "bare").unwrap();
        assert!(!bare.contains("armature"), "a model with no armature must not export one");

        let mut geared = robot.clone();
        for j in geared.joints.iter_mut() {
            *j = j.clone().with_armature(0.0119);
        }
        let xml = to_mjcf(&geared, &inertia, "geared").unwrap();
        assert_eq!(xml.matches("armature=").count(), geared.dof(), "one armature per joint");
        assert!(xml.contains("angle=\"radian\""), "angles must be explicit, not left to the reader's default");
    }

    /// A caller-supplied name is text, and `&` in it would produce a file this crate's own parser rejects.
    #[test]
    fn a_hostile_model_name_still_produces_parseable_xml() {
        let so101 = include_str!("../examples/so101.urdf");
        let (robot, inertia) = crate::from_urdf_full(so101, "base_link", "gripper_link").unwrap();
        let xml = to_mjcf(&robot, &inertia, r#"a & b <c> "d""#).unwrap();
        assert!(xml.contains("&amp;") && xml.contains("&lt;") && xml.contains("&quot;"));
        from_mjcf_full(&xml).expect("the escaped name must still parse");
    }

    /// Mismatched inertia count is a caller error and must be reported, not indexed past.
    #[test]
    fn a_wrong_inertia_count_is_an_error() {
        let so101 = include_str!("../examples/so101.urdf");
        let (robot, inertia) = crate::from_urdf_full(so101, "base_link", "gripper_link").unwrap();
        assert!(to_mjcf(&robot, &inertia[..2], "short").is_err());
    }

    /// **The repo now ships a model that passes its own plausibility check.**
    ///
    /// `so101.urdf` declares `effort="10"` on every joint and cannot state a rotor inertia, so
    /// `actuator_plausibility` flags its two distal joints at 12,049 and 289,728 rad/s². MJCF *can* state one.
    /// `so101_servo.mjcf` is that same arm with the STS3215's terms attached, and this test both regenerates it
    /// — so the committed file cannot drift from the code that produced it — and checks the property the file
    /// exists for.
    #[test]
    fn the_shipped_servo_mjcf_is_current_and_plausible() {
        const COMMITTED: &str = include_str!("../examples/so101_servo.mjcf");
        let urdf = include_str!("../examples/so101.urdf");
        let (mut robot, inertia) = crate::from_urdf_full(urdf, "base_link", "gripper_link").unwrap();
        let armature = 345.0f64.powi(2) * 1e-7; // N^2 J_rotor
        let damping = 3.0 / 4.7; // tau_stall / omega_0, the back-EMF speed droop
        for j in robot.joints.iter_mut() {
            *j = j.clone().with_armature(armature).with_damping(damping);
        }
        let regenerated = to_mjcf(&robot, &inertia, "so101_sts3215").unwrap();
        assert_eq!(
            regenerated, COMMITTED,
            "so101_servo.mjcf is stale; regenerate it from so101.urdf with the STS3215 terms"
        );

        // The property the file exists for: a realistic effort limit is now physically plausible on every
        // joint, which it is not on the URDF this was derived from.
        let (mut back, back_inertia) = from_mjcf_full(COMMITTED).unwrap();
        let n = back.dof();
        for j in back.joints.iter_mut() {
            *j = j.clone().with_effort(3.0); // STS3215 stall, the figure the URDF should have carried
        }
        let report = crate::actuator_plausibility(&back, &back_inertia, &vec![0.0; n]);
        let flagged: Vec<usize> = report
            .iter()
            .filter(|r| r.implied_acceleration.is_some_and(|a| a >= 1e4))
            .map(|r| r.joint)
            .collect();
        assert!(flagged.is_empty(), "no joint should be implausible now, flagged {flagged:?}");
        assert!(report.iter().all(|r| r.armature_stated), "every joint must carry the rotor term");

        // And the same URDF WITHOUT the term does flag, so the assertion above is not vacuous.
        let (mut bare, bare_inertia) = crate::from_urdf_full(urdf, "base_link", "gripper_link").unwrap();
        for j in bare.joints.iter_mut() {
            *j = j.clone().with_effort(3.0);
        }
        let bare_report = crate::actuator_plausibility(&bare, &bare_inertia, &vec![0.0; n]);
        assert!(
            bare_report.iter().any(|r| r.implied_acceleration.is_some_and(|a| a >= 1e4)),
            "the URDF this was derived from must still flag, or the check above proves nothing"
        );
    }
}
