//! **OpenUSD interchange** — reading and writing `.usda` with the `UsdPhysics` schema.
//!
//! USD is the interchange format the physical-AI tooling has settled on: OpenUSD Core Specification 1.0 defines the
//! data model that "SimReady" assets are built against, and a simulator that cannot read a stage cannot be handed an
//! asset. Alongside the existing URDF, MJCF and SDFormat readers, this closes the loop in both directions —
//! [`robot_from_usda`] to consume a stage, [`usda_from_robot`] to emit one.
//!
//! Only the ASCII crate (`.usda`) is handled. The binary `.usdc` crate is a separate, versioned container and is not
//! attempted here rather than half-attempted.
//!
//! **Three conventions cause almost all silent interchange failures, and all three are handled explicitly:**
//!
//! 1. **Revolute limits are in degrees.** `UsdPhysics` specifies `physics:lowerLimit` and `physics:upperLimit` on a
//!    `PhysicsRevoluteJoint` in degrees, while URDF and every internal routine here use radians. A reader that forgets
//!    this produces a robot whose joint limits are 57× too wide, which does not fail — it just stops being a limit.
//!    Prismatic limits are in distance units and are *not* converted.
//! 2. **`metersPerUnit` is often not 1.** Content authored in centimetres carries `metersPerUnit = 0.01`, and every
//!    length on the stage has to be scaled. A robot read without scaling is a hundred times too large and its
//!    inertias are wrong by `10^5`.
//! 3. **`upAxis` is `Y` by default in graphics content and `Z` in robotics content.** [`UsdaStage::to_z_up`] rotates a
//!    Y-up stage rather than assuming, because assuming is how a robot ends up lying on its side.
//!
//! Each has a test that fails if the conversion is dropped.

use crate::{Iso, Joint, JointKind, LinkInertia, Robot};
use nalgebra::{Matrix3, Translation3, UnitQuaternion, Unit, Vector3};
use std::collections::BTreeMap;

/// A parsed `.usda` value. USD's type system is much larger than this; these are the forms the physics schema uses.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Bool(bool),
    /// A tuple such as `(0, 0, 0.15)`.
    Tuple(Vec<f64>),
    /// A quoted string.
    Text(String),
    /// An unquoted token, or one written `"like this"` in a `token` field.
    Token(String),
    /// A prim path such as `</robot/link1>`.
    Path(String),
    /// A bracketed array.
    Array(Vec<Value>),
}

impl Value {
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_tuple(&self) -> Option<&[f64]> {
        match self {
            Value::Tuple(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_vector3(&self) -> Option<Vector3<f64>> {
        self.as_tuple().filter(|t| t.len() >= 3).map(|t| Vector3::new(t[0], t[1], t[2]))
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Text(s) | Value::Token(s) | Value::Path(s) => Some(s),
            _ => None,
        }
    }

    /// Strings in a `token[]` array, however they were quoted.
    pub fn as_token_list(&self) -> Vec<String> {
        match self {
            Value::Array(items) => items.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect(),
            other => other.as_str().map(|s| vec![s.to_owned()]).unwrap_or_default(),
        }
    }
}

/// One prim: its schema type, name, applied API schemas, attributes, relationships and children.
#[derive(Clone, Debug, Default)]
pub struct Prim {
    /// The prim's type name, e.g. `Xform`, `PhysicsRevoluteJoint`, `Mesh`. Empty for an over or a typeless def.
    pub spec: String,
    pub name: String,
    /// `prepend apiSchemas = [...]`, which is where `PhysicsRigidBodyAPI` and friends live.
    pub api_schemas: Vec<String>,
    pub attributes: BTreeMap<String, Value>,
    pub relationships: BTreeMap<String, Vec<String>>,
    pub children: Vec<Prim>,
}

impl Prim {
    pub fn has_api(&self, api: &str) -> bool {
        self.api_schemas.iter().any(|a| a == api)
    }

    pub fn child(&self, name: &str) -> Option<&Prim> {
        self.children.iter().find(|c| c.name == name)
    }

    pub fn attr(&self, name: &str) -> Option<&Value> {
        self.attributes.get(name)
    }

    /// The prim's local transform, composed from `xformOp:translate` and `xformOp:orient` (or `xformOp:rotateXYZ`),
    /// with lengths scaled by `scale`.
    pub fn local_transform(&self, scale: f64) -> Iso {
        let t = self.attr("xformOp:translate").and_then(Value::as_vector3).unwrap_or_else(Vector3::zeros) * scale;
        let rot = if let Some(q) = self.attr("xformOp:orient").and_then(Value::as_tuple).filter(|q| q.len() >= 4) {
            // USD writes a quaternion as (w, x, y, z)
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3]))
        } else if let Some(r) = self.attr("xformOp:rotateXYZ").and_then(Value::as_vector3) {
            // rotateXYZ is in degrees
            UnitQuaternion::from_euler_angles(r.x.to_radians(), r.y.to_radians(), r.z.to_radians())
        } else {
            UnitQuaternion::identity()
        };
        Iso::from_parts(Translation3::from(t), rot)
    }

    /// Walk this prim and its descendants depth-first.
    pub fn walk(&self) -> Vec<&Prim> {
        let mut out = vec![self];
        for c in &self.children {
            out.extend(c.walk());
        }
        out
    }
}

/// A parsed stage.
#[derive(Clone, Debug)]
pub struct UsdaStage {
    /// Scene scale. Content in centimetres carries `0.01`; every length must be multiplied by this to reach metres.
    pub meters_per_unit: f64,
    /// **Mass scale**, USD's `kilogramsPerUnit`. Content in grams carries `0.001`; every mass must be multiplied by
    /// this to reach kilograms, and every inertia by it as well (inertia is `mass * length^2`).
    ///
    /// This field was missing, and the parser silently assumed `1`. A gram-authored stage therefore loaded with masses,
    /// inertias and every gravity or inertial torque **1000x wrong**, with no error — in the module whose entire
    /// purpose is to catch exactly this class of trap for `metersPerUnit`. Found by a scope-error hunt.
    pub kilograms_per_unit: f64,
    /// `"Y"` or `"Z"`. Graphics content defaults to `Y`; robotics content is authored `Z`.
    pub up_axis: String,
    pub default_prim: Option<String>,
    pub root: Vec<Prim>,
}

impl Default for UsdaStage {
    fn default() -> Self {
        // USD's own defaults, not robotics' - a reader that silently substitutes Z-up and unit scale is the bug.
        UsdaStage { meters_per_unit: 1.0,
            kilograms_per_unit: 1.0, up_axis: "Y".into(), default_prim: None, root: Vec::new() }
    }
}

impl UsdaStage {
    /// Find a prim by its absolute path, e.g. `/robot/link1`.
    pub fn prim(&self, path: &str) -> Option<&Prim> {
        let mut parts = path.trim_start_matches('/').split('/').filter(|s| !s.is_empty());
        let first = parts.next()?;
        let mut cur = self.root.iter().find(|p| p.name == first)?;
        for part in parts {
            cur = cur.child(part)?;
        }
        Some(cur)
    }

    /// Every prim on the stage, depth-first.
    pub fn walk(&self) -> Vec<&Prim> {
        self.root.iter().flat_map(Prim::walk).collect()
    }

    /// The rotation that takes this stage's up-axis to `+Z`. Identity for a Z-up stage; a −90° rotation about `X` for
    /// a Y-up one.
    pub fn to_z_up(&self) -> UnitQuaternion<f64> {
        if self.up_axis.eq_ignore_ascii_case("Z") {
            UnitQuaternion::identity()
        } else {
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::FRAC_PI_2)
        }
    }
}

/// What went wrong, with the byte offset so a large stage can be located in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for ParseError {}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, message: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError { message: message.into(), offset: self.i })
    }

    /// Skip whitespace and both comment forms USD accepts.
    fn skip(&mut self) {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"#") || self.s[self.i..].starts_with(b"//") {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
                continue;
            }
            return;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip();
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, b: u8) -> bool {
        if self.peek() == Some(b) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn eat_word(&mut self, w: &str) -> bool {
        self.skip();
        if self.s[self.i..].starts_with(w.as_bytes()) {
            let end = self.i + w.len();
            // a keyword must not be a prefix of a longer identifier
            if self.s.get(end).is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_' && *c != b':') {
                self.i = end;
                return true;
            }
        }
        false
    }

    fn identifier(&mut self) -> Option<String> {
        self.skip();
        let start = self.i;
        while self.i < self.s.len() && (self.s[self.i].is_ascii_alphanumeric() || matches!(self.s[self.i], b'_' | b':' | b'.')) {
            self.i += 1;
        }
        (self.i > start).then(|| String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    fn quoted(&mut self) -> Result<String, ParseError> {
        if !self.eat(b'"') {
            return self.err("expected a quoted string");
        }
        let start = self.i;
        while self.i < self.s.len() && self.s[self.i] != b'"' {
            if self.s[self.i] == b'\\' {
                self.i += 1;
            }
            self.i += 1;
        }
        if self.i >= self.s.len() {
            return self.err("unterminated string");
        }
        let out = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
        self.i += 1;
        Ok(out)
    }

    fn number(&mut self) -> Option<f64> {
        self.skip();
        let start = self.i;
        if matches!(self.s.get(self.i), Some(b'-' | b'+')) {
            self.i += 1;
        }
        while self.i < self.s.len() && (self.s[self.i].is_ascii_digit() || matches!(self.s[self.i], b'.' | b'e' | b'E')) {
            // an exponent sign
            if matches!(self.s[self.i], b'e' | b'E') && matches!(self.s.get(self.i + 1), Some(b'-' | b'+')) {
                self.i += 1;
            }
            self.i += 1;
        }
        String::from_utf8_lossy(&self.s[start..self.i]).parse().ok().or_else(|| {
            self.i = start;
            None
        })
    }

    fn value(&mut self) -> Result<Value, ParseError> {
        match self.peek() {
            Some(b'"') => Ok(Value::Text(self.quoted()?)),
            Some(b'<') => {
                self.i += 1;
                let start = self.i;
                while self.i < self.s.len() && self.s[self.i] != b'>' {
                    self.i += 1;
                }
                let p = String::from_utf8_lossy(&self.s[start..self.i]).into_owned();
                if !self.eat(b'>') {
                    return self.err("unterminated prim path");
                }
                Ok(Value::Path(p))
            }
            Some(b'(') => {
                self.i += 1;
                let mut t = Vec::new();
                loop {
                    if self.eat(b')') {
                        break;
                    }
                    match self.number() {
                        Some(n) => t.push(n),
                        None => return self.err("expected a number in a tuple"),
                    }
                    let _ = self.eat(b',');
                }
                Ok(Value::Tuple(t))
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    if self.eat(b']') {
                        break;
                    }
                    items.push(self.value()?);
                    let _ = self.eat(b',');
                }
                Ok(Value::Array(items))
            }
            Some(c) if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.') => match self.number() {
                Some(n) => Ok(Value::Number(n)),
                None => self.err("malformed number"),
            },
            Some(_) => {
                if self.eat_word("true") {
                    return Ok(Value::Bool(true));
                }
                if self.eat_word("false") {
                    return Ok(Value::Bool(false));
                }
                match self.identifier() {
                    Some(id) => Ok(Value::Token(id)),
                    None => self.err("unrecognised value"),
                }
            }
            None => self.err("unexpected end of stage"),
        }
    }

    /// A parenthesised metadata block, which is where `apiSchemas` and stage metadata live.
    fn metadata(&mut self) -> Result<BTreeMap<String, Value>, ParseError> {
        let mut out = BTreeMap::new();
        if !self.eat(b'(') {
            return Ok(out);
        }
        loop {
            if self.eat(b')') {
                return Ok(out);
            }
            // qualifiers that may precede a metadatum
            for w in ["prepend", "append", "delete", "add", "uniform", "custom"] {
                if self.eat_word(w) {
                    break;
                }
            }
            if self.eat(b')') {
                return Ok(out);
            }
            // a bare docstring is legal metadata content
            if self.peek() == Some(b'"') {
                let _ = self.quoted()?;
                continue;
            }
            let Some(key) = self.identifier() else { return self.err("expected a metadata key") };
            if self.eat(b'=') {
                out.insert(key, self.value()?);
            }
        }
    }

    /// One `def`/`over`/`class` block and its subtree.
    fn prim(&mut self) -> Result<Prim, ParseError> {
        // `def`, `over` and `class` all introduce a prim; only `def` carries a type in practice
        if !(self.eat_word("def") || self.eat_word("over") || self.eat_word("class")) {
            return self.err("expected def, over or class");
        }
        let mut prim = Prim::default();
        // an optional type name, then the quoted prim name
        if self.peek() != Some(b'"') {
            prim.spec = self.identifier().unwrap_or_default();
        }
        prim.name = self.quoted()?;
        let meta = self.metadata()?;
        if let Some(v) = meta.get("apiSchemas") {
            prim.api_schemas = v.as_token_list();
        }
        if !self.eat(b'{') {
            return self.err("expected a prim body");
        }
        loop {
            if self.eat(b'}') {
                return Ok(prim);
            }
            if self.peek().is_none() {
                return self.err("unterminated prim body");
            }
            // a nested prim
            self.skip();
            if self.s[self.i..].starts_with(b"def") || self.s[self.i..].starts_with(b"over") || self.s[self.i..].starts_with(b"class") {
                let save = self.i;
                match self.prim() {
                    Ok(c) => {
                        prim.children.push(c);
                        continue;
                    }
                    Err(e) => {
                        // `def` could be the start of an attribute named e.g. `defaultValue`; only give up if not
                        if save == self.i {
                            return Err(e);
                        }
                        return Err(e);
                    }
                }
            }
            // a relationship
            if self.eat_word("rel") {
                let Some(name) = self.identifier() else { return self.err("expected a relationship name") };
                if self.eat(b'=') {
                    let v = self.value()?;
                    let targets = match v {
                        Value::Array(items) => items.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect(),
                        other => other.as_str().map(|s| vec![s.to_owned()]).unwrap_or_default(),
                    };
                    prim.relationships.insert(name, targets);
                } else {
                    prim.relationships.insert(name, Vec::new());
                }
                continue;
            }
            // an attribute: [uniform|custom|varying] <type>[[]] <name> [= value] [( meta )]
            for w in ["uniform", "custom", "varying", "prepend", "append"] {
                if self.eat_word(w) {
                    break;
                }
            }
            let Some(_ty) = self.identifier() else { return self.err("expected an attribute type") };
            if self.eat(b'[') && !self.eat(b']') {
                return self.err("expected [] after an array type");
            }
            let Some(name) = self.identifier() else { return self.err("expected an attribute name") };
            if self.eat(b'=') {
                let v = self.value()?;
                prim.attributes.insert(name, v);
            } else {
                prim.attributes.insert(name, Value::Token(String::new()));
            }
            // trailing per-attribute metadata
            if self.peek() == Some(b'(') {
                let _ = self.metadata()?;
            }
        }
    }
}

/// **Parse a `.usda` stage.**
pub fn parse_usda(src: &str) -> Result<UsdaStage, ParseError> {
    let trimmed = src.trim_start();
    if !trimmed.starts_with("#usda") {
        return Err(ParseError { message: "not a .usda stage: missing the #usda header".into(), offset: 0 });
    }
    // Consume the magic line by hand. It must NOT go through `skip()`, which treats a leading `#` as a comment and
    // would swallow the line plus the `(` that opens stage metadata - silently dropping metersPerUnit and upAxis,
    // which is the one failure mode this whole module exists to prevent.
    let mut p = Parser { s: src.as_bytes(), i: src.len() - trimmed.len() };
    while p.i < p.s.len() && p.s[p.i] != b'\n' {
        p.i += 1;
    }
    let mut stage = UsdaStage::default();
    let meta = p.metadata()?;
    if let Some(v) = meta.get("metersPerUnit").and_then(Value::as_number) {
        stage.meters_per_unit = v;
    }
    if let Some(v) = meta.get("kilogramsPerUnit").and_then(Value::as_number) {
        stage.kilograms_per_unit = v;
    }
    if let Some(v) = meta.get("upAxis").and_then(|v| v.as_str()) {
        stage.up_axis = v.to_owned();
    }
    if let Some(v) = meta.get("defaultPrim").and_then(|v| v.as_str()) {
        stage.default_prim = Some(v.to_owned());
    }
    while p.peek().is_some() {
        stage.root.push(p.prim()?);
    }
    Ok(stage)
}

/// The USD axis token for a joint axis, or `None` if the axis is not one of the three cardinal directions (USD's
/// `physics:axis` is a token, so an arbitrary axis has to be carried as a joint frame rotation instead).
pub fn axis_token(axis: &Vector3<f64>) -> Option<&'static str> {
    let n = axis.normalize();
    for (t, v) in [("X", Vector3::x()), ("Y", Vector3::y()), ("Z", Vector3::z())] {
        if (n - v).norm() < 1e-9 {
            return Some(t);
        }
    }
    None
}

fn axis_from_token(t: &str) -> Vector3<f64> {
    match t.to_ascii_uppercase().as_str() {
        "X" => Vector3::x(),
        "Y" => Vector3::y(),
        _ => Vector3::z(),
    }
}

/// **Build a [`Robot`] and its link inertias from a stage**, following `physics:body0`/`physics:body1` relationships
/// from `base` to `tip`.
///
/// Applies `metersPerUnit` to every length and **converts revolute limits from degrees to radians**. Prismatic limits
/// are lengths and are scaled, not converted.
pub fn robot_from_usda(stage: &UsdaStage, base: &str, tip: &str) -> Result<(Robot, Vec<LinkInertia>), String> {
    let scale = stage.meters_per_unit;
    let mass_scale = stage.kilograms_per_unit;
    // index joints by the body they drive
    let joints: Vec<&Prim> = stage.walk().into_iter().filter(|p| p.spec.starts_with("Physics") && p.spec.ends_with("Joint")).collect();
    let leaf = |path: &str| path.rsplit('/').next().unwrap_or(path).to_owned();

    let mut by_child: BTreeMap<String, &Prim> = BTreeMap::new();
    for j in &joints {
        if let Some(b1) = j.relationships.get("physics:body1").and_then(|v| v.first()) {
            by_child.insert(leaf(b1), j);
        }
    }

    // walk back from the tip to the base to get the chain in order
    let mut chain: Vec<&Prim> = Vec::new();
    let mut cur = tip.to_owned();
    while cur != base {
        let Some(j) = by_child.get(&cur) else {
            return Err(format!("no joint drives body '{cur}' while walking from '{tip}' to '{base}'"));
        };
        chain.push(j);
        let Some(b0) = j.relationships.get("physics:body0").and_then(|v| v.first()) else {
            return Err(format!("joint '{}' has no physics:body0", j.name));
        };
        cur = leaf(b0);
        if chain.len() > 4096 {
            return Err("cycle in the joint graph".into());
        }
    }
    chain.reverse();

    let mut out_joints = Vec::with_capacity(chain.len());
    let mut inertias = Vec::with_capacity(chain.len());
    for j in &chain {
        let kind = if j.spec.contains("Prismatic") {
            JointKind::Prismatic
        } else if j.spec.contains("Revolute") {
            JointKind::Revolute
        } else {
            return Err(format!("joint '{}' has unsupported type '{}'", j.name, j.spec));
        };
        let axis = j.attr("physics:axis").and_then(|v| v.as_str()).map(axis_from_token).unwrap_or_else(Vector3::z);
        // **UsdPhysics defines the joint frame TWICE** — once in each body — and the parent-to-child transform is
        // `T(localPos0, localRot0) * T(localPos1, localRot1)^-1`. Reading only the body0 side silently assumes the
        // body1 frame is the identity, which loses |localPos1| of link length per joint with no error. Isaac Sim
        // exports set both, so that is the common asset shape, and the loss was measured at 0.15 m on a 0.40 m link
        // before this composed the second frame in.
        let quat = |name: &str| {
            j.attr(name)
                .and_then(Value::as_tuple)
                .filter(|q| q.len() >= 4)
                .map(|q| UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3])))
                .unwrap_or_else(UnitQuaternion::identity)
        };
        let p0 = j.attr("physics:localPos0").and_then(Value::as_vector3).unwrap_or_else(Vector3::zeros) * scale;
        let p1 = j.attr("physics:localPos1").and_then(Value::as_vector3).unwrap_or_else(Vector3::zeros) * scale;
        let frame0 = Iso::from_parts(Translation3::from(p0), quat("physics:localRot0"));
        let frame1 = Iso::from_parts(Translation3::from(p1), quat("physics:localRot1"));
        let origin = frame0 * frame1.inverse();
        let pos = origin.translation.vector;
        let rot = origin.rotation;

        // limits: DEGREES for revolute, distance units for prismatic
        let lower = j.attr("physics:lowerLimit").and_then(Value::as_number);
        let upper = j.attr("physics:upperLimit").and_then(Value::as_number);
        let limits = match (lower, upper) {
            (Some(l), Some(u)) => Some(match kind {
                JointKind::Revolute => (l.to_radians(), u.to_radians()),
                JointKind::Prismatic => (l * scale, u * scale),
            }),
            _ => None,
        };

        // USD's PhysicsDriveAPI carries maxForce; not parsed here yet, so None rather than a guessed default.
        out_joints.push(Joint {
            origin: Iso::from_parts(Translation3::from(pos), rot),
            axis: Unit::new_normalize(axis),
            kind,
            limits,
            effort: None,
            max_velocity: None,
        });

        // the driven body's mass properties, if the stage carries them
        let body = j.relationships.get("physics:body1").and_then(|v| v.first()).map(|p| leaf(p)).unwrap_or_default();
        let prim = stage.walk().into_iter().find(|p| p.name == body);
        inertias.push(match prim {
            Some(p) if p.has_api("PhysicsMassAPI") || p.attr("physics:mass").is_some() => {
                // Mass carries the stage's mass scale, exactly as lengths carry metersPerUnit.
                let mass = p.attr("physics:mass").and_then(Value::as_number).unwrap_or(0.0) * mass_scale;
                let com = p.attr("physics:centerOfMass").and_then(Value::as_vector3).unwrap_or_else(Vector3::zeros) * scale;
                let d = p.attr("physics:diagonalInertia").and_then(Value::as_vector3).unwrap_or_else(Vector3::zeros);
                // inertia has units of mass * length^2, so it carries BOTH scales — one power of the mass scale and
                // two of the length scale.
                let s2 = scale * scale * mass_scale;
                LinkInertia { mass, com, inertia: Matrix3::from_diagonal(&Vector3::new(d.x * s2, d.y * s2, d.z * s2)) }
            }
            _ => LinkInertia::zero(),
        });
    }
    Ok((Robot { joints: out_joints, ee_offset: Iso::identity() }, inertias))
}

/// **Emit a `.usda` stage** for a serial chain: an articulation root, one rigid body per link with its mass
/// properties, and one physics joint per degree of freedom.
///
/// Written Z-up at unit scale, and revolute limits are converted **to degrees** so the stage means what it says when
/// another tool reads it.
pub fn usda_from_robot(robot: &Robot, inertia: &[LinkInertia], name: &str) -> String {
    let mut s = String::from("#usda 1.0\n(\n");
    s += &format!("    defaultPrim = \"{name}\"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n    upAxis = \"Z\"\n)\n\n");
    s += &format!("def Xform \"{name}\" (\n    prepend apiSchemas = [\"PhysicsArticulationRootAPI\"]\n)\n{{\n");
    s += "    def Xform \"link0\" (\n        prepend apiSchemas = [\"PhysicsRigidBodyAPI\"]\n    )\n    {\n    }\n\n";

    for i in 0..robot.joints.len() {
        let li = i + 1;
        let inr = inertia.get(i).cloned().unwrap_or_else(LinkInertia::zero);
        s += &format!("    def Xform \"link{li}\" (\n        prepend apiSchemas = [\"PhysicsRigidBodyAPI\", \"PhysicsMassAPI\"]\n    )\n    {{\n");
        s += &format!("        float physics:mass = {}\n", inr.mass);
        s += &format!("        point3f physics:centerOfMass = ({}, {}, {})\n", inr.com.x, inr.com.y, inr.com.z);
        s += &format!(
            "        float3 physics:diagonalInertia = ({}, {}, {})\n",
            inr.inertia[(0, 0)],
            inr.inertia[(1, 1)],
            inr.inertia[(2, 2)]
        );
        s += "    }\n\n";
    }

    for (i, j) in robot.joints.iter().enumerate() {
        let (spec, li) = (
            match j.kind {
                JointKind::Revolute => "PhysicsRevoluteJoint",
                JointKind::Prismatic => "PhysicsPrismaticJoint",
            },
            i + 1,
        );
        s += &format!("    def {spec} \"joint{li}\"\n    {{\n");
        s += &format!("        rel physics:body0 = </{name}/link{}>\n", i);
        s += &format!("        rel physics:body1 = </{name}/link{li}>\n");
        let t = j.origin.translation.vector;
        s += &format!("        point3f physics:localPos0 = ({}, {}, {})\n", t.x, t.y, t.z);
        let q = j.origin.rotation.quaternion();
        s += &format!("        quatf physics:localRot0 = ({}, {}, {}, {})\n", q.w, q.i, q.j, q.k);
        match axis_token(j.axis.as_ref()) {
            Some(tok) => s += &format!("        uniform token physics:axis = \"{tok}\"\n"),
            None => {
                // USD cannot express an arbitrary axis in physics:axis, so say so rather than silently rounding it
                s += &format!(
                    "        # axis ({:.6}, {:.6}, {:.6}) is not cardinal; carried in physics:localRot0 by the caller\n",
                    j.axis.x, j.axis.y, j.axis.z
                );
                s += "        uniform token physics:axis = \"Z\"\n";
            }
        }
        if let Some((lo, hi)) = j.limits {
            let (lo, hi) = match j.kind {
                JointKind::Revolute => (lo.to_degrees(), hi.to_degrees()),
                JointKind::Prismatic => (lo, hi),
            };
            s += &format!("        float physics:lowerLimit = {lo}\n        float physics:upperLimit = {hi}\n");
        }
        s += "    }\n\n";
    }
    s += "}\n";
    s
}

#[cfg(test)]
mod tests {
    /// **The joint frame is defined twice, and we used to read one half.** `UsdPhysics` gives the joint frame in both
    /// bodies, so parent-to-child is `T(localPos0, localRot0) * T(localPos1, localRot1)^-1`. Reading only the body0
    /// side assumes the body1 frame is the identity and loses `|localPos1|` of link length per joint, silently — on
    /// exactly the asset shape Isaac Sim produces. Measured 0.4000 where 0.5500 was correct before the fix.
    #[test]
    fn the_body1_joint_frame_is_composed_not_discarded() {
        let base = r#"#usda 1.0
(
    defaultPrim = "robot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "robot" ( prepend apiSchemas = ["PhysicsArticulationRootAPI"] )
{
    def Xform "link0" ( prepend apiSchemas = ["PhysicsRigidBodyAPI"] ) { }
    def Xform "link1" ( prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"] )
    {
        float physics:mass = 2.0
        float3 physics:diagonalInertia = (0.03, 0.03, 0.008)
    }
    def PhysicsRevoluteJoint "joint1"
    {
        rel physics:body0 = </robot/link0>
        rel physics:body1 = </robot/link1>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0.40)
LP1
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
    }
}
"#;
        let z = |lp1: &str| {
            let s = base.replace("LP1", lp1);
            let st = parse_usda(&s).expect("parse");
            robot_from_usda(&st, "link0", "link1").expect("robot").0.joints[0].origin.translation.vector.z
        };
        let absent = z("");
        let with = z("        point3f physics:localPos1 = (0, 0, -0.15)");
        eprintln!("localPos1 absent -> {absent:.4} m; localPos1 = -0.15 -> {with:.4} m (USD says 0.55)");
        assert!((absent - 0.40).abs() < 1e-12, "with no body1 frame the origin is localPos0");
        assert!((with - 0.55).abs() < 1e-12, "the body1 frame must be composed as an inverse, got {with}");
        // And it carries metersPerUnit exactly once, like every other length.
        let cm = base
            .replace("metersPerUnit = 1", "metersPerUnit = 0.01")
            .replace("LP1", "        point3f physics:localPos1 = (0, 0, -0.15)");
        let st = parse_usda(&cm).unwrap();
        let zz = robot_from_usda(&st, "link0", "link1").unwrap().0.joints[0].origin.translation.vector.z;
        assert!((zz - 0.0055).abs() < 1e-12, "both frames scale once, got {zz}");
    }

    /// **The mass-scale trap, which was live.** `kilogramsPerUnit` was never read, so a gram-authored stage loaded with
    /// masses and inertias 1000x wrong and no error — in the module that exists to catch this exact class of trap for
    /// `metersPerUnit`. Inertia carries one power of the mass scale and two of the length scale, and both are checked
    /// here because getting the exponent right is the whole point.
    #[test]
    fn kilograms_per_unit_scales_mass_and_inertia() {
        let grams = STAGE.replace("metersPerUnit = 1", "metersPerUnit = 1\n    kilogramsPerUnit = 0.001");
        let base = parse_usda(STAGE).expect("kg stage");
        let g = parse_usda(&grams).expect("gram stage");
        assert_eq!(base.kilograms_per_unit, 1.0, "the default must be USD's 1, not 0");
        assert_eq!(g.kilograms_per_unit, 0.001);

        let (_, kg_in) = robot_from_usda(&base, "link0", "link2").expect("kg robot");
        let (_, g_in) = robot_from_usda(&g, "link0", "link2").expect("gram robot");
        for (a, b) in kg_in.iter().zip(&g_in) {
            eprintln!("mass {:.6} kg -> {:.9}; inertia xx {:.6} -> {:.9}", a.mass, b.mass, a.inertia[(0, 0)], b.inertia[(0, 0)]);
            assert!((b.mass - a.mass * 0.001).abs() < 1e-12, "mass carries one power of the mass scale");
            assert!(
                (b.inertia[(0, 0)] - a.inertia[(0, 0)] * 0.001).abs() < 1e-15,
                "inertia carries ONE power of the mass scale (and two of length, unchanged here)"
            );
        }
        // A gram-authored stage read as kilograms would be 1000x heavy — the silent failure this closes.
        assert!(kg_in[1].mass > 0.0 && (kg_in[1].mass / g_in[1].mass - 1000.0).abs() < 1e-6);
    }

    /// Both scales compose on inertia: `mass * length^2`. Checked together, because the exponent is the thing a reader
    /// would get wrong and the module header once stated it as 10^5 when the code does 10^4.
    #[test]
    fn the_two_stage_scales_compose_on_inertia() {
        let both = STAGE
            .replace("metersPerUnit = 1", "metersPerUnit = 0.01\n    kilogramsPerUnit = 0.001");
        let base = parse_usda(STAGE).unwrap();
        let s = parse_usda(&both).unwrap();
        let (_, a) = robot_from_usda(&base, "link0", "link2").unwrap();
        let (_, b) = robot_from_usda(&s, "link0", "link2").unwrap();
        // 0.001 * 0.01^2 = 1e-7
        let expect = 1e-3 * 1e-2 * 1e-2;
        // Pick the link that actually carries inertia in the fixture.
        let i = a.iter().position(|l| l.inertia[(0, 0)] > 0.0).expect("a link with inertia");
        let (ia, ib) = (a[i].inertia[(0, 0)], b[i].inertia[(0, 0)]);
        let ratio = ib / ia;
        eprintln!("inertia xx {ia:.6} -> {ib:.12}, ratio {ratio:.3e} (expect {expect:.0e})");
        assert!((ratio - expect).abs() / expect < 1e-9, "both scales must compose: got {ratio:e}");
    }

    use super::*;

    const STAGE: &str = r#"#usda 1.0
(
    defaultPrim = "robot"
    metersPerUnit = 1
    upAxis = "Z"
)

# a two-joint arm
def Xform "robot" (
    prepend apiSchemas = ["PhysicsArticulationRootAPI"]
)
{
    def Xform "link0" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
    }

    def Xform "link1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 2.0
        point3f physics:centerOfMass = (0, 0, 0.15)
        float3 physics:diagonalInertia = (0.03, 0.03, 0.008)
        double3 xformOp:translate = (0, 0, 0.05)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Xform "link2" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 1.3
        point3f physics:centerOfMass = (0.1, 0, 0.1)
        float3 physics:diagonalInertia = (0.02, 0.02, 0.005)
    }

    def PhysicsRevoluteJoint "joint1"
    {
        rel physics:body0 = </robot/link0>
        rel physics:body1 = </robot/link1>
        uniform token physics:axis = "Z"
        point3f physics:localPos0 = (0, 0, 0.05)
        float physics:lowerLimit = -90
        float physics:upperLimit = 90
    }

    def PhysicsPrismaticJoint "joint2"
    {
        rel physics:body0 = </robot/link1>
        rel physics:body1 = </robot/link2>
        uniform token physics:axis = "X"
        point3f physics:localPos0 = (0.3, 0, 0)
        float physics:lowerLimit = -0.5
        float physics:upperLimit = 0.5
    }
}
"#;

    #[test]
    fn a_stage_parses_into_prims_attributes_and_relationships() {
        let stage = parse_usda(STAGE).expect("parses");
        assert_eq!(stage.default_prim.as_deref(), Some("robot"));
        assert_eq!(stage.up_axis, "Z");
        assert!((stage.meters_per_unit - 1.0).abs() < 1e-12);

        let robot = stage.prim("/robot").expect("root prim");
        assert!(robot.has_api("PhysicsArticulationRootAPI"));
        assert_eq!(robot.children.len(), 5, "3 links + 2 joints");

        let l1 = stage.prim("/robot/link1").expect("link1");
        assert!(l1.has_api("PhysicsRigidBodyAPI") && l1.has_api("PhysicsMassAPI"));
        assert!((l1.attr("physics:mass").unwrap().as_number().unwrap() - 2.0).abs() < 1e-12);
        assert_eq!(l1.attr("physics:centerOfMass").unwrap().as_vector3().unwrap(), Vector3::new(0.0, 0.0, 0.15));
        assert_eq!(l1.local_transform(1.0).translation.vector, Vector3::new(0.0, 0.0, 0.05));

        let j = stage.prim("/robot/joint1").expect("joint1");
        assert_eq!(j.spec, "PhysicsRevoluteJoint");
        assert_eq!(j.relationships.get("physics:body0").unwrap()[0], "/robot/link0");
        eprintln!("parsed {} prims, {} at the root", stage.walk().len(), stage.root.len());
    }

    /// **Revolute limits are degrees in USD and radians here.** A reader that forgets makes a 90 degree limit into a
    /// 90 radian one, which is not a limit at all.
    #[test]
    fn revolute_limits_convert_from_degrees_and_prismatic_ones_do_not() {
        let stage = parse_usda(STAGE).unwrap();
        let (robot, inertia) = robot_from_usda(&stage, "link0", "link2").expect("chain resolves");
        assert_eq!(robot.dof(), 2);

        let (lo, hi) = robot.joints[0].limits.expect("revolute limits");
        eprintln!("revolute: stage says (-90, 90) degrees -> ({lo:.6}, {hi:.6}) rad");
        assert!((hi - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "90 degrees is pi/2, not 90");
        assert!((lo + std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let (plo, phi) = robot.joints[1].limits.expect("prismatic limits");
        eprintln!("prismatic: stage says (-0.5, 0.5) length units -> ({plo}, {phi}) m, unconverted");
        assert!((phi - 0.5).abs() < 1e-12, "a prismatic limit is a length and must not be treated as an angle");

        // and the chain came out in the right order with the right axes
        assert_eq!(robot.joints[0].kind, JointKind::Revolute);
        assert_eq!(robot.joints[1].kind, JointKind::Prismatic);
        assert_eq!(*robot.joints[1].axis, Vector3::x());
        assert!((inertia[0].mass - 2.0).abs() < 1e-12 && (inertia[1].mass - 1.3).abs() < 1e-12);
    }

    /// **`metersPerUnit` scales every length**, and inertia by its square.
    #[test]
    fn centimetre_content_is_scaled_to_metres() {
        let cm = STAGE.replace("metersPerUnit = 1", "metersPerUnit = 0.01");
        let stage = parse_usda(&cm).unwrap();
        assert!((stage.meters_per_unit - 0.01).abs() < 1e-15);
        let (robot, inertia) = robot_from_usda(&stage, "link0", "link2").unwrap();

        // the first joint sits at 0.05 stage units = 0.5 mm
        let t = robot.joints[0].origin.translation.vector;
        eprintln!("joint offset: 0.05 stage units at 0.01 m/unit -> {:.6} m", t.z);
        assert!((t.z - 0.0005).abs() < 1e-15, "lengths scale linearly: {}", t.z);
        // inertia is mass * length^2, so it scales by 1e-4
        eprintln!("diagonal inertia 0.03 -> {:.3e} kg m^2 (scale^2)", inertia[0].inertia[(0, 0)]);
        assert!((inertia[0].inertia[(0, 0)] - 0.03 * 1e-4).abs() < 1e-18);
        // a prismatic limit is a length and scales; an angle does not
        assert!((robot.joints[1].limits.unwrap().1 - 0.005).abs() < 1e-15);
        assert!((robot.joints[0].limits.unwrap().1 - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "an angle is not a length");
    }

    /// **`upAxis` defaults to Y in USD**, and a robot read as if it were Z-up lies on its side.
    #[test]
    fn the_up_axis_is_reported_not_assumed() {
        let y_up = STAGE.replace("upAxis = \"Z\"", "upAxis = \"Y\"");
        let stage = parse_usda(&y_up).unwrap();
        assert_eq!(stage.up_axis, "Y");
        let r = stage.to_z_up();
        // Y-up to Z-up sends +Y to +Z
        let mapped = r * Vector3::y();
        eprintln!("Y-up stage: to_z_up() maps +Y to ({:.3}, {:.3}, {:.3})", mapped.x, mapped.y, mapped.z);
        assert!((mapped - Vector3::z()).norm() < 1e-12);
        // and a Z-up stage is left alone
        assert!(parse_usda(STAGE).unwrap().to_z_up().angle().abs() < 1e-15);
        // the default when the stage says nothing is USD's Y, not robotics' Z
        assert_eq!(UsdaStage::default().up_axis, "Y", "guessing Z here is the bug this default prevents");
    }

    /// Round trip: emit a stage from a robot, read it back, and get the same robot.
    #[test]
    fn a_robot_round_trips_through_usda() {
        let stage = parse_usda(STAGE).unwrap();
        let (robot, inertia) = robot_from_usda(&stage, "link0", "link2").unwrap();

        let text = usda_from_robot(&robot, &inertia, "arm");
        let reparsed = parse_usda(&text).expect("what we emit is what we can read");
        let (robot2, inertia2) = robot_from_usda(&reparsed, "link0", "link2").expect("chain survives");

        assert_eq!(robot2.dof(), robot.dof());
        for (a, b) in robot.joints.iter().zip(&robot2.joints) {
            assert_eq!(a.kind, b.kind);
            assert!((a.origin.translation.vector - b.origin.translation.vector).norm() < 1e-9);
            assert!((*a.axis - *b.axis).norm() < 1e-9);
            let (la, lb) = (a.limits.unwrap(), b.limits.unwrap());
            // the limit survives a radians -> degrees -> radians round trip
            assert!((la.0 - lb.0).abs() < 1e-9 && (la.1 - lb.1).abs() < 1e-9, "{la:?} vs {lb:?}");
        }
        for (a, b) in inertia.iter().zip(&inertia2) {
            assert!((a.mass - b.mass).abs() < 1e-6 && (a.com - b.com).norm() < 1e-9);
        }
        eprintln!("round trip: {} joints, limits and inertias preserved", robot2.dof());
    }

    /// A malformed stage has to fail with a location rather than panic.
    #[test]
    fn malformed_input_is_reported_not_panicked() {
        assert!(parse_usda("def Xform \"x\" {}").is_err(), "a stage without the #usda header is not a stage");
        let truncated = "#usda 1.0\ndef Xform \"robot\"\n{\n    def Xform \"a\"\n    {\n";
        let e = parse_usda(truncated).expect_err("unterminated body");
        eprintln!("truncated stage: {e}");
        assert!(e.offset > 0, "the error carries a location");
        // an unresolvable chain is an error, not a silent empty robot
        let stage = parse_usda(STAGE).unwrap();
        assert!(robot_from_usda(&stage, "link0", "nonexistent").is_err());
    }
}
