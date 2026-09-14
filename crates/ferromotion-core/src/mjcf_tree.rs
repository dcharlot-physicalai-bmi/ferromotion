//! **Loading a branched MJCF model as a kinematic tree — measured against MuJoCo itself.**
//!
//! [`from_mjcf_full`](crate::from_mjcf_full) reads the serial-chain subset of MJCF and refuses everything
//! else: branching, more than one joint per body, ball and free joints, `<default>` classes, `<include>`,
//! `<frame>`. That subset covers a servo arm written by hand and almost nothing that is actually published.
//! MuJoCo is the format the field runs — 3.6 million installs a month at the time of writing — and its
//! models arrive as `<default>`-classed, `<include>`-split, branched trees with free-floating roots.
//!
//! This loader reads that. The oracle is not plausibility but **MuJoCo's own `mj_kinematics`**: every
//! convention below was first probed on a hand-written model and the numbers MuJoCo produced are pinned in the
//! tests at the bottom of this file, and the whole of MuJoCo Menagerie is then swept by the
//! `menagerie_parity` example against the same oracle. Measured 2026-09-14: **201 of the 204 Menagerie models
//! MuJoCo 3.13.0 compiles load and reproduce every body and site pose to 1.7e-15 m** (3,979 bodies, 3,804
//! joints, 959 sites), and 257 of the 259 bodies whose inertia MuJoCo infers from geoms come out within 1e-5
//! of MuJoCo's; the refusals are `<attach>` and one 215-gon OBJ face. The full record, including the dynamics
//! parity that follows from it, is on the example.
//!
//! # The conventions, as MuJoCo applies them
//!
//! * A body frame is `parent · pos · orientation`. Orientation comes from exactly one of `quat` (w x y z),
//!   `axisangle`, `euler` (under `<compiler eulerseq>`: lower-case letters are intrinsic, upper-case
//!   extrinsic, default `xyz`), `xyaxes` (x, then y orthogonalised against it) or `zaxis` (the minimal rotation
//!   taking +z there).
//! * A joint in a body moves **that body** relative to its parent, about an anchor `pos` in the body's own
//!   frame: a hinge is `T(pos)·R(axis, q − ref)·T(−pos)`, a slide is `T(axis·(q − ref))`, and several joints
//!   in one body compose in file order, each anchored in the body frame the previous one left behind.
//! * A **ball** joint is `T(pos)·R(quat)·T(−pos)`; it is carried here as three hinges about `z`, `y`, `x`
//!   sharing the anchor, so `R = R_z(yaw)·R_y(pitch)·R_x(roll)` — exactly nalgebra's
//!   [`UnitQuaternion::euler_angles`] decomposition, which is how a caller maps MuJoCo's quaternion onto them.
//! * A **free** joint places the body frame *directly at* its 7 `qpos` values; the body's own `pos`/`quat`
//!   only seed `qpos0`. It is carried as three world-axis slides then the same three hinges.
//! * `<default>` resolution: an element's own attribute, else its `class`, else the nearest enclosing body's
//!   `childclass`, walking each class up to its parent and finally the unnamed main default. A `childclass`
//!   applies to the body's own children as well as its descendants — MuJoCo does this and the probe confirmed it.
//! * `<include file>` is textual insertion of the included root's children, with every path — nested ones too
//!   — relative to the **main** model's directory. The loader is string-based and WASM-clean, so the caller
//!   supplies the file contents through a resolver closure; [`tree_from_mjcf_str`] is the no-includes form.
//! * Several `<worldbody>` elements (a scene that includes its robot outside its own worldbody gets two) are
//!   walked in order, as MuJoCo merges them. `fromto` on a site puts the frame at the segment midpoint, +z along it.
//! * `<frame>` is a pure transform applied to its children. Jointless bodies weld into the nearest jointed
//!   ancestor exactly as the URDF tree loader welds fixed joints; their frames stay addressable.
//! * Angles default to **degrees** (`euler`, `axisangle`, hinge `range`, hinge `ref`); `<compiler
//!   angle="radian">` switches. `<compiler autolimits>` defaults to true, and MuJoCo itself refuses a `range`
//!   without `limited` when it is false, so that combination is refused here too rather than guessed at.
//!
//! # What is refused, loudly
//!
//! `<replicate>`, `<attach>`, `<composite>` and `<flexcomp>` (procedural model generation), a free joint on a
//! body that is not a child of the world, unknown joint types, `settotalmass`, `shellinertia` on a geom,
//! `inertia="convex"` on a mesh, and mesh files that are neither OBJ nor STL.
//!
//! # Inertia
//!
//! A stated `<inertial>` is read as written. A body without one gets what MuJoCo's compiler gives it, by
//! MuJoCo's own rules (read from `user_objects.cc` and `user_mesh.cc`, and measured against it): every geom
//! whose `group` lies in `<compiler inertiagrouprange>` weighs `density × volume` (or its stated `mass`), with
//! the closed-form inertia of its primitive, and a mesh is weighed by MuJoCo's **`legacy`** method — pyramids
//! from the area-weighted face centroid with **absolute** volumes, so an open mesh still has mass and a concave
//! one weighs more than its true volume. `exact` (signed) and `shell` (surface) are the other two methods.
//! `boundmass` and `boundinertia` floor the result. A geom of a primitive type that names a `mesh` is **fitted**
//! to it (`mjCMesh::FitGeom`: sized from the mesh's equivalent inertia box, or its bounds under `fitaabb`,
//! times `fitscale`, placed at the mesh's centre in its principal frame) — and a geom with a `mesh` and no
//! `type` is therefore a fitted sphere, which is what MuJoCo makes of it. Mesh coordinates are rounded to
//! `f32` as MuJoCo stores them, quads are split as tinyobjloader splits them, and the principal frame comes
//! from a port of MuJoCo's own Jacobi solver, whose absolute stopping tolerance is itself a behaviour (see
//! `eig3_mujoco`). Which bodies were inferred is in [`MjcfTree::inferred_from_geoms`]; which stated nothing
//! is in [`MjcfTree::no_inertial`].

use crate::dynamics::LinkInertia;
use crate::kinematic_tree::KinematicTree;
use crate::mjcf::{floats, parse_xml, vec3, El};
use crate::{Iso, Joint};
use nalgebra::{Matrix3, Translation3, Unit, UnitQuaternion, Vector3};
use std::collections::{BTreeMap, HashMap};

/// The MuJoCo joint kinds this loader carries, each as one or more single-DoF tree joints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MjcfJointKind {
    /// One tree joint; `qpos` width 1.
    Hinge,
    /// One tree joint; `qpos` width 1.
    Slide,
    /// Three hinges (`z`, `y`, `x` about the shared anchor); MuJoCo `qpos` width 4 (a unit quaternion).
    Ball,
    /// Three slides along world `x`, `y`, `z`, then three hinges; MuJoCo `qpos` width 7.
    Free,
}

impl MjcfJointKind {
    /// Number of single-DoF tree joints this kind expands to.
    pub fn dofs(self) -> usize {
        match self {
            Self::Hinge | Self::Slide => 1,
            Self::Ball => 3,
            Self::Free => 6,
        }
    }
    /// Width of this joint in MuJoCo's `qpos`.
    pub fn qpos_width(self) -> usize {
        match self {
            Self::Hinge | Self::Slide => 1,
            Self::Ball => 4,
            Self::Free => 7,
        }
    }
}

/// One MuJoCo-level joint and where its tree joints start.
#[derive(Clone, Debug)]
pub struct MjcfJoint {
    /// The joint's `name`, or `joint{i}` for an unnamed joint where `i` is its index in file order — the same
    /// numbering MuJoCo assigns, since both walk the bodies depth-first in file order.
    pub name: String,
    pub kind: MjcfJointKind,
    /// The body this joint moves.
    pub body: String,
    /// Index of the first tree joint in [`MjcfTree::tree`]; the next `kind.dofs() - 1` follow it.
    pub first: usize,
    /// MuJoCo `ref` (the `qpos` value at which the model is in its declared configuration), already in
    /// radians for a hinge. Folded into the tree joint's origin, so callers pass MuJoCo's raw `qpos`.
    pub reference: f64,
}

/// A branched MJCF model as a [`KinematicTree`], plus the bookkeeping MuJoCo-level parity needs.
#[derive(Clone, Debug)]
pub struct MjcfTree {
    /// The tree, one single-DoF joint per entry, in MuJoCo's joint order.
    pub tree: KinematicTree,
    /// MuJoCo-level joints in file order.
    pub joints: Vec<MjcfJoint>,
    /// Every body's frame: the tree joint it rides on and the fixed offset from that joint's frame. Jointless
    /// bodies ride on their nearest jointed ancestor. Also mirrored into `tree.tip_offsets` under the body name.
    pub body_frames: BTreeMap<String, (usize, Iso)>,
    /// Site frames, same form. Sites and bodies live in separate namespaces in MuJoCo and here.
    pub site_frames: BTreeMap<String, (usize, Iso)>,
    /// Bodies and sites fixed to the world (no jointed ancestor), by world pose. Keys are `body:<name>` and
    /// `site:<name>`.
    pub world_fixed: BTreeMap<String, Iso>,
    /// Bodies that declared no `<inertial>`. Under `inertiafromgeom="auto"` (the default) their inertia was
    /// inferred from their geoms exactly as MuJoCo infers it; those with a positive result are also listed in
    /// [`MjcfTree::inferred_from_geoms`], and the rest loaded massless.
    pub no_inertial: Vec<String>,
    /// Bodies whose inertia came from their geoms (MuJoCo's rules: density × volume per geom, `legacy` mesh
    /// volumes, `inertiagrouprange`), rather than from a stated `<inertial>`.
    pub inferred_from_geoms: Vec<String>,
    /// `<compiler angle>` as the radians-per-unit factor that was applied: `1` for radian, `π/180` for degree.
    pub angle_scale: f64,
}

impl MjcfTree {
    /// World pose of a named body, or of a world-fixed body.
    pub fn body_pose(&self, name: &str, q: &[f64]) -> Option<Iso> {
        if let Some((j, off)) = self.body_frames.get(name) {
            return Some(self.tree.frames(Iso::identity(), q)[*j] * off);
        }
        self.world_fixed.get(&format!("body:{name}")).copied()
    }

    /// World pose of a named site, or of a world-fixed site.
    pub fn site_pose(&self, name: &str, q: &[f64]) -> Option<Iso> {
        if let Some((j, off)) = self.site_frames.get(name) {
            return Some(self.tree.frames(Iso::identity(), q)[*j] * off);
        }
        self.world_fixed.get(&format!("site:{name}")).copied()
    }

    /// **Map MuJoCo's `qpos` onto this tree's `q`.** `qposadr` gives each MuJoCo joint's address, in the order
    /// of [`MjcfTree::joints`]. Hinge and slide copy through; a ball's quaternion becomes `(yaw, pitch, roll)`
    /// on its three hinges; a free joint copies its translation and does the same with its quaternion.
    pub fn q_from_qpos(&self, qpos: &[f64], qposadr: &[usize]) -> Result<Vec<f64>, String> {
        if qposadr.len() != self.joints.len() {
            return Err(format!("{} qpos addresses for {} joints", qposadr.len(), self.joints.len()));
        }
        let mut q = vec![0.0; self.tree.dof()];
        for (j, &adr) in self.joints.iter().zip(qposadr) {
            let need = adr + j.kind.qpos_width();
            if need > qpos.len() {
                return Err(format!("joint '{}' reads qpos[{adr}..{need}] of {}", j.name, qpos.len()));
            }
            match j.kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => q[j.first] = qpos[adr],
                MjcfJointKind::Ball => {
                    let (yaw, pitch, roll) = ypr(&qpos[adr..adr + 4]);
                    q[j.first] = yaw;
                    q[j.first + 1] = pitch;
                    q[j.first + 2] = roll;
                }
                MjcfJointKind::Free => {
                    q[j.first..j.first + 3].copy_from_slice(&qpos[adr..adr + 3]);
                    let (yaw, pitch, roll) = ypr(&qpos[adr + 3..adr + 7]);
                    q[j.first + 3] = yaw;
                    q[j.first + 4] = pitch;
                    q[j.first + 5] = roll;
                }
            }
        }
        Ok(q)
    }
}

/// `(yaw, pitch, roll)` of a MuJoCo `(w, x, y, z)` quaternion, in the order the three hinges take them.
fn ypr(wxyz: &[f64]) -> (f64, f64, f64) {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]));
    let (roll, pitch, yaw) = q.euler_angles();
    (yaw, pitch, roll)
}

// ---------------------------------------------------------------------------------------------
// <include>
// ---------------------------------------------------------------------------------------------

fn expand_includes(el: &mut El, resolve: &dyn Fn(&str) -> Option<Vec<u8>>, depth: usize) -> Result<(), String> {
    if depth > 32 {
        return Err("<include> nesting deeper than 32 — a cycle".into());
    }
    let mut out: Vec<El> = Vec::with_capacity(el.children.len());
    for child in std::mem::take(&mut el.children) {
        if child.name == "include" {
            let file = child.attr("file").ok_or("<include> needs a file attribute")?;
            let bytes = resolve(file).ok_or_else(|| format!("<include file=\"{file}\"> could not be resolved"))?;
            let text = String::from_utf8_lossy(&bytes);
            let mut inc = parse_xml(&text).map_err(|e| format!("in included file '{file}': {e}"))?;
            expand_includes(&mut inc, resolve, depth + 1)?;
            out.extend(inc.children);
        } else {
            let mut child = child;
            expand_includes(&mut child, resolve, depth + 1)?;
            out.push(child);
        }
    }
    el.children = out;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// <compiler> and <default>
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum InertiaFromGeom {
    Auto,
    True,
    False,
}

struct Compiler {
    /// radians per stated angle unit
    deg: f64,
    eulerseq: String,
    autolimits: bool,
    inertiafromgeom: InertiaFromGeom,
    inertiagrouprange: (i64, i64),
    boundmass: f64,
    boundinertia: f64,
    fitaabb: bool,
    meshdir: String,
}

fn compiler(root: &El) -> Result<Compiler, String> {
    let mut c = Compiler {
        deg: std::f64::consts::PI / 180.0,
        eulerseq: "xyz".into(),
        autolimits: true,
        inertiafromgeom: InertiaFromGeom::Auto,
        inertiagrouprange: (0, 5),
        boundmass: 0.0,
        boundinertia: 0.0,
        fitaabb: false,
        meshdir: String::new(),
    };
    // several <compiler> elements can arrive through includes; later ones override attribute by attribute
    for el in root.children_named("compiler") {
        match el.attr("angle") {
            Some("radian") => c.deg = 1.0,
            Some("degree") => c.deg = std::f64::consts::PI / 180.0,
            Some(other) => return Err(format!("unknown compiler angle '{other}'")),
            None => {}
        }
        if let Some(s) = el.attr("eulerseq") {
            if s.len() != 3 || !s.chars().all(|ch| "xyzXYZ".contains(ch)) {
                return Err(format!("eulerseq '{s}' is not three of xyzXYZ"));
            }
            c.eulerseq = s.to_string();
        }
        match el.attr("autolimits") {
            Some("true") => c.autolimits = true,
            Some("false") => c.autolimits = false,
            Some(other) => return Err(format!("compiler autolimits '{other}' is not true/false")),
            None => {}
        }
        match el.attr("inertiafromgeom") {
            Some("auto") => c.inertiafromgeom = InertiaFromGeom::Auto,
            Some("true") => c.inertiafromgeom = InertiaFromGeom::True,
            Some("false") => c.inertiafromgeom = InertiaFromGeom::False,
            Some(other) => return Err(format!("compiler inertiafromgeom '{other}' is not auto/true/false")),
            None => {}
        }
        if let Some(r) = el.attr("inertiagrouprange") {
            let v = floats(r)?;
            if v.len() != 2 {
                return Err("compiler inertiagrouprange needs 2 numbers".into());
            }
            c.inertiagrouprange = (v[0] as i64, v[1] as i64);
        }
        if let Some(b) = el.attr("boundmass") {
            c.boundmass = b.trim().parse().map_err(|e| format!("boundmass: {e}"))?;
        }
        if let Some(b) = el.attr("boundinertia") {
            c.boundinertia = b.trim().parse().map_err(|e| format!("boundinertia: {e}"))?;
        }
        match el.attr("fitaabb") {
            Some("true") => c.fitaabb = true,
            Some("false") => c.fitaabb = false,
            Some(other) => return Err(format!("compiler fitaabb '{other}' is not true/false")),
            None => {}
        }
        if el.attr("settotalmass").is_some_and(|v| v.trim().parse::<f64>().map(|x| x > 0.0).unwrap_or(true)) {
            return Err("compiler settotalmass rescales every body after compilation — outside this loader's subset".into());
        }
        // MuJoCo looks in `meshdir`, then `assetdir`, for mesh files
        if let Some(d) = el.attr("assetdir") {
            c.meshdir = d.to_string();
        }
        if let Some(d) = el.attr("meshdir") {
            c.meshdir = d.to_string();
        }
    }
    Ok(c)
}

// ---------------------------------------------------------------------------------------------
// Inertia from geoms — MuJoCo's own rules, read from its compiler (user_mesh.cc / user_objects.cc)
// ---------------------------------------------------------------------------------------------

/// MuJoCo's `<mesh inertia>` methods. `legacy` is the default and the one Menagerie uses on 3,426 of its
/// 3,449 meshes: pyramids from the area-weighted face centroid with **absolute** volumes, so a mesh that is
/// not watertight still gets a positive mass — and a concave one gets more than its true volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MeshInertia {
    Legacy,
    Exact,
    Shell,
}

/// What MuJoCo's `mjCMesh::Process` produces and its geoms consume.
#[derive(Clone, Copy)]
struct MeshData {
    /// volume (or area, for a shell): what the density multiplies
    volume: f64,
    /// centre of mass in the mesh frame (`pos_`)
    com: Vector3<f64>,
    /// principal frame (`quat_`), MuJoCo `(w x y z)`
    quat: [f64; 4],
    /// the equivalent inertia box half-sizes (`boxsz_`)
    boxsz: [f64; 3],
    /// axis-aligned bounds in the mesh frame after centring at the CoM and rotating into the principal frame
    /// (`aamm_`), which is what `fitaabb` fits
    aamm: [f64; 6],
}

struct MeshAsset {
    file: String,
    scale: Vector3<f64>,
    refpos: Vector3<f64>,
    refquat: UnitQuaternion<f64>,
    inertia: MeshInertia,
}

/// Volume (or area, for a shell), centre of mass and inertia per unit density about that centre, computed
/// exactly as `mjCMesh::Process` does: face centroid → one pass for the CoM → one pass for the second moments
/// about it. Both passes take pyramids from the apex to each face; `legacy` takes their volumes absolute.
fn mesh_inertia_mujoco(mesh: &crate::TriMesh3, method: MeshInertia) -> Result<(f64, Vector3<f64>, Matrix3<f64>), String> {
    const MINVAL: f64 = 1e-15;
    // triangle(): unit normal, centre, area — faces below MINVAL in |cross| are ignored (area 0)
    let tri = |a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>| -> (Vector3<f64>, Vector3<f64>, f64) {
        let n = (b - a).cross(&(c - a));
        let len = n.norm();
        if len < MINVAL {
            return (Vector3::zeros(), (a + b + c) / 3.0, 0.0);
        }
        (n / len, (a + b + c) / 3.0, 0.5 * len)
    };
    let v = |i: usize| mesh.verts[i];
    // area-weighted centroid of face centres
    let (mut facecen, mut total_area) = (Vector3::zeros(), 0.0);
    for t in &mesh.tris {
        let (_, c, area) = tri(v(t[0]), v(t[1]), v(t[2]));
        facecen += area * c;
        total_area += area;
    }
    if total_area < MINVAL {
        return Err("mesh surface area is too small".into());
    }
    facecen /= total_area;
    // pass 1: volume (or area) and CoM from pyramids at the face centroid
    let (mut vol, mut com) = (0.0, Vector3::zeros());
    for t in &mesh.tris {
        let (n, c, area) = tri(v(t[0]), v(t[1]), v(t[2]));
        let mut volume = match method {
            MeshInertia::Shell => area,
            _ => (c - facecen).dot(&n) * area / 3.0,
        };
        if method == MeshInertia::Legacy {
            volume = volume.abs();
        }
        vol += volume;
        com += volume * (c * 0.75 + facecen * 0.25);
    }
    if vol < MINVAL {
        return Err(if vol < 0.0 { "mesh volume is negative (misoriented triangles)".into() } else { "mesh volume is too small".into() });
    }
    com /= vol;
    // pass 2: products of inertia about the CoM, volume recomputed with the CoM as apex
    let (mut total, mut p) = (0.0, [0.0f64; 6]);
    let k = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let cdiv = if method == MeshInertia::Shell { 12.0 } else { 20.0 };
    for t in &mesh.tris {
        let (d, e, f) = (v(t[0]) - com, v(t[1]) - com, v(t[2]) - com);
        let (n, c, area) = tri(d, e, f);
        let mut volume = match method {
            MeshInertia::Shell => area,
            _ => c.dot(&n) * area / 3.0,
        };
        if method == MeshInertia::Legacy {
            volume = volume.abs();
        }
        total += volume;
        for (j, &(a, b)) in k.iter().enumerate() {
            p[j] += volume / cdiv * (2.0 * (d[a] * d[b] + e[a] * e[b] + f[a] * f[b]) + d[a] * e[b] + d[b] * e[a] + d[a] * f[b] + d[b] * f[a] + e[a] * f[b] + e[b] * f[a]);
        }
    }
    let inertia = Matrix3::new(p[1] + p[2], -p[3], -p[4], -p[3], p[0] + p[2], -p[5], -p[4], -p[5], p[0] + p[1]);
    Ok((total, com, inertia))
}

/// **MuJoCo's `mjuu_eig3`, ported line for line** — a Jacobi iteration from the identity quaternion with its
/// pivot rule, its 1e-12 **absolute** stopping tolerance, and a bubble sort into decreasing eigenvalues that
/// rotates the frame by 90° about the third axis at each swap. Returns `(eigenvalues, frame quaternion)` with
/// the eigenvectors as the columns of the quaternion's rotation matrix.
///
/// Ported rather than replaced by nalgebra's solver because MuJoCo's answer is what parity means, and the
/// absolute tolerance is a behaviour: a mesh whose unit-density inertia is below 1e-12 (a few-millimetre part
/// at unit scale, or a fly's claw) terminates at iteration zero, so MuJoCo takes the tensor's **diagonal** in
/// the mesh's own axes as its principal inertia and drops the off-diagonal terms. That is measurable against
/// MuJoCo and this port reproduces it.
fn eig3_mujoco(mat: &Matrix3<f64>) -> ([f64; 3], [f64; 4]) {
    const EPS: f64 = 1e-12;
    let quat2mat = |q: &[f64; 4]| -> Matrix3<f64> {
        if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
            return Matrix3::identity();
        }
        let (q00, q01, q02, q03) = (q[0] * q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3]);
        let (q11, q12, q13) = (q[1] * q[1], q[1] * q[2], q[1] * q[3]);
        let (q22, q23, q33) = (q[2] * q[2], q[2] * q[3], q[3] * q[3]);
        Matrix3::new(
            q00 + q11 - q22 - q33, 2.0 * (q12 - q03), 2.0 * (q13 + q02),
            2.0 * (q12 + q03), q00 - q11 + q22 - q33, 2.0 * (q23 - q01),
            2.0 * (q13 - q02), 2.0 * (q23 + q01), q00 - q11 - q22 + q33,
        )
    };
    let normalize = |q: &mut [f64; 4]| {
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        if n >= 1e-15 && (n - 1.0).abs() > 1e-15 {
            for x in q.iter_mut() {
                *x /= n;
            }
        }
    };
    let mulquat = |a: &[f64; 4], b: &[f64; 4]| -> [f64; 4] {
        let mut r = [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ];
        normalize(&mut r);
        r
    };
    let mut quat = [1.0, 0.0, 0.0, 0.0];
    let mut eigval = [0.0; 3];
    for _ in 0..500 {
        let ev = quat2mat(&quat);
        let d = ev.transpose() * mat * ev;
        eigval = [d[(0, 0)], d[(1, 1)], d[(2, 2)]];
        let (rk, ck, rotk) = if d[(0, 1)].abs() > d[(0, 2)].abs() && d[(0, 1)].abs() > d[(1, 2)].abs() {
            (0, 1, 2)
        } else if d[(0, 2)].abs() > d[(1, 2)].abs() {
            (0, 2, 1)
        } else {
            (1, 2, 0)
        };
        if d[(rk, ck)].abs() < EPS {
            break;
        }
        let tau = (d[(ck, ck)] - d[(rk, rk)]) / (2.0 * d[(rk, ck)]);
        let t = if tau >= 0.0 { 1.0 / (tau + (1.0 + tau * tau).sqrt()) } else { -1.0 / (-tau + (1.0 + tau * tau).sqrt()) };
        let c = 1.0 / (1.0 + t * t).sqrt();
        if c > 1.0 - EPS {
            break;
        }
        let mut tmp = [0.0; 4];
        tmp[rotk + 1] = if tau >= 0.0 { -(0.5 - 0.5 * c).sqrt() } else { (0.5 - 0.5 * c).sqrt() };
        if rotk == 1 {
            tmp[rotk + 1] = -tmp[rotk + 1];
        }
        tmp[0] = (1.0 - tmp[rotk + 1] * tmp[rotk + 1]).sqrt();
        normalize(&mut tmp);
        quat = mulquat(&quat, &tmp);
        normalize(&mut quat);
    }
    // bubble sort into decreasing order (0, 1, 0), rotating the frame by a quarter turn about the third axis
    for j in 0..3 {
        let j1 = j % 2;
        if eigval[j1] + EPS < eigval[j1 + 1] {
            eigval.swap(j1, j1 + 1);
            let mut tmp = [std::f64::consts::FRAC_1_SQRT_2, 0.0, 0.0, 0.0];
            tmp[(j1 + 2) % 3 + 1] = tmp[0];
            quat = mulquat(&quat, &tmp);
            normalize(&mut quat);
        }
    }
    (eigval, quat)
}

/// Rotation matrix of a MuJoCo `(w x y z)` quaternion, columns = axes.
fn quat_to_rotation(q: &[f64; 4]) -> Matrix3<f64> {
    *UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3])).to_rotation_matrix().matrix()
}

/// A mesh as MuJoCo's OBJ reader (tinyobjloader with `real_t = float`) delivers it: coordinates rounded to
/// `f32`, quads split along their **shorter diagonal** (measured in `f32`), larger polygons refused because
/// tinyobj's ear clipping is not reproduced here.
fn obj_as_mujoco_reads_it(text: &str) -> Result<crate::TriMesh3, String> {
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for line in text.lines() {
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("v") => {
                let mut c = [0.0f64; 3];
                for x in c.iter_mut() {
                    let t = tok.next().ok_or("OBJ vertex with fewer than 3 coordinates")?;
                    *x = t.parse::<f32>().map_err(|e| format!("OBJ vertex '{t}': {e}"))? as f64;
                }
                verts.push(Vector3::new(c[0], c[1], c[2]));
            }
            Some("f") => {
                let mut poly = Vec::new();
                for t in tok {
                    let raw: i64 = t.split('/').next().unwrap_or("").parse().map_err(|e| format!("OBJ face index '{t}': {e}"))?;
                    let idx = if raw > 0 { raw as usize - 1 } else if raw < 0 { verts.len() - raw.unsigned_abs() as usize } else { return Err("OBJ face index 0".into()) };
                    if idx >= verts.len() {
                        return Err("OBJ face index out of range".into());
                    }
                    poly.push(idx);
                }
                match poly.len() {
                    0..=2 => return Err("OBJ face with fewer than 3 vertices".into()),
                    3 => tris.push([poly[0], poly[1], poly[2]]),
                    4 => {
                        let (v0, v1, v2, v3) = (verts[poly[0]], verts[poly[1]], verts[poly[2]], verts[poly[3]]);
                        let sq = |a: Vector3<f64>, b: Vector3<f64>| {
                            let e = (b - a).map(|x| x as f32);
                            e.x * e.x + e.y * e.y + e.z * e.z
                        };
                        if sq(v0, v2) < sq(v1, v3) {
                            tris.push([poly[0], poly[1], poly[2]]);
                            tris.push([poly[0], poly[2], poly[3]]);
                        } else {
                            tris.push([poly[0], poly[1], poly[3]]);
                            tris.push([poly[1], poly[2], poly[3]]);
                        }
                    }
                    n => return Err(format!("OBJ face with {n} vertices: tinyobj's ear clipping is not reproduced here")),
                }
            }
            _ => {}
        }
    }
    if tris.is_empty() {
        return Err("OBJ has no faces".into());
    }
    Ok(crate::TriMesh3 { verts, tris })
}

/// One geom's contribution to its body: mass, centre in the body frame, inertia about that centre in the
/// body frame.
struct GeomMass {
    mass: f64,
    com: Vector3<f64>,
    inertia: Matrix3<f64>,
}

/// The default classes: for each class, its parent class and the attributes it sets per element kind.
#[derive(Default)]
struct Defaults {
    parent: HashMap<String, Option<String>>,
    /// (class, element kind) → attributes
    attrs: HashMap<(String, String), Vec<(String, String)>>,
}

const MAIN: &str = "";

impl Defaults {
    fn collect(root: &El) -> Result<Self, String> {
        let mut d = Defaults::default();
        d.parent.insert(MAIN.to_string(), None);
        for def in root.children_named("default") {
            d.collect_class(def, None)?;
        }
        Ok(d)
    }

    fn collect_class(&mut self, def: &El, parent: Option<&str>) -> Result<(), String> {
        let class = def.attr("class").unwrap_or(MAIN).to_string();
        if class != MAIN {
            match self.parent.get(&class) {
                // MuJoCo refuses a repeated class name; merging silently would hide a real authoring error
                Some(_) => return Err(format!("default class '{class}' is defined twice")),
                None => {
                    self.parent.insert(class.clone(), Some(parent.unwrap_or(MAIN).to_string()));
                }
            }
        }
        for el in &def.children {
            if el.name == "default" {
                self.collect_class(el, Some(&class))?;
            } else {
                self.attrs.entry((class.clone(), el.name.clone())).or_default().extend(el.attrs.iter().cloned());
            }
        }
        Ok(())
    }

    /// The value of `key` for `el` of kind `kind`: the element's own attribute, else its class chain, else the
    /// body's `childclass` chain, else main.
    fn get<'a>(&'a self, el: &'a El, kind: &str, key: &str, childclass: Option<&str>) -> Option<&'a str> {
        if let Some(v) = el.attr(key) {
            return Some(v);
        }
        let mut class: Option<&str> = el.attr("class").or(childclass).or(Some(MAIN));
        let mut guard = 0;
        while let Some(c) = class {
            if let Some(list) = self.attrs.get(&(c.to_string(), kind.to_string())) {
                // last definition of the key in the class wins, as it does when a file states an attribute twice
                if let Some((_, v)) = list.iter().rev().find(|(k, _)| k == key) {
                    return Some(v.as_str());
                }
            }
            class = self.parent.get(c).and_then(|p| p.as_deref());
            guard += 1;
            if guard > 64 {
                return None;
            }
        }
        None
    }

    fn known(&self, class: &str) -> bool {
        self.parent.contains_key(class)
    }
}

// ---------------------------------------------------------------------------------------------
// Orientation, with every MJCF form
// ---------------------------------------------------------------------------------------------

fn axis_of(ch: char) -> Unit<Vector3<f64>> {
    match ch.to_ascii_lowercase() {
        'x' => Vector3::x_axis(),
        'y' => Vector3::y_axis(),
        _ => Vector3::z_axis(),
    }
}

/// Orientation from `quat` / `axisangle` / `euler` / `xyaxes` / `zaxis`, reading each through the defaults.
fn orientation(get: &dyn Fn(&str) -> Option<String>, c: &Compiler) -> Result<UnitQuaternion<f64>, String> {
    let mut given = 0;
    let mut out = UnitQuaternion::identity();
    if let Some(q) = get("quat") {
        let v = floats(&q)?;
        if v.len() != 4 {
            return Err("quat needs 4 numbers (w x y z)".into());
        }
        out = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[0], v[1], v[2], v[3]));
        given += 1;
    }
    if let Some(aa) = get("axisangle") {
        let v = floats(&aa)?;
        if v.len() != 4 {
            return Err("axisangle needs 4 numbers".into());
        }
        let axis = Unit::try_new(Vector3::new(v[0], v[1], v[2]), 1e-12).ok_or("axisangle axis is zero")?;
        out = UnitQuaternion::from_axis_angle(&axis, v[3] * c.deg);
        given += 1;
    }
    if let Some(e) = get("euler") {
        let v = floats(&e)?;
        if v.len() != 3 {
            return Err("euler needs 3 numbers".into());
        }
        let mut r = UnitQuaternion::identity();
        for (ch, &ang) in c.eulerseq.chars().zip(&v) {
            let step = UnitQuaternion::from_axis_angle(&axis_of(ch), ang * c.deg);
            // lower-case: intrinsic, about the already-rotated axis (right-multiply); upper-case: extrinsic,
            // about the fixed axis (left-multiply)
            r = if ch.is_ascii_lowercase() { r * step } else { step * r };
        }
        out = r;
        given += 1;
    }
    if let Some(xy) = get("xyaxes") {
        let v = floats(&xy)?;
        if v.len() != 6 {
            return Err("xyaxes needs 6 numbers".into());
        }
        let x = Unit::try_new(Vector3::new(v[0], v[1], v[2]), 1e-12).ok_or("xyaxes x is zero")?;
        let y0 = Vector3::new(v[3], v[4], v[5]);
        let y = Unit::try_new(y0 - x.into_inner() * x.dot(&y0), 1e-12).ok_or("xyaxes y is parallel to x")?;
        let z = x.cross(&y);
        let m = Matrix3::from_columns(&[x.into_inner(), y.into_inner(), z]);
        out = UnitQuaternion::from_rotation_matrix(&nalgebra::Rotation3::from_matrix_unchecked(m));
        given += 1;
    }
    if let Some(za) = get("zaxis") {
        let v = vec3(&za)?;
        let z = Unit::try_new(v, 1e-12).ok_or("zaxis is zero")?;
        out = UnitQuaternion::rotation_between(&Vector3::z(), &z).unwrap_or_else(|| {
            // exactly antiparallel: no unique minimal rotation; MuJoCo picks one, and so does this
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI)
        });
        given += 1;
    }
    if given > 1 {
        return Err("more than one of quat/axisangle/euler/xyaxes/zaxis given — MuJoCo refuses this too".into());
    }
    Ok(out)
}

fn pose_of(el: &El, kind: &str, defaults: &Defaults, childclass: Option<&str>, c: &Compiler) -> Result<Iso, String> {
    let get = |k: &str| defaults.get(el, kind, k, childclass).map(|s| s.to_string());
    // `fromto` (sites and geoms): the frame sits at the segment's midpoint with +z along it, and pos and the
    // orientation attributes are ignored. flybody's claw sites use it; without this they sat 5.6 mm off.
    if let Some(ft) = get("fromto") {
        let v = floats(&ft)?;
        if v.len() != 6 {
            return Err(format!("{kind} fromto needs 6 numbers"));
        }
        let (p1, p2) = (Vector3::new(v[0], v[1], v[2]), Vector3::new(v[3], v[4], v[5]));
        let d = p2 - p1;
        if !(d.norm().is_finite() && d.norm() > 0.0) {
            return Err(format!("{kind} fromto endpoints coincide"));
        }
        let rot = UnitQuaternion::rotation_between(&Vector3::z(), &d)
            .unwrap_or_else(|| UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI));
        return Ok(Iso::from_parts(Translation3::from((p1 + p2) / 2.0), rot));
    }
    let p = get("pos").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::zeros);
    Ok(Iso::from_parts(Translation3::from(p), orientation(&get, c)?))
}

fn inertial_of(el: &El, c: &Compiler) -> Result<LinkInertia, String> {
    let mass = el.attr("mass").ok_or("inertial needs mass")?.trim().parse::<f64>().map_err(|e| e.to_string())?;
    let com = el.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
    let get = |k: &str| el.attr(k).map(|s| s.to_string());
    let rm = *orientation(&get, c)?.to_rotation_matrix().matrix();
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
    } else if mass == 0.0 {
        // MuJoCo accepts `<inertial pos="0 0 0" mass="0"/>` (Menagerie's rby1 uses it for its world body):
        // a massless body has no tensor to state
        Matrix3::zeros()
    } else {
        return Err("inertial needs diaginertia or fullinertia".into());
    };
    Ok(LinkInertia { mass, com, inertia: rm * ic * rm.transpose() })
}

// ---------------------------------------------------------------------------------------------
// The walk
// ---------------------------------------------------------------------------------------------

struct Walk<'a> {
    defaults: &'a Defaults,
    c: &'a Compiler,
    out: MjcfTree,
    unnamed_bodies: usize,
    unnamed_joints: usize,
    meshes: &'a HashMap<String, MeshAsset>,
    /// per mesh name: what `mjCMesh::Process` leaves behind, in the mesh's frame
    mesh_cache: HashMap<String, MeshData>,
    resolve: &'a dyn Fn(&str) -> Option<Vec<u8>>,
}

const REFUSED: [&str; 4] = ["replicate", "attach", "composite", "flexcomp"];

impl Walk<'_> {
    /// Visit the children of a body-like element (`worldbody`, `body`, `frame`).
    ///
    /// `parent` is the tree joint the enclosing frame rides on (`-1` = world), `carry` the fixed transform from
    /// that joint's frame to the element's own frame.
    fn children(&mut self, el: &El, parent: isize, carry: Iso, childclass: Option<&str>) -> Result<(), String> {
        for ch in &el.children {
            match ch.name.as_str() {
                "body" => self.body(ch, parent, carry, childclass)?,
                "frame" => {
                    let f = pose_of(ch, "frame", self.defaults, childclass, self.c)?;
                    let cc = ch.attr("childclass").or(childclass);
                    self.children(ch, parent, carry * f, cc)?;
                }
                "site" => {
                    let name = ch.attr("name").map(|s| s.to_string());
                    let sp = pose_of(ch, "site", self.defaults, childclass, self.c)?;
                    if let Some(name) = name {
                        self.place("site", name, parent, carry * sp)?;
                    }
                }
                n if REFUSED.contains(&n) => {
                    return Err(format!("<{n}> is procedural model generation, outside this loader's subset"));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn place(&mut self, kind: &str, name: String, parent: isize, pose: Iso) -> Result<(), String> {
        let map = if kind == "body" { &mut self.out.body_frames } else { &mut self.out.site_frames };
        if map.contains_key(&name) || self.out.world_fixed.contains_key(&format!("{kind}:{name}")) {
            return Err(format!("{kind} name '{name}' is used twice"));
        }
        if parent < 0 {
            self.out.world_fixed.insert(format!("{kind}:{name}"), pose);
        } else {
            map.insert(name.clone(), (parent as usize, pose));
            if kind == "body" {
                self.out.tree.tip_offsets.insert(name, (parent as usize, pose));
            }
        }
        Ok(())
    }

    fn body(&mut self, b: &El, parent: isize, carry: Iso, childclass: Option<&str>) -> Result<(), String> {
        let name = match b.attr("name") {
            Some(n) => n.to_string(),
            None => {
                self.unnamed_bodies += 1;
                format!("body{}", self.body_count())
            }
        };
        let childclass = b.attr("childclass").or(childclass);
        if let Some(cc) = childclass.filter(|cc| !self.defaults.known(cc)) {
            return Err(format!("body '{name}' names unknown childclass '{cc}'"));
        }
        let joints: Vec<&El> = b.children.iter().filter(|c| c.name == "joint" || c.name == "freejoint").collect();
        let has_free = joints.iter().any(|j| j.name == "freejoint" || self.defaults.get(j, "joint", "type", childclass) == Some("free"));
        if has_free && (parent >= 0 || joints.len() > 1) {
            return Err(format!("body '{name}': a free joint must be the only joint of a child of the world"));
        }
        // the body frame relative to the enclosing frame — dropped entirely for a free body, whose pose IS qpos
        let bpose = if has_free { Iso::identity() } else { pose_of(b, "body", self.defaults, None, self.c)? };

        let mut ride = parent; // tree joint the body frame currently rides on
        let mut pre = carry * bpose; // from that joint's frame to the body frame
        for j in &joints {
            let jname = match j.attr("name") {
                Some(n) => n.to_string(),
                None => {
                    self.unnamed_joints += 1;
                    format!("joint{}", self.out.joints.len())
                }
            };
            let kind = if j.name == "freejoint" {
                MjcfJointKind::Free
            } else {
                match self.defaults.get(j, "joint", "type", childclass).unwrap_or("hinge") {
                    "hinge" => MjcfJointKind::Hinge,
                    "slide" => MjcfJointKind::Slide,
                    "ball" => MjcfJointKind::Ball,
                    "free" => MjcfJointKind::Free,
                    other => return Err(format!("joint '{jname}': unknown type '{other}'")),
                }
            };
            let get = |k: &str| self.defaults.get(j, "joint", k, childclass).map(|s| s.to_string());
            let anchor = if kind == MjcfJointKind::Free {
                Vector3::zeros()
            } else {
                get("pos").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::zeros)
            };
            let axis = get("axis").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::z);
            if axis.norm() < 1e-12 {
                return Err(format!("joint '{jname}' has a zero axis"));
            }
            let angle_scale = if kind == MjcfJointKind::Hinge { self.c.deg } else { 1.0 };
            let reference = get("ref").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(0.0) * angle_scale;
            let first = self.out.tree.joints.len();
            let parse = |s: Option<String>| s.and_then(|v| v.trim().parse::<f64>().ok());
            let (armature, damping, frictionloss) = (parse(get("armature")), parse(get("damping")), parse(get("frictionloss")));
            let limits = {
                let limited = get("limited");
                let range = get("range");
                match (limited.as_deref(), range) {
                    (Some("true"), Some(r)) => Some(r),
                    (Some("false"), _) | (_, None) => None,
                    (Some("auto") | None, Some(r)) if self.c.autolimits => Some(r),
                    (None, Some(_)) => {
                        return Err(format!("joint '{jname}' has a range but no limited and autolimits is false — MuJoCo refuses this"))
                    }
                    (Some(other), _) => return Err(format!("joint '{jname}': limited '{other}' is not true/false/auto")),
                }
            };
            let limits = match limits {
                Some(r) if kind == MjcfJointKind::Hinge || kind == MjcfJointKind::Slide => {
                    let v = floats(&r)?;
                    if v.len() != 2 {
                        return Err(format!("joint '{jname}': range needs 2 numbers"));
                    }
                    Some((v[0] * angle_scale, v[1] * angle_scale))
                }
                _ => None,
            };
            let mut push = |joint: Joint| {
                self.out.tree.joints.push(joint);
                self.out.tree.parent.push(ride);
                self.out.tree.inertia.push(LinkInertia::zero());
                ride = (self.out.tree.joints.len() - 1) as isize;
            };
            match kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => {
                    // T(anchor)·R(axis, q − ref)·T(−anchor): the ref rotation folds into the origin, so the
                    // caller passes MuJoCo's raw q
                    let unref = if kind == MjcfJointKind::Hinge {
                        Iso::from_parts(Translation3::identity(), UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), -reference))
                    } else {
                        Iso::from_parts(Translation3::from(-axis.normalize() * reference), UnitQuaternion::identity())
                    };
                    let origin = pre * Iso::from_parts(Translation3::from(anchor), UnitQuaternion::identity()) * unref;
                    let mut joint = if kind == MjcfJointKind::Hinge { Joint::revolute(origin, axis) } else { Joint::prismatic(origin, axis) };
                    if let Some((lo, hi)) = limits {
                        joint = joint.with_limits(lo, hi);
                    }
                    if let Some(a) = armature {
                        joint = joint.with_armature(a);
                    }
                    if let Some(d) = damping {
                        joint = joint.with_damping(d);
                    }
                    if let Some(f) = frictionloss {
                        joint = joint.with_friction(f);
                    }
                    push(joint);
                }
                MjcfJointKind::Ball => {
                    let origin = pre * Iso::from_parts(Translation3::from(anchor), UnitQuaternion::identity());
                    push(Joint::revolute(origin, Vector3::z()));
                    push(Joint::revolute(Iso::identity(), Vector3::y()));
                    push(Joint::revolute(Iso::identity(), Vector3::x()));
                }
                MjcfJointKind::Free => {
                    push(Joint::prismatic(Iso::identity(), Vector3::x()));
                    push(Joint::prismatic(Iso::identity(), Vector3::y()));
                    push(Joint::prismatic(Iso::identity(), Vector3::z()));
                    push(Joint::revolute(Iso::identity(), Vector3::z()));
                    push(Joint::revolute(Iso::identity(), Vector3::y()));
                    push(Joint::revolute(Iso::identity(), Vector3::x()));
                }
            }
            self.out.tree.joint_names.insert(jname.clone(), first);
            self.out.joints.push(MjcfJoint { name: jname, kind, body: name.clone(), first, reference });
            // after the motion, the body frame sits at −anchor from the joint's frame
            pre = Iso::from_parts(Translation3::from(-anchor), UnitQuaternion::identity());
        }

        // inertia: a jointed body owns its last joint's link; a jointless one welds into the ancestor it rides on.
        // `inertiafromgeom`: auto = the <inertial> if stated, else the geoms; true = always the geoms; false = never
        let stated = b.child("inertial");
        let mut li = match (stated, self.c.inertiafromgeom) {
            (Some(el), InertiaFromGeom::Auto | InertiaFromGeom::False) => inertial_of(el, self.c).map_err(|e| format!("body '{name}': {e}"))?,
            (None, InertiaFromGeom::False) => {
                self.out.no_inertial.push(name.clone());
                LinkInertia::zero()
            }
            (_, _) => {
                if stated.is_none() {
                    self.out.no_inertial.push(name.clone());
                }
                match self.inertia_from_geoms(b, childclass, &name)? {
                    Some(li) => {
                        self.out.inferred_from_geoms.push(name.clone());
                        li
                    }
                    None => LinkInertia::zero(),
                }
            }
        };
        // mjCBody::Compile: floors on the mass and on each principal inertia
        if self.c.boundmass > 0.0 {
            li.mass = li.mass.max(self.c.boundmass);
        }
        if self.c.boundinertia > 0.0 {
            let eig = li.inertia.symmetric_eigen();
            let floored = eig.eigenvalues.map(|e| e.max(self.c.boundinertia));
            li.inertia = eig.eigenvectors * Matrix3::from_diagonal(&floored) * eig.eigenvectors.transpose();
        }
        if ride >= 0 {
            let idx = ride as usize;
            let moved = crate::dynamics::transform_inertia(&li, &pre);
            self.out.tree.inertia[idx] = crate::dynamics::combine_inertia(&self.out.tree.inertia[idx], &moved);
            if !joints.is_empty() {
                self.out.tree.link_names.insert(name.clone(), idx);
            }
        }
        self.place("body", name, ride, pre)?;
        self.children(b, ride, pre, childclass)
    }

    /// MuJoCo's `mjCGeom::Compile` + `SetInertia` for one geom, in the body frame; `None` for a geom MuJoCo
    /// gives no mass (planes, height fields, zero density or mass, outside the group range).
    fn geom_mass(&mut self, g: &El, childclass: Option<&str>, body: &str) -> Result<Option<GeomMass>, String> {
        let get = |k: &str| self.defaults.get(g, "geom", k, childclass).map(|s| s.to_string());
        let group: i64 = get("group").map(|s| s.trim().parse::<f64>().map(|x| x as i64).map_err(|e| e.to_string())).transpose()?.unwrap_or(0);
        if group < self.c.inertiagrouprange.0 || group > self.c.inertiagrouprange.1 {
            return Ok(None);
        }
        if get("shellinertia").as_deref() == Some("true") {
            return Err(format!("body '{body}': geom shellinertia is outside this loader's subset"));
        }
        let ty = get("type").unwrap_or_else(|| "sphere".into());
        let mut size = get("size").map(|s| floats(&s)).transpose()?.unwrap_or_default();
        let mut pose = pose_of(g, "geom", self.defaults, childclass, self.c)?;
        // **Mesh fitting.** A geom of a primitive type that names a `mesh` is sized from that mesh
        // (`mjCMesh::FitGeom`): from its equivalent inertia box, or its bounding box under `fitaabb`, times
        // `fitscale`, and placed at the mesh's centre of mass in its principal frame. Menagerie uses this for
        // collision capsules fitted to visual meshes, and a geom with a mesh and no type is a fitted SPHERE.
        let mesh_name = get("mesh");
        if let (Some(mname), true) = (&mesh_name, ty != "mesh") {
            if get("fromto").is_some() {
                return Err(format!("body '{body}': fromto cannot be used with a fitted mesh geom"));
            }
            let md = self.mesh_data(mname)?;
            let fitscale: f64 = get("fitscale").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(1.0);
            let mut center = Vector3::zeros();
            let mut sz = [0.0f64; 3];
            if !self.c.fitaabb {
                let b = md.boxsz;
                match ty.as_str() {
                    "sphere" => sz[0] = (b[0] + b[1] + b[2]) / 3.0,
                    "capsule" => {
                        sz[0] = (b[0] + b[1]) / 2.0;
                        sz[1] = (b[2] - sz[0] / 2.0).max(0.0);
                    }
                    "cylinder" => {
                        sz[0] = (b[0] + b[1]) / 2.0;
                        sz[1] = b[2];
                    }
                    "ellipsoid" | "box" => sz = b,
                    other => return Err(format!("body '{body}': invalid geom type '{other}' in fitting mesh '{mname}'")),
                }
            } else {
                let a = md.aamm;
                center = Vector3::new((a[0] + a[3]) / 2.0, (a[1] + a[4]) / 2.0, (a[2] + a[5]) / 2.0);
                let half = [a[3] - center.x, a[4] - center.y, a[5] - center.z];
                match ty.as_str() {
                    "sphere" => sz[0] = half[0].max(half[1]).max(half[2]),
                    "capsule" | "cylinder" => {
                        sz[0] = half[0].max(half[1]);
                        sz[1] = half[2];
                        if ty == "capsule" {
                            sz[1] -= sz[0];
                        }
                    }
                    "ellipsoid" | "box" => sz = half,
                    other => return Err(format!("body '{body}': invalid fittype '{other}' in mesh '{mname}'")),
                }
            }
            size = sz.iter().map(|v| v * fitscale).collect();
            // accumulate the mesh frame into the geom frame: pos += R·(R_mesh·center + com), quat ∘= quat_mesh
            let rm = quat_to_rotation(&md.quat);
            let meshpos = rm * center + md.com;
            let r = *pose.rotation.to_rotation_matrix().matrix();
            let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(md.quat[0], md.quat[1], md.quat[2], md.quat[3]));
            pose = Iso::from_parts(Translation3::from(pose.translation.vector + r * meshpos), pose.rotation * q);
        }
        if let Some(ft) = get("fromto") {
            // pose_of already placed the frame at the midpoint with +z along the segment; MuJoCo puts the
            // half-length into size[1] (capsule, cylinder) or size[2] with size[1] = size[0] (box, ellipsoid)
            let v = floats(&ft)?;
            let half = (Vector3::new(v[0], v[1], v[2]) - Vector3::new(v[3], v[4], v[5])).norm() / 2.0;
            match ty.as_str() {
                "capsule" | "cylinder" => {
                    size.resize(2, 0.0);
                    size[1] = half;
                }
                "box" | "ellipsoid" => {
                    size.resize(3, 0.0);
                    size[2] = half;
                    size[1] = size[0];
                }
                other => return Err(format!("body '{body}': fromto requires capsule, cylinder, box or ellipsoid, not {other}")),
            }
        }
        let need = |k: usize| -> Result<(), String> {
            if size.len() < k {
                Err(format!("body '{body}': geom type '{ty}' needs {k} size value(s), got {}", size.len()))
            } else {
                Ok(())
            }
        };
        let pi = std::f64::consts::PI;
        // (volume, inertia per unit MASS about the centre in the geom frame, centre offset in the geom frame)
        let (volume, unit_inertia, com_geom): (f64, Matrix3<f64>, Vector3<f64>) = match ty.as_str() {
            "sphere" => {
                need(1)?;
                let r = size[0];
                (4.0 * pi * r * r * r / 3.0, Matrix3::identity() * (2.0 * r * r / 5.0), Vector3::zeros())
            }
            "capsule" => {
                need(2)?;
                let (r, h) = (size[0], 2.0 * size[1]);
                let volume = pi * (r * r * h + 4.0 * r * r * r / 3.0);
                let sphere = 4.0 * r / (4.0 * r + 3.0 * h); // the sphere's share of the mass
                let cyl = 1.0 - sphere;
                let si = 2.0 * sphere * r * r / 5.0;
                let ixx = cyl * (3.0 * r * r + h * h) / 12.0 + si + sphere * h * (3.0 * r + 2.0 * h) / 8.0;
                (volume, Matrix3::from_diagonal(&Vector3::new(ixx, ixx, cyl * r * r / 2.0 + si)), Vector3::zeros())
            }
            "cylinder" => {
                need(2)?;
                let (r, h) = (size[0], 2.0 * size[1]);
                (pi * r * r * h, Matrix3::from_diagonal(&Vector3::new((3.0 * r * r + h * h) / 12.0, (3.0 * r * r + h * h) / 12.0, r * r / 2.0)), Vector3::zeros())
            }
            "ellipsoid" => {
                need(3)?;
                let (a, b, c) = (size[0], size[1], size[2]);
                (4.0 * pi * a * b * c / 3.0, Matrix3::from_diagonal(&Vector3::new((b * b + c * c) / 5.0, (a * a + c * c) / 5.0, (a * a + b * b) / 5.0)), Vector3::zeros())
            }
            "box" => {
                need(3)?;
                let (a, b, c) = (size[0], size[1], size[2]);
                (8.0 * a * b * c, Matrix3::from_diagonal(&Vector3::new((b * b + c * c) / 3.0, (a * a + c * c) / 3.0, (a * a + b * b) / 3.0)), Vector3::zeros())
            }
            "mesh" => {
                let name = mesh_name.clone().ok_or_else(|| format!("body '{body}': a mesh geom needs a `mesh` attribute"))?;
                if get("fromto").is_some() {
                    return Err(format!("body '{body}': fromto cannot be used with a mesh geom"));
                }
                let md = self.mesh_data(&name)?;
                if md.volume < 1e-15 {
                    return Ok(None);
                }
                // mjCGeom::SetInertia for a mesh: the equivalent box's inertia per unit mass, in the mesh's
                // principal frame — which is MuJoCo's own eigendecomposition, quirks included
                let b = md.boxsz;
                let diag = Vector3::new((b[1] * b[1] + b[2] * b[2]) / 3.0, (b[0] * b[0] + b[2] * b[2]) / 3.0, (b[0] * b[0] + b[1] * b[1]) / 3.0);
                let rm = quat_to_rotation(&md.quat);
                (md.volume, rm * Matrix3::from_diagonal(&diag) * rm.transpose(), md.com)
            }
            "plane" | "hfield" | "sdf" => return Ok(None),
            other => return Err(format!("body '{body}': geom type '{other}' is not one this loader can weigh")),
        };
        // an explicit `mass` fixes it; otherwise density × volume, default density 1000
        let mass = match get("mass").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()? {
            Some(0.0) => return Ok(None),
            Some(m) => m,
            None => {
                let density = get("density").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(1000.0);
                if density == 0.0 {
                    return Ok(None);
                }
                density * volume
            }
        };
        if mass.is_nan() || mass <= 1e-15 {
            return Ok(None);
        }
        // a mesh's frame is accumulated into the geom's: centre = geom pos + R·CoM, tensor rotated by R
        let r = *pose.rotation.to_rotation_matrix().matrix();
        Ok(Some(GeomMass { mass, com: pose.translation.vector + r * com_geom, inertia: r * (unit_inertia * mass) * r.transpose() }))
    }

    /// `mjCMesh::Process` for one mesh, loaded through the resolver once per mesh name.
    fn mesh_data(&mut self, name: &str) -> Result<MeshData, String> {
        if let Some(d) = self.mesh_cache.get(name) {
            return Ok(*d);
        }
        let asset = self.meshes.get(name).ok_or_else(|| format!("geom references mesh '{name}', which no <asset><mesh> declares"))?;
        let path = if asset.file.starts_with('/') || self.c.meshdir.is_empty() || self.c.meshdir.ends_with('/') {
            format!("{}{}", if asset.file.starts_with('/') { "" } else { self.c.meshdir.as_str() }, asset.file)
        } else {
            format!("{}/{}", self.c.meshdir, asset.file)
        };
        let bytes = (self.resolve)(&path).ok_or_else(|| format!("mesh '{name}': file '{path}' could not be resolved"))?;
        let lower = asset.file.to_ascii_lowercase();
        let mut mesh = if lower.ends_with(".obj") {
            obj_as_mujoco_reads_it(&String::from_utf8_lossy(&bytes)).map_err(|e| format!("mesh '{name}': '{path}': {e}"))?
        } else if lower.ends_with(".stl") {
            // MuJoCo reads binary STL only; its float32 coordinates are what both sides see
            crate::mesh_io::from_stl(&bytes).ok_or_else(|| format!("mesh '{name}': '{path}' did not parse"))?
        } else {
            return Err(format!("mesh '{name}': '{path}' is neither OBJ nor STL — .msh and others are outside this loader's subset"));
        };
        // MuJoCo stores vertices as float; the compiler works on a double copy of those
        for v in &mut mesh.verts {
            *v = v.map(|x| x as f32 as f64);
        }
        // mjCMesh::ApplyTransformations — refpos, then the inverse refquat rotation, then scale
        let rq = *asset.refquat.to_rotation_matrix().matrix();
        for v in &mut mesh.verts {
            let p = rq.transpose() * (*v - asset.refpos);
            *v = Vector3::new(p.x * asset.scale.x, p.y * asset.scale.y, p.z * asset.scale.z);
        }
        let (volume, com, unit) = mesh_inertia_mujoco(&mesh, asset.inertia).map_err(|e| format!("mesh '{name}': {e}"))?;
        let (eigval, quat) = eig3_mujoco(&unit);
        if eigval[2] <= 0.0 {
            return Err(format!("mesh '{name}': eigenvalue of mesh inertia must be positive"));
        }
        let (rtol, atol) = (1e-6, 1e-9);
        if eigval[0] + eigval[1] < eigval[2] * (1.0 - rtol) - atol || eigval[0] + eigval[2] < eigval[1] * (1.0 - rtol) - atol || eigval[1] + eigval[2] < eigval[0] * (1.0 - rtol) - atol {
            return Err(format!("mesh '{name}': eigenvalues of mesh inertia violate A + B >= C"));
        }
        let boxsz = [
            0.5 * (6.0 * (eigval[1] + eigval[2] - eigval[0]) / volume).sqrt(),
            0.5 * (6.0 * (eigval[0] + eigval[2] - eigval[1]) / volume).sqrt(),
            0.5 * (6.0 * (eigval[0] + eigval[1] - eigval[2]) / volume).sqrt(),
        ];
        // the mesh is re-expressed centred at its CoM in its principal frame; the bounds are taken there
        let r = quat_to_rotation(&quat);
        let mut aamm = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for v in &mesh.verts {
            let p = r.transpose() * (*v - com);
            for k in 0..3 {
                aamm[k] = aamm[k].min(p[k]);
                aamm[k + 3] = aamm[k + 3].max(p[k]);
            }
        }
        let d = MeshData { volume, com, quat, boxsz, aamm };
        self.mesh_cache.insert(name.to_string(), d);
        Ok(d)
    }

    /// `mjCBody::InertiaFromGeom`: the body's inertia from the geoms MuJoCo would weigh, in the body frame.
    fn inertia_from_geoms(&mut self, b: &El, childclass: Option<&str>, body: &str) -> Result<Option<LinkInertia>, String> {
        let mut parts = Vec::new();
        for g in b.children.iter().filter(|c| c.name == "geom") {
            if let Some(gm) = self.geom_mass(g, childclass, body)? {
                parts.push(gm);
            }
        }
        if parts.is_empty() {
            return Ok(None);
        }
        let mass: f64 = parts.iter().map(|p| p.mass).sum();
        let com = parts.iter().map(|p| p.com * p.mass).sum::<Vector3<f64>>() / mass;
        let mut inertia = Matrix3::zeros();
        for p in &parts {
            let d = p.com - com;
            inertia += p.inertia + p.mass * (Matrix3::identity() * d.dot(&d) - d * d.transpose());
        }
        Ok(Some(LinkInertia { mass, com, inertia }))
    }

    fn body_count(&self) -> usize {
        self.out.body_frames.len() + self.out.world_fixed.keys().filter(|k| k.starts_with("body:")).count() + 1
    }
}

/// **Load an MJCF model as a branched tree**, resolving `<include file>` and `<asset><mesh file>` through
/// `resolve`, which receives the path exactly as written (MuJoCo resolves every include, nested ones too,
/// against the main model's directory; a mesh path is `meshdir/file`) and returns the file's bytes.
pub fn tree_from_mjcf(xml: &str, resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<MjcfTree, String> {
    let mut root = parse_xml(xml)?;
    if root.name != "mujoco" {
        return Err(format!("root element is <{}>, expected <mujoco>", root.name));
    }
    expand_includes(&mut root, resolve, 0)?;
    let c = compiler(&root)?;
    let defaults = Defaults::collect(&root)?;
    // <asset><mesh name file scale refpos refquat inertia> — several <asset> blocks can arrive through includes
    let mut meshes: HashMap<String, MeshAsset> = HashMap::new();
    for asset in root.children_named("asset") {
        for m in asset.children_named("mesh") {
            let get = |k: &str| defaults.get(m, "mesh", k, None).map(|s| s.to_string());
            let file = get("file").ok_or("<mesh> needs a file")?;
            let stem = file.rsplit('/').next().unwrap_or(&file);
            let stem = stem.rsplit_once('.').map(|(s, _)| s).unwrap_or(stem);
            let name = m.attr("name").map(|s| s.to_string()).unwrap_or_else(|| stem.to_string());
            let scale = get("scale").map(|s| vec3(&s)).transpose()?.unwrap_or_else(|| Vector3::new(1.0, 1.0, 1.0));
            let refpos = get("refpos").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::zeros);
            let refquat = match get("refquat") {
                Some(q) => {
                    let v = floats(&q)?;
                    if v.len() != 4 {
                        return Err(format!("mesh '{name}': refquat needs 4 numbers"));
                    }
                    UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(v[0], v[1], v[2], v[3]))
                }
                None => UnitQuaternion::identity(),
            };
            let inertia = match get("inertia").as_deref() {
                None | Some("legacy") => MeshInertia::Legacy,
                Some("exact") => MeshInertia::Exact,
                Some("shell") => MeshInertia::Shell,
                Some("convex") => return Err(format!("mesh '{name}': inertia=\"convex\" needs the convex hull, outside this loader's subset")),
                Some(other) => return Err(format!("mesh '{name}': unknown inertia method '{other}'")),
            };
            meshes.insert(name, MeshAsset { file, scale, refpos, refquat, inertia });
        }
    }
    // an <include> outside the worldbody splices in a second <worldbody>; MuJoCo merges them in order
    let worlds: Vec<&El> = root.children_named("worldbody").collect();
    if worlds.is_empty() {
        return Err("no <worldbody>".into());
    }
    let mut walk = Walk {
        defaults: &defaults,
        c: &c,
        out: MjcfTree {
            tree: KinematicTree {
                joints: Vec::new(),
                parent: Vec::new(),
                inertia: Vec::new(),
                joint_names: BTreeMap::new(),
                link_names: BTreeMap::new(),
                tip_offsets: BTreeMap::new(),
            },
            joints: Vec::new(),
            body_frames: BTreeMap::new(),
            site_frames: BTreeMap::new(),
            world_fixed: BTreeMap::new(),
            no_inertial: Vec::new(),
            inferred_from_geoms: Vec::new(),
            angle_scale: c.deg,
        },
        unnamed_bodies: 0,
        unnamed_joints: 0,
        meshes: &meshes,
        mesh_cache: HashMap::new(),
        resolve,
    };
    for world in worlds {
        walk.children(world, -1, Iso::identity(), world.attr("childclass"))?;
    }
    let out = walk.out;
    for (i, p) in out.tree.parent.iter().enumerate() {
        if *p >= i as isize {
            return Err("internal error: joints are not in topological order".into());
        }
    }
    Ok(out)
}

/// [`tree_from_mjcf`] for a self-contained model: any `<include>` is an error naming the file it wanted.
pub fn tree_from_mjcf_str(xml: &str) -> Result<MjcfTree, String> {
    tree_from_mjcf(xml, &|_| None)
}

#[cfg(test)]
mod tests {
    //! Every number here was produced by MuJoCo 3.13.0 (`mj_kinematics`) on the model in the same test, with a
    //! unit sphere geom added to each body so the compiler accepted it. The tolerance is what MuJoCo's own
    //! single-precision-free double pipeline leaves: 1e-12 on positions, 1e-12 on quaternions up to sign.
    use super::*;
    use nalgebra::Point3;

    const TOL: f64 = 1e-12;

    fn check_body(t: &MjcfTree, q: &[f64], name: &str, xpos: [f64; 3], xquat: [f64; 4]) {
        let p = t.body_pose(name, q).unwrap_or_else(|| panic!("no body {name}"));
        check_pose(&p, name, xpos, xquat);
    }

    fn check_pose(p: &Iso, name: &str, xpos: [f64; 3], xquat: [f64; 4]) {
        let dp = (p.translation.vector - Vector3::new(xpos[0], xpos[1], xpos[2])).norm();
        let q = p.rotation.quaternion();
        let (w, x, y, z) = (q.w, q.i, q.j, q.k);
        let d1 = ((w - xquat[0]).powi(2) + (x - xquat[1]).powi(2) + (y - xquat[2]).powi(2) + (z - xquat[3]).powi(2)).sqrt();
        let d2 = ((w + xquat[0]).powi(2) + (x + xquat[1]).powi(2) + (y + xquat[2]).powi(2) + (z + xquat[3]).powi(2)).sqrt();
        assert!(dp < TOL, "{name}: position off by {dp:.3e} (got {:?}, MuJoCo {xpos:?})", p.translation.vector);
        assert!(d1.min(d2) < TOL, "{name}: quaternion off by {:.3e} (got [{w} {x} {y} {z}], MuJoCo {xquat:?})", d1.min(d2));
    }

    fn check_site(t: &MjcfTree, q: &[f64], name: &str, xpos: [f64; 3]) {
        let p = t.site_pose(name, q).unwrap_or_else(|| panic!("no site {name}"));
        let dp = (p.translation.vector - Vector3::new(xpos[0], xpos[1], xpos[2])).norm();
        assert!(dp < TOL, "site {name}: off by {dp:.3e}");
    }

    #[test]
    fn composite_joints_in_one_body_compose_in_file_order_about_their_anchors() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="b1" pos="0.1 0.2 0.3" euler="0.3 0.4 0.5">
  <joint name="j1" type="hinge" axis="0 0 1" pos="0.05 0 0"/>
  <joint name="j2" type="hinge" axis="0 1 0" pos="0 0.07 0"/>
  <joint name="j3" type="slide" axis="1 0 0" pos="0.01 0.02 0.03"/>
  <site name="s1" pos="0.1 0.1 0.1"/>
  <body name="b2" pos="0.5 0 0" quat="0.9 0.1 0.2 0.3"><joint name="j4" axis="1 1 0"/><site name="s2" pos="0 0.2 0"/></body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        assert_eq!(t.tree.dof(), 4);
        assert_eq!(t.joints.len(), 4);
        let q = [0.7, -0.4, 0.15, 1.1];
        check_body(&t, &q, "b1", [0.1925857829518738, 0.29422494862374926, 0.3545563982013887], [0.7832092098471127, 0.34074970957435335, -0.07771831313710306, 0.5142303305317358]);
        check_body(&t, &q, "b2", [0.4221128139162187, 0.6704923868488532, 0.5906499326071624], [0.35364234955873286, 0.13477932780837915, 0.5265429852707952, 0.7612648067894767]);
        check_site(&t, &q, "s1", [0.17551553187974175, 0.33200111185716835, 0.522727644315867]);
        check_site(&t, &q, "s2", [0.3428134677992626, 0.6314165575445739, 0.7700508614498136]);
    }

    #[test]
    fn degrees_are_the_default_and_uppercase_eulerseq_is_extrinsic() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler eulerseq="XYZ"/><worldbody>
<body name="b1" pos="0 0 0" euler="30 40 50"><joint name="j1" axis="0 0 1"/>
 <body name="b2" pos="1 0 0" axisangle="0 1 0 90"><joint name="j2" axis="1 0 0"/></body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let q = [20.0f64.to_radians(), -35.0f64.to_radians()];
        check_body(&t, &q, "b1", [0.0, 0.0, 0.0], [0.7942962447824292, 0.14941811936505986, 0.3820566077441136, 0.44810763172367496]);
        check_body(&t, &q, "b2", [0.3064645977401198, 0.8260327779131652, -0.47302145844036125], [0.2144953697710137, -0.2890851222668695, 0.666255947389569, 0.6530884633790229]);
    }

    #[test]
    fn xyaxes_and_zaxis_orientations() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="b1" pos="0 0 1" xyaxes="1 1 0 -1 1 0"><joint name="j1" axis="0 0 1"/>
 <body name="b2" pos="0.3 0 0" zaxis="1 0 1"><joint name="j2" axis="0 1 0"/></body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let q = [0.2, 0.3];
        check_body(&t, &q, "b1", [0.0, 0.0, 1.0], [0.8810593885167033, 0.0, 0.0, 0.47300565948683204]);
        check_body(&t, &q, "b2", [0.16575938765605625, 0.2500476462674449, 1.0], [0.7544668934405848, -0.24428336695438144, 0.4550223651597143, 0.40504319588903587]);
    }

    #[test]
    fn default_classes_resolve_element_class_then_childclass_then_main() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian" autolimits="true"/>
<default>
  <joint axis="0 1 0" pos="0 0 0.1" damping="1"/>
  <default class="arm"><joint axis="1 0 0" range="-1 1"/>
    <default class="wrist"><joint axis="0 0 1" pos="0.02 0 0"/></default>
  </default>
</default>
<worldbody>
<body name="b1" pos="0 0 0" childclass="arm"><joint name="j1"/>
 <body name="b2" pos="0.5 0 0"><joint name="j2" class="wrist"/>
  <body name="b3" pos="0.2 0 0" childclass="wrist"><joint name="j3"/><joint name="j4" class="arm"/></body>
 </body>
</body>
<body name="b0" pos="1 1 1"><joint name="j0"/></body>
</worldbody></mujoco>"#,
        )
        .unwrap();
        let q = [0.5, 0.6, 0.7, 0.8, 0.9];
        check_body(&t, &q, "b1", [0.0, 0.0479425538604203, 0.012241743810962727], [0.9689124217106447, 0.24740395925452294, 0.0, 0.0]);
        check_body(&t, &q, "b2", [0.5034932877018063, 0.038032146093337674, 0.006827663372438245], [0.9256373912272359, 0.23635402982999043, -0.07311286916773024, 0.2863331991006687]);
        check_body(&t, &q, "b3", [0.6105957130849323, 0.13243403914221866, 0.09295973888072757], [0.6337494143122936, 0.4817790927085964, 0.09043792627160364, 0.5983908147518627]);
        check_body(&t, &q, "b0", [0.9216673090372517, 1.0, 1.0378390031729336], [0.9004471023526769, 0.0, 0.43496553411123023, 0.0]);
        // the limits came through the class chain: j1 from arm, j2 from wrist's parent arm, j0 unlimited
        assert_eq!(t.tree.joints[0].limits, Some((-1.0, 1.0)));
        assert_eq!(t.tree.joints[1].limits, Some((-1.0, 1.0)));
        assert_eq!(t.tree.joints[4].limits, None);
        // and damping from main reached every joint
        assert!(t.tree.joints.iter().all(|j| j.damping == Some(1.0)));
    }

    #[test]
    fn a_free_root_is_placed_at_qpos_and_its_own_pose_is_ignored() {
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="fb" pos="1 2 3" euler="0.1 0.2 0.3"><freejoint name="root"/>
 <body name="leg" pos="0 0 -0.2"><joint name="hip" axis="0 1 0" pos="0 0 0.05"/></body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert_eq!(t.tree.dof(), 7);
        assert_eq!(t.joints[0].kind, MjcfJointKind::Free);
        let qpos = [0.5, -0.5, 2.0, 0.8, 0.0, 0.6, 0.0, 0.4];
        let q = t.q_from_qpos(&qpos, &[0, 7]).unwrap();
        check_body(&t, &q, "fb", [0.5, -0.5, 2.0], [0.8, 0.0, 0.6, 0.0]);
        check_body(&t, &q, "leg", [0.3063372154955404, -0.5, 1.9637972265147747], [0.6648516637959566, 0.0, 0.7469754113407939, 0.0]);
    }

    #[test]
    fn a_ball_joint_is_three_hinges_that_reproduce_the_quaternion() {
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="fb" pos="1 2 3"><freejoint/>
 <body name="leg" pos="0 0 -0.2"><joint name="hip" type="ball" pos="0 0 0.05"/></body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert_eq!(t.tree.dof(), 9);
        assert_eq!(t.joints[0].name, "joint0");
        let qpos = [0.5, -0.5, 2.0, 0.8, 0.0, 0.6, 0.0, 0.5, 0.5, 0.5, 0.5];
        let q = t.q_from_qpos(&qpos, &[0, 7]).unwrap();
        check_body(&t, &q, "leg", [0.34199999999999997, -0.5, 2.006], [0.10000000000000003, 0.7, 0.7, 0.10000000000000003]);
    }

    #[test]
    fn ref_shifts_the_zero_and_autolimits_false_refuses_an_unlabelled_range() {
        let xml = r#"<mujoco><compiler angle="radian" autolimits="false"/><worldbody>
<body name="b1"><joint name="j1" axis="0 0 1" ref="0.3" limited="true" range="-1 1"/>
 <body name="b2" pos="1 0 0"><joint name="j2" type="slide" axis="0 0 1" ref="0.2" limited="true" range="-1 1"/></body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let q = [0.5, 0.7];
        check_body(&t, &q, "b1", [0.0, 0.0, 0.0], [0.9950041652780258, 0.0, 0.0, 0.09983341664682815]);
        check_body(&t, &q, "b2", [0.9800665778412417, 0.19866933079506124, 0.49999999999999994], [0.9950041652780258, 0.0, 0.0, 0.09983341664682815]);
        let e = tree_from_mjcf_str(&xml.replace(r#"ref="0.3" limited="true" range"#, r#"ref="0.3" range"#)).unwrap_err();
        assert!(e.contains("autolimits"), "{e}");
    }

    #[test]
    fn a_frame_is_a_transform_and_a_jointless_body_welds_into_its_ancestor() {
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody>
<frame pos="0 0 0.5" euler="0 0 1.0">
 <body name="b1" pos="0.1 0 0"><joint name="j1" axis="0 0 1"/>
  <body name="static" pos="0 0.2 0" euler="0.5 0 0">
   <body name="b2" pos="0.3 0 0"><joint name="j2" axis="1 0 0"/></body>
  </body>
 </body>
</frame></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert_eq!(t.tree.dof(), 2);
        let q = [0.4, -0.6];
        check_body(&t, &q, "b1", [0.05403023058681398, 0.08414709848078966, 0.5], [0.7648421872844885, 0.0, 0.0, 0.644217687237691]);
        check_body(&t, &q, "static", [-0.14305971541087809, 0.11814052706083789, 0.5], [0.7410650959082802, 0.18922498533907178, 0.1593820064443967, 0.6241905194503019]);
        check_body(&t, &q, "b2", [-0.09206957254080578, 0.4137754460573759, 0.5], [0.7638863337114383, -0.03822617714364728, -0.03219746483761745, 0.6434125828796867]);
    }

    #[test]
    fn includes_are_textual_and_every_path_is_relative_to_the_main_model() {
        let files: BTreeMap<&str, &str> = [
            ("sub/defaults.xml", r#"<mujocoinclude><default><joint axis="0 1 0"/></default></mujocoinclude>"#),
            ("sub/arm.xml", r#"<mujocoinclude><body name="b1" pos="0 0 1"><joint name="j1"/><include file="sub/tip.xml"/></body></mujocoinclude>"#),
            ("sub/tip.xml", r#"<mujocoinclude><body name="b2" pos="0.4 0 0"><joint name="j2" axis="0 0 1"/></body></mujocoinclude>"#),
        ]
        .into_iter()
        .collect();
        let main = r#"<mujoco><compiler angle="radian"/><include file="sub/defaults.xml"/><worldbody>
<include file="sub/arm.xml"/></worldbody></mujoco>"#;
        let t = tree_from_mjcf(main, &|p| files.get(p).map(|s| s.as_bytes().to_vec())).unwrap();
        let q = [0.3, 0.6];
        check_body(&t, &q, "b1", [0.0, 0.0, 1.0], [0.9887710779360422, 0.0, 0.14943813247359922, 0.0]);
        check_body(&t, &q, "b2", [0.3821345956502424, 0.0, 0.8817919173354641], [0.9446090901443596, 0.04416198779168268, 0.14276370081881548, 0.29220183329241467]);
        // the same model without a resolver names the file it wanted
        let e = tree_from_mjcf_str(main).unwrap_err();
        assert!(e.contains("sub/defaults.xml"), "{e}");
    }

    #[test]
    fn a_serial_chain_agrees_with_the_serial_loader_to_machine_precision() {
        // the serial loader's own fixture shape: one chain, explicit inertials, joint offsets
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="l1" pos="0 0 0.1"><joint name="j1" axis="0 0 1" pos="0 0 0.02"/><inertial pos="0 0 0.05" mass="1" diaginertia="0.01 0.01 0.005"/>
 <body name="l2" pos="0.3 0 0" euler="0 0.2 0"><joint name="j2" axis="0 1 0"/><inertial pos="0.1 0 0" mass="0.5" diaginertia="0.002 0.003 0.003"/>
  <body name="l3" pos="0.25 0 0"><joint name="j3" type="slide" axis="1 0 0" pos="0.01 0 0"/><inertial pos="0.05 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
   <body name="tool" pos="0.1 0 0"/>
  </body>
 </body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let (serial, inertias) = crate::from_mjcf_full(xml).unwrap();
        assert_eq!(serial.dof(), t.tree.dof());
        for q in [[0.0, 0.0, 0.0], [0.3, -0.7, 0.12], [-1.2, 0.4, -0.05]] {
            let a = serial.fk(&q);
            let b = t.body_pose("tool", &q).unwrap();
            let p = Point3::new(0.0, 0.0, 0.0);
            assert!(((a * p) - (b * p)).norm() < 1e-14);
            assert!((a.rotation.angle_to(&b.rotation)) < 1e-14);
        }
        for (i, li) in inertias.iter().enumerate() {
            assert!((li.mass - t.tree.inertia[i].mass).abs() < 1e-15);
            assert!((li.com - t.tree.inertia[i].com).norm() < 1e-15);
            assert!((li.inertia - t.tree.inertia[i].inertia).norm() < 1e-15);
        }
    }

    #[test]
    fn refusals_are_loud() {
        let e = tree_from_mjcf_str(r#"<mujoco><worldbody><replicate count="3"><body name="b"><joint/></body></replicate></worldbody></mujoco>"#).unwrap_err();
        assert!(e.contains("replicate"), "{e}");
        let e = tree_from_mjcf_str(r#"<mujoco><worldbody><body name="a"><joint/><body name="b"><freejoint/></body></body></worldbody></mujoco>"#).unwrap_err();
        assert!(e.contains("free joint"), "{e}");
        let e = tree_from_mjcf_str(r#"<mujoco><worldbody><body name="a"><joint type="wheel"/></body></worldbody></mujoco>"#).unwrap_err();
        assert!(e.contains("wheel"), "{e}");
        let e = tree_from_mjcf_str(r#"<mujoco><worldbody><body name="a" childclass="nope"><joint/></body></worldbody></mujoco>"#).unwrap_err();
        assert!(e.contains("nope"), "{e}");
        let e = tree_from_mjcf_str(r#"<mujoco><default><default class="x"/><default class="x"/></default><worldbody/></mujoco>"#).unwrap_err();
        assert!(e.contains("twice"), "{e}");
        assert!(tree_from_mjcf_str(r#"<mujoco><worldbody><body name="a"><joint/><body name="a"/></body></worldbody></mujoco>"#).unwrap_err().contains("twice"));
    }

    #[test]
    fn a_body_without_inertial_is_weighed_from_its_geoms_as_mujoco_weighs_it() {
        let t = tree_from_mjcf_str(r#"<mujoco><worldbody><body name="a"><joint/><geom size="0.1"/><body name="b"><joint/><inertial mass="1" diaginertia="1 1 1"/></body></body></worldbody></mujoco>"#).unwrap();
        assert_eq!(t.no_inertial, vec!["a".to_string()]);
        assert_eq!(t.inferred_from_geoms, vec!["a".to_string()]);
        // a density-1000 sphere of radius 0.1: m = 4/3·π·0.001·1000, I = 2/5·m·r²
        let m = 4.0 / 3.0 * std::f64::consts::PI * 0.001 * 1000.0;
        assert!((t.tree.inertia[0].mass - m).abs() < 1e-12);
        assert!((t.tree.inertia[0].inertia[(0, 0)] - 0.4 * m * 0.01).abs() < 1e-12);
        assert_eq!(t.tree.inertia[1].mass, 1.0);
        // inertiafromgeom="false" keeps the massless behaviour and names the body
        let t = tree_from_mjcf_str(r#"<mujoco><compiler inertiafromgeom="false"/><worldbody><body name="a"><joint/><geom size="0.1"/></body></worldbody></mujoco>"#).unwrap();
        assert_eq!(t.tree.inertia[0].mass, 0.0);
        assert_eq!(t.no_inertial, vec!["a".to_string()]);
        assert!(t.inferred_from_geoms.is_empty());
    }

    /// Every number from MuJoCo 3.13.0 on this model: geoms of four kinds through two classes, one with a
    /// stated mass, one `fromto` capsule, one in a group outside `inertiagrouprange`, and an `<inertial>` that
    /// `inertiafromgeom="true"` overrides.
    #[test]
    fn geom_inference_matches_mujoco_on_primitives() {
        let xml = r#"<mujoco><compiler angle="radian" inertiafromgeom="true"/>
<default><geom density="500"/><default class="heavy"><geom density="2000" type="box"/></default></default>
<worldbody>
<body name="a" pos="0 0 1"><joint axis="0 0 1"/>
  <geom type="cylinder" size="0.05 0.1" pos="0.1 0 0" euler="0 1.2 0"/>
  <geom class="heavy" size="0.02 0.03 0.04" pos="0 0.2 0" quat="0.9 0.1 0.2 0.3"/>
  <geom type="capsule" size="0.03" fromto="0 0 0 0.3 0.1 0.2" mass="0.7"/>
  <geom type="ellipsoid" size="0.05 0.04 0.03" pos="-0.1 0 0" group="7"/>
  <inertial pos="1 1 1" mass="99" diaginertia="1 1 1"/>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let li = &t.tree.inertia[0];
        assert!((li.mass - 1.8693981633974484).abs() < 1e-12, "mass {}", li.mass);
        let com = Vector3::new(0.09818123283387589, 0.05980534387431644, 0.03744520636137881);
        assert!((li.com - com).norm() < 1e-12, "com {:?}", li.com);
        let full = Matrix3::new(
            0.020175673698348183, 0.0036390241227345524, -0.008632870168277268,
            0.0036390241227345532, 0.022840183602302153, -0.0007332479255163521,
            -0.008632870168277268, -0.0007332479255163513, 0.026398448280610037,
        );
        // MuJoCo hands back a principal frame from its Jacobi eigen-solver (1e-12 absolute stop): 1e-7 relative
        // is that solver's precision on this tensor, not this loader's
        assert!((li.inertia - full).norm() < 1e-7 * full.norm(), "inertia {:?}", li.inertia);
    }
}
