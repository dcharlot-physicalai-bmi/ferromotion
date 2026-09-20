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
//! * `<include file>` is textual insertion of the included root's children. A path is resolved against the
//!   **main** model's directory first and, failing that, against the directory of the file the `<include>` is
//!   written in — MuJoCo keeps both rules. The loader is string-based and WASM-clean, so the caller supplies
//!   the file contents through a resolver closure; [`tree_from_mjcf_str`] is the no-includes form.
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
use nalgebra::{Matrix3, Point3, Translation3, Unit, UnitQuaternion, Vector3};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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
    /// `range` when the joint is limited, in `qpos` units (radians for a hinge). NOT shifted by `ref`,
    /// because MuJoCo's range is in `qpos` and so is this tree's `q`.
    pub range: Option<(f64, f64)>,
    /// `actuatorfrcrange` when the joint limits it: the cap on the TOTAL actuator force this degree of
    /// freedom will pass, applied after every actuator driving it has been summed.
    pub actuator_force_range: Option<(f64, f64)>,
    /// ⛔ `damping` and `frictionloss` are NOT actuator terms in MuJoCo and are deliberately not folded into
    /// the tree's [`crate::Joint`]: damping is a PASSIVE force (`qfrc_passive`) and frictionloss is a
    /// CONSTRAINT row, neither of which belongs in `qfrc_bias`. Folding them in leaves the bias wrong the
    /// moment the model moves — 2.0e-1 on Menagerie — while every state sampled at rest agrees exactly.
    pub damping: f64,
    pub frictionloss: f64,
    /// `stiffness` and `springref`: the joint's spring, also a passive force. `springref` is in `qpos` units
    /// and defaults to `qpos0`, which is `q = 0` here.
    pub stiffness: f64,
    pub springref: f64,
    /// `margin`, `solreflimit` and `solimplimit`: what a limit row is built from.
    pub margin: f64,
    pub solref_limit: [f64; 2],
    pub solimp_limit: [f64; 5],
    /// `solreffriction` and `solimpfriction`: what a dof-friction row is built from.
    pub solref_friction: [f64; 2],
    pub solimp_friction: [f64; 5],
    /// `armature`, the constant this joint adds to the mass matrix's diagonal. ⚠ On a free or ball joint it
    /// matches MuJoCo only on the translations — see the note where the joints are built.
    pub armature: f64,
    /// A SLIDE joint whose anchor is at the body frame's origin and whose axis is a coordinate axis — one of
    /// the conditions for MuJoCo's "simple body" shortcut in [`MjcfTree::dof_invweight0`].
    pub aligned_slide: bool,
}

impl MjcfTree {
    /// **`qfrc_passive`**: the forces a MuJoCo model exerts without being asked — joint damping and joint
    /// springs.
    ///
    /// ⛔ These are NOT part of `qfrc_bias`. MuJoCo's bias is the rigid-body recursion alone; damping and
    /// springs are added separately, and a port that folds damping into the recursion's actuator terms is
    /// exactly right at rest and wrong the moment the model moves.
    ///
    /// ⚠ Short of a tendon's own spring and damper where the tendon is SPATIAL, and of the per-geom
    /// `fluidshape` model. `gravcomp` is included the way MuJoCo includes it (`d.qfrc_passive` carries
    /// `d.qfrc_gravcomp`), and so is the inertia-box fluid model — see [`MjcfTree::qfrc_fluid`].
    pub fn qfrc_passive(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        let mut out = self.qfrc_gravcomp(q);
        for (o, f) in out.iter_mut().zip(self.qfrc_fluid(q, qd)) {
            *o += f;
        }
        // a tendon's spring and damper, carried back through its constant Jacobian. ⛔ `springlength` is a
        // BAND: between its two values the tendon is slack and pulls nothing at all, which a single resting
        // length cannot express and a model that states two values is relying on.
        let ten = (!self.tendons.is_empty()).then(|| (self.ten_length(q), self.ten_moment(q)));
        for (i, t) in self.tendons.iter().enumerate() {
            if t.stiffness == 0.0 && t.damping == 0.0 {
                continue;
            }
            let (lengths, moments) = ten.as_ref().expect("built whenever there is a tendon");
            let (len, vel) = (lengths[i], moments[i].iter().map(|(d, c)| c * qd[*d]).sum::<f64>());
            let stretch = if len < t.springlength[0] {
                len - t.springlength[0]
            } else if len > t.springlength[1] {
                len - t.springlength[1]
            } else {
                0.0
            };
            let f = -t.stiffness * stretch - t.damping * vel;
            for (d, c) in &moments[i] {
                out[*d] += c * f;
            }
        }
        for j in &self.joints {
            if !matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) {
                continue;
            }
            let i = j.first;
            out[i] -= j.damping * qd[i];
            if j.stiffness != 0.0 {
                // `springref` is in `qpos`, and this tree's `q` is `qpos` because `ref` is folded into the
                // joint's origin — so the rest position is `springref` with no further shift
                out[i] -= j.stiffness * (q[i] - j.springref);
            }
        }
        out
    }

    /// **The change of basis a free or ball joint needs**: `v_mujoco = T·v_ours`, identity everywhere else.
    ///
    /// A free joint is six coordinates here and six there but not the same six. MuJoCo's are the body frame
    /// origin's world velocity and a BODY-FRAME angular velocity; this port's are three world slides and
    /// three Euler hinges with `R = Rz(ψ)·Ry(θ)·Rx(φ)`, so the Euler rates map to that angular velocity
    /// through the columns `[Rxᵀ Ryᵀ ẑ, Rxᵀ ŷ, x̂]`. A ball joint is the same three columns alone.
    ///
    /// ⛔ Verified the only way it can be: `M_ours = Tᵀ·M_mujoco·T` entry by entry, on 225 of 228 free-base
    /// states of Menagerie (`examples/menagerie_floating`).
    pub fn free_basis(&self, q: &[f64]) -> nalgebra::DMatrix<f64> {
        let nv = self.tree.joints.len();
        let mut t = nalgebra::DMatrix::identity(nv, nv);
        for j in &self.joints {
            let (base, rot) = match j.kind {
                MjcfJointKind::Free => (j.first, j.first + 3),
                MjcfJointKind::Ball => (j.first, j.first),
                _ => continue,
            };
            let _ = base;
            let (yaw, pitch, roll) = (q[rot], q[rot + 1], q[rot + 2]);
            let _ = yaw;
            let rx: Matrix3<f64> = *UnitQuaternion::from_euler_angles(roll, 0.0, 0.0).to_rotation_matrix().matrix();
            let ry: Matrix3<f64> = *UnitQuaternion::from_euler_angles(0.0, pitch, 0.0).to_rotation_matrix().matrix();
            let cols = [rx.transpose() * ry.transpose() * Vector3::z(), rx.transpose() * Vector3::y(), Vector3::x()];
            for (k, col) in cols.iter().enumerate() {
                for r in 0..3 {
                    t[(rot + r, rot + k)] = col[r];
                }
            }
        }
        t
    }

    /// **`Ṫ`, the time derivative of [`MjcfTree::free_basis`]** — the half of the change of basis that only
    /// exists when something is turning.
    ///
    /// `v_mujoco = T(q)·v_ours` differentiates to `a_mujoco = T·a_ours + Ṫ·v_ours`, and `Ṫ` depends on the
    /// PITCH and ROLL rates alone: the Euler-rate columns are
    /// `[(−sθ, sφcθ, cφcθ), (0, cφ, −sφ), (1, 0, 0)]`, and the yaw does not appear in any of them.
    ///
    /// ⛔ At rest this is identically zero, which is why a static sweep can compare a floating base without
    /// it and a moving one cannot.
    pub fn free_basis_dot(&self, q: &[f64], qd: &[f64]) -> nalgebra::DMatrix<f64> {
        let nv = self.tree.joints.len();
        let mut t = nalgebra::DMatrix::zeros(nv, nv);
        for j in &self.joints {
            let rot = match j.kind {
                MjcfJointKind::Free => j.first + 3,
                MjcfJointKind::Ball => j.first,
                _ => continue,
            };
            let (pitch, roll) = (q[rot + 1], q[rot + 2]);
            let (dpitch, droll) = (qd[rot + 1], qd[rot + 2]);
            let (st, ct, sp, cp) = (pitch.sin(), pitch.cos(), roll.sin(), roll.cos());
            // ∂E/∂θ·θ̇ + ∂E/∂φ·φ̇, column by column
            let de = [
                [-ct * dpitch, -sp * st * dpitch + cp * ct * droll, -cp * st * dpitch - sp * ct * droll],
                [0.0, -sp * droll, -cp * droll],
                [0.0, 0.0, 0.0],
            ];
            for (k, col) in de.iter().enumerate() {
                for (r, v) in col.iter().enumerate() {
                    t[(rot + r, rot + k)] = *v;
                }
            }
        }
        t
    }

    /// **The mass matrix, with armature where MuJoCo puts it.**
    ///
    /// ⛔⛔ `dof_armature` is a DIAGONAL in MuJoCo's basis, and a diagonal there is not one here. On a hinge
    /// or a slide the two coincide and nothing happens; on a free or a ball joint the correct contribution
    /// is `Tᵀ·diag(a)·T`, which is full. Adding `a` to this port's own diagonal instead is the error that
    /// made `rainbow_robotics_rby1`'s acceleration 8.2e-2 out — the one thing about a floating base this
    /// port could not previously match.
    pub fn mass_matrix(&self, q: &[f64]) -> nalgebra::DMatrix<f64> {
        let nv = self.tree.joints.len();
        let mut m = crate::tree_dynamics::tree_mass_matrix(&self.tree.joints, &self.tree.inertia, &self.tree.parent, q);
        let multi: Vec<(usize, usize, f64)> = self
            .joints
            .iter()
            .filter_map(|j| match j.kind {
                MjcfJointKind::Free => Some((j.first + 3, 3, j.armature)),
                MjcfJointKind::Ball => Some((j.first, 3, j.armature)),
                _ => None,
            })
            .filter(|(_, _, a)| *a != 0.0)
            .collect();
        if multi.is_empty() {
            return m;
        }
        let t = self.free_basis(q);
        for (first, n, a) in multi {
            for r in 0..n {
                m[(first + r, first + r)] -= a;
            }
            // Tᵀ diag(a) T restricted to those rows: the block is `a · Eᵀ E` with E the rotation columns
            for r in 0..n {
                for c in 0..n {
                    m[(first + r, first + c)] += a * (0..n).map(|k| t[(first + k, first + r)] * t[(first + k, first + c)]).sum::<f64>();
                }
            }
        }
        let _ = nv;
        m
    }

    /// **`qfrc_passive` in MUJOCO'S OWN COORDINATES.** A generalised force maps by `T⁻ᵀ`, except that a
    /// free or ball joint's DAMPING is not ours to map: MuJoCo applies `−dof_damping[i]·qvel[i]` per dof in
    /// its own basis, so it is added here and [`MjcfTree::qfrc_passive`] leaves it out.
    ///
    /// With no free or ball joint this is [`MjcfTree::qfrc_passive`] unchanged.
    pub fn qfrc_passive_mujoco(&self, q: &[f64], v_mujoco: &[f64]) -> Option<Vec<f64>> {
        let tinv = self.free_basis(q).try_inverse()?;
        let qd: Vec<f64> = (&tinv * nalgebra::DVector::from_row_slice(v_mujoco)).iter().copied().collect();
        let mut out = tinv.transpose() * nalgebra::DVector::from_row_slice(&self.qfrc_passive(q, &qd));
        for j in &self.joints {
            let n = match j.kind {
                MjcfJointKind::Free => 6,
                MjcfJointKind::Ball => 3,
                _ => continue,
            };
            if j.damping == 0.0 {
                continue;
            }
            for k in 0..n {
                out[j.first + k] -= j.damping * v_mujoco[j.first + k];
            }
        }
        Some(out.iter().copied().collect())
    }

    /// **`qacc_smooth` in MUJOCO'S OWN COORDINATES** — the unconstrained acceleration of a model with a free
    /// or ball joint, in the basis MuJoCo reports it in, from a velocity given in that basis.
    ///
    /// `q` is in this port's coordinates ([`MjcfTree::q_from_qpos`]); `v_mujoco` and the answer are in
    /// MuJoCo's. With no free or ball joint `T` is the identity and this is the ordinary computation.
    ///
    /// The derivation, because the velocity term is easy to drop: from `v_mujoco = T·v_ours` and
    /// `M_ours·a_ours + C_ours = τ_ours`,
    ///
    /// ```text
    /// M_mujoco = T⁻ᵀ·M_ours·T⁻¹              (plus the armature, which is a diagonal HERE)
    /// C_mujoco = T⁻ᵀ·(C_ours − M_rigid·T⁻¹·Ṫ·v_ours)
    /// ```
    ///
    /// ⛔ `M_rigid` in that correction is the mass matrix WITHOUT the free joint's armature: armature is a
    /// constant diagonal in MuJoCo's basis, so it contributes no velocity term there, however state-dependent
    /// it looks in ours.
    ///
    /// ⛔ Undefined at gimbal lock, where `T` is singular — the Euler base's one real limitation, and the
    /// reason this returns an `Option`.
    pub fn qacc_smooth_mujoco(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64]) -> Option<Vec<f64>> {
        let nv = self.tree.joints.len();
        let t = self.free_basis(q);
        let tinv = t.clone().try_inverse()?;
        let v = &tinv * nalgebra::DVector::from_row_slice(v_mujoco);
        let qd: Vec<f64> = v.iter().copied().collect();
        let m_full = self.mass_matrix(q);
        // the rigid-body half: `mass_matrix` differs from the tree's own only by where the armature sits
        let mut m_rigid = crate::tree_dynamics::tree_mass_matrix(&self.tree.joints, &self.tree.inertia, &self.tree.parent, q);
        for j in &self.joints {
            let (first, n) = match j.kind {
                MjcfJointKind::Free => (j.first + 3, 3),
                MjcfJointKind::Ball => (j.first, 3),
                _ => continue,
            };
            for r in 0..n {
                m_rigid[(first + r, first + r)] -= j.armature;
            }
        }
        let bias = crate::tree_dynamics::tree_inverse_dynamics(&self.tree.joints, &self.tree.inertia, &self.tree.parent, q, &qd, &vec![0.0; nv], self.gravity);
        let act = self.qfrc_actuator(q, &qd, ctrl);
        let tdot_v = self.free_basis_dot(q, &qd) * &v;
        let c_ours = nalgebra::DVector::from_iterator(nv, (0..nv).map(|i| bias[i])) - &m_rigid * (&tinv * tdot_v);
        let m_mj = tinv.transpose() * &m_full * &tinv;
        let passive = self.qfrc_passive_mujoco(q, v_mujoco)?;
        let rhs = nalgebra::DVector::from_row_slice(&passive) + tinv.transpose() * (nalgebra::DVector::from_iterator(nv, (0..nv).map(|i| act[i])) - c_ours);
        Some((m_mj.try_inverse()? * rhs).iter().copied().collect())
    }

    /// **`ten_length`** for every tendon. A FIXED tendon is `L = Σ coefₖ·qₖ`; a SPATIAL one is the length
    /// of the path through its sites, `Σ |pᵢ₊₁ − pᵢ|`.
    pub fn ten_length(&self, q: &[f64]) -> Vec<f64> {
        let frames = self.tendons.iter().any(|t| matches!(t.path, TendonPath::Spatial(_))).then(|| crate::tree_frames(&self.tree, q));
        self.tendons
            .iter()
            .map(|t| match &t.path {
                TendonPath::Fixed(j) => j.iter().map(|(d, c)| c * q[*d]).sum(),
                TendonPath::Spatial(pts) => {
                    let path = self.spatial_path(pts, frames.as_ref().expect("frames are built whenever a spatial tendon is present"));
                    path.windows(2).map(|w| w[0].arc.unwrap_or_else(|| (w[1].pos - w[0].pos).norm())).sum()
                }
            })
            .collect()
    }

    /// **`ten_velocity`**: `J·q̇` with the tendon's Jacobian at this state.
    pub fn ten_velocity(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        self.ten_moment(q).iter().map(|m| m.iter().map(|(d, c)| c * qd[*d]).sum()).collect()
    }

    /// **`ten_J`**: each tendon's moment at this state. Constant for a fixed tendon; for a spatial one it is
    /// `Σ ûᵢᵀ(Jᵢ₊₁ − Jᵢ)` over the path's segments, with `ûᵢ` the unit direction of segment `i`.
    pub fn ten_moment(&self, q: &[f64]) -> Vec<Vec<(usize, f64)>> {
        let nv = self.tree.joints.len();
        let frames = self.tendons.iter().any(|t| matches!(t.path, TendonPath::Spatial(_))).then(|| crate::tree_frames(&self.tree, q));
        self.tendons
            .iter()
            .map(|t| match &t.path {
                TendonPath::Fixed(j) => j.clone(),
                TendonPath::Spatial(pts) => {
                    let frames = frames.as_ref().expect("frames are built whenever a spatial tendon is present");
                    let path = self.spatial_path(pts, frames);
                    let jac: Vec<nalgebra::DMatrix<f64>> = path
                        .iter()
                        .map(|p| crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, frames, p.ride, p.pos))
                        .collect();
                    let mut out = vec![0.0; nv];
                    for i in 0..path.len().saturating_sub(1) {
                        if path[i].arc.is_some() {
                            continue;
                        }
                        let d = path[i + 1].pos - path[i].pos;
                        let n = d.norm();
                        if n < 1e-15 {
                            continue;
                        }
                        let u = d / n;
                        for k in 0..nv {
                            out[k] += (0..3).map(|r| u[r] * (jac[i + 1][(r, k)] - jac[i][(r, k)])).sum::<f64>();
                        }
                    }
                    out.iter().enumerate().filter(|(_, m)| **m != 0.0).map(|(k, m)| (k, *m)).collect()
                }
            })
            .collect()
    }

    /// **Where a spatial tendon actually touches the world at this state** — every point of the path, and
    /// whether the next one is the far side of the same obstacle. This is what MuJoCo publishes as
    /// `d.wrap_xpos`, and comparing against it is the only way to see WHICH wrap a disagreement is in.
    pub fn tendon_path_points(&self, i: usize, q: &[f64]) -> Vec<(Vector3<f64>, bool)> {
        let TendonPath::Spatial(pts) = &self.tendons[i].path else { return Vec::new() };
        let frames = crate::tree_frames(&self.tree, q);
        self.spatial_path(pts, &frames).iter().map(|p| (p.pos, p.arc.is_some())).collect()
    }

    /// The pose of a named site in the world, and the tree dof it rides on (`None` when welded to the world).
    fn site_ride(&self, name: &str, frames: &[Iso]) -> (Option<usize>, Iso) {
        match self.site_frames.get(name) {
            Some((ride, off)) => (Some(*ride), frames[*ride] * off),
            None => (None, self.world_fixed.get(&format!("site:{name}")).copied().unwrap_or_else(Iso::identity)),
        }
    }

    /// **A spatial tendon's path, resolved at this state**: every point the cable touches, the dof it rides
    /// on, and — for the pair of points an obstacle contributes — the ARC between them.
    ///
    /// ⛔ The arc is a length, not a segment: it adds to `ten_length` and contributes NOTHING to the moment.
    /// At a tangency the cable's direction is along the surface, so by the envelope theorem the sliding of
    /// the tangent point is first-order invisible and the derivative is the straight runs alone — which is
    /// what lets a wrapped tendon's Jacobian be written without differentiating the tangency condition.
    fn spatial_path(&self, pts: &[WrapPoint], frames: &[Iso]) -> Vec<PathPoint> {
        let mut out: Vec<PathPoint> = Vec::new();
        let mut pending: Option<&WrapPoint> = None;
        for p in pts {
            match p {
                WrapPoint::Geom { .. } => pending = Some(p),
                WrapPoint::Site(n) => {
                    let (ride, pose) = self.site_ride(n, frames);
                    let here = pose.translation.vector;
                    if let Some(WrapPoint::Geom { geom, sidesite }) = pending.take() {
                        let prev = out.last().expect("a geom is never first in the path").pos;
                        if let Some(g) = self.geoms.iter().find(|g| &g.name == geom) {
                            let gp = match g.joint {
                                Some(j) => frames[j] * g.pose,
                                None => g.pose,
                            };
                            let side = sidesite.as_ref().map(|s| self.site_ride(s, frames).1.translation.vector);
                            let radius = g.size[0];
                            let cyl = matches!(g.kind, crate::mujoco_collision::GeomType::Cylinder);
            // MuJoCo wraps a sphere or an infinite cylinder and nothing else
                            let wraps = matches!(g.kind, crate::mujoco_collision::GeomType::Sphere) || cyl;
                            if let Some(([w0, w1], arc)) = wraps.then(|| wrap_obstacle(prev, here, &gp, radius, cyl, side)).flatten() {
                                out.push(PathPoint { ride: g.joint, pos: w0, arc: Some(arc) });
                                out.push(PathPoint { ride: g.joint, pos: w1, arc: None });
                            }
                        }
                    }
                    out.push(PathPoint { ride, pos: here, arc: None });
                }
            }
        }
        out
    }

    /// **`tendon_invweight0`**: `J M⁻¹ Jᵀ` at `qpos0` for each fixed tendon's (constant) Jacobian row — the
    /// scale a tendon limit row regularises against, the tendon counterpart of `dof_invweight0`. Cached once
    /// by MuJoCo, like every other `*_invweight0`.
    pub fn tendon_invweight0(&self) -> Vec<f64> {
        let nv = self.tree.joints.len();
        let Some(chol) = self.mass_matrix(&self.reference_q).cholesky() else {
            return vec![0.0; self.tendons.len()];
        };
        self.ten_moment(&vec![0.0; nv])
            .iter()
            .map(|m| {
                let mut j = nalgebra::DVector::zeros(nv);
                for (d, c) in m {
                    j[*d] += c;
                }
                j.dot(&chol.solve(&j))
            })
            .collect()
    }

    /// **`actuator_moment` for every actuator**, at this state. A joint or fixed-tendon transmission carries
    /// its moment from the file; a SITE transmission's is a wrench in the site's frame and turns with the
    /// model, so it is rebuilt here: `mj_jacSite` at the site, and `gear` rotated into the world.
    pub fn actuator_state(&self, q: &[f64], qd: &[f64]) -> Vec<crate::mujoco_actuator::ActState> {
        use crate::mujoco_actuator::ActState;
        if self.actuators.iter().all(|a| a.dynamic.is_none()) {
            return self.actuators.iter().map(|a| ActState { moment: a.moment.clone(), length: a.length(q), velocity: a.velocity(qd) }).collect();
        }
        let frames = crate::tree_frames(&self.tree, q);
        // a spatial tendon's length is shared by every actuator that pulls on it, so build it once
        let ten = self.tendons.iter().any(|t| matches!(t.path, TendonPath::Spatial(_))).then(|| (self.ten_length(q), self.ten_moment(q)));
        self.actuators.iter().map(|a| self.act_state_of(a, &frames, ten.as_ref(), q, qd)).collect()
    }

    /// One actuator's transmission at a state — the per-actuator half of [`MjcfTree::actuator_state`], so a
    /// caller inside a loop does not pay for every other actuator.
    pub fn actuator_state_at(&self, a: &crate::mujoco_actuator::Actuator, q: &[f64], qd: &[f64]) -> crate::mujoco_actuator::ActState {
        use crate::mujoco_actuator::ActState;
        if a.dynamic.is_none() {
            return ActState { moment: a.moment.clone(), length: a.length(q), velocity: a.velocity(qd) };
        }
        let frames = crate::tree_frames(&self.tree, q);
        let ten = matches!(a.dynamic, Some(crate::mujoco_actuator::DynTransmission::SpatialTendon { .. })).then(|| (self.ten_length(q), self.ten_moment(q)));
        self.act_state_of(a, &frames, ten.as_ref(), q, qd)
    }

    #[allow(clippy::type_complexity)]
    fn act_state_of(&self, a: &crate::mujoco_actuator::Actuator, frames: &[Iso], ten: Option<&(Vec<f64>, Vec<Vec<(usize, f64)>>)>, q: &[f64], qd: &[f64]) -> crate::mujoco_actuator::ActState {
        use crate::mujoco_actuator::{ActState, DynTransmission};
        let nv = self.tree.joints.len();
        let vel = |m: &[(usize, f64)]| m.iter().map(|(d, c)| c * qd[*d]).sum::<f64>();
        let moment: Vec<(usize, f64)> = match &a.dynamic {
            None => return ActState { moment: a.moment.clone(), length: a.length(q), velocity: a.velocity(qd) },
            Some(DynTransmission::SpatialTendon { index }) => {
                let (lengths, moments) = ten.expect("tendon lengths are built whenever a spatial transmission is present");
                let m: Vec<(usize, f64)> = moments[*index].iter().map(|(d, c)| (*d, a.gear * c)).collect();
                let v = vel(&m);
                return ActState { moment: m, length: a.gear * lengths[*index], velocity: v };
            }
            Some(DynTransmission::Site { site, gear }) => {
                // a site welded to the world moves nothing, whatever wrench is applied to it
                let Some((ride, off)) = self.site_frames.get(site) else { return ActState::default() };
                let pose = frames[*ride] * off;
                let rot = pose.rotation.to_rotation_matrix().into_inner();
                let f = rot * Vector3::new(gear[0], gear[1], gear[2]);
                let t = rot * Vector3::new(gear[3], gear[4], gear[5]);
                let jp = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, frames, Some(*ride), pose.translation.vector);
                let ja = crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, frames, Some(*ride));
                (0..nv)
                    .filter_map(|k| {
                        let m: f64 = (0..3).map(|r| jp[(r, k)] * f[r] + ja[(r, k)] * t[r]).sum();
                        (m != 0.0).then_some((k, m))
                    })
                    .collect()
            }
        };
        // a site transmission has NO length, whatever the pose; its velocity is the ordinary moment·q̇
        let velocity = vel(&moment);
        ActState { moment, length: 0.0, velocity }
    }

    /// **`qfrc_actuator`** for this model at this state, with every transmission it carries.
    pub fn qfrc_actuator(&self, q: &[f64], qd: &[f64], ctrl: &[f64]) -> Vec<f64> {
        crate::mujoco_actuator::qfrc_actuator_with(&self.actuators, &self.actuator_state(q, qd), ctrl, &self.dof_actuator_force_range())
    }

    /// **`mj_inertiaBoxFluidModel`**: the ambient medium's drag on every body, from `<option density>`,
    /// `<option viscosity>` and `<option wind>`.
    ///
    /// Each body is replaced by the box with its mass and its inertia — full sides
    /// `bᵢ = √(6·(I_j + I_k − I_i)/m)` — and the four terms are written in the body's INERTIAL frame, on the
    /// velocity `mj_objectVelocity` reports there:
    ///
    /// - viscous torque `−π·d³·ν·ω` and viscous force `−3π·d·ν·v`, with `d` the mean of the three sides —
    ///   Stokes drag on the equivalent sphere;
    /// - blunt drag force `−½·ρ·b_j·b_k·|vᵢ|·vᵢ` on each face, and torque
    ///   `−ρ·bᵢ·(b_j⁴ + b_k⁴)·|ωᵢ|·ωᵢ / 64`.
    ///
    /// `wind` is subtracted from the LINEAR velocity only; it does not spin anything.
    ///
    /// ⛔ Every coefficient here was fitted against MuJoCo 3.13.0 one term at a time — angular viscous,
    /// linear viscous, blunt force, blunt torque, then all four together on a body with a rotated inertial
    /// frame and an offset centre of mass, where it agrees to 1.1e-16.
    ///
    /// ⚠ The per-geom `fluidshape="ellipsoid"` model is NOT this and is not carried. No Menagerie model
    /// uses it.
    pub fn qfrc_fluid(&self, q: &[f64], qd: &[f64]) -> Vec<f64> {
        let nv = self.tree.joints.len();
        let mut out = vec![0.0; nv];
        if self.density == 0.0 && self.viscosity == 0.0 {
            return out;
        }
        let frames = crate::tree_frames(&self.tree, q);
        for (name, (ride, pre)) in &self.body_frames {
            let mass = self.body_mass.get(name).copied().unwrap_or(0.0);
            let Some(inertia) = self.body_iinertia.get(name) else { continue };
            if mass <= 0.0 {
                continue;
            }
            let body = frames[*ride] * pre;
            let rot = body.rotation.to_rotation_matrix().into_inner() * self.body_iquat[name];
            let ipos = self.body_ipos.get(name).copied().unwrap_or_else(Vector3::zeros);
            let com = (body * Point3::from(ipos)).coords;
            let jp = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(*ride), com);
            let ja = crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(*ride));
            let vel = |j: &nalgebra::DMatrix<f64>| Vector3::new((0..nv).map(|k| j[(0, k)] * qd[k]).sum(), (0..nv).map(|k| j[(1, k)] * qd[k]).sum(), (0..nv).map(|k| j[(2, k)] * qd[k]).sum());
            let w = rot.transpose() * vel(&ja);
            let v = rot.transpose() * (vel(&jp) - self.wind);
            let b = [0, 1, 2].map(|i| (6.0 * (inertia[(i + 1) % 3] + inertia[(i + 2) % 3] - inertia[i]).max(1e-15) / mass).sqrt());
            let diam = (b[0] + b[1] + b[2]) / 3.0;
            let (mut force, mut torque) = (Vector3::zeros(), Vector3::zeros());
            if self.viscosity > 0.0 {
                torque -= w * (std::f64::consts::PI * diam * diam * diam * self.viscosity);
                force -= v * (3.0 * std::f64::consts::PI * diam * self.viscosity);
            }
            if self.density > 0.0 {
                for i in 0..3 {
                    force[i] -= 0.5 * self.density * b[(i + 1) % 3] * b[(i + 2) % 3] * v[i].abs() * v[i];
                    let (p, r) = (b[(i + 1) % 3], b[(i + 2) % 3]);
                    torque[i] -= self.density * b[i] * (p.powi(4) + r.powi(4)) * w[i].abs() * w[i] / 64.0;
                }
            }
            // `mj_applyFT` at the body's centre of mass, in the world
            let (fw, tw) = (rot * force, rot * torque);
            for k in 0..nv {
                out[k] += (0..3).map(|r| jp[(r, k)] * fw[r] + ja[(r, k)] * tw[r]).sum::<f64>();
            }
        }
        out
    }

    /// **`mj_gravcomp`**: `<body gravcomp>` as an upward force `−gravcomp·mass·g` applied at that body's
    /// centre of mass, mapped through the body's point Jacobian. `gravcomp = 1` cancels the body's weight
    /// exactly; MuJoCo allows more, which over-compensates.
    ///
    /// ⛔ It is applied per BODY, at the body's own centre of mass — so a jointless body welded onto a link
    /// contributes its own term at its own position, not the link's. That is what [`MjcfTree::body_ipos`]
    /// and [`MjcfTree::body_mass`] are for.
    ///
    /// ⛔ MuJoCo keeps this in its own field, `d.qfrc_gravcomp`, AND adds it into `d.qfrc_passive`. Nine
    /// Menagerie models set it, all of them arms whose controllers expect gravity to be handled for them.
    pub fn qfrc_gravcomp(&self, q: &[f64]) -> Vec<f64> {
        let nv = self.tree.joints.len();
        let mut out = vec![0.0; nv];
        if self.body_gravcomp.is_empty() {
            return out;
        }
        let frames = crate::tree_frames(&self.tree, q);
        for (name, &gc) in &self.body_gravcomp {
            // a body welded to the world has no dof to push against, and MuJoCo's force on it goes nowhere
            let Some((ride, pre)) = self.body_frames.get(name) else { continue };
            let mass = self.body_mass.get(name).copied().unwrap_or(0.0);
            if mass == 0.0 {
                continue;
            }
            let ipos = self.body_ipos.get(name).copied().unwrap_or_else(Vector3::zeros);
            let com = (frames[*ride] * pre * Point3::from(ipos)).coords;
            let jac = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(*ride), com);
            let f = self.gravity * (-gc * mass);
            for (i, o) in out.iter_mut().enumerate() {
                *o += jac.column(i).dot(&f);
            }
        }
        out
    }

    /// **MuJoCo's "simple body"** and its mass: a body whose inertial frame IS its body frame, which has no
    /// children, is a child of the world or of a dof-less child of the world, and whose every joint is a
    /// slide along a coordinate axis anchored at the origin.
    ///
    /// ⛔⛔ Such a body's inverse weights are NOT computed — MuJoCo writes `1/mass`, ignoring the armature
    /// and every coupling, and it does so for BOTH `dof_invweight0` and the translational half of
    /// `body_invweight0`. On `body_invweight0` the difference is a clean factor of THREE, because the value
    /// it replaces is a mean over the three world axes and a body on one slide moves along only one of them.
    /// It is not a fallback: it is the number MuJoCo ships.
    fn simple_bodies(&self) -> BTreeMap<&str, f64> {
        let mut dofs: BTreeMap<&str, Vec<&MjcfJoint>> = BTreeMap::new();
        for j in &self.joints {
            dofs.entry(j.body.as_str()).or_default().push(j);
        }
        let has_child = |b: &str| self.body_parent.values().any(|p| p == b);
        let mut out = BTreeMap::new();
        for (body, js) in &dofs {
            let parent = self.body_parent.get(*body).map(|s| s.as_str()).unwrap_or("world");
            let near_world = parent == "world" || (!dofs.contains_key(parent) && self.body_parent.get(parent).map(|s| s.as_str()).unwrap_or("world") == "world");
            let li = &self.tree.inertia[js[0].first];
            let frames_agree = li.com.norm() == 0.0 && (0..3).all(|r| (0..3).all(|c| r == c || li.inertia[(r, c)] == 0.0));
            if js.iter().all(|j| j.aligned_slide) && !has_child(body) && near_world && frames_agree && li.mass > 0.0 {
                out.insert(*body, li.mass);
            }
        }
        out
    }

    /// **`dof_invweight0`**: the diagonal of `J M⁻¹ Jᵀ` for each degree of freedom at `qpos0`, which for a
    /// single-dof joint is `(M⁻¹)ᵢᵢ`. It is the scale every joint constraint row regularises against, and
    /// MuJoCo computes it ONCE at the reference configuration — not at the current state — so a row's
    /// stiffness does not change as the model moves.
    pub fn dof_invweight0(&self) -> Vec<f64> {
        let nv = self.tree.joints.len();
        let mut out = self.dof_invweight0_general();
        // ⛔⛔ MuJoCo does NOT use `(M⁻¹)ᵢᵢ` for a "simple body with no rotations" — it uses `1/mass`, which
        // ignores the armature and every coupling. On `franka_emika_panda`'s fingers that is 66.67 against a
        // true 8.70, a factor of 7.7 in every constraint row on those dofs, and it is not an approximation
        // MuJoCo falls back to: it is the value it ships.
        //
        // A body qualifies when its inertial frame IS its body frame, it has no children, it is a child of
        // the world or of a dof-less child of the world, and every one of its joints is a slide along a
        // coordinate axis anchored at the origin.
        for (body, mass) in self.simple_bodies() {
            for j in self.joints.iter().filter(|j| j.body == body) {
                out[j.first] = 1.0 / mass;
            }
        }
        // ⛔⛔ a FREE or BALL joint's dofs share ONE inverse weight each way: MuJoCo averages `(M⁻¹)ᵢᵢ` over
        // the three translational dofs and over the three rotational ones, exactly as it does for
        // `body_invweight0`, so every row on that joint regularises against the same number. On
        // `unitree_go2` the three rotational diagonals are 12.43, 2.47 and 2.24 and MuJoCo ships 5.71 for
        // all three — their mean. Without this, `dof_invweight0` agrees with MuJoCo on NONE of Menagerie's
        // 76 free-base models.
        //
        // ⭐ The mean is what makes it basis-free: a trace is invariant under the orthogonal map between
        // this port's rotational dofs and MuJoCo's, and at the reference pose that map is a permutation.
        for j in &self.joints {
            let groups: &[(usize, usize)] = match j.kind {
                MjcfJointKind::Free => &[(0, 3), (3, 3)],
                MjcfJointKind::Ball => &[(0, 3)],
                _ => continue,
            };
            for (off, n) in groups {
                let mean = (0..*n).map(|k| out[j.first + off + k]).sum::<f64>() / *n as f64;
                for k in 0..*n {
                    out[j.first + off + k] = mean;
                }
            }
        }
        let _ = nv;
        out
    }

    /// **`dof_invweight0` in MUJOCO'S coordinates** — what every constraint row on a floating base actually
    /// regularises against.
    ///
    /// ⛔⛔ Even at the identity orientation `T` is NOT the identity: it is a PERMUTATION, because this
    /// port's rotational dofs are ordered (yaw, pitch, roll) and MuJoCo's are (ωx, ωy, ωz). So
    /// `diag(M⁻¹)` in one basis is not `diag(M⁻¹)` in the other, ever, and on Menagerie's free-base models
    /// the two agree on NONE of the 76 — worst a factor of 2.5. `M_mujoco⁻¹ = T·M_ours⁻¹·Tᵀ`, so the
    /// diagonal is `Σⱼₖ Tᵢⱼ (M_ours⁻¹)ⱼₖ Tᵢₖ` and the coupling is the whole of the difference.
    pub fn dof_invweight0_mujoco(&self) -> Vec<f64> {
        self.dof_invweight0()
    }

    /// `(M⁻¹)ᵢᵢ` at `qpos0` — the general case, before MuJoCo's simple-body shortcut.
    fn dof_invweight0_general(&self) -> Vec<f64> {
        let nv = self.tree.joints.len();
        // ⛔ `MjcfTree::mass_matrix`, not the tree's own: armature on a free or ball joint belongs on
        // MuJoCo's diagonal, and every inverse weight is read off the matrix that has it there — at
        // MuJoCo's `qpos0`, which is `reference_q` and not zero
        let m = self.mass_matrix(&self.reference_q);
        // ⛔ a CHOLESKY solve, not a general inverse: the mass matrix is symmetric positive definite, and
        // on an ill-conditioned one — a mobile base with light links — the general inverse loses digits that
        // show up directly in every constraint row's regularisation. `google_robot` moved by 5e-9 relative.
        match m.clone().cholesky() {
            Some(chol) => {
                let inv = chol.solve(&nalgebra::DMatrix::identity(nv, nv));
                (0..nv).map(|i| inv[(i, i)]).collect()
            }
            None => vec![0.0; nv],
        }
    }

    /// **`body_invweight0` for every body**, at `qpos0` — the contact half of the same rule `dof_invweight0`
    /// obeys for joint rows: MuJoCo evaluates these ONCE, when the model is compiled, and every contact's
    /// regularisation reads them for the rest of the simulation.
    ///
    /// ⛔ Evaluating them at the CURRENT state instead is the kind of mistake that hides: it is exact at
    /// `qpos0`, so a sweep that samples the reference pose sees nothing, and the error grows with how far the
    /// model has moved. It changes `efc_D` on every contact row, which changes the force the solver
    /// distributes, while leaving the contact's position, normal and frame untouched.
    ///
    /// ⛔⛔ Keyed by BODY, not by dof. A jointless body is welded into the link it rides on and shares every
    /// one of its dofs, but it has its own centre of mass, so `mj_jacBodyCom` gives it its own Jacobian and
    /// MuJoCo its own inverse weight. Four fingertips welded onto one link are four different numbers, and
    /// reading the link's instead gives every contact between them the same regularisation.
    ///
    /// A body welded to the world is absent from the map and takes [`InvWeight::STATIC`], which is what an
    /// immovable body means.
    pub fn body_invweight0(&self) -> BTreeMap<String, crate::mujoco_contact::InvWeight> {
        let nv = self.tree.joints.len();
        let zero = self.reference_q.clone();
        let _ = nv;
        let Some(chol) = self.mass_matrix(&zero).cholesky() else {
            return BTreeMap::new();
        };
        let minv = chol.solve(&nalgebra::DMatrix::identity(nv, nv));
        let frames = crate::tree_frames(&self.tree, &zero);
        let simple = self.simple_bodies();
        self.body_frames
            .iter()
            .map(|(name, (ride, pre))| {
                let ipos = self.body_ipos.get(name).copied().unwrap_or_else(Vector3::zeros);
                let com = (frames[*ride] * pre * Point3::from(ipos)).coords;
                let mut w = crate::tree_jacobian::body_invweight(&minv, &self.tree.joints, &self.tree.parent, &frames, Some(*ride), com);
                if let Some(mass) = simple.get(name.as_str()) {
                    w.tran = 1.0 / mass;
                }
                (name.clone(), w)
            })
            .collect()
    }

    /// **`mj_makeConstraint` for the rows a joint alone produces**: dof friction first, then limits, which is
    /// MuJoCo's own order and therefore part of the problem statement — `mj_constraintUpdate_impl` reads a
    /// row's law from its position. Contacts are appended by the caller AFTER these.
    ///
    /// ⛔ A limit produces a row per SIDE, and only when that side is within `margin`: `dist = side·(range −
    /// q)` for `side = ∓1`, with the row's Jacobian `−side`. Both sides can be active at once on a joint
    /// whose range is narrower than twice its margin, and a port that emits one row per joint is wrong there
    /// in a way no single-limit test can see.
    pub fn joint_constraint_rows(&self, q: &[f64], qd: &[f64], dof_invweight0: &[f64]) -> crate::mujoco_contact::AssembledRows {
        use crate::mujoco_contact::{row_reference, EfcBlock, SolImp, SolRef};
        let nv = self.tree.joints.len();
        let (mut blocks, mut rows, mut aref, mut d) = (Vec::new(), Vec::<Vec<f64>>::new(), Vec::new(), Vec::new());
        let solr = |v: [f64; 2]| SolRef(v[0], v[1]);
        let soli = |v: [f64; 5]| SolImp { d0: v[0], d_width: v[1], width: v[2], midpoint: v[3], power: v[4] };
        let mut push = |j: Vec<f64>, a: f64, r: f64, block: EfcBlock| {
            rows.push(j);
            aref.push(a);
            d.push(1.0 / r.max(1e-15));
            blocks.push(block);
        };
        // --- equality first: MuJoCo counts `ne` from the front, and a row's law is its position
        // a `connect` needs the CURRENT frames; a joint coupling does not, so only pay for them when asked
        let frames = self.equalities.iter().any(|e| matches!(e.kind, EqualityKind::Connect { .. })).then(|| crate::tree_frames(&self.tree, q));
        for e in &self.equalities {
            match &e.kind {
                EqualityKind::Joint { joint1, joint2, reference, polycoef: c } => {
                    let mut jac = vec![0.0; nv];
                    let (pos, diag) = match *joint2 {
                        Some(j2) => {
                            let y = q[j2] - reference.1;
                            // q₁ − ref₁ − (c₀ + c₁y + c₂y² + c₃y³ + c₄y⁴), and the Jacobian carries the derivative
                            let poly = c[0] + c[1] * y + c[2] * y * y + c[3] * y * y * y + c[4] * y * y * y * y;
                            let deriv = c[1] + 2.0 * c[2] * y + 3.0 * c[3] * y * y + 4.0 * c[4] * y * y * y;
                            jac[*joint1] += 1.0;
                            jac[j2] -= deriv;
                            (q[*joint1] - reference.0 - poly, dof_invweight0[*joint1] + dof_invweight0[j2])
                        }
                        None => {
                            jac[*joint1] = 1.0;
                            (q[*joint1] - reference.0 - c[0], dof_invweight0[*joint1])
                        }
                    };
                    let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                    // ⛔ an equality has NO margin: it is violated by however far it is from zero, both ways
                    let (a, r) = row_reference(&solr(e.solref), &soli(e.solimp), pos, 0.0, diag, jvel, self.timestep);
                    push(jac, a, r, EfcBlock::Equality { rows: 1 });
                }
                EqualityKind::Connect { side1, side2, diag_a } => {
                    let frames = frames.as_ref().expect("frames are built whenever a connect is present");
                    let at = |(ride, local): &(Option<usize>, Vector3<f64>)| -> (Vector3<f64>, nalgebra::DMatrix<f64>) {
                        let p = match ride {
                            Some(r) => (frames[*r] * Point3::from(*local)).coords,
                            None => *local,
                        };
                        (p, crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, frames, *ride, p))
                    };
                    let ((p1, j1), (p2, j2)) = (at(side1), at(side2));
                    // ⛔ `J̇·q̇`: a connect is NONLINEAR in q, so the residual's second derivative is
                    // `J·q̈ + J̇·q̇` and the reference has to carry the second term. Zero at rest, which is
                    // why a static sweep cannot see it missing.
                    let bias = |(ride, local): &(Option<usize>, Vector3<f64>), p: Vector3<f64>| match ride {
                        Some(_) => {
                            let _ = local;
                            crate::tree_jacobian::tree_point_bias_acceleration(&self.tree.joints, &self.tree.parent, frames, qd, *ride, p)
                        }
                        None => Vector3::zeros(),
                    };
                    let jdotqd = bias(side1, p1) - bias(side2, p2);
                    // one constraint, one impedance: taken at the NORM of the whole residual, not per row
                    let norm = -(p1 - p2).norm();
                    // three rows in WORLD axes; each body's Jacobian is taken at its OWN anchor, which is
                    // the same point only while the constraint is satisfied
                    for r in 0..3 {
                        let jac: Vec<f64> = (0..nv).map(|k| j1[(r, k)] - j2[(r, k)]).collect();
                        let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                        let (a, rr) = crate::mujoco_contact::row_reference_at(&solr(e.solref), &soli(e.solimp), norm, p1[r] - p2[r], 0.0, *diag_a, jvel, self.timestep);
                        push(jac, a - jdotqd[r], rr, EfcBlock::Equality { rows: 1 });
                    }
                }
            }
        }
        // --- dof friction, in dof order
        for j in &self.joints {
            if j.frictionloss <= 0.0 || !matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) {
                continue;
            }
            let i = j.first;
            let mut jac = vec![0.0; nv];
            jac[i] = 1.0;
            // a friction row has no position and no margin: it resists motion, not penetration
            let (a, r) = row_reference(&solr(j.solref_friction), &soli(j.solimp_friction), 0.0, 0.0, dof_invweight0[i], qd[i], self.timestep);
            push(jac, a, r, EfcBlock::Friction { loss: j.frictionloss });
        }
        // --- joint limits, in joint order, lower side then upper
        for j in &self.joints {
            let (Some((lo, hi)), true) = (j.range, matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide)) else {
                continue;
            };
            let i = j.first;
            for side in [-1.0f64, 1.0] {
                let dist = side * (if side < 0.0 { lo } else { hi } - q[i]);
                if dist >= j.margin {
                    continue;
                }
                let mut jac = vec![0.0; nv];
                jac[i] = -side;
                let (a, r) = row_reference(&solr(j.solref_limit), &soli(j.solimp_limit), dist, j.margin, dof_invweight0[i], -side * qd[i], self.timestep);
                push(jac, a, r, EfcBlock::Limit);
            }
        }
        // --- tendon limits, after every joint limit: MuJoCo's own order is limit_joint then limit_tendon
        if self.tendons.iter().any(|t| t.range.is_some()) {
            let tiw = self.tendon_invweight0();
            let len = self.ten_length(q);
            let mom = self.ten_moment(q);
            for (i, t) in self.tendons.iter().enumerate() {
                let Some((lo, hi)) = t.range else { continue };
                for side in [-1.0f64, 1.0] {
                    let dist = side * (if side < 0.0 { lo } else { hi } - len[i]);
                    if dist >= t.margin {
                        continue;
                    }
                    let mut jac = vec![0.0; nv];
                    for (d, c) in &mom[i] {
                        jac[*d] -= side * c;
                    }
                    let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                    let (a, r) = row_reference(&solr(t.solref_limit), &soli(t.solimp_limit), dist, t.margin, tiw[i], jvel, self.timestep);
                    push(jac, a, r, EfcBlock::Limit);
                }
            }
        }
        let jac = nalgebra::DMatrix::from_fn(rows.len(), nv, |r, c| rows[r][c]);
        crate::mujoco_contact::AssembledRows { blocks, jac, aref, d }
    }

    /// One entry per degree of freedom: the joint's cap on the total actuator force through it, for
    /// [`crate::qfrc_actuator`]. `None` where the joint does not limit it.
    pub fn dof_actuator_force_range(&self) -> Vec<Option<[f64; 2]>> {
        let mut out = vec![None; self.tree.joints.len()];
        for j in &self.joints {
            // MuJoCo clamps at the joint's `dofadr`, so a multi-dof joint caps only its first
            if let Some((lo, hi)) = j.actuator_force_range {
                out[j.first] = Some([lo, hi]);
            }
        }
        out
    }
}

/// **`<equality>`**, resolved. Only the `joint` kind is carried — 210 of Menagerie's 255 equality elements,
/// the coupled fingers and linkages — and `connect`, `weld` and `tendon` are left to
/// [`MjcfTree::equalities_unsupported`] rather than silently dropped.
#[derive(Clone, Debug)]
pub struct MjcfEquality {
    pub name: String,
    pub kind: EqualityKind,
    pub solref: [f64; 2],
    pub solimp: [f64; 5],
}

/// Which equality, and everything its rows need. ⛔ `<weld>` is absent on purpose: no Menagerie model uses
/// one, and its rotational residual is a quaternion difference this port has not measured against MuJoCo.
#[derive(Clone, Debug)]
pub enum EqualityKind {
    /// `<equality joint>`: a polynomial coupling between two single-dof joints. ONE row.
    Joint {
        /// The driven joint: the row constrains ITS value.
        joint1: usize,
        /// The driving joint, if any. With none, the constraint pins `joint1` to a constant.
        joint2: Option<usize>,
        /// Each joint's `ref`. MuJoCo writes the coupling in `qpos − qpos0`, so the references are part of it.
        reference: (f64, f64),
        /// `q₁ − ref₁ = c₀ + c₁·y + c₂·y² + c₃·y³ + c₄·y⁴` with `y = q₂ − ref₂`.
        polycoef: [f64; 5],
    },
    /// `<equality connect>`: a ball joint between two bodies at a shared point — the closed kinematic loop a
    /// tree cannot express, and what every four-bar gripper in Menagerie is built from. THREE rows, in world
    /// axes, whose residual is `p₁ − p₂` and whose Jacobian is each body's point Jacobian at ITS OWN anchor.
    ///
    /// ⛔ MJCF states the anchor in body1's frame only. MuJoCo's compiler resolves it to a world point at
    /// `qpos0` and stores body2's local copy; a port that uses body1's anchor for both bodies is exactly
    /// right at `qpos0` and wrong everywhere else. Each side here is `(the tree dof it rides, the anchor in
    /// THAT frame)`, with `None` for a body welded to the world.
    Connect {
        side1: (Option<usize>, Vector3<f64>),
        side2: (Option<usize>, Vector3<f64>),
        /// `mj_diagApprox`: the two bodies' translational `body_invweight0`, summed, cached at load.
        diag_a: f64,
    },
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
    /// Each body's own centre of mass in its OWN frame — MuJoCo's `body_ipos`. The tree merges a jointless
    /// body into the ancestor it rides on, so after loading there is no other way back to it, and
    /// [`MjcfTree::body_invweight0`] needs exactly this: two geoms welded onto the same link belong to
    /// DIFFERENT bodies and get different inverse weights.
    pub body_ipos: BTreeMap<String, Vector3<f64>>,
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
    /// `<option timestep>`, default 0.002. Needed where a constraint row is built, because MuJoCo floors a
    /// reference time constant at two timesteps.
    pub timestep: f64,
    /// `<option gravity>`, default `0 0 -9.81`.
    pub gravity: Vector3<f64>,
    /// **MuJoCo's `qpos0`, in THIS port's coordinates.** ⛔⛔ It is NOT `q = 0`. A joint's `ref` is folded
    /// into its origin here, so `q` IS `qpos` and the reference configuration is the vector of `ref` values;
    /// a free body's reference pose is the one its `<body>` element states, which the free joint carries.
    /// Everything MuJoCo evaluates "once, at `qpos0`" — every `*_invweight0`, a `dampratio`, a `connect`
    /// anchor's second copy, a tendon's automatic resting length — has to be evaluated HERE.
    /// `agility_cassie`'s knee and tarsus have `ref` of −0.785 and 1.012, and resolving its plantar rod's
    /// anchor at `q = 0` instead put the two ends of a closed loop 41 cm apart.
    pub reference_q: Vec<f64>,
    /// The pose a free-jointed body's `<body>` element states — the free joint carries it, so it is not in
    /// the tree's own geometry, and [`MjcfTree::reference_q`] is where it goes.
    pub free_base_pose: BTreeMap<String, Iso>,
    /// `<option density>`, `<option viscosity>` and `<option wind>` — the ambient medium. With both density
    /// and viscosity zero there is no fluid force at all, which is MuJoCo's default.
    pub density: f64,
    pub viscosity: f64,
    pub wind: Vector3<f64>,
    /// Each body's PRINCIPAL inertia and the orientation of its inertial frame in its body frame —
    /// MuJoCo's `body_inertia` and `body_iquat`. The fluid model is written in that frame and nowhere else.
    pub body_iinertia: BTreeMap<String, [f64; 3]>,
    pub body_iquat: BTreeMap<String, Matrix3<f64>>,
    /// Each body's mass after MuJoCo's `boundmass` floor, and its `<body gravcomp>` where it is non-zero —
    /// together with [`MjcfTree::body_ipos`] these are what [`MjcfTree::qfrc_gravcomp`] needs.
    pub body_mass: BTreeMap<String, f64>,
    pub body_gravcomp: BTreeMap<String, f64>,
    /// Each body's MJCF parent body (`world` for the children of the worldbody) — what MuJoCo's contact
    /// filtering reads through `body_parentid` and `body_weldid`.
    pub body_parent: BTreeMap<String, String>,
    /// Hull data for every mesh a collidable mesh geom (`contype` or `conaffinity` non-zero) references, as
    /// MuJoCo stores the mesh: `f32` vertices centred at the mesh's CoM in its principal frame. A mesh geom's
    /// `geom_xmat` already includes that frame, so these vertices go straight under the geom's world pose.
    pub mesh_hulls: BTreeMap<String, crate::mujoco_hull::MeshHull>,
    /// The same meshes before centring and rotation: after `scale`/`refpos`/`refquat`, `f32` vertices, in
    /// the mesh file's own frame — for placing them under a frame chosen elsewhere (MuJoCo's `mesh_pos`,
    /// `mesh_quat` when comparing against it).
    pub mesh_raw: BTreeMap<String, crate::TriMesh3>,
    /// Every geom, in file order — the order MuJoCo numbers them in.
    pub geoms: Vec<MjcfGeom>,
    /// `<actuator>` entries, resolved into MuJoCo's gain/bias form, in file order — the same order as
    /// `ctrl`. Only stateless joint transmissions are carried; the loader refuses the rest.
    pub actuators: Vec<crate::mujoco_actuator::Actuator>,
    /// `<equality>` entries the loader carries, in file order — which is the order their rows appear in.
    pub equalities: Vec<MjcfEquality>,
    /// `<equality>` entries it does not, as `(name, why)`: `connect`, `weld`, `tendon` and `flex`.
    pub equalities_unsupported: Vec<(String, String)>,
    /// Actuators the loader could not carry, as `(name, why)` — a tendon or site transmission, a muscle, an
    /// activation state. The model still loads; `qfrc_actuator` is simply short of these terms, and this
    /// list is how a caller finds that out instead of wondering.
    pub actuators_unsupported: Vec<(String, String)>,
    /// `<contact><exclude body1 body2/>`, in file order.
    pub contact_excludes: Vec<(String, String)>,
    /// `<contact><pair/>` entries with their parameters resolved (defaults classes applied).
    pub contact_pairs: Vec<MjcfContactPair>,
    /// `<tendon><fixed>` entries in file order — which is the order their limit rows appear in.
    pub tendons: Vec<MjcfTendon>,
    /// `<tendon><spatial>` and anything else in a `<tendon>` block, as `(name, why)`. A model still loads;
    /// what it loses is that tendon's length, its limit and any actuator driving it.
    pub tendons_unsupported: Vec<(String, String)>,
}

/// **`<tendon><fixed>`**: a tendon whose length is a fixed linear combination of single-dof joint positions,
/// `L = Σ coefₖ·qₖ`, with a constant Jacobian. MuJoCo uses it for the coupled linkages that a real gripper
/// has and a kinematic tree does not — two fingers driven by one motor, a telescoping stage, a differential.
///
/// A `<spatial>` tendon is a different object: its length is a PATH, and it moves with the model, so its
/// Jacobian is not a constant. ⛔ Only a path of sites is carried; a wrapping geom or a pulley is named in
/// [`MjcfTree::tendons_unsupported`] instead.
#[derive(Clone, Debug)]
pub enum TendonPath {
    /// `<fixed>`: `(tree dof, coefficient)` in file order. The Jacobian row IS these coefficients.
    Fixed(Vec<(usize, f64)>),
    /// `<spatial>`: the path, in order. `L = Σ |pᵢ₊₁ − pᵢ|` over the straight runs plus each obstacle's
    /// arc, and the moment is `Σ ûᵢᵀ(Jᵢ₊₁ − Jᵢ)` over the straight runs alone — a function of the state,
    /// not of the file.
    Spatial(Vec<WrapPoint>),
}

/// One element of a `<spatial>` tendon's path.
#[derive(Clone, Debug)]
pub enum WrapPoint {
    Site(String),
    /// A sphere or an infinite cylinder the cable bends around. ⛔ `sidesite` picks WHICH WAY round, and
    /// 240 of `ms_human_700`'s 252 wrap geoms give one — without it the shorter way is taken, which for a
    /// muscle routed deliberately the long way round a bone is the wrong side entirely.
    Geom {
        geom: String,
        sidesite: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct MjcfTendon {
    pub name: String,
    pub path: TendonPath,
    /// `range` when `limited`, in the tendon's own length units — NOT scaled by `<compiler angle>`, because
    /// a tendon length is a length even when every joint it names is a hinge.
    pub range: Option<(f64, f64)>,
    pub margin: f64,
    pub solref_limit: [f64; 2],
    pub solimp_limit: [f64; 5],
    pub stiffness: f64,
    /// `springlength`, MuJoCo's resting BAND: below `[0]` and above `[1]` the spring pulls back, between
    /// them it is slack. A single stated value fills both, and the default `-1 -1` means "the length at
    /// `qpos0`", resolved at load.
    pub springlength: [f64; 2],
    pub damping: f64,
    pub frictionloss: f64,
    pub solref_friction: [f64; 2],
    pub solimp_friction: [f64; 5],
}

/// **One geom as the collision pipeline needs it**: where it is, what shape, and the contact parameters
/// MJCF gave it. `joint` is the tree joint it rides on (`None` for a geom welded to the world) and `pose` is
/// its transform from that joint's frame, so its world pose is `frames[joint] * pose`.
#[derive(Clone, Debug)]
pub struct MjcfGeom {
    /// The geom's `name`, or `geom{i}` with `i` its index in [`MjcfTree::geoms`] — which is MuJoCo's own
    /// geom id, since the vector is in MuJoCo's body-major order, so an unnamed geom still lines up.
    pub name: String,
    /// The body it belongs to.
    pub body: String,
    pub joint: Option<usize>,
    pub pose: Iso,
    pub kind: crate::mujoco_collision::GeomType,
    /// MuJoCo's resolved `geom_size`, after `fromto` and after fitting to a mesh.
    pub size: [f64; 3],
    /// The mesh it references, for [`MjcfTree::mesh_hulls`]; set for a `type="mesh"` geom, and also for a
    /// primitive geom fitted to a mesh, where the SHAPE is the primitive and this is only provenance.
    pub mesh: Option<String>,
    pub params: crate::mujoco_collision::GeomParams,
}

/// An explicit `<contact><pair>`: the two geoms and the contact parameters MuJoCo uses for that pair
/// instead of mixing the geoms' own (`mj_collideGeomPair`).
#[derive(Clone, Debug, PartialEq)]
pub struct MjcfContactPair {
    pub geom1: String,
    pub geom2: String,
    pub condim: usize,
    /// `(slide1, slide2, spin, roll1, roll2)`
    pub friction: [f64; 5],
    pub solref: [f64; 2],
    pub solreffriction: [f64; 2],
    pub solimp: [f64; 5],
    pub margin: f64,
    pub gap: f64,
    pub adhesion: f64,
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

/// ⛔ An include path is tried against the MAIN model's directory FIRST and, only if that fails, against the
/// directory of the file the `<include>` is written in — `IncludeXML` calls the first "legacy behavior" and
/// the second "new behavior", and keeps both. The subtree then inherits the resolved file's own directory,
/// so the rule compounds: `ms_human_700` reaches `assets/body_primary/Body_Torso_Simple.xml` from the model
/// root and that file says `<include file="Body_Arm_r.xml"/>`, meaning its own neighbour. Five of Menagerie's
/// models are written this way and a model-relative-only loader cannot open any of them.
/// Record which file's directory an element arrived from, without overwriting a deeper include's answer.
fn stamp_include_dir(el: &mut El, dir: &str) {
    if el.attr(INCLUDE_DIR).is_none() {
        el.attrs.push((INCLUDE_DIR.to_string(), dir.to_string()));
    }
    for c in el.children.iter_mut() {
        stamp_include_dir(c, dir);
    }
}

/// The `<actuator>` children MuJoCo accepts. The tag chooses the force law's SHAPE — which gain and bias
/// type — and the attributes only fill it in, so the tag has to be read even though the defaults are shared.
const ACTUATOR_TAGS: [&str; 12] = ["general", "motor", "position", "velocity", "damper", "intvelocity", "cylinder", "muscle", "adhesion", "pid", "dcmotor", "orientation"];

/// The attribute `stamp_include_dir` writes. Not an MJCF attribute — the leading underscores keep it out of
/// the way of anything a model can say.
const INCLUDE_DIR: &str = "__include_dir";

fn expand_includes(el: &mut El, resolve: &dyn Fn(&str) -> Option<Vec<u8>>, dir: &str, depth: usize, included: &mut std::collections::HashSet<String>) -> Result<(), String> {
    // ⛔ `depth` counts INCLUDE nesting, not element nesting. Counting elements caps how deep a model's
    // bodies may go: `ms_human_700` is 40-odd bodies deep and was refused as "a cycle".
    if depth > 32 {
        return Err("<include> nesting deeper than 32 — a cycle".into());
    }
    let mut out: Vec<El> = Vec::with_capacity(el.children.len());
    for child in std::mem::take(&mut el.children) {
        if child.name == "include" {
            let file = child.attr("file").ok_or("<include> needs a file attribute")?;
            // MuJoCo refuses a file included twice, testing the path AS WRITTEN against a set it fills with
            // the path AS RESOLVED — the asymmetry is its own, and is kept here so the refusals agree
            if included.contains(file) {
                return Err(format!("file '{file}' is already included"));
            }
            let nested = format!("{dir}{file}");
            let (path, bytes) = match resolve(file) {
                Some(b) => (file.to_string(), b),
                None if !file.starts_with('/') && !dir.is_empty() => {
                    let b = resolve(&nested).ok_or_else(|| format!("<include file=\"{file}\"> could not be resolved, from the model root or from '{dir}'"))?;
                    (nested, b)
                }
                None => return Err(format!("<include file=\"{file}\"> could not be resolved")),
            };
            let next = match path.rfind('/') {
                Some(i) => path[..=i].to_string(),
                None => String::new(),
            };
            let text = String::from_utf8_lossy(&bytes);
            let mut inc = parse_xml(&text).map_err(|e| format!("in included file '{file}': {e}"))?;
            included.insert(path);
            expand_includes(&mut inc, resolve, &next, depth + 1, included)?;
            // ⛔ An asset's `file` is resolved against `meshdir` FIRST and, failing that, against the
            // directory of the file that declares it — `ResolveFilePath` walks up to the nearest `<include>`
            // and reads the `dir` MuJoCo stamped on it. Splicing the subtree in loses that ancestry, so the
            // directory is stamped onto the elements themselves, innermost include first.
            for c in inc.children.iter_mut() {
                stamp_include_dir(c, &next);
            }
            out.extend(inc.children);
        } else {
            let mut child = child;
            expand_includes(&mut child, resolve, dir, depth, included)?;
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
pub(crate) enum MeshInertia {
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

/// What [`Walk::geom_spec`] resolves: MuJoCo's own `geom_type`, `geom_size` and pose for one geom.
struct GeomSpec {
    ty: String,
    size: Vec<f64>,
    pose: Iso,
    mesh: Option<String>,
    /// For a `type="mesh"` geom, the mesh's OWN stored frame — the centre of mass and the principal rotation
    /// `mjCMesh::Process` left it in — which MuJoCo folds into `geom_xmat`. Kept apart from `pose` because
    /// [`Walk::geom_mass`] applies the same frame itself, through the inertia, and would double-count it.
    mesh_frame: Option<Iso>,
}

struct MeshAsset {
    file: String,
    /// `<mesh maxhullvert>`: the collision hull is capped at this many vertices (MuJoCo's qhull `Q9 TA`).
    /// `None` is MJCF's `-1`, an uncapped hull. Three Menagerie families set it to 64.
    maxhullvert: Option<usize>,
    /// The directory of the XML file that declared this mesh, if it arrived through an `<include>`; the
    /// fallback `ResolveFilePath` uses when `meshdir/file` does not exist.
    include_dir: String,
    scale: Vector3<f64>,
    refpos: Vector3<f64>,
    refquat: UnitQuaternion<f64>,
    inertia: MeshInertia,
}

/// Volume (or area, for a shell), centre of mass and inertia per unit density about that centre, computed
/// exactly as `mjCMesh::Process` does: face centroid → one pass for the CoM → one pass for the second moments
/// about it. Both passes take pyramids from the apex to each face; `legacy` takes their volumes absolute.
pub(crate) fn mesh_inertia_mujoco(mesh: &crate::TriMesh3, method: MeshInertia) -> Result<(f64, Vector3<f64>, Matrix3<f64>), String> {
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
pub(crate) fn eig3_mujoco(mat: &Matrix3<f64>) -> ([f64; 3], [f64; 4]) {
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
pub(crate) fn quat_to_rotation(q: &[f64; 4]) -> Matrix3<f64> {
    *UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(q[0], q[1], q[2], q[3])).to_rotation_matrix().matrix()
}

/// A mesh as MuJoCo's OBJ reader (tinyobjloader with `real_t = float`) delivers it: coordinates rounded to
/// `f32`, quads split along their **shorter diagonal** (measured in `f32`), larger polygons refused because
/// tinyobj's ear clipping is not reproduced here.
/// **tinyobjloader's built-in ear clipping, ported branch for branch** — how a face with more than four
/// vertices becomes triangles, which is what MuJoCo gets because its OBJ decoder builds tinyobj without
/// `TINYOBJLOADER_USE_MAPBOX_EARCUT`. Three Menagerie families (`arx_l5`, `hello_robot_stretch_3`,
/// `trossen_wxai`) have such faces and were refused outright until this existed.
///
/// Measured against MuJoCo's own `mesh_face` on `hello_robot_stretch_3`, the only Menagerie model with faces
/// this large: the 49-, 72-, 73- and 85-gons come out **triangle for triangle identical**, all 271 of them.
/// The 215-gon does not — MuJoCo emits NOTHING for it (its face count is short by exactly that polygon's
/// 213) while this emits 201, none of them MuJoCo's. That polygon is degenerate at the `f32` noise floor:
/// its corner cross products run 1e-10 to 1e-9 against an `f32::EPSILON` of 1.19e-7, so which corner first
/// clears the threshold, and every ear test after it, is decided in the last bits. It moves four geoms of
/// one model by 1.29e-4 m and nothing else in the corpus. Fused multiply-add in `cross`/`area` is not the
/// cause — tried, no change.
///
/// It is not a textbook ear clip and the differences are the whole point of porting it rather than writing
/// one: the working plane is chosen from the FIRST corner of the polygon whose cross product clears
/// `f32::EPSILON`, not from a fitted normal; convexity is judged by `cross * area < 0` where `area` is one
/// term of the shoelace sum and not the polygon's area; the ear test is `pnpoly` on the other vertices; and
/// when no ear is found for a full cycle of `guess_vert` the remaining vertices are silently DROPPED. All of
/// it runs in `f32`, because tinyobj's `real_t` is `float`.
fn triangulate_as_tinyobj(poly: &[usize], verts: &[Vector3<f64>]) -> Vec<[usize; 3]> {
    let vf = |i: usize| -> [f32; 3] {
        let v = verts[i];
        [v.x as f32, v.y as f32, v.z as f32]
    };
    // the two axes to work in: the first corner that is not degenerate picks the plane its normal is most
    // aligned with, so the polygon is projected along its largest component
    let n = poly.len();
    let mut axes = [1usize, 2usize];
    for k in 0..n {
        let (a, b, c) = (vf(poly[k % n]), vf(poly[(k + 1) % n]), vf(poly[(k + 2) % n]));
        let e0 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let e1 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let cx = (e0[1] * e1[2] - e0[2] * e1[1]).abs();
        let cy = (e0[2] * e1[0] - e0[0] * e1[2]).abs();
        let cz = (e0[0] * e1[1] - e0[1] * e1[0]).abs();
        if cx > f32::EPSILON || cy > f32::EPSILON || cz > f32::EPSILON {
            if !(cx > cy && cx > cz) {
                axes[0] = 0;
                if cz > cx && cz > cy {
                    axes[1] = 1;
                }
            }
            break;
        }
    }
    // the crossing-number point-in-triangle test tinyobj carries verbatim from pnpoly
    let pnpoly = |vx: &[f32; 3], vy: &[f32; 3], tx: f32, ty: f32| -> bool {
        let mut c = false;
        let mut j = 2usize;
        for i in 0..3 {
            if ((vy[i] > ty) != (vy[j] > ty)) && (tx < (vx[j] - vx[i]) * (ty - vy[i]) / (vy[j] - vy[i]) + vx[i]) {
                c = !c;
            }
            j = i;
        }
        c
    };
    let mut out: Vec<[usize; 3]> = Vec::new();
    let mut rem: Vec<usize> = poly.to_vec();
    let mut guess_vert = 0usize;
    // how many turns are left before giving up: reset whenever an ear is actually taken
    let mut remaining_iterations = poly.len();
    let mut previous = rem.len();
    while rem.len() > 3 && remaining_iterations > 0 {
        let m = rem.len();
        if guess_vert >= m {
            guess_vert -= m;
        }
        if previous != m {
            previous = m;
            remaining_iterations = m;
        } else {
            remaining_iterations -= 1;
        }
        let mut ind = [0usize; 3];
        let (mut vx, mut vy) = ([0f32; 3], [0f32; 3]);
        for k in 0..3 {
            ind[k] = rem[(guess_vert + k) % m];
            let p = vf(ind[k]);
            vx[k] = p[axes[0]];
            vy[k] = p[axes[1]];
        }
        let (e0x, e0y) = (vx[1] - vx[0], vy[1] - vy[0]);
        let (e1x, e1y) = (vx[2] - vx[1], vy[2] - vy[1]);
        let cross = e0x * e1y - e0y * e1x;
        let area = (vx[0] * vy[1] - vy[0] * vx[1]) * 0.5;
        if cross * area < 0.0 {
            guess_vert += 1;
            continue;
        }
        let mut overlap = false;
        for other in 3..m {
            let p = vf(rem[(guess_vert + other) % m]);
            if pnpoly(&vx, &vy, p[axes[0]], p[axes[1]]) {
                overlap = true;
                break;
            }
        }
        if overlap {
            guess_vert += 1;
            continue;
        }
        out.push(ind);
        rem.remove((guess_vert + 1) % m);
    }
    if rem.len() == 3 {
        out.push([rem[0], rem[1], rem[2]]);
    }
    out
}

pub(crate) fn obj_as_mujoco_reads_it(text: &str) -> Result<crate::TriMesh3, String> {
    let mut verts: Vec<Vector3<f64>> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    // ⛔ MuJoCo reads the faces of the FIRST SHAPE ONLY: `obj_decoder.cc` takes `GetShapes()[0]` and drops
    // the rest, while tinyobj starts a new shape at each `o`/`g` tag that follows a face. Its vertex array,
    // by contrast, is the whole file's, so the later objects' vertices stay — and they still count towards
    // the mesh's bounds only through faces that no longer exist. Menagerie's `distal.obj` is one object of
    // 9,802 faces followed by a 1,450-face tip: reading both moved the mesh's centre of mass 7.9 mm and
    // turned its principal frame 90°, which is 9.6 mm of geom placement and a permuted `geom_size`.
    let mut first_shape_done = false;
    for line in text.lines() {
        let mut tok = line.split_whitespace();
        match tok.next() {
            Some("o") | Some("g") if !tris.is_empty() => first_shape_done = true,
            Some("v") => {
                let mut c = [0.0f64; 3];
                for x in c.iter_mut() {
                    let t = tok.next().ok_or("OBJ vertex with fewer than 3 coordinates")?;
                    *x = t.parse::<f32>().map_err(|e| format!("OBJ vertex '{t}': {e}"))? as f64;
                }
                verts.push(Vector3::new(c[0], c[1], c[2]));
            }
            Some("f") if !first_shape_done => {
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
                    _ => tris.extend(triangulate_as_tinyobj(&poly, &verts)),
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

/// One point a spatial tendon touches: where it is, which dof carries it, and — when the next point is the
/// far side of the same obstacle — the arc length between the two.
struct PathPoint {
    ride: Option<usize>,
    pos: Vector3<f64>,
    arc: Option<f64>,
}

/// **`mju_wrap`**: where a cable leaving `x0` for `x1` first touches an obstacle and where it leaves it.
///
/// Returns the two tangent points in the world and the arc length between them, or `None` when the straight
/// run misses the obstacle entirely — which is most of the time, and is why a wrapped tendon's length is not
/// a smooth function of the state.
///
/// The construction is two-dimensional. For a SPHERE the plane is the one through `x0`, `x1` and the centre,
/// and the answer lifts straight back. For a CYLINDER it is the geom's own xy-plane, the radius is `size[0]`
/// and the axis is infinite, so the two tangent points get their heights by interpolating `z` along the
/// 2-D path — which is what makes the arc a helix and not a circle.
///
/// ⛔ There are always TWO solutions, one each way round. `sidesite` picks between them; without one the
/// shorter path wins. Measured against MuJoCo 3.13.0, which stores its own answer in `d.wrap_xpos`.
fn wrap_obstacle(x0: Vector3<f64>, x1: Vector3<f64>, pose: &Iso, radius: f64, cylinder: bool, side: Option<Vector3<f64>>) -> Option<([Vector3<f64>; 2], f64)> {
    let c = pose.translation.vector;
    let (e1, e2, axis) = if cylinder {
        let m = pose.rotation.to_rotation_matrix().into_inner();
        (m.column(0).into_owned(), m.column(1).into_owned(), Some(m.column(2).into_owned()))
    } else {
        // the plane through the two endpoints and the centre; collinear means no plane, so no wrap
        let n = (x1 - x0).cross(&(x0 - c));
        if n.norm() < 1e-12 {
            return None;
        }
        let n = n.normalize();
        let e1 = (x1 - x0 - n * (x1 - x0).dot(&n)).normalize();
        (e1, n.cross(&e1), None)
    };
    let to2 = |p: Vector3<f64>| [(p - c).dot(&e1), (p - c).dot(&e2)];
    let (p0, p1) = (to2(x0), to2(x1));
    let (d0, d1) = ((p0[0] * p0[0] + p0[1] * p0[1]).sqrt(), (p1[0] * p1[0] + p1[1] * p1[1]).sqrt());
    if d0 <= radius || d1 <= radius {
        return None;
    }
    let dif = [p1[0] - p0[0], p1[1] - p0[1]];
    let a = dif[0] * dif[0] + dif[1] * dif[1];
    if a < 1e-24 {
        return None;
    }
    let side2 = side.map(to2);
    let b = p0[0] * dif[0] + p0[1] * dif[1];
    let det = b * b - a * (d0 * d0 - radius * radius);
    let crosses = det > 0.0 && (-b - det.max(0.0).sqrt()) / a <= 1.0 && (-b + det.max(0.0).sqrt()) / a >= 0.0;
    match side2 {
        // ⛔⛔ a `sidesite` does not merely choose between two wraps — it DEMANDS a side, and the cable is
        // pulled all the way round when it is on the wrong one, however far off the obstacle sits. The test
        // is the CLOSEST POINT OF THE RUN, clamped to its ends: if it lies on the opposite side of the
        // obstacle's centre from the side site, the cable is on the wrong side and must come round.
        //
        // Measured on MuJoCo 3.13.0: an obstacle two-thirds of a segment length PAST the end still wraps
        // when the side site is across from it, and one sitting right beside the run does not when the side
        // site agrees with where the cable already is. Requiring an intersection loses `iit_softfoot`'s
        // wraps entirely; requiring the obstacle to be alongside the run loses `ms_human_700`'s knee.
        Some(sp) => {
            let t = (-b / a).clamp(0.0, 1.0);
            let near = [p0[0] + t * dif[0], p0[1] + t * dif[1]];
            if !crosses && near[0] * sp[0] + near[1] * sp[1] >= 0.0 {
                return None;
            }
        }
        None if !crosses => return None,
        None => {}
    }
    // the tangent points from a point at distance `d`, one each way round
    let tang = |p: [f64; 2], d: f64, sgn: f64| -> [f64; 2] {
        let ca = radius / d;
        let sa = sgn * (1.0 - ca * ca).max(0.0).sqrt();
        let k = radius / d;
        [(ca * p[0] - sa * p[1]) * k, (sa * p[0] + ca * p[1]) * k]
    };
    // ⛔⛔ the arc is NOT "the shorter way round". Which way the cable turns is fixed by how it arrives: it
    // leaves `p0` on a tangent, so at the touch point it is already travelling one way around the circle and
    // cannot reverse. A cable that comes in at one angle and leaves at another 147° away may well take the
    // 213° arc, and on `ms_human_700`'s rectus femoris it does — taking the short arc instead lost 28 mm of
    // a 690 mm tendon and put the wrap points 1.7 mm out along the cylinder's axis, because the heights
    // interpolate along that same path.
    let arc_of = |t0: [f64; 2], t1: [f64; 2]| -> f64 {
        let din = [t0[0] - p0[0], t0[1] - p0[1]];
        let ccw = t0[0] * din[1] - t0[1] * din[0] > 0.0;
        let (a0, a1) = (t0[1].atan2(t0[0]), t1[1].atan2(t1[0]));
        let mut da = if ccw { a1 - a0 } else { a0 - a1 };
        while da < 0.0 {
            da += std::f64::consts::TAU;
        }
        radius * da
    };
    let dist = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let cands = [(tang(p0, d0, 1.0), tang(p1, d1, -1.0)), (tang(p0, d0, -1.0), tang(p1, d1, 1.0))];
    let pick = match side2 {
        // the side point chooses: the wrap whose midpoint on the circle is nearer to it
        Some(sp) => {
            let score = |(t0, t1): ([f64; 2], [f64; 2])| dist([(t0[0] + t1[0]) * 0.5, (t0[1] + t1[1]) * 0.5], sp);
            usize::from(score(cands[1]) < score(cands[0]))
        }
        None => {
            let total = |(t0, t1): ([f64; 2], [f64; 2])| dist(p0, t0) + arc_of(t0, t1) + dist(t1, p1);
            usize::from(total(cands[1]) < total(cands[0]))
        }
    };
    let (t0, t1) = cands[pick];
    let arc2d = arc_of(t0, t1);
    let lift = |t: [f64; 2], h: f64| c + e1 * t[0] + e2 * t[1] + axis.map(|ax| ax * h).unwrap_or_else(Vector3::zeros);
    let Some(ax) = axis else {
        return Some(([lift(t0, 0.0), lift(t1, 0.0)], arc2d));
    };
    // a cylinder is infinite along its axis: the heights follow the 2-D path, so the arc is a helix
    let (h0, h1) = ((x0 - c).dot(&ax), (x1 - c).dot(&ax));
    let (s0, s1) = (dist(p0, t0), dist(p0, t0) + arc2d);
    let total = s1 + dist(t1, p1);
    if total < 1e-15 {
        return None;
    }
    let (z0, z1) = (h0 + (h1 - h0) * s0 / total, h0 + (h1 - h0) * s1 / total);
    Some(([lift(t0, z0), lift(t1, z1)], (arc2d * arc2d + (z1 - z0) * (z1 - z0)).sqrt()))
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
    /// ⛔ A class's actuator default is a RECORD, not a bag of attributes: `<default><position kp="80"/>`
    /// has already been through the shortcut resolution by the time an element inherits it, so `kp` and `kv`
    /// no longer exist — they are `gainprm[0]` and `biasprm[2]`. Keeping the declarations in order, with
    /// their tags, is what lets the record be rebuilt the way MuJoCo built it.
    actuator_decls: HashMap<String, Vec<(String, Vec<(String, String)>)>>,
    parent: HashMap<String, Option<String>>,
    /// (class, element kind) → attributes
    attrs: HashMap<(String, String), Vec<(String, String)>>,
}

/// ⛔ MJCF's top-level default class is NAMED, and the name is `main`. A file may write `<default>` or
/// `<default class="main">` and mean the same class. Holding the root under a different key — the empty
/// string, say — makes the second spelling register a separate class that nothing inherits from, so every
/// default in it is silently dropped: on `pal_tiago_dual` that turned `type="mesh"` back into the sphere
/// default and `contype="0"` back into 1, on 77 of Menagerie's geoms, with no error anywhere.
const MAIN: &str = "main";

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
                // ⛔ MuJoCo keeps ONE actuator record per default class, not one per shortcut tag: a class
                // that says `<position kp="80"/>` is setting the same record a `<general>` in that class
                // would read. Keying by element name would make the two invisible to each other.
                if ACTUATOR_TAGS.contains(&el.name.as_str()) {
                    self.actuator_decls.entry(class.clone()).or_default().push((el.name.clone(), el.attrs.clone()));
                } else {
                    self.attrs.entry((class.clone(), el.name.clone())).or_default().extend(el.attrs.iter().cloned());
                }
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

    /// Every value of `key` that applies to `el`, OUTERMOST FIRST: main's, then each class down the chain,
    /// then the element's own. ⛔ MuJoCo's `ReadAttr` writes only as many numbers as the attribute supplies
    /// and leaves the rest of the array as the default class left it, so a numeric array attribute overrides
    /// only its LEADING entries. `robotiq_2f85_v4` inherits `size="0.004 0.011 0.01875"` from a class and
    /// writes `size="0.009 0.02"` on the geom; MuJoCo's `geom_size` is `0.009 0.02 0.01875`.
    fn chain<'a>(&'a self, el: &'a El, kind: &str, key: &str, childclass: Option<&str>) -> Vec<&'a str> {
        let mut classes: Vec<&str> = Vec::new();
        let mut class: Option<&str> = el.attr("class").or(childclass).or(Some(MAIN));
        while let Some(c) = class {
            if classes.len() > 64 {
                break;
            }
            classes.push(c);
            class = self.parent.get(c).and_then(|p| p.as_deref());
        }
        let mut out: Vec<&str> = Vec::new();
        for c in classes.iter().rev() {
            if let Some(list) = self.attrs.get(&((*c).to_string(), kind.to_string()))
                && let Some((_, v)) = list.iter().rev().find(|(k, _)| k == key)
            {
                out.push(v.as_str());
            }
        }
        if let Some(v) = el.attr(key) {
            out.push(v);
        }
        out
    }

    /// ⛔ The orientation attributes are ONE choice, not five independent ones. MuJoCo stores an element's
    /// orientation as a tagged union: a default class sets the union, and an element that states any
    /// alternative REPLACES it whole — `ReadAlternative` only refuses two alternatives written on the same
    /// element. Resolving `quat`, `euler`, `xyaxes`… separately through the class chain both invents
    /// conflicts (a geom with `quat` inside a class with `euler` read as "more than one given", which is why
    /// `anybotics_anymal_b` was refused while MuJoCo compiles it) and, the other way round, silently
    /// COMPOSES two rotations that were never meant to meet. Returns the winning layer's attributes.
    fn orientation_attrs(&self, el: &El, kind: &str, childclass: Option<&str>) -> Vec<(String, String)> {
        const ORIENT: [&str; 5] = ["quat", "axisangle", "euler", "xyaxes", "zaxis"];
        let own: Vec<(String, String)> = ORIENT.iter().filter_map(|k| el.attr(k).map(|v| ((*k).to_string(), v.to_string()))).collect();
        if !own.is_empty() {
            return own;
        }
        let mut class: Option<&str> = el.attr("class").or(childclass).or(Some(MAIN));
        let mut guard = 0;
        while let Some(c) = class {
            if let Some(list) = self.attrs.get(&(c.to_string(), kind.to_string())) {
                let found: Vec<(String, String)> = ORIENT
                    .iter()
                    .filter_map(|k| list.iter().rev().find(|(kk, _)| kk == k).map(|(_, v)| ((*k).to_string(), v.clone())))
                    .collect();
                if !found.is_empty() {
                    return found;
                }
            }
            class = self.parent.get(c).and_then(|p| p.as_deref());
            guard += 1;
            if guard > 64 {
                break;
            }
        }
        Vec::new()
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
        // ⛔ MuJoCo's +z runs from the SECOND endpoint to the FIRST (`mjCGeom::Compile` and `mjCSite::Compile`
        // both take `vec = fromto[0..3] − fromto[3..6]`, then `mjuu_z2quat`). Taking it the other way puts the
        // frame 180° out: the position, the size and the shape are all still right, because a capsule is
        // symmetric about its axis, so nothing fails until something reads `geom_xmat` — 77 of Menagerie's
        // 11,912 geoms, every one of them a `fromto` capsule.
        let d = p1 - p2;
        if !(d.norm().is_finite() && d.norm() > 0.0) {
            return Err(format!("{kind} fromto endpoints coincide"));
        }
        let rot = UnitQuaternion::rotation_between(&Vector3::z(), &d)
            .unwrap_or_else(|| UnitQuaternion::from_axis_angle(&Vector3::x_axis(), std::f64::consts::PI));
        return Ok(Iso::from_parts(Translation3::from((p1 + p2) / 2.0), rot));
    }
    let p = get("pos").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::zeros);
    let orient = defaults.orientation_attrs(el, kind, childclass);
    let get_orient = |k: &str| orient.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
    Ok(Iso::from_parts(Translation3::from(p), orientation(&get_orient, c)?))
}

#[allow(clippy::type_complexity)]
fn inertial_of(el: &El, c: &Compiler) -> Result<(LinkInertia, [f64; 3], Matrix3<f64>), String> {
    let mass = el.attr("mass").ok_or("inertial needs mass")?.trim().parse::<f64>().map_err(|e| e.to_string())?;
    let com = el.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
    let get = |k: &str| el.attr(k).map(|s| s.to_string());
    let rm = *orientation(&get, c)?.to_rotation_matrix().matrix();
    let (ic, principal, iframe) = if let Some(d) = el.attr("diaginertia") {
        let v = floats(d)?;
        if v.len() != 3 {
            return Err("diaginertia needs 3 numbers".into());
        }
        // stated diagonal: this IS `body_inertia`, and `body_iquat` is the orientation as written
        (Matrix3::from_diagonal(&Vector3::new(v[0], v[1], v[2])), [v[0], v[1], v[2]], rm)
    } else if let Some(f) = el.attr("fullinertia") {
        let v = floats(f)?;
        if v.len() != 6 {
            return Err("fullinertia needs 6 numbers (xx yy zz xy xz yz)".into());
        }
        // ⛔ MuJoCo does not keep the tensor you wrote. `mjCBody::Compile` diagonalises a `fullinertia` with
        // `mjuu_eig3` and stores `body_inertia` (the eigenvalues) beside `body_iquat`, so the tensor the
        // dynamics actually see is the ROUND TRIP — and `mjuu_eig3` stops on an ABSOLUTE 1e-12, which on an
        // off-diagonal term of 8.3e-7 is 2.3e-7 RELATIVE. Keeping the exact input instead leaves the mass
        // matrix out by 1.1e-7 relative at the root of `franka_emika_panda` — invisible in `qfrc_bias`,
        // because at that pose joint 1's axis is vertical and gravity exerts no torque about it.
        let (d, r) = mujoco_stored_inertia_parts(&Matrix3::new(v[0], v[3], v[4], v[3], v[1], v[5], v[4], v[5], v[2]));
        (r * Matrix3::from_diagonal(&Vector3::new(d[0], d[1], d[2])) * r.transpose(), d, rm * r)
    } else if mass == 0.0 {
        // MuJoCo accepts `<inertial pos="0 0 0" mass="0"/>` (Menagerie's rby1 uses it for its world body):
        // a massless body has no tensor to state
        (Matrix3::zeros(), [0.0; 3], rm)
    } else {
        return Err("inertial needs diaginertia or fullinertia".into());
    };
    Ok((LinkInertia { mass, com, inertia: rm * ic * rm.transpose() }, principal, iframe))
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
    /// per mesh name: the vertices as MuJoCo stores them after processing (CoM-centred, principal frame)
    mesh_stored: HashMap<String, crate::TriMesh3>,
    /// per mesh name: the vertices after `ApplyTransformations`, before centring
    mesh_raw: HashMap<String, crate::TriMesh3>,
    /// meshes referenced by a collidable mesh geom; their hulls are built once the walk is done
    collidable_meshes: BTreeSet<String>,
    /// the enclosing bodies, innermost last (empty at the worldbody)
    body_stack: Vec<String>,
    /// MuJoCo's body ids: the world is 0 and the rest are numbered on first entry, depth-first in file order
    body_ids: HashMap<String, usize>,
    /// geoms with the key MuJoCo sorts them by — `(body id, order within the file)` — and the name the MJCF
    /// gave them, if any; the auto-generated `geom{i}` needs the FINAL index, so it is assigned after sorting
    geom_records: Vec<(usize, usize, Option<String>, MjcfGeom)>,
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
                "geom" => self.record_geom(ch, parent, carry, childclass)?,
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

    /// **Record one geom**, resolved the way MuJoCo resolves it, and note the mesh a collidable mesh geom
    /// needs a hull for. `parent` is the tree joint the enclosing frame rides on and `carry` the transform
    /// from that joint's frame to it, so the geom's own pose composes on the right.
    fn record_geom(&mut self, g: &El, parent: isize, carry: Iso, childclass: Option<&str>) -> Result<(), String> {
        use crate::mujoco_collision::{GeomParams, GeomType};
        let body = self.body_stack.last().cloned().unwrap_or_else(|| "world".to_string());
        let spec = self.geom_spec(g, childclass, &body)?;
        let index = self.geom_records.len();
        let get = |k: &str| self.defaults.get(g, "geom", k, childclass).map(|s| s.to_string());
        let num = |k: &str, dflt: f64| -> Result<f64, String> { get(k).map(|s| s.trim().parse::<f64>().map_err(|e| format!("geom {k}: {e}"))).transpose().map(|v| v.unwrap_or(dflt)) };
        let int = |k: &str, dflt: i64| -> Result<i64, String> { get(k).map(|s| s.trim().parse::<f64>().map(|x| x as i64).map_err(|e| format!("geom {k}: {e}"))).transpose().map(|v| v.unwrap_or(dflt)) };
        // each layer of the default chain overwrites only the entries it states, as `ReadAttr` does
        let list = |k: &str, dflt: &[f64]| -> Result<Vec<f64>, String> {
            let mut out = dflt.to_vec();
            for layer in self.defaults.chain(g, "geom", k, childclass) {
                let f = floats(layer)?;
                if f.is_empty() || f.len() > dflt.len() {
                    return Err(format!("geom {k}: expected 1..{} numbers", dflt.len()));
                }
                out[..f.len()].copy_from_slice(&f);
            }
            Ok(out)
        };
        let kind = match spec.ty.as_str() {
            "plane" => GeomType::Plane,
            "hfield" => GeomType::HField,
            "sphere" => GeomType::Sphere,
            "capsule" => GeomType::Capsule,
            "ellipsoid" => GeomType::Ellipsoid,
            "cylinder" => GeomType::Cylinder,
            "box" => GeomType::Box,
            "mesh" | "sdf" => GeomType::Mesh,
            other => return Err(format!("body '{body}': unknown geom type '{other}'")),
        };
        let friction = list("friction", &[1.0, 0.005, 0.0001])?;
        let solref = list("solref", &[0.02, 1.0])?;
        let solimp = list("solimp", &[0.9, 0.95, 0.001, 0.5, 2.0])?;
        let params = GeomParams {
            condim: int("condim", 3)? as usize,
            priority: int("priority", 0)? as i32,
            solmix: num("solmix", 1.0)?,
            solref: [solref[0], solref[1]],
            solimp: [solimp[0], solimp[1], solimp[2], solimp[3], solimp[4]],
            friction: [friction[0], friction[1], friction[2]],
            margin: num("margin", 0.0)?,
            gap: num("gap", 0.0)?,
            contype: int("contype", 1)? as u32,
            conaffinity: int("conaffinity", 1)? as u32,
            adhesion: num("adhesion", 0.0)?,
        };
        if let Some(mname) = spec.mesh.as_ref().filter(|_| kind == GeomType::Mesh && (params.contype != 0 || params.conaffinity != 0)) {
            self.mesh_data(mname)?;
            self.collidable_meshes.insert(mname.clone());
        }
        let body_id = self.body_stack.last().and_then(|b| self.body_ids.get(b).copied()).unwrap_or(0);
        self.geom_records.push((body_id, index, g.attr("name").map(|s| s.to_string()), MjcfGeom {
            name: String::new(),
            body,
            joint: (parent >= 0).then_some(parent as usize),
            pose: carry * spec.pose * spec.mesh_frame.unwrap_or_else(Iso::identity),
            kind,
            size: [spec.size.first().copied().unwrap_or(0.0), spec.size.get(1).copied().unwrap_or(0.0), spec.size.get(2).copied().unwrap_or(0.0)],
            mesh: spec.mesh,
            params,
        }));
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
        let parent_name = self.body_stack.last().cloned().unwrap_or_else(|| "world".to_string());
        self.out.body_parent.insert(name.clone(), parent_name);
        // MuJoCo numbers bodies depth-first in file order, the world being 0
        let next_id = self.body_ids.len() + 1;
        self.body_ids.entry(name.clone()).or_insert(next_id);
        let joints: Vec<&El> = b.children.iter().filter(|c| c.name == "joint" || c.name == "freejoint").collect();
        let has_free = joints.iter().any(|j| j.name == "freejoint" || self.defaults.get(j, "joint", "type", childclass) == Some("free"));
        if has_free && (parent >= 0 || joints.len() > 1) {
            return Err(format!("body '{name}': a free joint must be the only joint of a child of the world"));
        }
        // the body frame relative to the enclosing frame — dropped entirely for a free body, whose pose IS qpos
        let stated_pose = pose_of(b, "body", self.defaults, None, self.c)?;
        if has_free {
            self.out.free_base_pose.insert(name.clone(), stated_pose);
        }
        let bpose = if has_free { Iso::identity() } else { stated_pose };

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
            // ⛔⛔ `<freejoint/>` DOES NOT TAKE THE JOINT DEFAULTS, and `<joint type="free"/>` does. The same
            // joint written the two ways gets different `armature`, `damping` and `frictionloss` — MuJoCo
            // gives the shortcut element only `name`, `group` and `align`, and reads nothing from a class.
            // On `booster_t1`, whose default class says `armature="0.005"`, inheriting it added 0.005 to all
            // six base dofs and the model came out 5 g heavy: `M[0][0]` read 31.619357 against MuJoCo's
            // 31.614357, which is the whole of a 1.6e-4 error in the mass matrix of a 32 kg humanoid.
            let shortcut = j.name == "freejoint";
            let get = |k: &str| if shortcut { j.attr(k).map(|s| s.to_string()) } else { self.defaults.get(j, "joint", k, childclass).map(|s| s.to_string()) };
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
            // a fixed-length numeric attribute, filled from the left and defaulted for what is missing
            let pair = |v: Option<String>, d: [f64; 2]| -> Result<[f64; 2], String> {
                let mut out = d;
                if let Some(t) = v {
                    for (i, x) in floats(&t)?.iter().take(2).enumerate() {
                        out[i] = *x;
                    }
                }
                Ok(out)
            };
            let five = |v: Option<String>, d: [f64; 5]| -> Result<[f64; 5], String> {
                let mut out = d;
                if let Some(t) = v {
                    for (i, x) in floats(&t)?.iter().take(5).enumerate() {
                        out[i] = *x;
                    }
                }
                Ok(out)
            };
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
            // `actuatorfrcrange`: the joint's own cap on the total actuator force through it
            let actfrc = {
                let r = get("actuatorfrcrange").map(|s| floats(&s)).transpose()?;
                let r = match r {
                    Some(v) if v.len() == 2 => Some((v[0], v[1])),
                    Some(_) => return Err(format!("joint '{jname}': actuatorfrcrange needs 2 numbers")),
                    None => None,
                };
                match (get("actuatorfrclimited").as_deref(), r) {
                    (Some("false"), _) | (_, None) => None,
                    (Some("true"), Some(v)) => Some(v),
                    (_, Some(v)) if v != (0.0, 0.0) && self.c.autolimits => Some(v),
                    (None, Some(v)) if v != (0.0, 0.0) => {
                        return Err(format!("joint '{jname}' has an actuatorfrcrange but no actuatorfrclimited and autolimits is false"))
                    }
                    _ => None,
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
                    push(joint);
                }
                // ⛔ ARMATURE APPLIES TO EVERY DOF OF A MULTI-DOF JOINT, not only to hinges and slides:
                // MuJoCo writes it into all six of a free joint's `dof_armature`, and ten Menagerie models
                // set it there. Dropping it left the base six mass-matrix entries short.
                //
                // ⚠ It can only be made to match on the THREE TRANSLATIONS. Armature is a constant added to
                // the mass matrix's diagonal, and a diagonal is only a diagonal in one basis: MuJoCo's three
                // body-frame angular velocities are not these three Euler-rate hinges, so the same number
                // builds a different matrix on the rotations. That is a property of the coordinates, not of
                // the recursion, and only a base in MuJoCo's own coordinates removes it.
                MjcfJointKind::Ball => {
                    let origin = pre * Iso::from_parts(Translation3::from(anchor), UnitQuaternion::identity());
                    let arm = |j: Joint| match armature {
                        Some(a) => j.with_armature(a),
                        None => j,
                    };
                    push(arm(Joint::revolute(origin, Vector3::z())));
                    push(arm(Joint::revolute(Iso::identity(), Vector3::y())));
                    push(arm(Joint::revolute(Iso::identity(), Vector3::x())));
                }
                MjcfJointKind::Free => {
                    let arm = |j: Joint| match armature {
                        Some(a) => j.with_armature(a),
                        None => j,
                    };
                    push(arm(Joint::prismatic(Iso::identity(), Vector3::x())));
                    push(arm(Joint::prismatic(Iso::identity(), Vector3::y())));
                    push(arm(Joint::prismatic(Iso::identity(), Vector3::z())));
                    push(arm(Joint::revolute(Iso::identity(), Vector3::z())));
                    push(arm(Joint::revolute(Iso::identity(), Vector3::y())));
                    push(arm(Joint::revolute(Iso::identity(), Vector3::x())));
                }
            }
            self.out.tree.joint_names.insert(jname.clone(), first);
            self.out.joints.push(MjcfJoint {
                name: jname,
                kind,
                body: name.clone(),
                first,
                reference,
                range: limits,
                actuator_force_range: actfrc,
                damping: damping.unwrap_or(0.0),
                frictionloss: frictionloss.unwrap_or(0.0),
                stiffness: parse(get("stiffness")).unwrap_or(0.0),
                springref: parse(get("springref")).unwrap_or(0.0) * angle_scale,
                margin: parse(get("margin")).unwrap_or(0.0) * angle_scale,
                solref_limit: pair(get("solreflimit"), [0.02, 1.0])?,
                solimp_limit: five(get("solimplimit"), [0.9, 0.95, 0.001, 0.5, 2.0])?,
                solref_friction: pair(get("solreffriction"), [0.02, 1.0])?,
                solimp_friction: five(get("solimpfriction"), [0.9, 0.95, 0.001, 0.5, 2.0])?,
                armature: armature.unwrap_or(0.0),
                aligned_slide: kind == MjcfJointKind::Slide
                    && anchor.norm() == 0.0
                    && axis.iter().filter(|x| x.abs() > f64::EPSILON).count() == 1,
            });
            // after the motion, the body frame sits at −anchor from the joint's frame
            pre = Iso::from_parts(Translation3::from(-anchor), UnitQuaternion::identity());
        }

        // inertia: a jointed body owns its last joint's link; a jointless one welds into the ancestor it rides on.
        // `inertiafromgeom`: auto = the <inertial> if stated, else the geoms; true = always the geoms; false = never
        let stated = b.child("inertial");
        let (mut li, mut principal, mut iframe) = match (stated, self.c.inertiafromgeom) {
            (Some(el), InertiaFromGeom::Auto | InertiaFromGeom::False) => inertial_of(el, self.c).map_err(|e| format!("body '{name}': {e}"))?,
            (None, InertiaFromGeom::False) => {
                self.out.no_inertial.push(name.clone());
                (LinkInertia::zero(), [0.0; 3], Matrix3::identity())
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
                    None => (LinkInertia::zero(), [0.0; 3], Matrix3::identity()),
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
            // the floor rewrites the principal values, so the stored pair has to follow it
            let (d, r) = mujoco_stored_inertia_parts(&li.inertia);
            principal = d;
            iframe = r;
        }
        if ride >= 0 {
            let idx = ride as usize;
            let moved = crate::dynamics::transform_inertia(&li, &pre);
            self.out.tree.inertia[idx] = crate::dynamics::combine_inertia(&self.out.tree.inertia[idx], &moved);
            if !joints.is_empty() {
                self.out.tree.link_names.insert(name.clone(), idx);
            }
        }
        self.out.body_ipos.insert(name.clone(), li.com);
        self.out.body_mass.insert(name.clone(), li.mass);
        // MuJoCo keeps a body's inertia as a DIAGONAL plus the frame it is diagonal in; the fluid model
        // reads both, and these are the two halves as the compiler computed them — not a second
        // diagonalisation of the tensor they were reassembled into
        self.out.body_iinertia.insert(name.clone(), principal);
        self.out.body_iquat.insert(name.clone(), iframe);
        // `<body gravcomp>` is a plain body attribute: no default class carries it, and it does not inherit
        if let Some(v) = b.attr("gravcomp") {
            let g = v.trim().parse::<f64>().map_err(|e| format!("body '{name}' gravcomp: {e}"))?;
            if g != 0.0 {
                self.out.body_gravcomp.insert(name.clone(), g);
            }
        }
        self.place("body", name.clone(), ride, pre)?;
        self.body_stack.push(name);
        let r = self.children(b, ride, pre, childclass);
        self.body_stack.pop();
        r
    }

    /// MuJoCo's `mjCGeom::Compile` + `SetInertia` for one geom, in the body frame; `None` for a geom MuJoCo
    /// gives no mass (planes, height fields, zero density or mass, outside the group range).
    /// **Resolve one geom the way MuJoCo's compiler does**: its type, its `geom_size` after `fromto` and
    /// after fitting to a mesh, and its pose in the body frame. Shared by the inertia pass and by
    /// [`Walk::record_geom`], because a geom fitted to a mesh must be the SAME primitive in both.
    fn geom_spec(&mut self, g: &El, childclass: Option<&str>, body: &str) -> Result<GeomSpec, String> {
        let get = |k: &str| self.defaults.get(g, "geom", k, childclass).map(|s| s.to_string());
        let ty = get("type").unwrap_or_else(|| "sphere".into());
        // ⛔ layered, not last-wins: a `size` on the geom overrides only as many entries as it supplies
        let mut size: Vec<f64> = Vec::new();
        for layer in self.defaults.chain(g, "geom", "size", childclass) {
            let v = floats(layer)?;
            if v.len() > size.len() {
                size.resize(v.len(), 0.0);
            }
            size[..v.len()].copy_from_slice(&v);
        }
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
        // a mesh geom takes MuJoCo's `geom_size` from the mesh's own bounds, and carries the mesh frame
        let mut mesh_frame = None;
        if ty == "mesh" {
            let name = mesh_name.clone().ok_or_else(|| format!("body '{body}': a mesh geom needs a `mesh` attribute"))?;
            let md = self.mesh_data(&name)?;
            size = (0..3).map(|k| md.aamm[k].abs().max(md.aamm[k + 3].abs())).collect();
            mesh_frame = Some(Iso::from_parts(Translation3::from(md.com), UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(md.quat[0], md.quat[1], md.quat[2], md.quat[3]))));
        }
        Ok(GeomSpec { ty, size, pose, mesh: mesh_name, mesh_frame })
    }

    fn geom_mass(&mut self, g: &El, childclass: Option<&str>, body: &str) -> Result<Option<GeomMass>, String> {
        let get = |k: &str| self.defaults.get(g, "geom", k, childclass).map(|s| s.to_string());
        let group: i64 = get("group").map(|s| s.trim().parse::<f64>().map(|x| x as i64).map_err(|e| e.to_string())).transpose()?.unwrap_or(0);
        if group < self.c.inertiagrouprange.0 || group > self.c.inertiagrouprange.1 {
            return Ok(None);
        }
        if get("shellinertia").as_deref() == Some("true") {
            return Err(format!("body '{body}': geom shellinertia is outside this loader's subset"));
        }
        let GeomSpec { ty, size, pose, mesh: mesh_name, mesh_frame: _ } = self.geom_spec(g, childclass, body)?;
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
        // `meshdir/file` first; then, for a mesh declared inside an included file, the same name relative to
        // THAT file's own directory, with `meshdir` deliberately not applied — `ms_human_700` says
        // `file="../geometry/r_pelvis.stl"` from `assets/asset/`, which means nothing from the model root
        let bytes = match (self.resolve)(&path) {
            Some(b) => b,
            None => {
                let alt = format!("{}{}", asset.include_dir, asset.file);
                (self.resolve)(&alt).ok_or_else(|| format!("mesh '{name}': file '{path}' could not be resolved, nor '{alt}'"))?
            }
        };
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
        self.mesh_raw.insert(name.to_string(), crate::TriMesh3 { verts: mesh.verts.iter().map(|v| v.map(|x| x as f32 as f64)).collect(), tris: mesh.tris.clone() });
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
        // what `mjCMesh::Process` stores back into its float vertex array: CoM-centred, in the principal frame
        let stored = crate::TriMesh3 { verts: mesh.verts.iter().map(|v| (r.transpose() * (*v - com)).map(|x| x as f32 as f64)).collect(), tris: mesh.tris.clone() };
        self.mesh_stored.insert(name.to_string(), stored);
        self.mesh_cache.insert(name.to_string(), d);
        Ok(d)
    }

    /// `mjCBody::InertiaFromGeom`: the body's inertia from the geoms MuJoCo would weigh, in the body frame.
    #[allow(clippy::type_complexity)]
    fn inertia_from_geoms(&mut self, b: &El, childclass: Option<&str>, body: &str) -> Result<Option<(LinkInertia, [f64; 3], Matrix3<f64>)>, String> {
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
        let (d, r) = mujoco_stored_inertia_parts(&inertia);
        Ok(Some((LinkInertia { mass, com, inertia: r * Matrix3::from_diagonal(&Vector3::new(d[0], d[1], d[2])) * r.transpose() }, d, r)))
    }

    fn body_count(&self) -> usize {
        self.out.body_frames.len() + self.out.world_fixed.keys().filter(|k| k.starts_with("body:")).count() + 1
    }
}

/// **`<actuator>`**, resolved the way MuJoCo's compiler resolves it: the shortcut tag picks a gain and bias
/// TYPE, the attributes fill in the parameters, and everything downstream sees only `gain·ctrl + bias`.
///
/// Refuses, rather than approximating, anything this port does not carry: a transmission other than a joint
/// (tendon, site, body, slider-crank), a target that is not a hinge or a slide, an activation state
/// (`dyntype`), and the tags whose force law is not affine (`muscle`, `adhesion`, `damper`, `cylinder`,
/// `pid`, `dcmotor`, `orientation`). A silently dropped actuator is a robot that does not move for reasons
/// nobody can see.
/// Why an actuator was not carried: a feature outside this port's subset, or a malformed file. The first is
/// recorded and the model still loads; the second refuses the model, as any other parse error does.
enum ActErr {
    Unsupported(String),
    Bad(String),
}
use ActErr::{Bad, Unsupported};

/// **An actuator under construction**, in MuJoCo's own form. A default class holds one of these, an element
/// inherits it, and both are built by the same two steps: the general attribute table, then the shortcut tag.
#[derive(Clone, Debug)]
struct ActRecord {
    gain: crate::mujoco_actuator::ActGain,
    gainprm: [f64; 3],
    bias: crate::mujoco_actuator::ActBias,
    biasprm: [f64; 3],
    gear: f64,
    /// the whole `gear` vector: a site transmission reads all six, as a wrench in the site's frame
    gear6: [f64; 6],
    ctrlrange: [f64; 2],
    ctrllimited: Option<bool>,
    forcerange: [f64; 2],
    forcelimited: Option<bool>,
    inheritrange: f64,
    /// the transmission as `(kind, target)`; only `joint`/`jointinparent` are carried
    trn: Option<(String, String)>,
    /// set the moment a feature outside the subset is seen, so the reason survives to the caller
    unsupported: Option<String>,
}

impl Default for ActRecord {
    fn default() -> Self {
        Self {
            gain: crate::mujoco_actuator::ActGain::Fixed,
            gainprm: [1.0, 0.0, 0.0],
            bias: crate::mujoco_actuator::ActBias::None,
            biasprm: [0.0; 3],
            gear: 1.0,
            gear6: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrlrange: [0.0; 2],
            ctrllimited: None,
            forcerange: [0.0; 2],
            forcelimited: None,
            inheritrange: 0.0,
            trn: None,
            unsupported: None,
        }
    }
}

impl ActRecord {
    /// One `<motor>`/`<position>`/`<general>`/… applied to this record, exactly as `mjXReader::OneActuator`
    /// applies it: **the general attribute table first, the shortcut tag second**. The order is the whole
    /// point — a class that sets `biasprm="0 -200 -100"` and an element that says `kp="400"` must end at
    /// `biasprm[1] = −400`, because the shortcut overwrites what the table left. Reading it the other way
    /// round leaves `−200`, which is a servo with half the stiffness it was asked for and no error anywhere.
    ///
    /// ⛔ `kp`, `kv` and `dampratio` are ELEMENT-ONLY. They are not in the general table, so a class never
    /// hands them down — by the time a class is inherited they have already become `gainprm[0]` and
    /// `biasprm[2]`. Reading them through the default chain both invents conflicts (`pal_tiago` states `kv`
    /// on the element and `dampratio` on the class, which MuJoCo accepts and a chained read rejects) and
    /// silently uses a parent's `kv` where MuJoCo used the child's.
    fn apply(&mut self, tag: &str, attrs: &[(String, String)], c: &Compiler) -> Result<(), ActErr> {
        let at = |k: &str| attrs.iter().rev().find(|(a, _)| a == k).map(|(_, v)| v.as_str());
        let num = |k: &str| -> Result<Option<f64>, ActErr> {
            at(k).map(|v| v.trim().parse::<f64>().map_err(|e| Bad(format!("actuator {k}: {e}")))).transpose()
        };
        let vec3 = |k: &str, p: &mut [f64; 3]| -> Result<(), ActErr> {
            if let Some(v) = at(k) {
                for (i, x) in floats(v).map_err(Bad)?.iter().take(3).enumerate() {
                    p[i] = *x;
                }
            }
            Ok(())
        };
        // --- the general attribute table
        for (k, kind) in [("joint", "joint"), ("jointinparent", "joint"), ("tendon", "tendon"), ("site", "site"), ("body", "body"), ("cranksite", "slider-crank")] {
            if let Some(v) = at(k) {
                self.trn = Some((kind.to_string(), v.to_string()));
            }
        }
        if let Some(g) = at("gear") {
            let v = floats(g).map_err(Bad)?;
            self.gear = v.first().copied().unwrap_or(1.0);
            self.gear6 = [0.0; 6];
            for (o, x) in self.gear6.iter_mut().zip(&v) {
                *o = *x;
            }
        }
        match at("gaintype") {
            None => {}
            Some("fixed") => self.gain = crate::mujoco_actuator::ActGain::Fixed,
            Some("affine") => self.gain = crate::mujoco_actuator::ActGain::Affine,
            Some(o) => self.unsupported = Some(format!("gaintype '{o}'")),
        }
        match at("biastype") {
            None => {}
            Some("none") => self.bias = crate::mujoco_actuator::ActBias::None,
            Some("affine") => self.bias = crate::mujoco_actuator::ActBias::Affine,
            Some(o) => self.unsupported = Some(format!("biastype '{o}'")),
        }
        match at("dyntype") {
            None | Some("none") => {}
            Some(o) => self.unsupported = Some(format!("dyntype '{o}' carries an activation state")),
        }
        vec3("gainprm", &mut self.gainprm)?;
        vec3("biasprm", &mut self.biasprm)?;
        let rng = |k: &str, lk: &str, r: &mut [f64; 2], lim: &mut Option<bool>| -> Result<(), ActErr> {
            if let Some(v) = at(k) {
                let f = floats(v).map_err(Bad)?;
                if f.len() != 2 {
                    return Err(Bad(format!("actuator {k} needs 2 numbers")));
                }
                *r = [f[0], f[1]];
            }
            match at(lk) {
                Some("true") => *lim = Some(true),
                Some("false") => *lim = Some(false),
                Some("auto") | None => {}
                Some(o) => return Err(Bad(format!("actuator {lk} '{o}' is not true/false/auto"))),
            }
            Ok(())
        };
        rng("ctrlrange", "ctrllimited", &mut self.ctrlrange, &mut self.ctrllimited)?;
        rng("forcerange", "forcelimited", &mut self.forcerange, &mut self.forcelimited)?;
        if let Some(v) = num("inheritrange")? {
            self.inheritrange = v;
        }
        let _ = c;
        // --- the shortcut tag
        use crate::mujoco_actuator::{ActBias, ActGain};
        match tag {
            "general" => {}
            "motor" => {
                self.gainprm[0] = 1.0;
                self.gain = ActGain::Fixed;
                self.bias = ActBias::None;
            }
            "position" => {
                let kp = num("kp")?.unwrap_or(self.gainprm[0]);
                let (kv, dr) = (num("kv")?, num("dampratio")?);
                if kv.is_some() && dr.is_some() {
                    return Err(Bad("actuator: kv and dampratio cannot both be given".into()));
                }
                if num("timeconst")?.unwrap_or(0.0) != 0.0 {
                    self.unsupported = Some("a position actuator with timeconst has a filter state".into());
                }
                self.gainprm[0] = kp;
                self.biasprm[1] = -kp;
                // ⛔ a POSITIVE biasprm[2] is MuJoCo's marker for an unresolved `dampratio`
                if let Some(v) = kv {
                    self.biasprm[2] = -v;
                } else if let Some(v) = dr {
                    self.biasprm[2] = v;
                }
                self.gain = ActGain::Fixed;
                self.bias = ActBias::Affine;
            }
            "velocity" => {
                let kv = num("kv")?.unwrap_or(self.gainprm[0]);
                self.biasprm = [0.0; 3];
                self.gainprm[0] = kv;
                self.biasprm[2] = -kv;
                self.gain = ActGain::Fixed;
                self.bias = ActBias::Affine;
            }
            other => self.unsupported = Some(format!("<{other}>")),
        }
        Ok(())
    }
}

/// ⛔ **The inertia MuJoCo actually stores for a body, which is not the one the file states.**
/// `mjCBody::Compile` ends by diagonalising with `mjuu_eig3` and keeping `body_inertia` (the eigenvalues)
/// beside `body_iquat`, so the tensor the dynamics see is the ROUND TRIP — and `mjuu_eig3` stops on an
/// ABSOLUTE 1e-12, which on an off-diagonal of 8.3e-7 is 2.3e-7 RELATIVE.
///
/// ⚠ It applies wherever MuJoCo had to DIAGONALISE — a `fullinertia`, and an inertia inferred from geoms —
/// and NOT to a `diaginertia`, which is already in the stored form and is kept verbatim. Applying it to all
/// three is worse than applying it to one: it puts error into the case MuJoCo leaves exact.
/// `google_robot` declares four `<inertial>` elements for twelve bodies, so eight go through the geom path;
/// round-tripping only the stated ones left its root dof's inverse weight out by 2.1e-9, which lands
/// straight in every constraint row's regularisation.
/// The two halves MuJoCo actually stores: `body_inertia` (the eigenvalues) and `body_iquat` (the frame they
/// are diagonal in). ⛔ Recovering them by diagonalising the reassembled tensor a second time is NOT the
/// same number — `mjuu_eig3` stops on an absolute 1e-12, and a second pass moves the principal values by
/// enough to shift a fluid force by 5e-10 relative. Take them where they are computed.
fn mujoco_stored_inertia_parts(full: &Matrix3<f64>) -> ([f64; 3], Matrix3<f64>) {
    let (eigval, quat) = eig3_mujoco(full);
    (eigval, quat_to_rotation(&quat))
}

/// **`<tendon>`**: the `<fixed>` tendons, with their defaults class applied. A `<spatial>` tendon is
/// reported as unsupported rather than approximated — a path length around wrapping geoms is not something
/// a linear combination can stand in for, and a wrong tendon length is a wrong actuator force.
#[allow(clippy::type_complexity)]
fn parse_tendons(root: &El, defaults: &Defaults, c: &Compiler, joints: &[MjcfJoint]) -> Result<(Vec<MjcfTendon>, Vec<(String, String)>), String> {
    let by_name: HashMap<&str, &MjcfJoint> = joints.iter().map(|j| (j.name.as_str(), j)).collect();
    let (mut out, mut unsupported) = (Vec::new(), Vec::new());
    for section in root.children_named("tendon") {
        for el in &section.children {
            let name = el.attr("name").map(|s| s.to_string()).unwrap_or_else(|| format!("tendon{}", out.len() + unsupported.len()));
            if el.name != "fixed" && el.name != "spatial" {
                unsupported.push((name, format!("<{}>", el.name)));
                continue;
            }
            let get = |k: &str| defaults.get(el, "tendon", k, section.attr("childclass"));
            let num = |k: &str, dflt: f64| -> Result<f64, String> {
                match get(k) {
                    Some(v) => v.trim().parse::<f64>().map_err(|e| format!("tendon '{name}' {k}: {e}")),
                    None => Ok(dflt),
                }
            };
            let mut path: Option<TendonPath> = None;
            if el.name == "spatial" {
                // ⛔ a pulley splits the tendon into branches with their own divisors, which is a different
                // object from a path; say so rather than pretend the branches are one cable
                if let Some(other) = el.children.iter().find(|x| x.name != "site" && x.name != "geom") {
                    unsupported.push((name.clone(), format!("a spatial tendon with <{}>", other.name)));
                    continue;
                }
                let mut pts: Vec<WrapPoint> = Vec::new();
                for c in &el.children {
                    match c.name.as_str() {
                        "site" => pts.push(WrapPoint::Site(c.attr("site").unwrap_or_default().to_string())),
                        "geom" => pts.push(WrapPoint::Geom { geom: c.attr("geom").unwrap_or_default().to_string(), sidesite: c.attr("sidesite").map(|s| s.to_string()) }),
                        _ => {}
                    }
                }
                // MuJoCo requires a wrapping geom to sit BETWEEN two sites, and never two in a row
                let bad = pts.len() < 2
                    || !matches!(pts.first(), Some(WrapPoint::Site(_)))
                    || !matches!(pts.last(), Some(WrapPoint::Site(_)))
                    || pts.windows(2).any(|w| matches!((&w[0], &w[1]), (WrapPoint::Geom { .. }, WrapPoint::Geom { .. })));
                if bad {
                    unsupported.push((name.clone(), "a spatial path that is not sites with single geoms between them".into()));
                    continue;
                }
                path = Some(TendonPath::Spatial(pts));
            }
            let mut links: Vec<(usize, f64)> = Vec::new();
            for j in el.children.iter().filter(|x| x.name == "joint") {
                let jn = j.attr("joint").ok_or_else(|| format!("tendon '{name}': a <joint> with no joint"))?;
                let target = *by_name.get(jn).ok_or_else(|| format!("tendon '{name}': no joint named '{jn}'"))?;
                if !matches!(target.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) {
                    unsupported.push((name.clone(), format!("joint '{jn}' is a {:?} joint", target.kind)));
                    links.clear();
                    break;
                }
                let coef: f64 = match j.attr("coef") {
                    Some(v) => v.trim().parse().map_err(|e| format!("tendon '{name}' coef: {e}"))?,
                    None => return Err(format!("tendon '{name}': a <joint> with no coef")),
                };
                links.push((target.first, coef));
            }
            let path = match path {
                Some(p) => p,
                None if !links.is_empty() => TendonPath::Fixed(links),
                None => continue,
            };
            let rng = match get("range") {
                Some(v) => {
                    let f = floats(v)?;
                    [f.first().copied().unwrap_or(0.0), f.get(1).copied().unwrap_or(0.0)]
                }
                None => [0.0, 0.0],
            };
            let limited = match get("limited") {
                Some("true") => true,
                Some("false") => false,
                // `limited="auto"` and an unstated `limited` both defer to `autolimits`
                _ => c.autolimits && rng != [0.0, 0.0],
            };
            let spring = match get("springlength") {
                Some(v) => {
                    let f = floats(v)?;
                    let a = f.first().copied().unwrap_or(-1.0);
                    [a, f.get(1).copied().unwrap_or(a)]
                }
                None => [-1.0, -1.0],
            };
            let record = MjcfTendon {
                name: name.clone(),
                path,
                range: limited.then_some((rng[0], rng[1])),
                margin: num("margin", 0.0)?,
                solref_limit: pair(get("solreflimit"), [0.02, 1.0])?,
                solimp_limit: five(get("solimplimit"), [0.9, 0.95, 0.001, 0.5, 2.0])?,
                stiffness: num("stiffness", 0.0)?,
                springlength: spring,
                damping: num("damping", 0.0)?,
                frictionloss: num("frictionloss", 0.0)?,
                solref_friction: pair(get("solreffriction"), [0.02, 1.0])?,
                solimp_friction: five(get("solimpfriction"), [0.9, 0.95, 0.001, 0.5, 2.0])?,
            };
            out.push(record);
        }
    }
    Ok((out, unsupported))
}

fn pair(v: Option<&str>, dflt: [f64; 2]) -> Result<[f64; 2], String> {
    let Some(v) = v else { return Ok(dflt) };
    let f = floats(v)?;
    Ok([f.first().copied().unwrap_or(dflt[0]), f.get(1).copied().unwrap_or(dflt[1])])
}

fn five(v: Option<&str>, dflt: [f64; 5]) -> Result<[f64; 5], String> {
    let Some(v) = v else { return Ok(dflt) };
    let f = floats(v)?;
    let mut out = dflt;
    for (o, x) in out.iter_mut().zip(&f) {
        *o = *x;
    }
    Ok(out)
}

/// **`<actuator>`**, resolved the way MuJoCo's compiler resolves it.
///
/// ⛔ An actuator this port does not carry does not cost the whole model — a tendon-driven hand still has
/// geoms to collide and a tree to move — but it is NAMED, with its reason, in the second return value. A
/// silently dropped actuator is a robot that does not move for reasons nobody can see.
#[allow(clippy::type_complexity)]
fn parse_actuators(root: &El, defaults: &Defaults, c: &Compiler, joints: &[MjcfJoint], tendons: &[MjcfTendon]) -> Result<(Vec<crate::mujoco_actuator::Actuator>, Vec<(String, String)>), String> {
    let by_name: HashMap<&str, &MjcfJoint> = joints.iter().map(|j| (j.name.as_str(), j)).collect();
    let ten_by_name: HashMap<&str, (usize, &MjcfTendon)> = tendons.iter().enumerate().map(|(i, t)| (t.name.as_str(), (i, t))).collect();
    let (mut out, mut unsupported) = (Vec::new(), Vec::new());
    for section in root.children_named("actuator") {
        for el in &section.children {
            let name = el.attr("name").map(|s| s.to_string()).unwrap_or_else(|| format!("actuator{}", out.len() + unsupported.len()));
            match one_actuator(el, defaults, c, &by_name, &ten_by_name, section.attr("childclass"), &name) {
                Ok(a) => out.push(a),
                Err(Unsupported(why)) => unsupported.push((name, why)),
                Err(Bad(why)) => return Err(format!("actuator '{name}': {why}")),
            }
        }
    }
    Ok((out, unsupported))
}

#[allow(clippy::too_many_arguments)]
fn one_actuator(el: &El, defaults: &Defaults, c: &Compiler, by_name: &HashMap<&str, &MjcfJoint>, ten_by_name: &HashMap<&str, (usize, &MjcfTendon)>, childclass: Option<&str>, name: &str) -> Result<crate::mujoco_actuator::Actuator, ActErr> {
    use crate::mujoco_actuator::Actuator;
    let tag = el.name.as_str();
    if !ACTUATOR_TAGS.contains(&tag) {
        // `<plugin>` and anything else MuJoCo accepts here is an actuator we do not model, not a broken file
        return Err(Unsupported(format!("<{tag}>")));
    }
    // the class chain, outermost first: each class's declarations rebuild its record on top of its parent's
    let mut chain: Vec<&str> = Vec::new();
    let mut class: Option<&str> = el.attr("class").or(childclass).or(Some(MAIN));
    while let Some(k) = class {
        if chain.len() > 64 {
            break;
        }
        chain.push(k);
        class = defaults.parent.get(k).and_then(|p| p.as_deref());
    }
    let mut rec = ActRecord::default();
    for k in chain.iter().rev() {
        for (dtag, dattrs) in defaults.actuator_decls.get(*k).into_iter().flatten() {
            rec.apply(dtag, dattrs, c)?;
        }
    }
    rec.apply(tag, &el.attrs, c)?;
    if let Some(why) = rec.unsupported {
        return Err(Unsupported(why));
    }
    let (kind, target) = rec.trn.ok_or_else(|| Unsupported("no transmission named".into()))?;
    let mut dynamic = None;
    // ⛔ a TENDON transmission is the same actuator with a different moment: `length = gear·L`, and the
    // force comes back through `gear·coef` on every joint the tendon names, not through one dof. Everything
    // downstream — gain, bias, the clamps — is untouched, which is exactly MuJoCo's factoring.
    let (j, moment) = match kind.as_str() {
        "joint" => {
            let j = *by_name.get(target.as_str()).ok_or_else(|| Bad(format!("no joint named '{target}'")))?;
            if !matches!(j.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) {
                return Err(Unsupported(format!("joint '{target}' is a {:?} joint", j.kind)));
            }
            (Some(j), vec![(j.first, rec.gear)])
        }
        "tendon" => {
            let (i, t) = ten_by_name.get(target.as_str()).copied().ok_or_else(|| Unsupported(format!("tendon '{target}' is not one this loader carries")))?;
            match &t.path {
                TendonPath::Fixed(j) => (None, j.iter().map(|(d, c)| (*d, rec.gear * c)).collect()),
                // a spatial tendon's moment is a path derivative: it belongs to the state, not to the file
                TendonPath::Spatial(_) => {
                    dynamic = Some(crate::mujoco_actuator::DynTransmission::SpatialTendon { index: i });
                    (None, Vec::new())
                }
            }
        }
        // ⛔ a SITE transmission has no length at all — MuJoCo reports `actuator_length = 0` and
        // `actuator_velocity = 0` — and its moment is not a constant: the gear is a wrench in the SITE's
        // frame, so it turns with the model and has to be rebuilt at every state.
        "site" => {
            if el.attr("refsite").is_some() {
                return Err(Unsupported("a site transmission relative to another site".into()));
            }
            dynamic = Some(crate::mujoco_actuator::DynTransmission::Site { site: target.clone(), gear: rec.gear6 });
            (None, Vec::new())
        }
        _ => return Err(Unsupported(format!("a {kind} transmission"))),
    };
    // `limited="auto"`: under `autolimits` a stated range limits and an unstated one does not, and with
    // autolimits off MuJoCo refuses a range with no `limited`
    let limit = |r: [f64; 2], lim: Option<bool>, what: &str| -> Result<Option<[f64; 2]>, ActErr> {
        let on = match lim {
            Some(v) => v,
            None if r != [0.0, 0.0] => {
                if !c.autolimits {
                    return Err(Bad(format!("{what}range is given but {what}limited is not, and autolimits is off")));
                }
                true
            }
            None => false,
        };
        Ok(on.then_some(r))
    };
    let mut ctrlrange = limit(rec.ctrlrange, rec.ctrllimited, "ctrl")?;
    // ⭐ `inheritrange` takes the ctrlrange from the joint's own range, scaled about its midpoint. Without it
    // a position servo is UNCLAMPED, and the difference is not subtle: on `agilex_piper` the force ran to its
    // ±100 limit where MuJoCo's sat at −20.6, because a control of 2 rad is far outside a ±0.4 rad joint.
    if rec.inheritrange > 0.0 {
        if ctrlrange.is_some() {
            return Err(Bad("ctrlrange and inheritrange cannot both be given".into()));
        }
        let (lo, hi) = j.and_then(|j| j.range).ok_or_else(|| Bad(format!("inheritrange, but '{target}' has no range")))?;
        let (mean, radius) = (0.5 * (hi + lo), 0.5 * (hi - lo) * rec.inheritrange);
        ctrlrange = Some([mean - radius, mean + radius]);
    }
    let forcerange = limit(rec.forcerange, rec.forcelimited, "force")?;
    Ok(Actuator {
        name: name.to_string(),
        moment,
        dynamic,
        gear: rec.gear,
        gain: rec.gain,
        gainprm: rec.gainprm,
        bias: rec.bias,
        biasprm: rec.biasprm,
        ctrlrange,
        forcerange,
    })
}

/// **Load an MJCF model as a branched tree**, resolving `<include file>` and `<asset><mesh file>` through
/// `resolve`, which receives the path exactly as written (MuJoCo resolves every include, nested ones too,
/// against the main model's directory; a mesh path is `meshdir/file`) and returns the file's bytes.
pub fn tree_from_mjcf(xml: &str, resolve: &dyn Fn(&str) -> Option<Vec<u8>>) -> Result<MjcfTree, String> {
    let mut root = parse_xml(xml)?;
    if root.name != "mujoco" {
        return Err(format!("root element is <{}>, expected <mujoco>", root.name));
    }
    expand_includes(&mut root, resolve, "", 0, &mut std::collections::HashSet::new())?;
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
            let include_dir = m.attr(INCLUDE_DIR).or_else(|| asset.attr(INCLUDE_DIR)).unwrap_or("").to_string();
            let maxhullvert = get("maxhullvert")
                .map(|v| v.trim().parse::<i64>().map_err(|e| format!("mesh '{name}': maxhullvert: {e}")))
                .transpose()?
                .filter(|&v| v > -1)
                .map(|v| v.max(4) as usize);
            meshes.insert(name, MeshAsset { file, include_dir, maxhullvert, scale, refpos, refquat, inertia });
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
            body_ipos: BTreeMap::new(),
            site_frames: BTreeMap::new(),
            world_fixed: BTreeMap::new(),
            no_inertial: Vec::new(),
            inferred_from_geoms: Vec::new(),
            angle_scale: c.deg,
            timestep: 0.002,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            reference_q: Vec::new(),
            free_base_pose: BTreeMap::new(),
            density: 0.0,
            viscosity: 0.0,
            wind: Vector3::zeros(),
            body_iinertia: BTreeMap::new(),
            body_iquat: BTreeMap::new(),
            body_mass: BTreeMap::new(),
            body_gravcomp: BTreeMap::new(),
            tendons: Vec::new(),
            tendons_unsupported: Vec::new(),
            body_parent: BTreeMap::new(),
            mesh_hulls: BTreeMap::new(),
            mesh_raw: BTreeMap::new(),
            geoms: Vec::new(),
            actuators: Vec::new(),
            actuators_unsupported: Vec::new(),
            equalities: Vec::new(),
            equalities_unsupported: Vec::new(),
            contact_excludes: Vec::new(),
            contact_pairs: Vec::new(),
        },
        unnamed_bodies: 0,
        unnamed_joints: 0,
        meshes: &meshes,
        mesh_cache: HashMap::new(),
        mesh_stored: HashMap::new(),
        mesh_raw: HashMap::new(),
        collidable_meshes: BTreeSet::new(),
        body_stack: Vec::new(),
        body_ids: HashMap::new(),
        geom_records: Vec::new(),
        resolve,
    };
    for world in worlds {
        walk.children(world, -1, Iso::identity(), world.attr("childclass"))?;
    }
    let mut out = walk.out;
    // ⛔ MuJoCo orders geoms BODY-MAJOR — every geom of body 0 (the world), then of body 1, and so on —
    // not in file order. A scene that `<include>`s a robot and then declares its own floor puts that floor
    // LAST in the file and FIRST in `geom_xpos`, because the floor belongs to the world body. Recording in
    // file order gives the right set with the wrong indices, which every by-index comparison then reads as a
    // pose error on almost every geom.
    walk.geom_records.sort_by_key(|(body, seq, _, _)| (*body, *seq));
    out.geoms = walk
        .geom_records
        .into_iter()
        .enumerate()
        .map(|(i, (_, _, name, mut g))| {
            g.name = name.unwrap_or_else(|| format!("geom{i}"));
            g
        })
        .collect();
    for name in &walk.collidable_meshes {
        let stored = &walk.mesh_stored[name];
        let cap = walk.meshes.get(name).and_then(|a| a.maxhullvert);
        let hull = crate::mujoco_hull::MeshHull::with_max_verts(stored, cap).ok_or_else(|| format!("mesh '{name}': no 3-D convex hull (MuJoCo refuses such a collision mesh)"))?;
        out.mesh_hulls.insert(name.clone(), hull);
        out.mesh_raw.insert(name.clone(), walk.mesh_raw[name].clone());
    }
    for el in root.children_named("option") {
        if let Some(v) = el.attr("timestep") {
            out.timestep = v.trim().parse::<f64>().map_err(|e| format!("<option timestep>: {e}"))?;
        }
        if let Some(v) = el.attr("gravity") {
            out.gravity = vec3(v).map_err(|e| format!("<option gravity>: {e}"))?;
        }
        if let Some(v) = el.attr("wind") {
            out.wind = vec3(v).map_err(|e| format!("<option wind>: {e}"))?;
        }
        for (k, dst) in [("density", &mut out.density), ("viscosity", &mut out.viscosity)] {
            if let Some(v) = el.attr(k) {
                *dst = v.trim().parse::<f64>().map_err(|e| format!("<option {k}>: {e}"))?;
            }
        }
    }
    // ⛔⛔ MuJoCo's `qpos0` in this port's coordinates, computed BEFORE anything that is "evaluated once at
    // qpos0" reads it: the inverse weights, a `dampratio`, a `connect`'s second anchor, a tendon's automatic
    // resting length. `q = 0` is not it whenever a joint states a `ref`.
    {
        let nv = out.tree.joints.len();
        let mut rq = vec![0.0; nv];
        for j in &out.joints {
            match j.kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => rq[j.first] = j.reference,
                MjcfJointKind::Ball => {}
                MjcfJointKind::Free => {
                    let pose = out.free_base_pose.get(&j.body).copied().unwrap_or_else(Iso::identity);
                    rq[j.first..j.first + 3].copy_from_slice(pose.translation.vector.as_slice());
                    let (roll, pitch, yaw) = pose.rotation.euler_angles();
                    rq[j.first + 3] = yaw;
                    rq[j.first + 4] = pitch;
                    rq[j.first + 5] = roll;
                }
            }
        }
        out.reference_q = rq;
    }
    let (tendons, ten_unsupported) = parse_tendons(&root, &defaults, &c, &out.joints)?;
    out.tendons = tendons;
    out.tendons_unsupported = ten_unsupported;
    // `springlength = -1` means "however long the tendon is at qpos0", which MuJoCo resolves when it
    // compiles — and `qpos0` is `reference_q`, not zero
    let l0 = out.ten_length(&out.reference_q);
    for (i, t) in out.tendons.iter_mut().enumerate() {
        if t.springlength[0] < 0.0 {
            t.springlength = [l0[i], l0[i]];
        }
    }
    let (acts, unsupported) = parse_actuators(&root, &defaults, &c, &out.joints, &out.tendons)?;
    out.actuators = acts;
    out.actuators_unsupported = unsupported;
    // ⛔ `dampratio` cannot be resolved while reading the file: it is a ratio against the inertia the
    // transmission reflects, so it needs the mass matrix. MuJoCo does it in `mj_setConst`, at `qpos0` —
    // which is `q = 0` here, because a joint's `ref` is folded into its origin.
    if out.actuators.iter().any(|a| a.biasprm[2] > 0.0) {
        let nv = out.tree.joints.len();
        let m0 = out.mass_matrix(&out.reference_q);
        let diag: Vec<f64> = (0..nv).map(|i| m0[(i, i)]).collect();
        crate::mujoco_actuator::resolve_dampratio(&mut out.actuators, &diag);
    }
    // <equality>: the `joint` coupling and `connect`; the rest are named rather than dropped
    {
        let by_name: HashMap<&str, &MjcfJoint> = out.joints.iter().map(|j| (j.name.as_str(), j)).collect();
        // the reference pose, where MuJoCo resolves a connect's second anchor and every inverse weight
        let frames0 = crate::tree_frames(&out.tree, &out.reference_q);
        let invw = out.body_invweight0();
        let biw0 = |b: Option<&str>| b.and_then(|b| invw.get(b)).map(|w| w.tran).unwrap_or(0.0);
        for section in root.children_named("equality") {
            for el in &section.children {
                let name = el.attr("name").map(|s| s.to_string()).unwrap_or_else(|| format!("equality{}", out.equalities.len() + out.equalities_unsupported.len()));
                let get = |k: &str| defaults.get(el, "equality", k, section.attr("childclass")).map(|s| s.to_string());
                // `active="false"` is a constraint the model has switched off; it produces no rows
                if get("active").as_deref() == Some("false") {
                    continue;
                }
                if el.name != "joint" && el.name != "connect" {
                    out.equalities_unsupported.push((name, format!("<{}>", el.name)));
                    continue;
                }
                let mut solref = [0.02, 1.0];
                if let Some(v) = get("solref") {
                    for (i, x) in floats(&v)?.iter().take(2).enumerate() {
                        solref[i] = *x;
                    }
                }
                let mut solimp = [0.9, 0.95, 0.001, 0.5, 2.0];
                if let Some(v) = get("solimp") {
                    for (i, x) in floats(&v)?.iter().take(5).enumerate() {
                        solimp[i] = *x;
                    }
                }
                if el.name == "connect" {
                    let Some(anchor) = el.attr("anchor") else {
                        out.equalities_unsupported.push((name, "a site anchor".into()));
                        continue;
                    };
                    let anchor = vec3(anchor)?;
                    let side = |b: Option<&str>| -> Result<Option<(Option<usize>, Iso)>, String> {
                        // the world, named or omitted: no dof to ride, and its frame IS the world's
                        let Some(b) = b.filter(|b| *b != "world") else { return Ok(Some((None, Iso::identity()))) };
                        if let Some((ride, pre)) = out.body_frames.get(b) {
                            return Ok(Some((Some(*ride), frames0[*ride] * pre)));
                        }
                        match out.world_fixed.get(&format!("body:{b}")) {
                            Some(iso) => Ok(Some((None, *iso))),
                            None => Err(format!("equality '{name}': no body named '{b}'")),
                        }
                    };
                    let (Some((ride1, world1)), Some((ride2, world2))) = (side(el.attr("body1"))?, side(el.attr("body2"))?) else {
                        continue;
                    };
                    // ⛔ the anchor is stated in body1's frame; body2's copy is the SAME WORLD POINT at
                    // `qpos0`, which is how MuJoCo stores it and the only reading that moves correctly
                    let p0 = world1 * Point3::from(anchor);
                    let local = |ride: Option<usize>, w: Iso| match ride {
                        Some(r) => (Some(r), (frames0[r].inverse() * p0).coords),
                        None => {
                            let _ = w;
                            (None, p0.coords)
                        }
                    };
                    let diag_a = biw0(el.attr("body1")) + biw0(el.attr("body2"));
                    out.equalities.push(MjcfEquality { name, kind: EqualityKind::Connect { side1: local(ride1, world1), side2: local(ride2, world2), diag_a }, solref, solimp });
                    continue;
                }
                let j1 = el.attr("joint1").ok_or_else(|| format!("equality '{name}': needs joint1"))?;
                let Some(a) = by_name.get(j1) else { return Err(format!("equality '{name}': no joint named '{j1}'")) };
                let b = match el.attr("joint2") {
                    Some(j2) => match by_name.get(j2) {
                        Some(b) => Some(*b),
                        None => return Err(format!("equality '{name}': no joint named '{j2}'")),
                    },
                    None => None,
                };
                if !matches!(a.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide) || b.is_some_and(|b| !matches!(b.kind, MjcfJointKind::Hinge | MjcfJointKind::Slide)) {
                    out.equalities_unsupported.push((name, "a ball or free joint".into()));
                    continue;
                }
                let mut polycoef = [0.0, 1.0, 0.0, 0.0, 0.0];
                if let Some(v) = get("polycoef") {
                    for (i, x) in floats(&v)?.iter().take(5).enumerate() {
                        polycoef[i] = *x;
                    }
                }
                out.equalities.push(MjcfEquality {
                    name,
                    kind: EqualityKind::Joint {
                        joint1: a.first,
                        joint2: b.map(|b| b.first),
                        reference: (a.reference, b.map(|b| b.reference).unwrap_or(0.0)),
                        polycoef,
                    },
                    solref,
                    solimp,
                });
            }
        }
    }
    // <contact><exclude> and <contact><pair> — several <contact> blocks can arrive through includes
    for contact in root.children_named("contact") {
        for ex in contact.children_named("exclude") {
            let (b1, b2) = (ex.attr("body1").ok_or("<exclude> needs body1")?, ex.attr("body2").ok_or("<exclude> needs body2")?);
            out.contact_excludes.push((b1.to_string(), b2.to_string()));
        }
        for p in contact.children_named("pair") {
            let get = |k: &str| defaults.get(p, "pair", k, None).map(|s| s.to_string());
            let nums = |k: &str, n: usize, dflt: &[f64]| -> Result<Vec<f64>, String> {
                match get(k) {
                    Some(v) => {
                        let f = floats(&v)?;
                        if f.len() > n || f.is_empty() {
                            return Err(format!("<pair> {k}: expected up to {n} numbers"));
                        }
                        let mut out = dflt.to_vec();
                        out[..f.len()].copy_from_slice(&f);
                        Ok(out)
                    }
                    None => Ok(dflt.to_vec()),
                }
            };
            let condim = get("condim").map(|s| s.trim().parse::<usize>().map_err(|e| e.to_string())).transpose()?.unwrap_or(3);
            let friction = nums("friction", 5, &[1.0, 1.0, 0.005, 0.0001, 0.0001])?;
            let solref = nums("solref", 2, &[0.02, 1.0])?;
            let solreffriction = nums("solreffriction", 2, &[0.0, 0.0])?;
            let solimp = nums("solimp", 5, &[0.9, 0.95, 0.001, 0.5, 2.0])?;
            let margin = get("margin").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(0.0);
            let gap = get("gap").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(0.0);
            let adhesion = get("adhesion").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(0.0);
            out.contact_pairs.push(MjcfContactPair {
                geom1: p.attr("geom1").ok_or("<pair> needs geom1")?.to_string(),
                geom2: p.attr("geom2").ok_or("<pair> needs geom2")?.to_string(),
                condim,
                friction: [friction[0], friction[1], friction[2], friction[3], friction[4]],
                solref: [solref[0], solref[1]],
                solreffriction: [solreffriction[0], solreffriction[1]],
                solimp: [solimp[0], solimp[1], solimp[2], solimp[3], solimp[4]],
                margin,
                gap,
                adhesion,
            });
        }
    }
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
        // ⛔ on `MjcfJoint`, not on the tree's `Joint`: MuJoCo's damping is a passive force, not an
        // actuator term, and folding it into the tree would put it in `qfrc_bias` where MuJoCo has none
        assert!(t.joints.iter().all(|j| j.damping == 1.0));
        assert!(t.tree.joints.iter().all(|j| j.damping.is_none()));
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

    /// `fromto`'s +z runs from the second endpoint to the first, which is the opposite of the reading a
    /// left-to-right name suggests. MuJoCo 3.13.0's `geom_xmat` for this model, via
    /// `scripts/mujoco_geom_oracle.py`.
    #[test]
    fn a_fromto_geoms_axis_runs_from_the_second_endpoint_to_the_first() {
        let xml = r#"<mujoco><worldbody><body name="b">
            <joint name="j" type="hinge" axis="0 0 1"/>
            <geom name="rod" type="capsule" fromto="0 0 0  0 0 0.4" size="0.05"/>
        </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads");
        let g = t.geoms.iter().find(|g| g.name == "rod").expect("the geom");
        // half the segment length goes into size[1], and the frame sits at the midpoint
        assert!((g.size[0] - 0.05).abs() < 1e-15 && (g.size[1] - 0.2).abs() < 1e-15, "{:?}", g.size);
        assert!((g.pose.translation.vector - Vector3::new(0.0, 0.0, 0.2)).norm() < 1e-15);
        // the segment runs +z, so MuJoCo's `vec` is −z and the frame's own +z points DOWN
        let z = g.pose.rotation.to_rotation_matrix() * Vector3::z();
        assert!((z + Vector3::z()).norm() < 1e-12, "the capsule's axis is {z:?}, expected −z");
    }

    /// A non-convex polygon is where fan triangulation and ear clipping part company: fanning from vertex 0
    /// of an L puts a triangle outside the shape. Both the count and the SHAPE are checked — the triangles
    /// must tile the polygon, so their total area equals the polygon's.
    #[test]
    fn a_polygon_is_ear_clipped_and_the_pieces_stay_inside_it() {
        // an L in the z = 0 plane, wound counter-clockwise
        let ring = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (1.0, 1.0), (1.0, 3.0), (0.0, 3.0)];
        let verts: Vec<Vector3<f64>> = ring.iter().map(|&(x, y)| Vector3::new(x, y, 0.0)).collect();
        let poly: Vec<usize> = (0..ring.len()).collect();
        let tris = triangulate_as_tinyobj(&poly, &verts);
        assert_eq!(tris.len(), ring.len() - 2, "an n-gon becomes n−2 triangles, got {}", tris.len());
        let area: f64 = tris.iter().map(|t| {
            let (a, b, c) = (verts[t[0]], verts[t[1]], verts[t[2]]);
            0.5 * ((b - a).cross(&(c - a))).norm()
        }).sum();
        // the L is 3×3 minus the 2×2 notch
        assert!((area - 5.0).abs() < 1e-12, "the pieces cover {area}, the polygon is 5 — a triangle escaped it");
        // every triangle is wound the same way as the polygon, so none of them is inside out
        for t in &tris {
            let (a, b, c) = (verts[t[0]], verts[t[1]], verts[t[2]]);
            assert!((b - a).cross(&(c - a)).z > 0.0, "triangle {t:?} is wound backwards");
        }
    }

    /// An `<include>` resolves against the MAIN model's directory first and, failing that, against the
    /// directory of the file it is written in — and an asset's `file` follows the same two-step rule.
    #[test]
    fn an_include_falls_back_to_the_directory_of_the_file_it_is_written_in() {
        let files: std::collections::HashMap<&str, &str> = [
            ("parts/arm.xml", r#"<mujoco><body name="arm"><joint name="a" type="hinge" axis="0 0 1"/><geom name="ga" type="sphere" size="0.1"/><include file="hand.xml"/></body></mujoco>"#),
            ("parts/hand.xml", r#"<mujoco><body name="hand" pos="0 0 0.3"><joint name="h" type="hinge" axis="0 1 0"/><geom name="gh" type="sphere" size="0.05"/></body></mujoco>"#),
        ]
        .into_iter()
        .collect();
        let xml = r#"<mujoco><worldbody><include file="parts/arm.xml"/></worldbody></mujoco>"#;
        let t = tree_from_mjcf(xml, &|p: &str| files.get(p).map(|s| s.as_bytes().to_vec())).expect("loads");
        assert!(t.body_frames.contains_key("hand"), "the nested include named its own neighbour and was not found");
        // and the failure is reported with BOTH places that were tried, not just one
        let e = tree_from_mjcf(r#"<mujoco><worldbody><include file="nope.xml"/></worldbody></mujoco>"#, &|_: &str| None).unwrap_err();
        assert!(e.contains("nope.xml"), "{e}");
    }

    /// ⛔ The orientation attributes are one choice, not five. A geom with its own `quat` inside a class that
    /// states `euler` must use the geom's `quat` and MUST NOT be read as "two orientations given" — that
    /// refusal kept `anybotics_anymal_b` out while MuJoCo compiles it.
    #[test]
    fn an_elements_own_orientation_replaces_its_classs_whole() {
        let xml = r#"<mujoco>
            <default><default class="turned"><geom type="cylinder" size="0.05 0.05" euler="0 90 0"/></default></default>
            <worldbody><body name="b"><joint name="j" type="hinge" axis="0 0 1"/>
              <geom name="own" class="turned" quat="1 0 0 0"/>
              <geom name="inherited" class="turned"/>
            </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads: the two are alternatives, not a conflict");
        let axis = |name: &str| t.geoms.iter().find(|g| g.name == name).unwrap().pose.rotation.to_rotation_matrix() * Vector3::z();
        assert!((axis("own") - Vector3::z()).norm() < 1e-12, "the geom's own quat lost to its class: {:?}", axis("own"));
        assert!((axis("inherited") - Vector3::x()).norm() < 1e-12, "the class's euler was dropped: {:?}", axis("inherited"));
    }

    /// MuJoCo's `ReadAttr` writes only as many numbers as the attribute supplies, so a geom that states a
    /// shorter `size` than its default class keeps the class's trailing entries. `robotiq_2f85_v4` does
    /// exactly this and MuJoCo's `geom_size` for those capsules is `0.009 0.02 0.01875`.
    #[test]
    fn a_shorter_size_overrides_only_the_entries_it_states() {
        let xml = r#"<mujoco>
            <default><default class="pad"><geom type="box" size="0.004 0.011 0.01875" friction="1 0.02 0.03"/></default></default>
            <worldbody><body name="b"><joint name="j" type="hinge" axis="0 0 1"/>
              <geom name="g" class="pad" type="capsule" size="0.009 0.02" friction="2"/>
            </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads");
        let g = t.geoms.iter().find(|g| g.name == "g").expect("the geom");
        assert_eq!(g.size, [0.009, 0.02, 0.01875], "size");
        assert_eq!(g.params.friction, [2.0, 0.02, 0.03], "friction");
    }

    /// ⛔ **Armature applies to EVERY dof of a multi-dof joint.** MuJoCo writes it into all six of a free
    /// joint's `dof_armature`; dropping it left the base's six mass-matrix entries short, which showed up
    /// nowhere in `qfrc_bias` and only in the accelerations.
    ///
    /// ⚠ It can only be made to AGREE on the three translations. Armature is a constant on the mass
    /// matrix's diagonal, and a diagonal belongs to a basis: MuJoCo's three body-frame angular velocities
    /// are not these three Euler-rate hinges. Ten Menagerie models set it there and they are exactly the ten
    /// whose chain accelerations disagree.
    #[test]
    fn armature_reaches_every_dof_of_a_free_joint() {
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody><body name="b">
            <freejoint name="root"/>
            <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
            <geom name="g" type="sphere" size="0.05"/>
        </body></worldbody></mujoco>"#;
        let plain = tree_from_mjcf_str(xml).expect("loads");
        assert!(plain.tree.joints.iter().all(|j| j.armature.is_none()));
        let with_arm = tree_from_mjcf_str(&xml.replace("<freejoint name=\"root\"/>", "<joint name=\"root\" type=\"free\" armature=\"0.01\"/>")).expect("loads");
        assert_eq!(with_arm.tree.joints.len(), 6);
        assert!(with_arm.tree.joints.iter().all(|j| j.armature == Some(0.01)), "armature reached only some of the six");
        assert_eq!(with_arm.joints[0].armature, 0.01);
        // and it lands on the mass matrix: a unit sphere's translational inertia goes from 1 to 1.01
        let m = crate::tree_dynamics::tree_mass_matrix(&with_arm.tree.joints, &with_arm.tree.inertia, &with_arm.tree.parent, &[0.0; 6]);
        assert!((m[(0, 0)] - 1.01).abs() < 1e-12, "{}", m[(0, 0)]);
    }

    /// ⛔ A limit produces a row per SIDE and only when that side is within `margin`, so a joint whose range
    /// is narrower than twice its margin has BOTH rows live at once. A port that emits one row per joint is
    /// wrong exactly there, and no test with a comfortable range can see it.
    #[test]
    fn a_joint_limit_makes_a_row_per_side_and_both_at_once_when_the_range_is_narrow() {
        let model = |range: &str, margin: &str| {
            format!(
                r#"<mujoco><compiler angle="radian"/><worldbody><body name="b">
                <joint name="j" type="hinge" axis="0 0 1" range="{range}" margin="{margin}" frictionloss="0.4"/>
                <inertial mass="1" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
                <geom name="g" type="sphere" size="0.05"/>
            </body></worldbody></mujoco>"#
            )
        };
        let t = tree_from_mjcf_str(&model("-1 1", "0.1")).expect("loads");
        let iw = t.dof_invweight0();
        let rows = |q: f64| t.joint_constraint_rows(&[q], &[0.0], &iw);
        // ⭐ the friction row is always there and comes FIRST — MuJoCo reads a row's law from its position
        let mid = rows(0.0);
        assert_eq!(mid.blocks.len(), 1, "well inside the range, only friction");
        assert!(matches!(mid.blocks[0], crate::mujoco_contact::EfcBlock::Friction { loss } if loss == 0.4));
        assert_eq!(mid.jac[(0, 0)], 1.0);
        // near the upper limit: a second row, with Jacobian −1 because the limit pushes the other way
        let hi = rows(0.95);
        assert_eq!(hi.blocks.len(), 2);
        assert!(matches!(hi.blocks[1], crate::mujoco_contact::EfcBlock::Limit));
        assert_eq!(hi.jac[(1, 0)], -1.0);
        // near the lower limit: Jacobian +1
        let lo = rows(-0.95);
        assert_eq!(lo.blocks.len(), 2);
        assert_eq!(lo.jac[(1, 0)], 1.0);
        // ⛔ a range narrower than twice the margin: BOTH sides are within it everywhere
        let narrow = tree_from_mjcf_str(&model("-0.05 0.05", "0.2")).expect("loads");
        let both = narrow.joint_constraint_rows(&[0.0], &[0.0], &narrow.dof_invweight0());
        assert_eq!(both.blocks.len(), 3, "one friction row and a limit row on each side");
        assert_eq!((both.jac[(1, 0)], both.jac[(2, 0)]), (1.0, -1.0), "lower then upper");
    }

    /// ⛔⛔ **MuJoCo's `damping` and `frictionloss` are not actuator terms**: damping is a passive force and
    /// frictionloss is a constraint row, and neither belongs in `qfrc_bias`. Folding them into the tree's
    /// `Joint` — where this crate's own convention puts them — is exactly right at rest and wrong the moment
    /// the model moves.
    ///
    /// ⚠ That is why it survived: every state the sweeps sampled had `qvel = 0`, and both terms are zero
    /// there. A state at rest cannot tell a velocity-dependent term from a missing one.
    #[test]
    fn damping_and_frictionloss_stay_out_of_the_bias_and_show_up_where_mujoco_puts_them() {
        // ⛔ `angle="radian"`, because `springref` is an ANGLE and is converted while `stiffness` is a
        // torque per radian and is not — in the default degree mode the two are in different units
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody><body name="b" pos="0 0 1">
            <joint name="j" type="hinge" axis="0 1 0" damping="3" frictionloss="0.7" stiffness="5" springref="0.2"/>
            <inertial mass="2" pos="0 0 -0.5" diaginertia="0.1 0.1 0.1"/>
            <geom name="g" type="sphere" size="0.05"/>
        </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads");
        assert_eq!(t.tree.joints[0].damping, None, "damping must not become an actuator term");
        assert_eq!(t.tree.joints[0].friction, None, "frictionloss must not become an actuator term");
        assert_eq!((t.joints[0].damping, t.joints[0].frictionloss), (3.0, 0.7));
        // the bias at speed is the rigid-body recursion alone, so it is unchanged by either
        let bias = |q: f64, qd: f64| crate::tree_inverse_dynamics(&t.tree.joints, &t.tree.inertia, &t.tree.parent, &[q], &[qd], &[0.0], Vector3::new(0.0, 0.0, -9.81))[0];
        assert!((bias(0.3, 2.0) - bias(0.3, -2.0)).abs() < 1e-12, "the bias depends on the SIGN of the velocity, so a friction term is in it");
        // and the passive force is damping plus the spring, about `springref` and not about zero
        let p = t.qfrc_passive(&[0.5], &[2.0]);
        assert!((p[0] - (-3.0 * 2.0 - 5.0 * (0.5 - 0.2))).abs() < 1e-12, "{p:?}");
        assert!((t.qfrc_passive(&[0.2], &[0.0])[0]).abs() < 1e-15, "at the spring's rest position and at rest there is no passive force");
    }

    /// ⛔ **A `fullinertia` is not stored, its EIGENDECOMPOSITION is.** `mjCBody::Compile` diagonalises with
    /// `mjuu_eig3` and keeps `body_inertia` beside `body_iquat`, so the tensor the dynamics see is the round
    /// trip — and `mjuu_eig3` stops on an ABSOLUTE 1e-12, which on an off-diagonal of 8.3e-7 is 2.3e-7
    /// RELATIVE. The numbers below are MuJoCo 3.13.0's own, read back from `franka_emika_panda`.
    ///
    /// ⚠ Why this hid: with the exact input the mass matrix was out by 1.1e-7 relative at the arm's root
    /// while `qfrc_bias` matched to 1e-14, because at that pose joint 1's axis is vertical and gravity
    /// exerts no torque about it. An inertia error invisible in the force it produces is still an inertia
    /// error in everything that divides by it.
    #[test]
    fn a_fullinertia_is_stored_as_mujoco_stores_it_which_is_its_eigendecomposition() {
        let xml = r#"<mujoco><worldbody><body name="b">
            <joint name="j" type="hinge" axis="0 0 1"/>
            <inertial mass="0.629769" pos="-0.041018 -0.00014 0.049974"
              fullinertia="0.00315 0.00388 0.004285 8.2904e-7 0.00015 8.2299e-6"/>
            <geom name="g" type="sphere" size="0.1"/>
        </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads");
        let i = t.tree.inertia[0].inertia;
        // MuJoCo's own round trip, to every digit it prints — kept verbatim rather than shortened to the
        // shortest round-tripping literal, because the point of the test is that these ARE MuJoCo's digits
        #[allow(clippy::excessive_precision)]
        let want = [
            [3.1499999999977401e-03, 8.2903980985604243e-07, 1.4999999999510235e-04],
            [8.2903980985604232e-07, 3.8799999999426333e-03, 8.2298985390907685e-06],
            [1.4999999999510229e-04, 8.2298985390907499e-06, 4.2850000000596985e-03],
        ];
        for r in 0..3 {
            for c in 0..3 {
                assert!((i[(r, c)] - want[r][c]).abs() <= 1e-9 * want[r][c].abs(), "({r},{c}): {} vs MuJoCo {}", i[(r, c)], want[r][c]);
            }
        }
        // ⛔ and it is NOT the tensor the file states — that is the whole point, so a port that skips the
        // round trip cannot pass by accident
        assert!((i[(0, 1)] - 8.2904e-7).abs() > 1e-14, "the off-diagonal came back unchanged, so nothing was diagonalised");
        assert!((i[(0, 0)] - 0.00315).abs() > 1e-17, "the diagonal came back unchanged");
    }

    /// MuJoCo's OBJ decoder keeps `GetShapes()[0]` — the faces up to the second `o`/`g` tag — while its
    /// vertex array stays the whole file's. A mesh whose later objects are read too has the wrong centre of
    /// mass, the wrong principal frame and the wrong `geom_size`, all silently.
    #[test]
    fn an_obj_contributes_only_its_first_shapes_faces_and_all_of_its_vertices() {
        let obj = "o first\n\
            v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n\
            f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n\
            o second\n\
            v 10 0 0\nv 11 0 0\nv 10 1 0\nv 10 0 1\n\
            f 5 7 6\nf 5 6 8\nf 6 7 8\nf 7 5 8\n";
        let m = obj_as_mujoco_reads_it(obj).expect("parses");
        assert_eq!(m.verts.len(), 8, "every vertex in the file is kept, whichever shape declared it");
        assert_eq!(m.tris.len(), 4, "only the first shape's faces are read");
        assert!(m.tris.iter().flatten().all(|&i| i < 4), "a face of the second shape survived: {:?}", m.tris);
        // and the split is what makes the frame right: the pair read together would sit near x = 5
        let (_, com, _) = mesh_inertia_mujoco(&m, MeshInertia::Legacy).expect("inertia");
        assert!(com.x < 0.5, "centre of mass at {com:?} — the second object is still being weighed");
    }

    /// `<default>` and `<default class="main">` are the same class, and a geom with no class inherits from
    /// it either way.
    #[test]
    fn the_top_level_default_class_is_named_main_however_it_is_written() {
        for spelling in ["<default>", r#"<default class="main">"#] {
            let xml = format!(
                r#"<mujoco>{spelling}<geom contype="0" conaffinity="1" type="capsule" size="0.1 0.2" condim="6"/></default>
                <worldbody><body name="b"><joint name="j" type="hinge" axis="0 0 1"/><geom name="g"/></body></worldbody></mujoco>"#
            );
            let t = tree_from_mjcf_str(&xml).unwrap_or_else(|e| panic!("{spelling}: {e}"));
            let g = t.geoms.iter().find(|g| g.name == "g").expect("the geom");
            assert_eq!(g.kind, crate::mujoco_collision::GeomType::Capsule, "{spelling}: type");
            assert_eq!((g.params.contype, g.params.conaffinity, g.params.condim), (0, 1, 6), "{spelling}: contact parameters");
            assert!((g.size[0] - 0.1).abs() < 1e-15 && (g.size[1] - 0.2).abs() < 1e-15, "{spelling}: size {:?}", g.size);
        }
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

    /// **`gravcomp = 1` holds an arm up by itself.** The test is the physical statement, not the formula:
    /// with no damping, no spring and no actuator, a fully gravity-compensated arm at rest has zero smooth
    /// acceleration in every dof and at every pose. That is `qfrc_passive == qfrc_bias`, and it can only
    /// hold if the force is applied at each body's OWN centre of mass with each body's OWN mass.
    ///
    /// The second body is welded (no joint of its own), which is where a port that works per-link rather
    /// than per-body goes wrong: it would apply one force at the merged link's centre of mass, and the arm
    /// would sag or lift.
    #[test]
    fn gravcomp_one_holds_the_arm_up_by_itself() {
        let xml = r#"<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>
<body name="upper" gravcomp="1"><joint name="s" type="hinge" axis="0 1 0"/>
  <inertial pos="0.15 0 0" mass="2" diaginertia="0.02 0.02 0.02"/>
  <body name="fore" pos="0.3 0 0" gravcomp="1"><joint name="e" type="hinge" axis="0 1 0"/>
    <inertial pos="0.2 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <body name="tool" pos="0.4 0.05 0" gravcomp="1">
      <inertial pos="0.05 0 0.02" mass="0.5" diaginertia="0.001 0.001 0.001"/>
    </body>
  </body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert_eq!(t.body_gravcomp.len(), 3, "all three bodies ask for compensation");
        let nv = t.tree.joints.len();
        for pose in [[0.0, 0.0], [0.7, -1.1], [-1.3, 0.4]] {
            let q = pose.to_vec();
            let bias = crate::tree_dynamics::tree_inverse_dynamics(&t.tree.joints, &t.tree.inertia, &t.tree.parent, &q, &vec![0.0; nv], &vec![0.0; nv], t.gravity);
            let passive = t.qfrc_passive(&q, &vec![0.0; nv]);
            for i in 0..nv {
                assert!((passive[i] - bias[i]).abs() < 1e-12, "dof {i} at {pose:?}: passive {} vs bias {}", passive[i], bias[i]);
            }
            // and it is not vacuous: without gravity compensation the arm does not hold itself up
            assert!(bias.iter().any(|b| b.abs() > 1.0), "the arm should be loaded at {pose:?}");
        }
        // ⛔ holding the arm up does NOT by itself pin WHERE the force is applied: for one hinge the
        // generalised force is linear in the centre of mass, so summing the bodies of a link at the link's
        // own centre gives the same answer whenever they share a `gravcomp`. What separates the two is a
        // link whose bodies DISAGREE — here a compensated body at x = 0.3 welded beside an uncompensated
        // one at x = 0.8, with the joint's own mass sitting on the axis where it exerts no torque.
        let mixed = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>
<body name="link"><joint name="h" type="hinge" axis="0 1 0"/>
  <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  <body name="a" pos="0.3 0 0" gravcomp="1"><inertial pos="0 0 0" mass="2" diaginertia="0.01 0.01 0.01"/></body>
  <body name="b" pos="0.8 0 0"><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/></body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        assert_eq!(mixed.body_gravcomp.len(), 1, "only 'a' asks for compensation");
        let g = mixed.qfrc_gravcomp(&[0.0]);
        // torque about +y of an upward force m·g at x = 0.3 is −0.3·m·g; 'b' contributes nothing at all
        assert!((g[0] + 0.3 * 2.0 * 9.81).abs() < 1e-12, "{}", g[0]);

        // half compensation leaves exactly half the load
        let half = tree_from_mjcf_str(&xml.replace("gravcomp=\"1\"", "gravcomp=\"0.5\"")).unwrap();
        let q = vec![0.7, -1.1];
        let bias = crate::tree_dynamics::tree_inverse_dynamics(&half.tree.joints, &half.tree.inertia, &half.tree.parent, &q, &vec![0.0; nv], &vec![0.0; nv], half.gravity);
        let passive = half.qfrc_passive(&q, &vec![0.0; nv]);
        for i in 0..nv {
            assert!((passive[i] - 0.5 * bias[i]).abs() < 1e-12, "dof {i}: {} vs half of {}", passive[i], bias[i]);
        }
    }

    /// **A SPATIAL tendon through sites, against MuJoCo's own numbers.** Its length is the path
    /// `Σ |pᵢ₊₁ − pᵢ|` and its moment `Σ ûᵢᵀ(Jᵢ₊₁ − Jᵢ)`, so unlike a fixed tendon neither is a constant —
    /// the limit row, the spring, the damper and any actuator pulling on it all have to be rebuilt per state.
    ///
    /// ⛔ 590 of Menagerie's 881 spatial tendons run through sites alone; the other 291 wrap around geoms,
    /// and those are reported as not carried rather than run straight through the obstacle.
    #[test]
    fn a_spatial_tendon_is_the_length_of_its_path() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <site name="anchor" pos="0.1 -0.05 0.9"/>
  <body name="a" pos="0 0 0.5" euler="0.1 -0.2 0.3"><joint name="j1" type="hinge" axis="0 1 0"/>
    <inertial pos="0.2 0 0" mass="1.5" diaginertia="0.03 0.05 0.07"/>
    <site name="s1" pos="0.12 0.04 0.03"/>
    <body name="b" pos="0.4 0 0" euler="0.3 0.1 -0.2"><joint name="j2" type="slide" axis="1 0 0"/>
      <inertial pos="0.1 0 0" mass="0.8" diaginertia="0.01 0.014 0.02"/>
      <site name="s2" pos="0.15 -0.03 0.02"/>
      <site name="s3" pos="0.25 0.01 -0.04"/>
    </body>
  </body>
</worldbody>
<tendon><spatial name="cable" limited="true" range="0 0.9" stiffness="30" springlength="0.5" damping="2">
  <site site="anchor"/><site site="s1"/><site site="s2"/><site site="s3"/>
</spatial></tendon></mujoco>"#,
        )
        .unwrap();
        assert!(t.tendons_unsupported.is_empty(), "{:?}", t.tendons_unsupported);
        let (q, qd) = ([0.4, 0.15], [0.9, -0.6]);
        assert!((t.ten_length(&q)[0] - 1.118181624639552).abs() < 1e-12, "{:?}", t.ten_length(&q));
        assert!((t.ten_velocity(&q, &qd)[0] + 0.49105407378281224).abs() < 1e-12, "{:?}", t.ten_velocity(&q, &qd));
        for (k, want) in [0.11969509707489943, 0.9979661019170363].iter().enumerate() {
            let got = t.ten_moment(&q)[0].iter().find(|(d, _)| *d == k).map(|(_, m)| *m).unwrap_or(0.0);
            assert!((got - want).abs() < 1e-12, "ten_J[{k}] {got} vs {want}");
        }
        assert!((t.tendon_invweight0()[0] - 1.2485132).abs() < 1e-6, "{:?}", t.tendon_invweight0());
        // the spring and damper, carried back through that same moment
        let passive = t.qfrc_passive(&q, &qd);
        for (k, want) in [-2.1022457570735895, -17.5276185468612].iter().enumerate() {
            assert!((passive[k] - want).abs() < 1e-10, "qfrc_passive[{k}] {} vs {want}", passive[k]);
        }
        // a PULLEY still is not carried: it splits the cable into branches with their own divisors, which
        // is a different object from a path, and naming it is better than quietly halving a force
        let pulley = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><worldbody>
  <site name="anchor" pos="0 0 1"/>
  <body name="a"><joint name="j1" type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <site name="s1" pos="0.2 0 0"/><site name="s2" pos="0.3 0 0"/></body>
</worldbody>
<tendon><spatial name="split"><site site="anchor"/><pulley divisor="2"/><site site="s1"/><site site="s2"/></spatial></tendon></mujoco>"#,
        )
        .unwrap();
        assert_eq!(pulley.tendons.len(), 0);
        assert_eq!(pulley.tendons_unsupported.len(), 1, "{:?}", pulley.tendons_unsupported);
    }

    /// **A tendon wrapping a sphere and a cylinder, against MuJoCo's own numbers.** 291 of Menagerie's 881
    /// spatial tendons bend around a geom; every muscle in `ms_human_700` that crosses a joint does.
    #[test]
    fn a_spatial_tendon_wraps_a_sphere_and_a_cylinder() {
        let t = tree_from_mjcf_str(WRAP_MODEL).unwrap();
        assert!(t.tendons_unsupported.is_empty(), "{:?}", t.tendons_unsupported);
        let q = [0.35, 0.12];
        let (len, mom) = (t.ten_length(&q), t.ten_moment(&q));
        let want: [(f64, [f64; 2]); 2] = [
            (1.2200391927982945, [-0.05058065845892602, 0.9632246154822138]),
            (1.217413004230632, [-0.026493351837825998, 0.9684715020258167]),
        ];
        for (i, (wl, wj)) in want.iter().enumerate() {
            assert!((len[i] - wl).abs() < 1e-9, "tendon {i} length {} vs {wl}", len[i]);
            for (k, w) in wj.iter().enumerate() {
                let got = mom[i].iter().find(|(d, _)| *d == k).map(|(_, m)| *m).unwrap_or(0.0);
                assert!((got - w).abs() < 1e-9, "tendon {i} J[{k}] {got} vs {w}");
            }
        }
        // ⛔ and the OTHER way round. A `sidesite` on the far side of both obstacles must send the cable
        // there instead — same model, same state, a longer path and a different sign on the first dof.
        let flipped = tree_from_mjcf_str(&WRAP_MODEL.replace(r#"<site name="up" pos="0 0.5 0"/>"#, r#"<site name="down" pos="0 -0.5 0"/>"#).replace(r#"<geom geom="ball"/>"#, r#"<geom geom="ball" sidesite="down"/>"#).replace(r#"<geom geom="drum"/>"#, r#"<geom geom="drum" sidesite="down"/>"#)).unwrap();
        let (len, mom) = (flipped.ten_length(&q), flipped.ten_moment(&q));
        let want: [(f64, [f64; 2]); 2] = [
            (1.2848340762814132, [0.047479985220722644, 0.994814026993209]),
            (1.270119237142905, [-0.05860572178093858, 0.9986136828818115]),
        ];
        for (i, (wl, wj)) in want.iter().enumerate() {
            assert!((len[i] - wl).abs() < 1e-9, "sided tendon {i} length {} vs {wl}", len[i]);
            for (k, w) in wj.iter().enumerate() {
                let got = mom[i].iter().find(|(d, _)| *d == k).map(|(_, m)| *m).unwrap_or(0.0);
                assert!((got - w).abs() < 1e-9, "sided tendon {i} J[{k}] {got} vs {w}");
            }
        }
    }

    const WRAP_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <site name="a0" pos="-0.4 0.05 0.3"/>
  <body name="arm" pos="0 0 0.28" euler="0.1 -0.2 0.3"><joint name="j1" type="hinge" axis="0 1 0"/>
    <inertial pos="0.1 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom name="ball" type="sphere" size="0.12" pos="0.02 0.01 0"/>
    <geom name="drum" type="cylinder" size="0.1 0.3" pos="0.05 -0.02 0.02" euler="0.3 0.2 -0.1"/>
    <site name="up" pos="0 0.5 0"/>
    <body name="fore" pos="0.5 0 0"><joint name="j2" type="slide" axis="1 0 0"/>
      <inertial pos="0.1 0 0" mass="0.6" diaginertia="0.01 0.01 0.01"/>
      <site name="a1" pos="0.2 -0.1 -0.03"/>
    </body>
  </body>
</worldbody>
<tendon>
  <spatial name="sph"><site site="a0"/><geom geom="ball"/><site site="a1"/></spatial>
  <spatial name="cyl"><site site="a0"/><geom geom="drum"/><site site="a1"/></spatial>
</tendon></mujoco>"#;

    /// **A site transmission, against MuJoCo's own numbers.** `gear` is a WRENCH in the site's frame, so
    /// the moment turns with the model and cannot be resolved when the file is read — every quadrotor in
    /// Menagerie is actuated this way.
    ///
    /// ⛔ `actuator_length` is ZERO however the model is posed, and `actuator_velocity` is NOT: it is the
    /// ordinary `moment·q̇`. Zeroing both would put a velocity servo on a thruster exactly backwards, and
    /// nothing in a `<motor>` — fixed gain, no bias — would show it.
    #[test]
    fn a_site_transmission_is_a_wrench_in_the_sites_frame() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <body name="a" pos="0 0 0.5" euler="0.1 -0.2 0.3"><joint name="j1" type="hinge" axis="0 1 0"/>
    <inertial pos="0.2 0 0" mass="1.5" diaginertia="0.03 0.05 0.07"/>
    <body name="b" pos="0.4 0 0" euler="0.3 0.1 -0.2"><joint name="j2" type="slide" axis="1 0 0"/>
      <inertial pos="0.1 0 0" mass="0.8" diaginertia="0.01 0.014 0.02"/>
      <site name="tip" pos="0.15 -0.03 0.02" euler="0.4 -0.1 0.25"/>
    </body>
  </body>
</worldbody>
<actuator><motor name="thrust" site="tip" gear="0 0 1 0.02 -0.01 0.03" ctrlrange="-5 5"/></actuator></mujoco>"#,
        )
        .unwrap();
        assert!(t.actuators_unsupported.is_empty(), "{:?}", t.actuators_unsupported);
        let (q, qd) = ([0.4, 0.15], [0.9, -0.6]);
        let st = t.actuator_state(&q, &qd).remove(0);
        for (k, want) in [-0.5622785719716145, -0.09983341664682814].iter().enumerate() {
            let got = st.moment.iter().find(|(d, _)| *d == k).map(|(_, m)| *m).unwrap_or(0.0);
            assert!((got - want).abs() < 1e-12, "moment[{k}] {got} vs {want}");
        }
        assert_eq!(st.length, 0.0, "a site transmission has no length");
        assert!((st.velocity + 0.44615066449714825).abs() < 1e-9, "{}", st.velocity);
        let f = t.qfrc_actuator(&q, &qd, &[2.5]);
        for (k, want) in [-1.4056964299290362, -0.24958354161707036].iter().enumerate() {
            assert!((f[k] - want).abs() < 1e-12, "qfrc[{k}] {} vs {want}", f[k]);
        }
    }

    /// **A floating base in MuJoCo's basis, against MuJoCo's own mass matrix.** A free joint is six
    /// coordinates here and six there but not the same six, so `M_ours = Tᵀ·M_mujoco·T` is the identity that
    /// says the two are the same physics — and it is also the only way to put `dof_armature` where MuJoCo
    /// puts it, because a DIAGONAL in its basis is not one in ours.
    ///
    /// Numbers from MuJoCo 3.13.0 on this model, at a pose with the base rotated so the two bases differ.
    #[test]
    fn a_free_joints_mass_matrix_is_mujocos_through_the_basis() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>
<body name="base" pos="0 0 1"><joint name="root" type="free" armature="0.02"/>
  <inertial pos="0.05 -0.02 0.01" quat="0.9 0.2 -0.3 0.1" mass="3" diaginertia="0.05 0.08 0.11"/>
  <body name="arm" pos="0.2 0 0"><joint name="j1" type="hinge" axis="0 1 0" armature="0.004"/>
    <inertial pos="0.1 0 0" mass="1.2" diaginertia="0.01 0.012 0.014"/>
  </body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let qpos = [0.3, -0.2, 1.1, 0.9233805168766387, 0.20519567041703082, -0.3077935056255462, 0.10259783520851541, 0.4];
        let q = t.q_from_qpos(&qpos, &[0, 7]).unwrap();
        #[rustfmt::skip]
        let mj: [[f64; 7]; 7] = [
            [4.22, 0.0, 0.0, 0.026295725975672157, 0.25022737771830994, -0.11069283766747774, 0.021280009297257315],
            [0.0, 4.22, 0.0, 0.04149544306892891, 0.22022911792960326, 0.4516297067245203, 0.04591332845591904],
            [0.0, 0.0, 4.22, -0.038295725975672164, -0.3737551230822326, 0.1946928376674777, -0.10880775466117998],
            [0.026295725975672157, 0.04149544306892891, -0.038295725975672164, 0.10353853469087368, 0.00848476454293631, -0.012343088896731358, 0.0],
            [0.25022737771830994, 0.22022911792960326, -0.3737551230822326, 0.00848476454293631, 0.22975497203346804, -0.010190027700831016, 0.04610546385606923],
            [-0.11069283766747774, 0.4516297067245203, 0.1946928376674777, -0.012343088896731358, -0.010190027700831016, 0.22912834869993517, 0.0],
            [0.021280009297257315, 0.04591332845591904, -0.10880775466117998, 0.0, 0.04610546385606923, 0.0, 0.028000000000000004],
        ];
        let theirs = nalgebra::DMatrix::from_fn(7, 7, |r, c| mj[r][c]);
        let tm = t.free_basis(&q);
        let mapped = tm.transpose() * &theirs * &tm;
        let ours = t.mass_matrix(&q);
        for r in 0..7 {
            for c in 0..7 {
                assert!((ours[(r, c)] - mapped[(r, c)]).abs() < 1e-12, "M[{r},{c}] {} vs {}", ours[(r, c)], mapped[(r, c)]);
            }
        }
        // and the acceleration follows: at REST the basis change has no velocity term, so `a_mujoco = T·a_ours`
        let zero = vec![0.0; 7];
        let bias = crate::tree_dynamics::tree_inverse_dynamics(&t.tree.joints, &t.tree.inertia, &t.tree.parent, &q, &zero, &zero, t.gravity);
        let a_ours = ours.clone().try_inverse().unwrap() * nalgebra::DVector::from_iterator(7, bias.iter().map(|b| -b));
        let a_mj = tm * a_ours;
        let want = [-0.006566036987102614, -0.00018945353024978957, -9.751510865034783, 0.011093993260240298, 0.0795308742763936, -0.04836289437347041, 0.10163169173360131];
        for (i, w) in want.iter().enumerate() {
            assert!((a_mj[i] - w).abs() < 1e-9, "qacc_smooth[{i}] {} vs {w}", a_mj[i]);
        }
        // ⛔ and the armature is NOT a diagonal here: putting it on this port's own diagonal instead is what
        // made a free-based model's acceleration 8.2e-2 out, and nothing about the pose says so
        assert!(ours[(3, 4)].abs() > 1e-6, "the base block is coupled: {}", ours[(3, 4)]);

        // ⛔⛔ MOVING, with damping too. `a_mujoco = T·a_ours + Ṫ·v_ours` — and a free joint's DAMPING is a
        // per-dof force in MuJoCo's basis exactly like its armature, so both halves of the change of basis
        // are exercised here and neither is at rest. Numbers from MuJoCo 3.13.0 on the same model.
        let moving = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>
<body name="base" pos="0 0 1"><joint name="root" type="free" armature="0.02" damping="1.5"/>
  <inertial pos="0.05 -0.02 0.01" quat="0.9 0.2 -0.3 0.1" mass="3" diaginertia="0.05 0.08 0.11"/>
  <body name="arm" pos="0.2 0 0"><joint name="j1" type="hinge" axis="0 1 0" armature="0.004" damping="0.2"/>
    <inertial pos="0.1 0 0" mass="1.2" diaginertia="0.01 0.012 0.014"/>
  </body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let qm = moving.q_from_qpos(&qpos, &[0, 7]).unwrap();
        let v = [0.4, -0.3, 0.25, 0.7, -0.5, 0.9, -0.6];
        let got = moving.qacc_smooth_mujoco(&qm, &v, &[]).unwrap();
        let want = [-0.40776163054058184, 1.0160972487730715, -9.159571401180923, -11.697022186283133, 4.64153544898943, -8.757526118453406, -1.8570604740614982];
        for (i, w) in want.iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-9, "moving qacc_smooth[{i}] {} vs {w}", got[i]);
        }
        // without the velocity term the answer is wrong, and only while moving — the check that it is wired
        let still = moving.qacc_smooth_mujoco(&qm, &[0.0; 7], &[]).unwrap();
        assert!(still.iter().zip(&got).any(|(a, b)| (a - b).abs() > 1.0), "the velocity term does something");
    }

    /// **A BALL joint's mass matrix, through the same change of basis.** MuJoCo's three dofs are an angular
    /// velocity; this port's are three Euler hinges. If `M_ours = Tᵀ·M_mujoco·T` holds here too, the map is
    /// the same one a free joint needs, and a model with both — `agility_cassie` — is comparable.
    #[test]
    fn a_ball_joints_mass_matrix_is_mujocos_through_the_basis() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 -9.81"/><worldbody>
<body name="a" pos="0 0 1" euler="0.2 -0.1 0.3"><joint name="b1" type="ball"/>
  <inertial pos="0.1 0.02 -0.03" quat="0.9 0.2 -0.3 0.1" mass="2" diaginertia="0.04 0.06 0.09"/>
  <body name="c" pos="0.25 0 0"><joint name="h" type="hinge" axis="0 1 0"/>
    <inertial pos="0.1 0 0" mass="0.8" diaginertia="0.01 0.012 0.015"/></body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let qpos = [0.8295613557843402, 0.3110855084191276, -0.4147806778921701, 0.20739033894608505, 0.35];
        let q = t.q_from_qpos(&qpos, &[0, 4]).unwrap();
        #[rustfmt::skip]
        let mj: [[f64; 4]; 4] = [
            [0.0699734011289112, 0.0019833795013850285, -0.004062662124843231, 0.0],
            [0.0019833795013850285, 0.19515884203190065, -0.009204432132963989, 0.038787454256947594],
            [-0.004062662124843231, -0.009204432132963989, 0.19821757386697844, 0.0],
            [0.0, 0.038787454256947594, 0.0, 0.02000000000000001],
        ];
        let theirs = nalgebra::DMatrix::from_fn(4, 4, |r, c| mj[r][c]);
        let tm = t.free_basis(&q);
        let mapped = tm.transpose() * &theirs * &tm;
        let ours = t.mass_matrix(&q);
        for r in 0..4 {
            for c in 0..4 {
                assert!((ours[(r, c)] - mapped[(r, c)]).abs() < 1e-12, "M[{r},{c}] {} vs {}", ours[(r, c)], mapped[(r, c)]);
            }
        }
        let want = [-6.46750362599345, 9.001436644526741, -6.399013262157424, 4.021208453117406];
        let got = t.qacc_smooth_mujoco(&q, &[0.0; 4], &[]).unwrap();
        for (i, w) in want.iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-9, "qacc_smooth[{i}] {} vs {w}", got[i]);
        }
    }

    /// **The inertia-box fluid model, against MuJoCo's own numbers.** Every coefficient was fitted one term
    /// at a time — angular viscous, linear viscous, blunt force, blunt torque — and then checked together on
    /// a body with a rotated inertial frame and an offset centre of mass, where the two agree to 1.1e-16.
    ///
    /// ⛔ `wind` makes the force NON-ZERO AT REST, which is the one state a sweep of static poses samples:
    /// a port that omitted the whole model, or that dropped only the wind, would look identical at `q̇ = 0`
    /// without it. Both states are checked here for that reason.
    #[test]
    fn the_inertia_box_fluid_model_is_mujocos() {
        let xml = r#"<mujoco><compiler angle="radian"/>
<option gravity="0 0 0" density="1.3" viscosity="0.7" wind="0.5 -0.3 0.2"/>
<worldbody>
  <body name="a" pos="0 0 0.5" euler="0.1 -0.2 0.3"><joint name="j1" type="hinge" axis="0 1 0"/>
    <inertial pos="0.2 0.05 -0.03" euler="0.2 0.5 -0.1" mass="1.5" diaginertia="0.03 0.05 0.07"/>
    <body name="b" pos="0.4 0 0" euler="0.3 0.1 -0.2"><joint name="j2" type="slide" axis="1 0 0"/>
      <inertial pos="0.1 -0.02 0.04" euler="-0.3 0.2 0.4" mass="0.8" diaginertia="0.01 0.014 0.02"/>
    </body>
  </body>
  <body name="fixed" pos="1 0 0.5"><inertial pos="0 0 0" mass="3" diaginertia="0.1 0.1 0.1"/></body>
</worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert_eq!((t.density, t.viscosity), (1.3, 0.7));
        let q = [0.4, 0.15];
        let moving = t.qfrc_passive(&q, &[0.9, -0.6]);
        for (i, want) in [-1.6053610426517502, 1.863097416870217].iter().enumerate() {
            assert!((moving[i] - want).abs() < 1e-12, "dof {i} moving: {} vs {want}", moving[i]);
        }
        // at rest the WIND still blows, and a body welded to the world still contributes nothing
        let at_rest = t.qfrc_passive(&q, &[0.0, 0.0]);
        for (i, want) in [-0.5802116054482513, 0.7616013062115348].iter().enumerate() {
            assert!((at_rest[i] - want).abs() < 1e-12, "dof {i} at rest: {} vs {want}", at_rest[i]);
        }
        // with no medium there is no force at all, whatever the state
        let still = tree_from_mjcf_str(&xml.replace(r#"density="1.3" viscosity="0.7" wind="0.5 -0.3 0.2""#, "")).unwrap();
        assert_eq!(still.qfrc_passive(&q, &[0.9, -0.6]), vec![0.0, 0.0]);
    }

    /// **A `connect` equality's three rows, against MuJoCo's own numbers.** This is the closed kinematic
    /// loop a tree cannot express — every four-bar gripper in Menagerie is built from it.
    ///
    /// ⛔ The anchor is stated in body1's frame ONLY. MuJoCo resolves it to a world point at `qpos0` and
    /// stores body2's local copy, so the two sides track different points as the model moves and the
    /// residual `p₁ − p₂` is what closes the loop. A port that uses body1's anchor for both is exactly right
    /// at `qpos0`, which is the one pose a careless test would sample.
    ///
    /// Numbers from MuJoCo 3.13.0 on exactly this model, at a pose and a velocity away from the reference.
    #[test]
    fn a_connect_equalitys_rows_are_mujocos() {
        let xml = r#"<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
<body name="a" pos="0 0 0.5"><joint name="j1" type="hinge" axis="0 1 0"/>
  <inertial pos="0.2 0 0" mass="1.5" diaginertia="0.03 0.02 0.04"/>
  <body name="b" pos="0.4 0 0"><joint name="j2" type="hinge" axis="0 1 0"/>
    <inertial pos="0.15 0 0" mass="0.7" diaginertia="0.01 0.012 0.008"/>
  </body>
</body>
<body name="c" pos="0.3 0.2 0.5"><joint name="j3" type="slide" axis="1 0 0"/>
  <inertial pos="0 0 0" mass="0.9" diaginertia="0.02 0.02 0.02"/>
</body></worldbody>
<equality><connect name="cn" body1="b" body2="c" anchor="0.05 -0.02 0.03" solref="0.01 0.9" solimp="0.8 0.93 0.002 0.4 3"/></equality>
</mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert!(t.equalities_unsupported.is_empty(), "{:?}", t.equalities_unsupported);
        let (q, qd) = ([0.3, -0.2, 0.07], [0.4, -0.25, 0.1]);
        let set = t.joint_constraint_rows(&q, &qd, &t.dof_invweight0());
        assert_eq!(set.blocks.len(), 3, "a connect is three rows");
        let want_j = [[-0.09334963, 0.02485845, -1.0], [0.0, 0.0, 0.0], [-0.43487981, -0.05274521, 0.0]];
        let want_pos = [-0.08512019358645151, -1.0408340855860843e-17, -0.12334962853853648];
        let want_aref = [1160.8983288148454, 1.3816993038445295e-13, 1672.0118010798021];
        for r in 0..3 {
            for k in 0..3 {
                assert!((set.jac[(r, k)] - want_j[r][k]).abs() < 1e-8, "J[{r},{k}] {} vs {}", set.jac[(r, k)], want_j[r][k]);
            }
            assert!((set.aref[r] - want_aref[r]).abs() < 1e-6 * want_aref[r].abs().max(1.0), "aref[{r}] {}", set.aref[r]);
            // every row of one connect shares a diagonal: the two bodies' translational invweight0, summed
            assert!((set.d[r] - 9.105824175824182).abs() < 1e-8, "D[{r}] {}", set.d[r]);
        }
        let _ = want_pos;
        // ⛔ the same model with a `width` wide enough that the impedance is NOT saturated — the only
        // setting where "the norm of the residual" and "this row's own component" give different numbers.
        // All three rows share one `efc_D`, because a ball joint is one constraint.
        let wide = tree_from_mjcf_str(&xml.replace(r#"solimp="0.8 0.93 0.002 0.4 3""#, r#"solimp="0.1 0.95 0.5 0.5 2""#)).unwrap();
        let w = wide.joint_constraint_rows(&q, &qd, &wide.dof_invweight0());
        for r in 0..3 {
            assert!((w.d[r] - 0.23180246194304452).abs() < 1e-9, "D[{r}] {}", w.d[r]);
        }
        for (r, want) in [324.5641467374884, 3.598399633537263e-14, 460.2746976464331].iter().enumerate() {
            assert!((w.aref[r] - want).abs() < 1e-6 * want.abs().max(1.0), "aref[{r}] {} vs {want}", w.aref[r]);
        }
        // and at qpos0 the loop is closed, which is exactly where the wrong reading also looks right
        let z = [0.0, 0.0, 0.0];
        let at0 = t.joint_constraint_rows(&z, &z, &t.dof_invweight0());
        for r in 0..3 {
            assert!(at0.aref[r].abs() < 1e-9, "row {r} at qpos0: {}", at0.aref[r]);
        }
    }

    /// **A fixed tendon's limit row, against MuJoCo's own numbers.** Nothing in Menagerie can check this:
    /// the two models whose tendons produce `limit_tendon` rows are `toddlerbot`, which has a free base and
    /// so no dof correspondence, and `robotiq_2f85`, whose rows come with `connect` equalities this loader
    /// does not build. A corpus that exercises a feature is not the same as a corpus that CHECKS it, so the
    /// numbers below come from MuJoCo 3.13.0 run on exactly this model.
    ///
    /// ⛔ The row's Jacobian is `−side·coef`, its position `side·(range − L)` and its velocity `J·q̇`, all in
    /// the tendon's own length units — `<compiler angle>` does not touch a tendon range even when every
    /// joint it names is a hinge.
    #[test]
    fn a_fixed_tendons_limit_row_is_mujocos() {
        let xml = r#"<mujoco><compiler angle="radian"/><option timestep="0.002"/><worldbody>
<body name="l1" pos="0 0 0.5"><joint name="j1" type="hinge" axis="0 1 0" damping="0"/>
  <inertial pos="0.2 0 0" mass="1.5" diaginertia="0.03 0.02 0.04"/>
  <body name="l2" pos="0.4 0 0"><joint name="j2" type="slide" axis="1 0 0"/>
    <inertial pos="0.1 0 0" mass="0.7" diaginertia="0.01 0.012 0.008"/>
  </body>
</body></worldbody>
<tendon><fixed name="coup" limited="true" range="-0.15 0.25" margin="0.02" solreflimit="0.01 0.9" solimplimit="0.8 0.93 0.002 0.4 3">
  <joint joint="j1" coef="0.6"/><joint joint="j2" coef="-1.3"/>
</fixed></tendon></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let (q, qd) = ([0.7, -0.12], [0.35, -0.2]);
        assert!((t.ten_length(&q)[0] - 0.576).abs() < 1e-15, "{:?}", t.ten_length(&q));
        assert!((t.ten_velocity(&q, &qd)[0] - 0.47).abs() < 1e-15, "{:?}", t.ten_velocity(&q, &qd));
        assert!((t.tendon_invweight0()[0] - 3.76260032).abs() < 1e-8, "{:?}", t.tendon_invweight0());
        let set = t.joint_constraint_rows(&q, &qd, &t.dof_invweight0());
        assert_eq!(set.blocks.len(), 1, "only the UPPER side is within margin");
        assert!((set.jac[(0, 0)] + 0.6).abs() < 1e-15 && (set.jac[(0, 1)] - 1.3).abs() < 1e-15, "{}", set.jac);
        assert!((set.aref[0] - 4694.1988583565635).abs() < 1e-8, "aref {}", set.aref[0]);
        assert!((set.d[0] - 3.5309927050893757).abs() < 1e-9, "D {}", set.d[0]);
        // at a length INSIDE the range by more than the margin there is no row at all
        assert_eq!(t.joint_constraint_rows(&[0.1, 0.0], &qd, &t.dof_invweight0()).blocks.len(), 0);
    }

    /// **`springlength` is a BAND**: between its two values a tendon is slack and pulls nothing. A single
    /// stated value fills both ends, which is the ordinary spring; two values are a deadband, and a port
    /// that keeps one resting length turns a slack tendon into one that is always pulling.
    #[test]
    fn a_tendon_spring_is_slack_inside_its_band() {
        let model = |spring: &str| {
            let xml = format!(
                r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="l1"><joint name="j1" type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/></body>
</worldbody><tendon><fixed name="s" stiffness="10" damping="2" springlength="{spring}">
  <joint joint="j1" coef="1"/></fixed></tendon></mujoco>"#
            );
            tree_from_mjcf_str(&xml).unwrap()
        };
        let band = model("0.1 0.3");
        assert_eq!(band.qfrc_passive(&[0.2], &[0.0])[0], 0.0, "slack inside the band");
        assert!((band.qfrc_passive(&[0.5], &[0.0])[0] + 10.0 * 0.2).abs() < 1e-12, "pulled back from above");
        assert!((band.qfrc_passive(&[0.0], &[0.0])[0] - 10.0 * 0.1).abs() < 1e-12, "pushed out from below");
        // the damper does not care where in the band it is
        assert!((band.qfrc_passive(&[0.2], &[3.0])[0] + 2.0 * 3.0).abs() < 1e-12);
        // one value is one resting length, and then there is no slack anywhere
        let single = model("0.2");
        assert_eq!(single.qfrc_passive(&[0.2], &[0.0])[0], 0.0);
        assert!((single.qfrc_passive(&[0.25], &[0.0])[0] + 10.0 * 0.05).abs() < 1e-12);
    }

    /// **A tendon transmission spreads one actuator's force over every joint the tendon names.** Menagerie
    /// checks the numbers (2,814 of 2,814 actuators, `actuator_length` to 5.6e-17); this pins the shape.
    #[test]
    fn a_tendon_actuator_pulls_every_joint_the_tendon_names() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="a"><joint name="j1" type="slide" axis="1 0 0"/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  <body name="b"><joint name="j2" type="slide" axis="0 1 0"/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/></body>
</body></worldbody>
<tendon><fixed name="split"><joint joint="j1" coef="0.5"/><joint joint="j2" coef="0.25"/></fixed></tendon>
<actuator><general name="drive" tendon="split" gear="3"/></actuator></mujoco>"#,
        )
        .unwrap();
        assert_eq!(t.actuators.len(), 1, "{:?}", t.actuators_unsupported);
        let a = &t.actuators[0];
        assert_eq!(a.moment, vec![(0, 1.5), (1, 0.75)], "gear times each coefficient");
        // actuator_length = gear·(0.5·q₁ + 0.25·q₂)
        assert!((a.length(&[2.0, 4.0]) - 3.0 * (0.5 * 2.0 + 0.25 * 4.0)).abs() < 1e-15);
        let f = crate::mujoco_actuator::qfrc_actuator(&t.actuators, &[0.0, 0.0], &[0.0, 0.0], &[2.0], &[None, None]);
        assert_eq!(f, vec![3.0, 1.5], "one control, two dofs, in the tendon's proportions");
    }

    /// **Two geoms welded onto the same link belong to two different BODIES**, and MuJoCo gives each its own
    /// `body_invweight0` — the mean diagonal of `J M⁻¹ Jᵀ` at THAT body's centre of mass.
    ///
    /// For one hinge about `z`, the point Jacobian at radius `r` is `(−y, x, 0)`, so the translational
    /// inverse weight is `r²/(3M)` and two welded tips at radii 0.5 and 0.2 stand in the ratio 6.25 exactly.
    /// The mutation this refuses: keying the inverse weights by the tree's dof instead of by the body. Both
    /// tips ride the same dof, so they would come out equal, every contact between them would get the same
    /// regularisation, and nothing about the load would look wrong.
    #[test]
    fn a_welded_body_has_its_own_inverse_weight() {
        let xml = r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="link"><joint name="j" type="hinge" axis="0 0 1"/>
  <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
  <body name="far" pos="0.5 0 0"><inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/></body>
  <body name="near" pos="0.2 0 0"><inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01"/></body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let iw = t.body_invweight0();
        let (far, near, link) = (iw["far"], iw["near"], iw["link"]);
        assert!((far.tran / near.tran - 6.25).abs() < 1e-12, "r² scaling: {} vs {}", far.tran, near.tran);
        assert_eq!(link.tran, 0.0, "the link's own centre of mass is ON the hinge axis");
        // the angular Jacobian belongs to the dof, not to the body, so THAT part they do share
        assert_eq!(far.rot, near.rot);
        assert_eq!(far.rot, link.rot);
        // a body welded to the world has no entry at all, and a contact on it takes InvWeight::STATIC
        let fixed = tree_from_mjcf_str(r#"<mujoco><worldbody><body name="post"><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"#).unwrap();
        assert!(!fixed.body_invweight0().contains_key("post"), "a world-welded body is not movable");
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


