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
    /// `springdamper="timeconst dampratio"`: when given, `stiffness` and `damping` are NOT what the file
    /// says but are COMPUTED at compile from the joint's inverse weight (`mjCModel::AutoSpringDamper`), and
    /// the two fields above already hold the computed values.
    pub springdamper: Option<[f64; 2]>,
}

impl MjcfTree {
    /// A body's inertial frame in its own frame as the kinematics use it
    /// ([`MjcfTree::body_inertial_runtime`]); the body frame for a body with none recorded.
    pub(crate) fn inertial_rt(&self, body: &str) -> (Vector3<f64>, Matrix3<f64>) {
        self.body_inertial_runtime.get(body).copied().unwrap_or((Vector3::zeros(), Matrix3::identity()))
    }

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
    pub fn qacc_smooth_mujoco(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64]) -> Option<Vec<f64>> {
        self.qacc_smooth_mujoco_at(q, v_mujoco, ctrl, act, 0.0)
    }

    /// [`MjcfTree::qacc_smooth_mujoco`] at `time` (`d.time`, which a PID's slew limiter reads).
    pub fn qacc_smooth_mujoco_at(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Option<Vec<f64>> {
        self.qacc_smooth_impl(q, v_mujoco, ctrl, act, time, &MjcfApplied::default())
    }

    fn qacc_smooth_impl(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied) -> Option<Vec<f64>> {
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
        let act = self.qfrc_actuator_at(q, &qd, ctrl, act, time);
        let tdot_v = self.free_basis_dot(q, &qd) * &v;
        let c_ours = nalgebra::DVector::from_iterator(nv, (0..nv).map(|i| bias[i])) - &m_rigid * (&tinv * tdot_v);
        let m_mj = tinv.transpose() * &m_full * &tinv;
        let passive = self.qfrc_passive_mujoco(q, v_mujoco)?;
        let mut rhs = nalgebra::DVector::from_row_slice(&passive) + tinv.transpose() * (nalgebra::DVector::from_iterator(nv, (0..nv).map(|i| act[i])) - c_ours);
        // `qfrc_applied` (MuJoCo's basis already) and `xfrc_applied` (`mj_xfrcAccumulate`: each body's
        // world force and torque at its centre of mass, through its Jacobians)
        for (i, f) in applied.qfrc.iter().enumerate().take(nv) {
            rhs[i] += f;
        }
        if !applied.xfrc.is_empty() {
            let frames = crate::tree_frames(&self.tree, q);
            let mut xfrc = nalgebra::DVector::<f64>::zeros(nv);
            for (body, w) in &applied.xfrc {
                let Some((ride, pre)) = self.body_frames.get(body) else { continue };
                let com = (frames[*ride] * pre * Point3::from(self.inertial_rt(body).0)).coords;
                let jl = crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(*ride), com);
                let ja = crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, &frames, Some(*ride));
                xfrc += jl.transpose() * Vector3::new(w[0], w[1], w[2]) + ja.transpose() * Vector3::new(w[3], w[4], w[5]);
            }
            rhs += tinv.transpose() * xfrc;
        }
        Some((m_mj.try_inverse()? * rhs).iter().copied().collect())
    }

    /// **`qfrc_bias` in MUJOCO'S coordinates** — the rigid-body Coriolis, centrifugal and gravity force at a
    /// velocity given in MuJoCo's basis: `T⁻ᵀ·(C_ours − M_rigid·T⁻¹·Ṫ·v_ours)`, as
    /// [`MjcfTree::qacc_smooth_mujoco`] forms it. `None` at gimbal lock.
    pub fn qfrc_bias_mujoco(&self, q: &[f64], v_mujoco: &[f64]) -> Option<nalgebra::DVector<f64>> {
        let nv = self.tree.joints.len();
        let tinv = self.free_basis(q).try_inverse()?;
        let v = &tinv * nalgebra::DVector::from_row_slice(v_mujoco);
        let qd: Vec<f64> = v.iter().copied().collect();
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
        let tdot_v = self.free_basis_dot(q, &qd) * &v;
        let c_ours = nalgebra::DVector::from_iterator(nv, (0..nv).map(|i| bias[i])) - &m_rigid * (&tinv * tdot_v);
        Some(tinv.transpose() * c_ours)
    }

    /// **`body_weldid`, `body_dofnum` and the parent's weld, per body** — what MuJoCo's contact filter reads.
    ///
    /// A body's WELD body is itself when it has joints, and otherwise the nearest ancestor that has any;
    /// `0` means welded to the world. Two geoms are not collided when their weld bodies are the same, when
    /// neither weld has dofs, or when one weld is the other's parent weld.
    ///
    /// ⛔⛔ The obvious shortcut — "the tree dof a geom rides on IS its body" — is wrong the moment a body
    /// has MORE THAN ONE joint: the ride dof's PARENT is then the body's own previous dof, not the parent
    /// body's, so a child reads as unrelated to its parent and the pair is collided. On
    /// `hello_robot_stretch` the rubber tips have two joints each and collided with the fingers they hang
    /// off, which MuJoCo filters out.
    ///
    /// Returned as `(weld, parent weld, dofnum of the weld)` keyed by body name, with the same numbering on
    /// both sides so they can be compared.
    pub fn body_welds(&self) -> BTreeMap<String, (usize, usize, usize)> {
        let mut dofs: BTreeMap<&str, usize> = BTreeMap::new();
        for j in &self.joints {
            *dofs.entry(j.body.as_str()).or_default() += 1;
        }
        // stable ids: 0 is the world, then bodies in name order
        let ids: BTreeMap<&str, usize> = self.body_parent.keys().enumerate().map(|(i, b)| (b.as_str(), i + 1)).collect();
        fn weld_of<'a>(mut b: &'a str, dofs: &BTreeMap<&str, usize>, parents: &'a BTreeMap<String, String>) -> &'a str {
            let mut guard = 0;
            while dofs.get(b).copied().unwrap_or(0) == 0 {
                match parents.get(b) {
                    Some(p) if p != "world" => b = p.as_str(),
                    _ => return "world",
                }
                guard += 1;
                if guard > 1024 {
                    return "world";
                }
            }
            b
        }
        self.body_parent
            .keys()
            .map(|b| {
                let w = weld_of(b, &dofs, &self.body_parent);
                // ⛔⛔ the parent of the WELD, not the parent of the BODY. MuJoCo's filter asks
                // `body_weldid[body_parentid[body_weldid[b]]]`, and for a JOINTLESS body those are
                // different bodies: `aloha`'s `left/gripper_base` hangs off `left/gripper_link`, so the
                // parent of the body is its own weld and the pair with `left/wrist_link` never looks like
                // parent and child. MuJoCo filters it; this port collided it, at 3.7 mm of penetration.
                // For a body that HAS joints the two readings coincide, which is why it hid.
                let pw = match self.body_parent.get(w) {
                    Some(p) if p != "world" => weld_of(p.as_str(), &dofs, &self.body_parent),
                    _ => "world",
                };
                (b.clone(), (ids.get(w).copied().unwrap_or(0), ids.get(pw).copied().unwrap_or(0), dofs.get(w).copied().unwrap_or(0)))
            })
            .collect()
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

    /// **Which obstacles a spatial tendon actually wraps at this state, and where it touches them** — the
    /// obstacle's name with its two tangent points, in path order. MuJoCo publishes the same thing in
    /// `d.wrap_obj`/`d.wrap_xpos`, so the two paths can be compared by WHAT they wrapped and not merely by
    /// how many: two paths that wrap the same number of obstacles can be wrapping different ones.
    pub fn tendon_wraps(&self, i: usize, q: &[f64]) -> Vec<(String, [Vector3<f64>; 2])> {
        let TendonPath::Spatial(pts) = &self.tendons[i].path else { return Vec::new() };
        let frames = crate::tree_frames(&self.tree, q);
        let path = self.spatial_path(pts, &frames);
        path.windows(2)
            .filter(|w| w[0].arc.is_some())
            .filter_map(|w| w[0].obstacle.clone().map(|g| (g, [w[0].pos, w[1].pos])))
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
                                out.push(PathPoint { ride: g.joint, pos: w0, arc: Some(arc), obstacle: Some(geom.clone()) });
                                out.push(PathPoint { ride: g.joint, pos: w1, arc: None, obstacle: Some(geom.clone()) });
                            }
                        }
                    }
                    out.push(PathPoint { ride, pos: here, arc: None, obstacle: None });
                }
            }
        }
        out
    }

    /// **`tendon_invweight0`**: `J M⁻¹ Jᵀ` at `qpos0` for each fixed tendon's (constant) Jacobian row — the
    /// scale a tendon limit row regularises against, the tendon counterpart of `dof_invweight0`. Cached once
    /// by MuJoCo, like every other `*_invweight0`.
    pub fn tendon_invweight0(&self) -> Vec<f64> {
        self.invweight0().tendon.clone()
    }

    fn compute_tendon_invweight0(&self) -> Vec<f64> {
        let nv = self.tree.joints.len();
        let Some(chol) = self.mass_matrix(&self.reference_q).cholesky() else {
            return vec![0.0; self.tendons.len()];
        };
        // ⛔ BOTH at the reference pose. Taking the moment at `q = 0` and the mass matrix at `qpos0` is the
        // kind of half-migration that reads right on a model with no `ref` and wrong on one with.
        self.ten_moment(&self.reference_q)
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
        self.actuator_state_with(q, qd, true)
    }

    /// [`MjcfTree::actuator_state`], with the contact set optional. MuJoCo's `mj_setConst` runs
    /// `mj_transmission` WITHOUT `mj_collision`, so an adhesion actuator's moment is zero there and so is its
    /// `acc0`; the loader asks for exactly that.
    fn actuator_state_with(&self, q: &[f64], qd: &[f64], collide: bool) -> Vec<crate::mujoco_actuator::ActState> {
        use crate::mujoco_actuator::{ActState, DynTransmission};
        if self.actuators.iter().all(|a| a.dynamic.is_none()) {
            return self.actuators.iter().map(|a| ActState { moment: a.moment.clone(), length: a.length(q), velocity: a.velocity(qd) }).collect();
        }
        let frames = crate::tree_frames(&self.tree, q);
        // a spatial tendon's length is shared by every actuator that pulls on it, so build it once
        let ten = self.tendons.iter().any(|t| matches!(t.path, TendonPath::Spatial(_))).then(|| (self.ten_length(q), self.ten_moment(q)));
        // and so is the contact set, which only an adhesion actuator reads
        let con = (collide && self.actuators.iter().any(|a| matches!(a.dynamic, Some(DynTransmission::Body { .. })))).then(|| self.collide(q));
        self.actuators.iter().map(|a| self.act_state_of(a, &frames, ten.as_ref(), con.as_ref(), q, qd)).collect()
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
        let con = matches!(a.dynamic, Some(crate::mujoco_actuator::DynTransmission::Body { .. })).then(|| self.collide(q));
        self.act_state_of(a, &frames, ten.as_ref(), con.as_ref(), q, qd)
    }

    #[allow(clippy::type_complexity)]
    fn act_state_of(&self, a: &crate::mujoco_actuator::Actuator, frames: &[Iso], ten: Option<&(Vec<f64>, Vec<Vec<(usize, f64)>>)>, con: Option<&MjcfCollision>, q: &[f64], qd: &[f64]) -> crate::mujoco_actuator::ActState {
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
            // ⭐ `mj_transmission`'s `mjTRN_BODY`, from MuJoCo's source: every contact whose geom belongs to
            // the body counts once, an ACTIVE one through its normal constraint row and one found in the GAP
            // band through the same normal projection of `J₂ − J₁` at the contact point; the moment is minus
            // their sum over the count. A pyramidal contact's rows are averaged there, `0.5/(dim−1)` on each
            // of `2(dim−1)`, and the frictional halves cancel, leaving the same normal row.
            Some(DynTransmission::Body { body }) => {
                let Some(con) = con else { return ActState::default() };
                let (mut sum, mut counter) = (vec![0.0; nv], 0usize);
                for c in &con.contacts {
                    let (gi, gj) = (&self.geoms[c.geom[0]], &self.geoms[c.geom[1]]);
                    if &gi.body != body && &gj.body != body {
                        continue;
                    }
                    counter += 1;
                    let row = crate::contact_jacobian(&self.tree.joints, &self.tree.parent, frames, gi.joint, gj.joint, c.record.pos, &c.record.frame, 1);
                    for (k, s) in sum.iter_mut().enumerate() {
                        *s += row[(0, k)];
                    }
                }
                if counter == 0 {
                    return ActState::default();
                }
                let k = -1.0 / counter as f64;
                sum.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(d, v)| (d, v * k)).collect()
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
        self.qfrc_actuator_act(q, qd, ctrl, &[])
    }

    /// [`MjcfTree::qfrc_actuator`] with the ACTIVATIONS given, in MuJoCo's own layout: `d.act`, each
    /// actuator's activations in actuator order ([`MjcfTree::na`] in all). An empty slice is every
    /// activation at zero, which is what `mj_resetData` leaves and therefore what a state sampled by
    /// `mj_forward` on a fresh `mjData` has. At time zero; see [`MjcfTree::qfrc_actuator_at`].
    pub fn qfrc_actuator_act(&self, q: &[f64], qd: &[f64], ctrl: &[f64], act: &[f64]) -> Vec<f64> {
        self.qfrc_actuator_at(q, qd, ctrl, act, 0.0)
    }

    /// [`MjcfTree::qfrc_actuator_act`] at `time` (`d.time`), which a PID's slew limiter reads: it holds no
    /// previous setpoint until time has started.
    pub fn qfrc_actuator_at(&self, q: &[f64], qd: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Vec<f64> {
        let states = self.actuator_state(q, qd);
        let forces = self.actuator_forces(&states, ctrl, act, time);
        crate::mujoco_actuator::qfrc_from_forces(&states, &forces, &self.dof_actuator_force_range())
    }

    /// **`actuator_force`**, one per actuator, at this state: gain and bias, or a PID plugin, then
    /// `forcerange` — before any joint's `actuatorfrcrange`. `act` is `d.act`; `time` is `d.time`.
    pub fn actuator_force_at(&self, q: &[f64], qd: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Vec<f64> {
        self.actuator_forces(&self.actuator_state(q, qd), ctrl, act, time)
    }

    /// `actuator_force`, one per actuator: gain and bias, or a PID plugin, then `forcerange`.
    fn actuator_forces(&self, states: &[crate::mujoco_actuator::ActState], ctrl: &[f64], act: &[f64], time: f64) -> Vec<f64> {
        let inputs = self.act_inputs(ctrl, act);
        let act = self.act_full(act);
        self.actuators
            .iter()
            .zip(self.act_adr())
            .enumerate()
            .map(|(i, (a, (adr, num)))| {
                let Some(st) = states.get(i) else { return 0.0 };
                let u = ctrl.get(i).copied().unwrap_or(0.0);
                match &a.pid {
                    Some(pid) => {
                        let f = pid.force(a, st.length, st.velocity, u, &act[adr..adr + num], time, self.timestep);
                        a.forcerange.map_or(f, |[lo, hi]| f.clamp(lo, hi))
                    }
                    None => a.force(st.length, st.velocity, u, inputs[i]),
                }
            })
            .collect()
    }

    /// `(actuator_actadr, actuator_actnum)` per actuator: where its activations sit in `d.act`.
    pub fn act_adr(&self) -> Vec<(usize, usize)> {
        let mut adr = 0usize;
        self.actuators
            .iter()
            .map(|a| {
                let here = (adr, a.actnum());
                adr += a.actnum();
                here
            })
            .collect()
    }

    /// `act` padded with zeros to [`MjcfTree::na`] entries (an empty slice is `mj_resetData`'s zeros).
    fn act_full(&self, act: &[f64]) -> Vec<f64> {
        (0..self.na()).map(|k| act.get(k).copied().unwrap_or(0.0)).collect()
    }

    /// **`na`**: how many activations the model carries — one for each actuator with an activation law,
    /// and a PID plugin's own (its integral, its previous setpoint).
    pub fn na(&self) -> usize {
        self.actuators.iter().map(|a| a.actnum()).sum()
    }

    /// **`act_dot`**, in MuJoCo's layout ([`MjcfTree::na`] entries): every activation's rate at this state,
    /// as `mj_fwdActuation` computes it. `act` is `d.act`; empty means zeros.
    pub fn act_dot(&self, q: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Vec<f64> {
        let act = self.act_full(act);
        let lengths: Option<Vec<f64>> = self.actuators.iter().any(|a| a.pid.is_some()).then(|| self.actuator_state(q, &vec![0.0; q.len()]).iter().map(|s| s.length).collect());
        let mut out = Vec::with_capacity(act.len());
        for (i, (a, (adr, num))) in self.actuators.iter().zip(self.act_adr()).enumerate() {
            let u = ctrl.get(i).copied().unwrap_or(0.0);
            match &a.pid {
                Some(pid) => out.extend(pid.act_dot(a, lengths.as_ref().map_or(0.0, |l| l[i]), u, &act[adr..adr + num], time, self.timestep)),
                None if num > 0 => out.push(a.act_dot(u, act[adr + num - 1])),
                None => {}
            }
        }
        out
    }

    /// The activation each actuator's force READS, one entry per actuator: its last activation, or under
    /// `actearly` that activation at the end of the step; zero (unread) where it has none or a plugin reads
    /// its own.
    fn act_inputs(&self, ctrl: &[f64], act: &[f64]) -> Vec<f64> {
        let act = self.act_full(act);
        self.actuators
            .iter()
            .zip(self.act_adr())
            .enumerate()
            .map(|(i, (a, (adr, num)))| {
                if num == 0 || a.pid.is_some() {
                    return 0.0;
                }
                let x = act[adr + num - 1];
                if a.actearly {
                    a.next_activation(x, a.act_dot(ctrl.get(i).copied().unwrap_or(0.0), x), self.timestep)
                } else {
                    x
                }
            })
            .collect()
    }

    /// **`mj_advance`'s activation half**: every activation one step of `h` on, from `act` and its rate
    /// `act_dot` (both MuJoCo's layout) — [`crate::mujoco_actuator::Actuator::next_activation`] in turn.
    pub fn next_activation(&self, act: &[f64], act_dot: &[f64], h: f64) -> Vec<f64> {
        let act = self.act_full(act);
        let mut out = Vec::with_capacity(act.len());
        for (a, (adr, num)) in self.actuators.iter().zip(self.act_adr()) {
            for k in adr..adr + num {
                out.push(a.next_activation(act[k], act_dot.get(k).copied().unwrap_or(0.0), h));
            }
        }
        out
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
            let (ipos, irot) = self.inertial_rt(name);
            let rot = body.rotation.to_rotation_matrix().into_inner() * irot;
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
            let ipos = self.inertial_rt(name).0;
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
        self.invweight0().dof.clone()
    }

    /// The three `*_invweight0`, computed once. ⛔ In the chart the model was LOADED in: the free joint's
    /// rotational mean below is basis-free only where the map to MuJoCo's basis is orthogonal, so a copy
    /// re-charted by [`MjcfTree::charted`] must inherit these, never recompute them.
    fn invweight0(&self) -> &Invweight0 {
        self.invweight0_cache.get_or_init(|| Invweight0 { dof: self.compute_dof_invweight0(), body: self.compute_body_invweight0(), tendon: self.compute_tendon_invweight0() })
    }

    fn compute_dof_invweight0(&self) -> Vec<f64> {
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
    /// A body welded to the world is absent from the map and takes [`InvWeight::STATIC`](crate::mujoco_contact::InvWeight::STATIC), which is what an
    /// immovable body means.
    pub fn body_invweight0(&self) -> BTreeMap<String, crate::mujoco_contact::InvWeight> {
        self.invweight0().body.clone()
    }

    fn compute_body_invweight0(&self) -> BTreeMap<String, crate::mujoco_contact::InvWeight> {
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
                let ipos = self.inertial_rt(name).0;
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
        let frames = self.equalities.iter().any(|e| matches!(e.kind, EqualityKind::Connect { .. } | EqualityKind::Weld { .. })).then(|| crate::tree_frames(&self.tree, q));
        // the tendon lengths and moments, built once if a tendon equality needs them
        let mut ten: Option<(Vec<f64>, Vec<Vec<(usize, f64)>>)> = None;
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
                EqualityKind::Connect { side1, side2, diag_a, .. } => {
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
                EqualityKind::Tendon { tendon1, tendon2, length0, polycoef: c } => {
                    let (len, mom) = ten.get_or_insert_with(|| (self.ten_length(q), self.ten_moment(q)));
                    let tiw = self.tendon_invweight0();
                    let mut jac = vec![0.0; nv];
                    for (d, cf) in &mom[*tendon1] {
                        jac[*d] += cf;
                    }
                    let (pos, diag) = match *tendon2 {
                        Some(t2) => {
                            let y = len[t2] - length0.1;
                            let poly = c[0] + c[1] * y + c[2] * y * y + c[3] * y * y * y + c[4] * y * y * y * y;
                            let deriv = c[1] + 2.0 * c[2] * y + 3.0 * c[3] * y * y + 4.0 * c[4] * y * y * y;
                            for (d, cf) in &mom[t2] {
                                jac[*d] -= deriv * cf;
                            }
                            (len[*tendon1] - length0.0 - poly, tiw[*tendon1] + tiw[t2])
                        }
                        None => (len[*tendon1] - length0.0 - c[0], tiw[*tendon1]),
                    };
                    let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                    let (a, r) = row_reference(&solr(e.solref), &soli(e.solimp), pos, 0.0, diag, jvel, self.timestep);
                    push(jac, a, r, EfcBlock::Equality { rows: 1 });
                }
                EqualityKind::Weld { side1, side2, rot1, rot2, relpose, torquescale: ts, diag, .. } => {
                    let frames = frames.as_ref().expect("frames are built whenever a connect or weld is present");
                    let world_point = |(ride, local): &(Option<usize>, Vector3<f64>)| match ride {
                        Some(r) => (frames[*r] * Point3::from(*local)).coords,
                        None => *local,
                    };
                    let world_quat = |(ride, local): &(Option<usize>, UnitQuaternion<f64>)| {
                        let q = match ride {
                            Some(r) => frames[*r].rotation * local,
                            None => *local,
                        };
                        [q.w, q.i, q.j, q.k]
                    };
                    let (p1, p2) = (world_point(side1), world_point(side2));
                    let jlin = |ride: Option<usize>, p: Vector3<f64>| crate::tree_jacobian::tree_point_jacobian(&self.tree.joints, &self.tree.parent, frames, ride, p);
                    let jang = |ride: Option<usize>| crate::tree_jacobian::tree_angular_jacobian(&self.tree.joints, &self.tree.parent, frames, ride);
                    let (j1, j2, jr1, jr2) = (jlin(side1.0, p1), jlin(side2.0, p2), jang(rot1.0), jang(rot2.0));
                    // the orientation residual, `mj_instantiateEquality`: quat = q0·relpose, quat1 = neg(q1)
                    let q0 = world_quat(rot1);
                    let q1 = world_quat(rot2);
                    let q0r = quat_mul(&q0, relpose);
                    let negq1 = quat_neg(&q1);
                    let err = quat_mul(&negq1, &q0r);
                    let cpos = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2], ts * err[1], ts * err[2], ts * err[3]];
                    // one impedance for the whole weld, at the norm of all six residuals
                    let norm = -cpos.iter().map(|x| x * x).sum::<f64>().sqrt();
                    // `mj_Jdotv`: the translational J̇·v as a connect's, and the rotational one by MuJoCo's
                    // product rule (with its own quaternion rates, `mju_derivQuat` on the world ω)
                    let bias = |ride: Option<usize>, p: Vector3<f64>| crate::tree_jacobian::tree_spatial_bias_acceleration(&self.tree.joints, &self.tree.parent, frames, qd, ride, p);
                    let ((ab1, lb1), (ab2, lb2)) = (bias(side1.0, p1), bias(side2.0, p2));
                    let (ab1, ab2) = (if rot1.0 == side1.0 { ab1 } else { bias(rot1.0, p1).0 }, if rot2.0 == side2.0 { ab2 } else { bias(rot2.0, p2).0 });
                    let qdv = nalgebra::DVector::from_row_slice(qd);
                    let (w1, w2) = (&jr1 * &qdv, &jr2 * &qdv);
                    let (w1, w2) = ([w1[0], w1[1], w1[2]], [w2[0], w2[1], w2[2]]);
                    let domega = [w1[0] - w2[0], w1[1] - w2[1], w1[2] - w2[2]];
                    let djrdv = [ab1[0] - ab2[0], ab1[1] - ab2[1], ab1[2] - ab2[2]];
                    let qdot0r = quat_mul(&quat_deriv(&q0, &w1), relpose);
                    let negqdot1 = quat_neg(&quat_deriv(&q1, &w2));
                    let t1 = quat_mul(&quat_mul_axis(&negqdot1, &domega), &q0r);
                    let t2 = quat_mul(&quat_mul_axis(&negq1, &djrdv), &q0r);
                    let t3 = quat_mul(&quat_mul_axis(&negq1, &domega), &qdot0r);
                    let jdot_lin = lb1 - lb2;
                    for r in 0..3 {
                        let jac: Vec<f64> = (0..nv).map(|k| j1[(r, k)] - j2[(r, k)]).collect();
                        let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                        let (a, rr) = crate::mujoco_contact::row_reference_at(&solr(e.solref), &soli(e.solimp), norm, cpos[r], 0.0, diag[0], jvel, self.timestep);
                        push(jac, a - jdot_lin[r], rr, EfcBlock::Equality { rows: 1 });
                    }
                    // the rotation rows: `0.5·neg(q1)·(J₀ − J₁)·q0·relpose`, column by column
                    let rot_cols: Vec<[f64; 4]> = (0..nv).map(|k| quat_mul(&quat_mul_axis(&negq1, &[jr1[(0, k)] - jr2[(0, k)], jr1[(1, k)] - jr2[(1, k)], jr1[(2, k)] - jr2[(2, k)]]), &q0r)).collect();
                    for r in 0..3 {
                        let jac: Vec<f64> = rot_cols.iter().map(|c| 0.5 * c[1 + r] * ts).collect();
                        let jvel: f64 = (0..nv).map(|k| jac[k] * qd[k]).sum();
                        let (a, rr) = crate::mujoco_contact::row_reference_at(&solr(e.solref), &soli(e.solimp), norm, cpos[3 + r], 0.0, diag[1], jvel, self.timestep);
                        push(jac, a - 0.5 * (t1[1 + r] + t2[1 + r] + t3[1 + r]) * ts, rr, EfcBlock::Equality { rows: 1 });
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

    /// **`mj_collision`**: every contact this state produces, geom pair by geom pair, in MuJoCo's order —
    /// the same gates in the same sequence (`contype`/`conaffinity`, the weld-tree body filter,
    /// `<contact><exclude>`), an explicit `<pair>` bypassing them with its own parameters, detection at
    /// `margin + gap`, and `mj_setContact` completing each one.
    ///
    /// Contacts found in the gap band are KEPT, flagged by `record.exclude`: the solver skips them, but an
    /// adhesion actuator counts them. A pair that needs a collider this port does not carry (a height
    /// field) is listed in `refused` rather than silently dropped, and every pair that produced nothing
    /// says why in `why`, so a missing contact names the rule that removed it.
    ///
    /// This is the pipeline `examples/menagerie_forward` verifies against MuJoCo 3.13.0 on Menagerie.
    pub fn collide(&self, q: &[f64]) -> MjcfCollision {
        self.collide_impl(q, false)
    }

    /// [`MjcfTree::collide`] with [`MjcfCollision::why`] filled in: for every pair considered, the gate that
    /// dropped it or what the collider found. A diagnostic — it costs a string per pair, which on a model
    /// with a few thousand geom pairs is most of the collision time.
    pub fn collide_explained(&self, q: &[f64]) -> MjcfCollision {
        self.collide_impl(q, true)
    }

    fn collide_static(&self) -> &CollideStatic {
        use crate::mujoco_collision::GeomType;
        self.collide_cache.get_or_init(|| {
            let welds = self.body_welds();
            let body_id: HashMap<&str, usize> = self.geoms.iter().map(|g| g.body.as_str()).chain(self.contact_excludes.iter().flat_map(|(a, b)| [a.as_str(), b.as_str()])).collect::<BTreeSet<&str>>().into_iter().enumerate().map(|(i, b)| (b, i)).collect();
            let geom_id: HashMap<&str, usize> = self.geoms.iter().enumerate().map(|(i, g)| (g.name.as_str(), i)).collect();
            let mut pairs = HashMap::new();
            for (k, p) in self.contact_pairs.iter().enumerate() {
                if let (Some(&a), Some(&b)) = (geom_id.get(p.geom1.as_str()), geom_id.get(p.geom2.as_str())) {
                    pairs.insert((a, b), k);
                    pairs.insert((b, a), k);
                }
            }
            let rbound = self
                .geoms
                .iter()
                .map(|g| {
                    let s = g.size;
                    match g.kind {
                        GeomType::Sphere => s[0],
                        GeomType::Capsule => s[0] + s[1],
                        GeomType::Cylinder => (s[0] * s[0] + s[1] * s[1]).sqrt(),
                        GeomType::Ellipsoid => s[0].max(s[1]).max(s[2]),
                        GeomType::Box => (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt(),
                        GeomType::Mesh => g.mesh.as_ref().and_then(|m| self.mesh_hulls.get(m)).map_or(0.0, |h| h.rbound()),
                        _ => 0.0,
                    }
                })
                .collect();
            CollideStatic {
                weld: self.geoms.iter().map(|g| welds.get(&g.body).copied().unwrap_or((0, 0, 0))).collect(),
                excludes: self.contact_excludes.iter().flat_map(|(a, b)| [(body_id[a.as_str()], body_id[b.as_str()]), (body_id[b.as_str()], body_id[a.as_str()])]).collect(),
                body: self.geoms.iter().map(|g| body_id[g.body.as_str()]).collect(),
                pairs,
                rbound,
            }
        })
    }

    /// **Every geom's world frame as MuJoCo's `mj_kinematics` computes it** (`geom_xpos`, `geom_xmat`), from
    /// `qpos` in MuJoCo's layout, in MuJoCo's own arithmetic (the crate's `mujoco_kinematics`) — bit for bit.
    /// [`MjcfTree::geoms`]' poses composed through this crate's tree agree to a few ulps; a collision between
    /// two nearly parallel surfaces can read those ulps.
    pub fn geom_frames_mujoco(&self, qpos: &[f64]) -> Vec<(Vector3<f64>, Matrix3<f64>)> {
        self.mj_kin.geom_frames(qpos).into_iter().map(|(p, m)| (Vector3::from(p), Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]))).collect()
    }

    /// Every body's compiled frame in its parent, `(body_pos, body_quat)`, in MuJoCo's body order (the world
    /// first) and MuJoCo's own arithmetic.
    pub fn body_frames_compiled(&self) -> Vec<([f64; 3], [f64; 4])> {
        self.mj_kin.bodies.iter().map(|b| (b.pos, b.quat)).collect()
    }

    /// [`MjcfTree::collide`] from `qpos` in MuJoCo's layout, with the geoms where MuJoCo's own kinematics put
    /// them ([`MjcfTree::geom_frames_mujoco`]).
    pub fn collide_qpos(&self, qpos: &[f64]) -> MjcfCollision {
        self.collide_qpos_impl(qpos, false)
    }

    /// [`MjcfTree::collide_qpos`] with [`MjcfCollision::why`] filled in.
    pub fn collide_qpos_explained(&self, qpos: &[f64]) -> MjcfCollision {
        self.collide_qpos_impl(qpos, true)
    }

    fn collide_qpos_impl(&self, qpos: &[f64], explain: bool) -> MjcfCollision {
        use crate::mujoco_collision::GeomPose;
        let poses: Vec<GeomPose> = self.geom_frames_mujoco(qpos).into_iter().map(|(pos, mat)| GeomPose { pos, mat }).collect();
        self.collide_at(&poses, explain)
    }

    fn collide_impl(&self, q: &[f64], explain: bool) -> MjcfCollision {
        use crate::mujoco_collision::GeomPose;
        let frames = crate::tree_frames(&self.tree, q);
        let poses: Vec<GeomPose> = self
            .geoms
            .iter()
            .map(|g| {
                let world = match g.joint {
                    Some(j) => frames[j] * g.pose,
                    None => g.pose,
                };
                GeomPose { pos: world.translation.vector, mat: *world.rotation.to_rotation_matrix().matrix() }
            })
            .collect();
        self.collide_at(&poses, explain)
    }

    fn collide_at(&self, poses: &[crate::mujoco_collision::GeomPose], explain: bool) -> MjcfCollision {
        use crate::mujoco_collision::{can_collide, collide_pair_with, contact_param, filter_body_pair, margin_and_gap, set_contact, CollideOptions, CollisionGeom, GeomType, PairParams};
        let st = self.collide_static();
        let opts = CollideOptions::default();
        let geoms: Vec<Option<CollisionGeom>> = self
            .geoms
            .iter()
            .zip(poses)
            .map(|(g, pose)| {
                let hull = match (&g.mesh, g.kind) {
                    (Some(mn), GeomType::Mesh) => Some(self.mesh_hulls.get(mn)?),
                    _ => None,
                };
                Some(CollisionGeom { kind: g.kind, pose: *pose, size: g.size, hull })
            })
            .collect();
        let mut out = MjcfCollision::default();
        let why = |out: &mut MjcfCollision, key: (usize, usize), f: &dyn Fn() -> String| {
            if explain {
                out.why.insert(key, f());
            }
        };
        for i in 0..self.geoms.len() {
            for j in i + 1..self.geoms.len() {
                let (gi, gj) = (&self.geoms[i], &self.geoms[j]);
                let (Some(ci), Some(cj)) = (&geoms[i], &geoms[j]) else { continue };
                if ci.kind == GeomType::HField || cj.kind == GeomType::HField {
                    out.refused.push([i, j]);
                    continue;
                }
                let key = (i, j);
                let pair = st.pairs.get(&key).map(|&k| &self.contact_pairs[k]);
                let (margin, gap, params) = if let Some(p) = pair {
                    (p.margin, p.gap, PairParams { condim: p.condim, solref: p.solref, solimp: p.solimp, friction: p.friction, adhesion: p.adhesion })
                } else {
                    if !can_collide(gi.params.contype, gi.params.conaffinity, gj.params.contype, gj.params.conaffinity) {
                        why(&mut out, key, &|| "contype/conaffinity".into());
                        continue;
                    }
                    let ((w1, pw1, n1), (w2, pw2, n2)) = (st.weld[i], st.weld[j]);
                    if filter_body_pair(w1, pw1, n1, w2, pw2, n2, true) {
                        why(&mut out, key, &|| "the body filter (same weld, both static, or parent and child)".into());
                        continue;
                    }
                    if st.excludes.contains(&(st.body[i], st.body[j])) {
                        why(&mut out, key, &|| "<contact><exclude>".into());
                        continue;
                    }
                    let (m, g) = margin_and_gap(&gi.params, &gj.params);
                    (m, g, contact_param(&gi.params, &gj.params))
                };
                // `mj_filterSphere`: two bounding spheres further apart than the margin, or a plane further
                // from a geom's centre than its bound, cannot touch — the pair is never handed to a collider.
                // A pure saving: the test is conservative, so no contact is lost to it.
                let rb = &st.rbound;
                let far = if rb[i] > 0.0 && rb[j] > 0.0 {
                    let b = rb[i] + rb[j] + margin + gap;
                    (ci.pose.pos - cj.pose.pos).norm_squared() > b * b
                } else if ci.kind == GeomType::Plane && rb[j] > 0.0 {
                    (cj.pose.pos - ci.pose.pos).dot(&ci.pose.mat.column(2)) > margin + gap + rb[j]
                } else if cj.kind == GeomType::Plane && rb[i] > 0.0 {
                    (ci.pose.pos - cj.pose.pos).dot(&cj.pose.mat.column(2)) > margin + gap + rb[i]
                } else {
                    false
                };
                if far {
                    why(&mut out, key, &|| "the bounding-sphere filter".into());
                    continue;
                }
                let Ok(pre) = collide_pair_with(&opts, margin + gap, ci, cj) else {
                    out.refused.push([i, j]);
                    why(&mut out, key, &|| "a pair the collider refuses".into());
                    continue;
                };
                if pre.is_empty() {
                    why(&mut out, key, &|| "no contact: the collider found them apart".into());
                } else {
                    why(&mut out, key, &|| format!("the collider found {} witness(es) at dist {:?}, includemargin {margin}", pre.len(), pre.iter().map(|p| p.dist).collect::<Vec<_>>()));
                }
                out.contacts.extend(pre.iter().map(|p| MjcfContact { geom: [i, j], record: set_contact(p, &params, margin) }));
            }
        }
        out
    }

    /// One contact as the constraint rows need it: its Jacobian at this state (`contact_jacobian` on the
    /// two geoms' dofs) and its two bodies' inverse weights. ⛔ The weights are `body_invweight0`, taken at
    /// `qpos0` when MuJoCo compiles the model and never recomputed, so pass the table computed ONCE.
    pub fn contact_spec(&self, c: &MjcfContact, frames: &[Iso], invweight0: &BTreeMap<String, crate::mujoco_contact::InvWeight>) -> crate::mujoco_contact::ContactSpec {
        use crate::mujoco_contact::{InvWeight, SolImp, SolRef};
        let (gi, gj) = (&self.geoms[c.geom[0]], &self.geoms[c.geom[1]]);
        let r = &c.record;
        let jac = crate::contact_jacobian(&self.tree.joints, &self.tree.parent, frames, gi.joint, gj.joint, r.pos, &r.frame, r.dim);
        let iw = |g: &MjcfGeom| invweight0.get(&g.body).copied().unwrap_or(InvWeight::STATIC);
        crate::mujoco_contact::ContactSpec {
            jac,
            dist: r.dist,
            margin: r.includemargin,
            condim: r.dim,
            friction: r.friction,
            solref: SolRef(r.solref[0], r.solref[1]),
            solimp: SolImp { d0: r.solimp[0], d_width: r.solimp[1], width: r.solimp[2], midpoint: r.solimp[3], power: r.solimp[4] },
            invweight: [iw(gi), iw(gj)],
        }
    }

    /// [`MjcfTree::joint_constraint_rows`] posed in MuJoCo's coordinates: every Jacobian mapped as `J·T⁻¹`,
    /// with `v_mujoco` brought into this port's basis first.
    ///
    /// ⛔⛔ And the `J̇·q̇` a connect row carries has to move with it. This port builds that term in its own
    /// (Euler) basis, `J̇_ours·q̇_ours`. The residual's second derivative is basis-free, but splitting it into
    /// `J·q̈ + J̇·q̇` is not: with `v_mj = T·q̇`, `J̇_mj·v_mj = J̇_ours·q̇ − J_mj·Ṫ·q̇`. So every row whose reference
    /// was built with `J̇_ours·q̇` needs `+J_mj·Ṫ·q̇` in MuJoCo's basis — and ONLY those: MuJoCo carries `J̇·v`
    /// on its connect and weld rows alone. Missing it put cassie's moving constrained acceleration 2.65e-3
    /// off; with it, 6.7e-10.
    ///
    /// ⛔⛔ CORRECTED 2026-09-27: this used to add the term to EVERY joint row, on the argument that it "is
    /// exactly zero on any row without a free or ball dof in its Jacobian". A tendon limit row is not one:
    /// `robotiq_2f85/scene` ties a site on a FREE body to the world with a limited spatial tendon, and its
    /// limit row picked up a `J̇·v` MuJoCo never puts there — `aref` 1.9e-4 off while its `J` and `D` matched
    /// to the last bit.
    pub fn joint_constraint_rows_mujoco(&self, q: &[f64], v_mujoco: &[f64], dof_invweight0: &[f64]) -> Result<crate::mujoco_contact::AssembledRows, String> {
        let tinv = self.free_basis(q).try_inverse().ok_or("gimbal lock: the Euler base's basis map is singular")?;
        let qvel = &tinv * nalgebra::DVector::from_row_slice(v_mujoco);
        let qd: Vec<f64> = qvel.iter().copied().collect();
        let mut rows = self.joint_constraint_rows(q, &qd, dof_invweight0);
        rows.jac = &rows.jac * &tinv;
        let basis_term = &rows.jac * (self.free_basis_dot(q, &qd) * &qvel);
        // the rows that carry `J̇·q̇`: each connect's three, in equality order
        let carries: Vec<bool> = self
            .equalities
            .iter()
            .flat_map(|e| match e.kind {
                EqualityKind::Connect { .. } => std::iter::repeat_n(true, 3),
                EqualityKind::Weld { .. } => std::iter::repeat_n(true, 6),
                _ => std::iter::repeat_n(false, 1),
            })
            .collect();
        for (i, (a, c)) in rows.aref.iter_mut().zip(basis_term.iter()).enumerate() {
            if carries.get(i).copied().unwrap_or(false) {
                *a += c;
            }
        }
        Ok(rows)
    }

    /// **The constrained problem `mj_forward` solves at one state, in MUJOCO's coordinates**: the mass
    /// matrix, the unconstrained acceleration, and every constraint row in MuJoCo's order (equality,
    /// friction, limits, then `contacts`), each with its `efc_aref`, `efc_D` and law.
    ///
    /// `q` is this port's coordinates (from [`MjcfTree::q_from_qpos`]); `v_mujoco`, `ctrl` and `act` (`d.act`,
    /// empty for zeros) are MuJoCo's.
    /// ⛔ The problem is posed where MuJoCo's answer lives. On a free or ball joint this port's velocity
    /// basis is not MuJoCo's, so a row's Jacobian maps as `J·T⁻¹` and the mass matrix as `T⁻ᵀ·M·T⁻¹`, while
    /// `efc_aref` and `efc_D` do not move (`J·q̇` is the same number in either basis). With no such joint `T`
    /// is the identity.
    pub fn constraint_problem(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], contacts: &[crate::mujoco_contact::ContactSpec]) -> Result<MjcfConstraintProblem, String> {
        self.constraint_problem_at(q, v_mujoco, ctrl, act, 0.0, contacts)
    }

    /// [`MjcfTree::constraint_problem`] at `time` (`d.time`, which a PID's slew limiter reads).
    pub fn constraint_problem_at(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64, contacts: &[crate::mujoco_contact::ContactSpec]) -> Result<MjcfConstraintProblem, String> {
        self.constraint_problem_impl(q, v_mujoco, ctrl, act, time, &MjcfApplied::default(), contacts)
    }

    #[allow(clippy::too_many_arguments)]
    fn constraint_problem_impl(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied, contacts: &[crate::mujoco_contact::ContactSpec]) -> Result<MjcfConstraintProblem, String> {
        let nv = self.tree.joints.len();
        let tinv = self.free_basis(q).try_inverse().ok_or("gimbal lock: the Euler base's basis map is singular")?;
        let qvel: Vec<f64> = (&tinv * nalgebra::DVector::from_row_slice(v_mujoco)).iter().copied().collect();
        let mut rows = self.joint_constraint_rows_mujoco(q, v_mujoco, &self.dof_invweight0())?;
        let contact_blocks_from = rows.blocks.len();
        let mut contact = crate::mujoco_contact::contact_rows(contacts, nv, &qvel, self.cone, self.impratio, self.timestep)?;
        contact.jac = &contact.jac * &tinv;
        rows.append(contact);
        let m = self.mass_matrix(q);
        let a0 = self.qacc_smooth_impl(q, v_mujoco, ctrl, act, time, applied).ok_or("gimbal lock: the Euler base's basis map is singular")?;
        Ok(MjcfConstraintProblem { m: tinv.transpose() * m * &tinv, a0: nalgebra::DVector::from_vec(a0), rows, contact_blocks_from })
    }

    /// **`mj_forward`**: the constrained acceleration at one state, in MuJoCo's coordinates — contacts from
    /// [`MjcfTree::collide`], rows from [`MjcfTree::constraint_problem`], and MuJoCo's Newton solver run to
    /// convergence.
    ///
    /// ⛔ To CONVERGENCE, not to MuJoCo's iteration budget. The objective is strictly convex, so this is the
    /// one optimum of the rows MuJoCo builds; a model whose `<option iterations>` caps the solve (every
    /// `*_mjx.xml` in Menagerie) gets an answer MuJoCo itself only approaches. `Err` for a state this port
    /// cannot pose: a height field, or a gimbal-locked Euler base.
    ///
    /// `act` is MuJoCo's `d.act` ([`MjcfTree::na`] entries; empty for zeros).
    ///
    /// Verified against MuJoCo 3.13.0 on Menagerie by `examples/menagerie_forward`, which runs this and
    /// checks it against the pipeline it assembles itself.
    pub fn forward_mujoco(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64]) -> Result<MjcfForward, String> {
        self.forward_mujoco_at(q, v_mujoco, ctrl, act, 0.0)
    }

    /// [`MjcfTree::forward_mujoco`] at `time` (`d.time`). Only a PID plugin's slew limiter reads it: at
    /// `time = 0` it has no previous setpoint to limit against, as a fresh `mjData` has none.
    pub fn forward_mujoco_at(&self, q: &[f64], v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Result<MjcfForward, String> {
        self.forward_impl(q, None, v_mujoco, ctrl, act, time, &MjcfApplied::default())
    }

    /// `qpos`, where the caller has it in MuJoCo's layout, places the geoms for the collision pass by MuJoCo's
    /// own kinematics ([`MjcfTree::collide_qpos`]) rather than through this crate's tree.
    #[allow(clippy::too_many_arguments)]
    fn forward_impl(&self, q: &[f64], qpos: Option<&[f64]>, v_mujoco: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied) -> Result<MjcfForward, String> {
        let collision = match qpos {
            Some(qp) => self.collide_qpos(qp),
            None => self.collide(q),
        };
        if let Some([i, j]) = collision.refused.first() {
            return Err(format!("geoms '{}' and '{}' need a collider this port does not carry", self.geoms[*i].name, self.geoms[*j].name));
        }
        let frames = crate::tree_frames(&self.tree, q);
        let invweight0 = self.body_invweight0();
        let specs: Vec<crate::mujoco_contact::ContactSpec> = collision.contacts.iter().filter(|c| !c.record.exclude).map(|c| self.contact_spec(c, &frames, &invweight0)).collect();
        let p = self.constraint_problem_impl(q, v_mujoco, ctrl, act, time, applied, &specs)?;
        let mut solve = crate::mujoco_contact::solve_constraints_newton_blocks(&p.m, &p.a0, &p.rows.jac, &p.rows.aref, &p.rows.d, &p.rows.blocks, 1e-13, 200)?;
        // ⭐ `<option noslip_iterations>`: MuJoCo re-solves the friction forces after the main solve, so the
        // acceleration it reports is NOT the optimum of the main problem. Part of the problem statement.
        if self.noslip_iterations > 0 {
            let mut force = solve.force.clone();
            solve.qacc = crate::mujoco_contact::mujoco_noslip(&p.m, &p.a0, &p.rows.jac, &p.rows.aref, &p.rows.d, &p.rows.blocks, &mut force, self.noslip_iterations, self.noslip_tolerance, self.meaninertia)?;
            solve.force = force;
        }
        Ok(MjcfForward { qacc: solve.qacc.iter().copied().collect(), qacc_smooth: p.a0.iter().copied().collect(), nefc: p.rows.aref.len(), collision, solve, m: p.m, act_dot: self.act_dot(q, ctrl, act, time) })
    }

    /// MuJoCo's `qpos` address of each joint, in joint order.
    pub fn qposadr(&self) -> Vec<usize> {
        self.joints
            .iter()
            .scan(0usize, |adr, j| {
                let here = *adr;
                *adr += j.kind.qpos_width();
                Some(here)
            })
            .collect()
    }

    /// **`mj_step`**, for every integrator MuJoCo has but `discrete`, ported from `mj_EulerSkip`, `mj_implicitSkip`,
    /// `mjd_smooth_vel`, `mj_advance` and `mj_integratePos`: one `<option timestep>` from a state in MuJoCo's
    /// own layout (`qpos` with its quaternions, `qvel` in its basis).
    ///
    /// Both take the forward pass's force, `qfrc_smooth + qfrc_constraint = M·qacc`, and divide it by a
    /// modified mass matrix before integrating:
    /// * `Euler` (the default): `M + h·diag(damping)` when any dof is damped, so the velocity is implicit in
    ///   the dof damping alone; otherwise `M`, and the applied acceleration is `qacc` itself.
    /// * `implicitfast`: `M − h·∂qfrc/∂qvel` without the Coriolis term, where the derivative is every
    ///   actuator's velocity gain `(biasprm[2] + gainprm[2]·ctrl)·m mᵀ`, minus the dof damping and each
    ///   tendon's `damping·J Jᵀ`.
    /// * `implicit`: the same with the Coriolis term, `−∂qfrc_bias/∂qvel` (`mjd_rne_vel`), solved by LU.
    /// * `RK4`: see [`MjcfTree::step_mujoco_at`]'s four-stage path.
    ///
    /// Then `qvel += h·qacc`, and positions move with the NEW velocity: a hinge or slide by `h·qvel`, a free
    /// joint's position by `h·v`, every quaternion by `mju_quatIntegrate` (the rotation `h·ω` in the body's
    /// own frame, applied on the right after normalising). Each activation moves by `mj_nextActivation` on
    /// the rate the forward pass computed at the START of the step (`act` is `d.act`; empty for zeros).
    ///
    /// ⛔ The modified mass matrix is inverted in THIS port's coordinates, `T·(M − h·Q)⁻¹·M·T⁻¹`, where the
    /// damping becomes `Tᵀ·D·T` — the same answer as MuJoCo's basis, without forming `T⁻ᵀ·M·T⁻¹`, which on a
    /// tilted Euler base is badly conditioned. The constraint solve is converged (see
    /// [`MjcfTree::forward_mujoco`]). Refused: `implicit` or `implicitfast` with a fluid (its derivative is
    /// not carried).
    ///
    /// ⭐ And MuJoCo's AUTO-RESET, from `mj_step`: a `qpos` or `qvel` entry that is NaN or beyond ±1e10
    /// (`mju_isBad`) is replaced by `mj_resetData`'s state before anything runs, and a `qacc` that is bad
    /// after the forward pass resets the state and runs the forward pass again, which is then integrated
    /// without a second check. [`MjcfStep::reset`] says which. ⛔ `mj_resetData` also zeroes `d.ctrl`, so a
    /// caller following MuJoCo's trajectory past a reset drives it with zero control from then on.
    pub fn step_mujoco(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64]) -> Result<MjcfStep, String> {
        self.step_mujoco_at(qpos, qvel, ctrl, act, 0.0)
    }

    /// [`MjcfTree::step_mujoco`] from `time` (`d.time`); [`MjcfStep::time`] is where it ends. Only a PID
    /// plugin's slew limiter reads the time — it has no previous setpoint until time has started — so a
    /// model without one steps the same from any time. Chain `step.time` into the next call to follow
    /// MuJoCo's trajectory.
    pub fn step_mujoco_at(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64) -> Result<MjcfStep, String> {
        self.step_impl(qpos, qvel, ctrl, act, time, &MjcfApplied::default())
    }

    fn step_impl(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied) -> Result<MjcfStep, String> {
        let bad = |x: &f64| x.is_nan() || *x > MJ_MAXVAL || *x < -MJ_MAXVAL;
        if let Some(i) = qpos.iter().position(bad) {
            return self.step_after_reset(MjcfReset::BadQpos(i), true);
        }
        if let Some(i) = qvel.iter().position(bad) {
            return self.step_after_reset(MjcfReset::BadQvel(i), true);
        }
        self.step_checked(qpos, qvel, ctrl, act, time, applied, true)
    }

    /// One step from `mj_resetData`'s state — `qpos0`, time zero, everything else zero, the control included.
    fn step_after_reset(&self, why: MjcfReset, check_acc: bool) -> Result<MjcfStep, String> {
        let nv = self.tree.joints.len();
        let mut s = self.step_checked(&self.qpos0(), &vec![0.0; nv], &vec![0.0; self.actuators.len()], &vec![0.0; self.na()], 0.0, &MjcfApplied::default(), check_acc)?;
        s.reset = Some(why);
        Ok(s)
    }

    /// [`MjcfTree::step_mujoco`] once `qpos` and `qvel` have passed `mj_checkPos`/`mj_checkVel`; `check_acc`
    /// is `mj_checkAcc`, which MuJoCo does not repeat after the reset it triggers.
    #[allow(clippy::too_many_arguments)]
    fn step_checked(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied, check_acc: bool) -> Result<MjcfStep, String> {
        use crate::mujoco_actuator::{ActBias, ActDyn, ActGain};
        // ⭐ near gimbal lock, the same step in a chart where the angles are zero
        if let Some(t) = self.chart_if_needed(qpos)? {
            return t.step_checked(qpos, qvel, ctrl, act, time, applied, check_acc);
        }
        match self.integrator {
            MjcfIntegrator::Euler => {}
            MjcfIntegrator::ImplicitFast | MjcfIntegrator::Implicit if self.density > 0.0 || self.viscosity > 0.0 => {
                return Err("an implicit integrator with a fluid: the fluid force's velocity derivative is not carried".into());
            }
            MjcfIntegrator::ImplicitFast | MjcfIntegrator::Implicit => {}
            MjcfIntegrator::Rk4 => return self.step_rk4(qpos, qvel, ctrl, act, time, applied, check_acc),
        }
        let qposadr = self.qposadr();
        let q = self.q_from_qpos(qpos, &qposadr)?;
        let forward = self.forward_impl(&q, Some(qpos), qvel, ctrl, act, time, applied)?;
        if let Some(i) = forward.qacc.iter().position(|x| check_acc && (x.is_nan() || x.abs() > MJ_MAXVAL)) {
            return self.step_after_reset(MjcfReset::BadQacc(i), false);
        }
        let h = self.timestep;
        let nv = self.tree.joints.len();
        // dof damping in MuJoCo's dof order: a joint's own value on each of its dofs
        let damping: Vec<f64> = self.joints.iter().flat_map(|j| std::iter::repeat_n(j.damping, j.kind.dofs())).collect();
        let tb = self.free_basis(&q);
        let tinv = tb.clone().try_inverse().ok_or("gimbal lock: the Euler base's basis map is singular")?;
        let m = self.mass_matrix(&q);
        let qacc_mj = nalgebra::DVector::from_row_slice(&forward.qacc);
        // ⛔ `<flag eulerdamp="disable">` turns the Euler integrator's implicit damping off entirely
        let damped = damping.iter().any(|d| *d > 0.0) && (self.eulerdamp || self.integrator != MjcfIntegrator::Euler);
        // `∂qfrc/∂qvel` beyond the dof damping, as rank-one terms `c·v·vᵀ` in this port's coordinates:
        // `mjd_actuator_vel` and the tendon half of `mjd_passive_vel` (MuJoCo 3.13.0's rules)
        let mut rank_one: Vec<(f64, nalgebra::DVector<f64>)> = Vec::new();
        let implicit = matches!(self.integrator, MjcfIntegrator::ImplicitFast | MjcfIntegrator::Implicit);
        if implicit {
            let qd_ours: Vec<f64> = (&tinv * nalgebra::DVector::from_row_slice(qvel)).iter().copied().collect();
            let act_in = self.act_inputs(ctrl, act);
            for (u, (a, st)) in self.actuators.iter().zip(self.actuator_state(&q, &qd_ours)).enumerate() {
                // a plugin's velocity derivative is not carried by MuJoCo either (`pid.cc`: "allow actuator
                // plugins to compute their derivatives wrt qvel" is a TODO)
                if a.pid.is_some() {
                    continue;
                }
                let input = ctrl.get(u).copied().unwrap_or(0.0);
                // `actuatorDerivSkip`: a force pinned at its `forcerange` has no velocity derivative
                if let Some([lo, hi]) = a.forcerange {
                    let f = a.force(st.length, st.velocity, input, act_in[u]);
                    if f <= lo || f >= hi {
                        continue;
                    }
                }
                // `actuatorInput`: the CLAMPED control, or where there is a state the activation the force
                // reads (the next one under `actearly`)
                let input = if a.dynamics == ActDyn::None { a.ctrlrange.map_or(input, |[lo, hi]| input.clamp(lo, hi)) } else { act_in[u] };
                // `actuatorVelDeriv`: kv from an affine bias, plus the gain's velocity derivative times the input
                let mut rate = if a.bias == ActBias::Affine { a.biasprm[2] } else { 0.0 };
                match a.gain {
                    ActGain::Affine => rate += a.gainprm[2] * input,
                    ActGain::Muscle => {
                        let mut prm = [0.0; 9];
                        prm.copy_from_slice(&a.gainprm[..9]);
                        rate += crate::mujoco_actuator::muscle_gain_vel(st.length, st.velocity, &a.lengthrange, a.acc0, &prm) * input;
                    }
                    ActGain::Fixed => {}
                }
                if rate != 0.0 {
                    let mut mv = nalgebra::DVector::<f64>::zeros(nv);
                    for (d, c) in &st.moment {
                        mv[*d] += c;
                    }
                    rank_one.push((rate, mv));
                }
            }
            for (t, row) in self.tendons.iter().zip(self.ten_moment(&q)) {
                if t.damping != 0.0 {
                    let mut jv = nalgebra::DVector::<f64>::zeros(nv);
                    for (d, c) in row {
                        jv[d] += c;
                    }
                    rank_one.push((-t.damping, jv));
                }
            }
        }
        let mut applied = if self.integrator == MjcfIntegrator::Euler && !damped {
            qacc_mj.clone()
        } else {
            // the modified mass matrix `M − h·∂qfrc/∂qvel`, in this port's coordinates
            let mut mh = m.clone();
            if damped {
                mh += (tb.transpose() * nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(damping.clone())) * &tb) * h;
            }
            // `implicit` (not `implicitfast`) also differentiates the bias: `qDeriv −= ∂qfrc_bias/∂qvel`
            // (`mjd_rne_vel`). The bias is QUADRATIC in the velocity (plus gravity, which does not move), so a
            // central difference of it is its exact derivative whatever the step — up to rounding.
            let rne: Option<nalgebra::DMatrix<f64>> = (self.integrator == MjcfIntegrator::Implicit).then(|| {
                let mut d = nalgebra::DMatrix::<f64>::zeros(nv, nv);
                for j in 0..nv {
                    let (mut up, mut down) = (qvel.to_vec(), qvel.to_vec());
                    up[j] += 1.0;
                    down[j] -= 1.0;
                    let (bu, bd) = (self.qfrc_bias_mujoco(&q, &up), self.qfrc_bias_mujoco(&q, &down));
                    if let (Some(bu), Some(bd)) = (bu, bd) {
                        d.set_column(j, &(-(bu - bd) * 0.5));
                    }
                }
                d
            });
            if !rank_one.is_empty() || rne.is_some() {
                // ⛔ MuJoCo keeps `qDeriv` in the mass matrix's SPARSITY: an entry survives only between a dof
                // and one of its ancestors (or itself). Two sibling fingers pulled by one tendon actuator
                // get their diagonal terms and NOT the cross term between them — `addJTBJSparse` has no slot
                // for it. So the rank-one terms are masked in MuJoCo's basis before being mapped back here.
                let ancestor = |a: usize, b: usize| {
                    let mut p = self.tree.parent[b];
                    while p >= 0 {
                        if p as usize == a {
                            return true;
                        }
                        p = self.tree.parent[p as usize];
                    }
                    false
                };
                let tinv_t = tinv.transpose();
                let mut q_mj = nalgebra::DMatrix::<f64>::zeros(nv, nv);
                for (c, v) in &rank_one {
                    let vm = &tinv_t * v;
                    for i in 0..nv {
                        for j in 0..nv {
                            if i == j || ancestor(i, j) || ancestor(j, i) {
                                q_mj[(i, j)] += c * vm[i] * vm[j];
                            }
                        }
                    }
                }
                if let Some(d) = &rne {
                    for i in 0..nv {
                        for j in 0..nv {
                            if i == j || ancestor(i, j) || ancestor(j, i) {
                                q_mj[(i, j)] += d[(i, j)];
                            }
                        }
                    }
                }
                mh -= (tb.transpose() * q_mj * &tb) * h;
            }
            let rhs = &m * (&tinv * &qacc_mj);
            // LU, not Cholesky: MuJoCo factors this matrix without asking it to be positive definite, and a
            // destabilising velocity gain or an Euler base near its singularity can leave it merely invertible
            &tb * mh.lu().solve(&rhs).ok_or("the integrator's modified mass matrix is singular")?
        };
        // ⭐ MuJoCo 3.13.0, `implicitfast`: a STANDALONE free body (one free joint, no other dof in its tree,
        // no mass below it) is solved on its own, as an unsymmetric 6×6 `A = M − h·∂qfrc/∂qvel` that also
        // carries the gyroscopic derivative the global symmetric solve drops (`mjd_freeMhat`). Its six rows are
        // decoupled from every other dof, so overwriting them changes nothing else.
        if self.integrator == MjcfIntegrator::ImplicitFast {
            let frames = crate::tree_frames(&self.tree, &q);
            let rhs_mj = &forward.m * &qacc_mj;
            let tinv_t = tinv.transpose();
            for j in self.joints.iter().filter(|j| j.kind == MjcfJointKind::Free) {
                let (adr, block) = (j.first, j.first..j.first + 6);
                let child_dof = self.tree.joints.iter().enumerate().any(|(k, _)| !block.contains(&k) && {
                    let mut p = self.tree.parent[k];
                    loop {
                        if p < 0 {
                            break false;
                        }
                        if block.contains(&(p as usize)) {
                            break true;
                        }
                        p = self.tree.parent[p as usize];
                    }
                });
                let heavy_child = self.body_frames.iter().any(|(b, (ride, _))| b != &j.body && block.contains(ride) && self.body_mass.get(b).copied().unwrap_or(0.0) != 0.0);
                if child_dof || heavy_child {
                    continue;
                }
                let (Some((ride, pre)), Some(inertia)) = (self.body_frames.get(&j.body), self.body_iinertia.get(&j.body)) else { continue };
                let mass = self.body_mass.get(&j.body).copied().unwrap_or(0.0);
                let mut a = nalgebra::Matrix6::<f64>::zeros();
                for r in 0..6 {
                    for c in 0..6 {
                        a[(r, c)] = forward.m[(adr + r, adr + c)];
                    }
                    a[(r, r)] += h * damping[adr + r];
                }
                for (coef, v) in &rank_one {
                    let vm = &tinv_t * v;
                    for r in 0..6 {
                        for c in 0..6 {
                            a[(r, c)] -= h * coef * vm[adr + r] * vm[adr + c];
                        }
                    }
                }
                // `freeBias_vel_blocks`: the derivative of the free body's bias force in MuJoCo's basis
                let body = frames[*ride] * pre;
                let rm = body.rotation.to_rotation_matrix().into_inner();
                let (ipos, irot) = self.inertial_rt(&j.body);
                let xi = rm * irot;
                let s_off = rm * ipos;
                let w = rm * Vector3::new(qvel[adr + 3], qvel[adr + 4], qvel[adr + 5]);
                let iw = xi * Matrix3::from_diagonal(&Vector3::new(inertia[0], inertia[1], inertia[2])) * xi.transpose();
                let (ws, iww) = (w.cross(&s_off), iw * w);
                let k = s_off * w.transpose() - Matrix3::identity() * w.dot(&s_off) + ws.cross_matrix();
                let lin = k * rm;
                let cmat = -(s_off.cross_matrix() * k) * mass + w.cross_matrix() * iw - iww.cross_matrix();
                let rot = rm.transpose() * cmat * rm;
                for r in 0..3 {
                    for c in 0..3 {
                        a[(r, 3 + c)] += -h * mass * lin[(r, c)];
                        a[(3 + r, 3 + c)] += h * rot[(r, c)];
                    }
                }
                let b = nalgebra::Vector6::from_iterator((0..6).map(|r| rhs_mj[adr + r]));
                if let Some(x) = a.lu().solve(&b) {
                    for r in 0..6 {
                        applied[adr + r] = x[r];
                    }
                }
            }
        }
        // `mj_advance`: the activations first, on the rate at the start of the step
        let act_next = self.next_activation(act, &forward.act_dot, h);
        // `mju_addToScl(qvel, qacc, h)`: fused, as it compiles
        let qvel_next: Vec<f64> = qvel.iter().zip(applied.iter()).map(|(v, a)| a.mul_add(h, *v)).collect();
        // `mj_integratePos`, with the new velocity
        let qpos_next = self.integrate_pos(qpos, &qvel_next, h);
        Ok(MjcfStep { qpos: qpos_next, qvel: qvel_next, act: act_next, time: time + h, qacc_integrated: applied.iter().copied().collect(), forward, reset: None })
    }

    /// **`mj_integratePos`**: `qpos` moved by `h·v`, a velocity in MuJoCo's basis — a hinge or slide by
    /// `h·v`, a free joint's position by `h·v` and every quaternion by `mju_quatIntegrate`.
    pub fn integrate_pos(&self, qpos: &[f64], v: &[f64], h: f64) -> Vec<f64> {
        let mut out = qpos.to_vec();
        let mut vadr = 0usize;
        for (j, padr) in self.joints.iter().zip(self.qposadr()) {
            match j.kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => out[padr] = h.mul_add(v[vadr], out[padr]),
                MjcfJointKind::Free => {
                    for k in 0..3 {
                        out[padr + k] = h.mul_add(v[vadr + k], out[padr + k]);
                    }
                    quat_integrate(&mut out[padr + 3..padr + 7], &v[vadr + 3..vadr + 6], h);
                }
                MjcfJointKind::Ball => quat_integrate(&mut out[padr..padr + 4], &v[vadr..vadr + 3], h),
            }
            vadr += j.kind.dofs();
        }
        out
    }

    /// **`mj_RungeKutta`, `N = 4`**, ported from MuJoCo 3.13.0: four forward passes at the stage states
    /// `X₀ ⊕ h·Σⱼ Aᵢⱼ·(v, a)ⱼ` of the classic tableau (A = ½, ½, 1 on the sub-diagonal; B = ⅙, ⅓, ⅓, ⅙),
    /// positions composed through `mj_integratePos` from the ORIGINAL `qpos` at every stage, and the final
    /// step taken with the B-weighted velocity and acceleration: `qvel += h·Σ Bⱼ aⱼ`, and `qpos` moved by
    /// `h·Σ Bⱼ vⱼ` — the WEIGHTED velocity, not the new one. There is no implicit damping. Accumulations are
    /// fused multiply-adds, as `mju_addToScl` compiles. The activations ride along as the last block of the
    /// state: each stage sets `act₀ + h·Σⱼ Aᵢⱼ·act'ⱼ` (no clamp), and the final step is `mj_nextActivation` on
    /// the B-weighted rate, which does clamp.
    #[allow(clippy::too_many_arguments)]
    fn step_rk4(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], time: f64, applied: &MjcfApplied, check_acc: bool) -> Result<MjcfStep, String> {
        const A: [[f64; 3]; 3] = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];
        const B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
        let h = self.timestep;
        let qposadr = self.qposadr();
        let (nv, na) = (qvel.len(), self.na());
        let act0: Vec<f64> = (0..na).map(|k| act.get(k).copied().unwrap_or(0.0)).collect();
        let forward0 = self.forward_impl(&self.q_from_qpos(qpos, &qposadr)?, Some(qpos), qvel, ctrl, &act0, time, applied)?;
        if let Some(i) = forward0.qacc.iter().position(|x| check_acc && (x.is_nan() || x.abs() > MJ_MAXVAL)) {
            return self.step_after_reset(MjcfReset::BadQacc(i), false);
        }
        let mut xv: Vec<Vec<f64>> = vec![qvel.to_vec()];
        let mut fa: Vec<Vec<f64>> = vec![forward0.qacc.clone()];
        let mut fx: Vec<Vec<f64>> = vec![forward0.act_dot.clone()];
        for i in 1..4 {
            let (mut dv, mut da, mut dx) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; na]);
            for j in 0..i {
                for k in 0..nv {
                    dv[k] = xv[j][k].mul_add(A[i - 1][j], dv[k]);
                    da[k] = fa[j][k].mul_add(A[i - 1][j], da[k]);
                }
                for k in 0..na {
                    dx[k] = fx[j][k].mul_add(A[i - 1][j], dx[k]);
                }
            }
            let xq = self.integrate_pos(qpos, &dv, h);
            let v: Vec<f64> = (0..nv).map(|k| da[k].mul_add(h, qvel[k])).collect();
            let x: Vec<f64> = (0..na).map(|k| dx[k].mul_add(h, act0[k])).collect();
            // the stage's time, `t + Cᵢ·h` with `Cᵢ` the row sum of A
            let f = self.forward_impl(&self.q_from_qpos(&xq, &qposadr)?, Some(&xq), &v, ctrl, &x, A[i - 1].iter().sum::<f64>().mul_add(h, time), applied)?;
            xv.push(v);
            fa.push(f.qacc);
            fx.push(f.act_dot);
        }
        let (mut dv, mut da, mut dx) = (vec![0.0; nv], vec![0.0; nv], vec![0.0; na]);
        for j in 0..4 {
            for k in 0..nv {
                dv[k] = xv[j][k].mul_add(B[j], dv[k]);
                da[k] = fa[j][k].mul_add(B[j], da[k]);
            }
            for k in 0..na {
                dx[k] = fx[j][k].mul_add(B[j], dx[k]);
            }
        }
        let act_next = self.next_activation(&act0, &dx, h);
        let qvel_next: Vec<f64> = (0..nv).map(|k| da[k].mul_add(h, qvel[k])).collect();
        let qpos_next = self.integrate_pos(qpos, &dv, h);
        Ok(MjcfStep { qpos: qpos_next, qvel: qvel_next, act: act_next, time: time + h, qacc_integrated: da, forward: forward0, reset: None })
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

/// Which equality, and everything its rows need: `joint`, `tendon`, `connect` and `weld` (the flex
/// equalities are named in [`MjcfTree::equalities_unsupported`] instead).
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
        /// the two BODIES (empty for the world): the constraint's force acts on them, which is where a
        /// force sensor's subtree sum has to see it
        bodies: [String; 2],
    },
    /// `<equality tendon>`: a polynomial coupling between two tendon LENGTHS, each measured from its length at
    /// `qpos0` (`tendon_length0`), or one tendon held at that length. ONE row, Jacobian `J₁ − p′(y)·J₂`.
    Tendon {
        /// index into [`MjcfTree::tendons`]
        tendon1: usize,
        tendon2: Option<usize>,
        /// the two `tendon_length0`
        length0: (f64, f64),
        polycoef: [f64; 5],
    },
    /// `<equality weld>`: the two bodies (or sites) held at a fixed relative POSE. SIX rows: the three of a
    /// connect between the two anchors, then three for orientation, `torquescale·Im(q̄₂·q₁·relpose)`, with
    /// MuJoCo's `J̇·v` on all six.
    Weld {
        /// each anchor as `(the tree dof it rides, the point in THAT frame)`, `None` for the world
        side1: (Option<usize>, Vector3<f64>),
        side2: (Option<usize>, Vector3<f64>),
        /// each object's orientation as `(the tree dof it rides, its rotation in THAT frame)`
        rot1: (Option<usize>, UnitQuaternion<f64>),
        rot2: (Option<usize>, UnitQuaternion<f64>),
        /// `eq_data[6..10]`: the relative orientation held, `(w, x, y, z)` — the identity for sites
        relpose: [f64; 4],
        torquescale: f64,
        /// `mj_diagApprox`: the two bodies' `body_invweight0`, translational and rotational, summed
        diag: [f64; 2],
        bodies: [String; 2],
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
    /// The body each named site sits on.
    pub site_body: BTreeMap<String, String>,
    /// Each named site's shape: its `type` and `size` (MuJoCo's `site_type`, `site_size`) — the zone a
    /// touch sensor counts contacts in.
    pub site_shapes: BTreeMap<String, (crate::mujoco_collision::GeomType, [f64; 3])>,
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
    /// `<sensor>`, every child in file order — MuJoCo's sensor ids. See [`crate::mjcf_sensor`].
    pub sensors: Vec<crate::mjcf_sensor::MjcfSensor>,
    /// `<keyframe><key>`, in file order, completed as `mjCKey::Compile` completes them.
    pub keyframes: Vec<MjcfKeyframe>,
    /// `<option magnetic>`, the field a magnetometer reads; MuJoCo's default `0 -0.5 0`.
    pub magnetic: Vector3<f64>,
    /// **The chart each free or ball joint's three Euler hinges are measured from**, keyed by index into
    /// [`MjcfTree::joints`]: the joint's rotation is `chart · Rz(yaw)·Ry(pitch)·Rx(roll)`. Empty (every chart
    /// the identity) as loaded; [`MjcfTree::charted`] moves it. See there for why.
    pub chart: BTreeMap<usize, UnitQuaternion<f64>>,
    /// `dof_invweight0`, `body_invweight0` and `tendon_invweight0`, computed on first use at `qpos0` in the
    /// LOADED chart and kept, as MuJoCo compiles them once
    invweight0_cache: std::sync::OnceLock<Invweight0>,
    /// what `collide` reads that does not move — the weld filter, excludes, explicit pairs, bounds — by index
    collide_cache: std::sync::OnceLock<CollideStatic>,
    /// `<option density>`, `<option viscosity>` and `<option wind>` — the ambient medium. With both density
    /// and viscosity zero there is no fluid force at all, which is MuJoCo's default.
    pub density: f64,
    pub viscosity: f64,
    pub wind: Vector3<f64>,
    /// `<option cone>`: which friction cone the contact rows model. MuJoCo's default is PYRAMIDAL.
    pub cone: crate::mujoco_contact::Cone,
    /// `<option impratio>`, default 1: how much stiffer the frictional rows are made than the normal one.
    pub impratio: f64,
    /// `<option integrator>`: which of MuJoCo's integrators [`MjcfTree::step_mujoco`] must reproduce.
    pub integrator: MjcfIntegrator,
    /// `<option noslip_iterations>` (default 0) and `noslip_tolerance` (default 1e-6): when non-zero, every
    /// forward pass re-solves the friction forces after the main solve ([`crate::mujoco_noslip`]).
    pub noslip_iterations: usize,
    pub noslip_tolerance: f64,
    /// MuJoCo's `stat.meaninertia`: the mean diagonal of the mass matrix at `qpos0`, in MuJoCo's basis.
    pub meaninertia: f64,
    /// `<option><flag eulerdamp>`, default enabled: whether the `Euler` integrator treats dof damping
    /// implicitly. Every Menagerie file written for MJX disables it.
    pub eulerdamp: bool,
    /// Each body's PRINCIPAL inertia and the orientation of its inertial frame in its body frame —
    /// MuJoCo's `body_inertia` and `body_iquat`. The fluid model is written in that frame and nowhere else.
    pub body_iinertia: BTreeMap<String, [f64; 3]>,
    pub body_iquat: BTreeMap<String, Matrix3<f64>>,
    /// The same frame as MuJoCo stores it: `body_iquat`, `(w x y z)`, bit for bit what its compiler computes.
    pub body_iquat_wxyz: BTreeMap<String, [f64; 4]>,
    /// ⛔ The inertial frame the KINEMATICS use (`xipos`, `ximat` in the body frame), which is not always the
    /// stored one: where `body_ipos`/`body_iquat` are within 1e-6 of the body frame (`body_sameframe`),
    /// MuJoCo copies the body frame instead. Everything that runs on `xipos` — the dynamics, gravity
    /// compensation, the fluid model, the sensors — reads this.
    pub body_inertial_runtime: BTreeMap<String, (Vector3<f64>, Matrix3<f64>)>,
    /// MuJoCo's compiled kinematic data, for placing geoms in MuJoCo's own arithmetic
    pub(crate) mj_kin: crate::mujoco_kinematics::MjKinematics,
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
    /// The same meshes before centring and rotation: after `scale`/`refpos`/`refquat`, in `f64` as the
    /// compiler holds them, with a mirrored mesh's faces rewound as MuJoCo rewinds them — for placing them
    /// under a frame chosen elsewhere.
    pub mesh_raw: BTreeMap<String, crate::TriMesh3>,
    /// ⭐ And as the FILE states them: `f32` vertices before `refpos`, `refquat` and `scale`. This is the frame
    /// MuJoCo builds the convex hull and merges its polygons in (`MakeGraph` and `MakePolygons` run before
    /// `ApplyTransformations`), so it decides which vertices are on the hull and how faces group.
    pub mesh_file: BTreeMap<String, crate::TriMesh3>,
    /// **What MuJoCo's mesh compiler produced for every mesh** — volume, centre of mass and principal
    /// frame, which it publishes as `mesh_pos` and `mesh_quat`. Every body's `ipos` and inertia is built
    /// from these, so this is the level at which a mesh-integral residual can be seen at all. ⭐ Bit for bit
    /// MuJoCo's on all 2,339 Menagerie meshes, and so are the 788 collision meshes' stored `f32` vertices
    /// (`examples/menagerie_mesh_store`).
    pub mesh_props: BTreeMap<String, MeshData>,
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

/// The problem [`MjcfTree::constraint_problem`] poses, in MuJoCo's coordinates.
#[derive(Clone, Debug)]
pub struct MjcfConstraintProblem {
    /// `T⁻ᵀ·M·T⁻¹`
    pub m: nalgebra::DMatrix<f64>,
    /// `qacc_smooth`, the acceleration with no constraint at all
    pub a0: nalgebra::DVector<f64>,
    /// every row (Jacobian already `J·T⁻¹`), with `efc_aref`, `efc_D` and its law, in MuJoCo's order
    pub rows: crate::mujoco_contact::AssembledRows,
    /// the index of the first CONTACT block in `rows.blocks`; the equality, friction and limit blocks
    /// come before it
    pub contact_blocks_from: usize,
}

/// What [`MjcfTree::forward_mujoco`] returns: the answer and the pieces that produced it.
#[derive(Clone, Debug)]
pub struct MjcfForward {
    /// the constrained acceleration, in MuJoCo's coordinates
    pub qacc: Vec<f64>,
    /// the unconstrained one
    pub qacc_smooth: Vec<f64>,
    /// how many constraint rows (`d.nefc`)
    pub nefc: usize,
    pub collision: MjcfCollision,
    /// the solve itself: `efc_force`, `efc_state`, the cost and the gradient at exit
    pub solve: crate::mujoco_contact::NewtonSolve,
    /// the mass matrix in MuJoCo's coordinates, `T⁻ᵀ·M·T⁻¹` — what an integrator needs next
    pub m: nalgebra::DMatrix<f64>,
    /// `d.act_dot`: each activation's rate ([`MjcfTree::na`] entries)
    pub act_dot: Vec<f64>,
}

/// `<option integrator>`. Menagerie uses `implicitfast` on 109 models, `Euler` (the default) on 99 and `RK4`
/// on 2; `implicit` on none, and it is carried all the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MjcfIntegrator {
    Euler,
    Rk4,
    Implicit,
    ImplicitFast,
}

/// One [`MjcfTree::step_mujoco`]: the next state, in MuJoCo's layout, and the forward pass it came from.
#[derive(Clone, Debug)]
pub struct MjcfStep {
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    /// `d.act`, one step on ([`MjcfTree::na`] entries)
    pub act: Vec<f64>,
    /// `d.time` after the step: the start time plus `<option timestep>`, or one timestep after an auto-reset
    pub time: f64,
    /// the acceleration the integrator actually applied: `qacc` itself, or with damping the implicit
    /// `(M + h·D)⁻¹·M·qacc`
    pub qacc_integrated: Vec<f64>,
    pub forward: MjcfForward,
    /// `Some` when MuJoCo's auto-reset fired this step, and why; the state was then stepped from
    /// `mj_resetData`'s, with zero control.
    pub reset: Option<MjcfReset>,
}

/// Why `mj_step` reset the state (`mjWARN_BADQPOS`, `mjWARN_BADQVEL`, `mjWARN_BADQACC`), with the first bad
/// index — see [`MjcfTree::step_mujoco`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MjcfReset {
    BadQpos(usize),
    BadQvel(usize),
    BadQacc(usize),
}

/// **The forces a caller applies**, MuJoCo's `qfrc_applied` (generalised, in MuJoCo's `qvel` basis) and
/// `xfrc_applied` (per body: a world force and torque, `[fx, fy, fz, τx, τy, τz]`, acting at the body's centre
/// of mass).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MjcfApplied {
    pub qfrc: Vec<f64>,
    pub xfrc: BTreeMap<String, [f64; 6]>,
}

/// **The state and inputs of one simulation — MuJoCo's `mjData`, the part a loop sets and reads.** Make it
/// with [`MjcfTree::make_data`], advance it with [`MjcfTree::step`], evaluate it with [`MjcfTree::forward`]
/// and [`MjcfTree::sensors_of`]. `qpos` and `qvel` are in MuJoCo's layout and basis; `act` is `d.act`.
#[derive(Clone, Debug, PartialEq)]
pub struct MjcfData {
    pub time: f64,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub act: Vec<f64>,
    pub ctrl: Vec<f64>,
    pub applied: MjcfApplied,
}

impl MjcfTree {
    /// `mj_makeData`: the reset state — `qpos0`, everything else zero.
    pub fn make_data(&self) -> MjcfData {
        MjcfData { time: 0.0, qpos: self.qpos0(), qvel: vec![0.0; self.tree.joints.len()], act: vec![0.0; self.na()], ctrl: vec![0.0; self.actuators.len()], applied: MjcfApplied::default() }
    }

    /// `mj_resetData`.
    pub fn reset_data(&self, d: &mut MjcfData) {
        *d = self.make_data();
    }

    /// `mj_resetDataKeyframe`: the reset state, then the key's time, `qpos`, `qvel`, `act` and `ctrl`.
    pub fn reset_data_keyframe(&self, d: &mut MjcfData, key: usize) -> Result<(), String> {
        let k = self.keyframes.get(key).ok_or_else(|| format!("no keyframe {key}"))?;
        *d = self.make_data();
        d.time = k.time;
        d.qpos = k.qpos.clone();
        d.qvel = k.qvel.clone();
        d.act = k.act.clone();
        d.ctrl = k.ctrl.clone();
        Ok(())
    }

    /// `mj_forward` on `d`: its constrained acceleration and everything the solve found.
    pub fn forward(&self, d: &MjcfData) -> Result<MjcfForward, String> {
        let t = self.chart_if_needed(&d.qpos)?;
        let t = t.as_ref().unwrap_or(self);
        t.forward_impl(&t.q_from_qpos(&d.qpos, &t.qposadr())?, Some(&d.qpos), &d.qvel, &d.ctrl, &d.act, d.time, &d.applied)
    }

    /// `mj_step` on `d`: advances `time`, `qpos`, `qvel` and `act` by one `<option timestep>`, and returns
    /// the step's details. ⛔ After an auto-reset ([`MjcfStep::reset`]) `d` is what MuJoCo's is: the control
    /// and the applied forces zeroed too.
    pub fn step(&self, d: &mut MjcfData) -> Result<MjcfStep, String> {
        let s = self.step_impl(&d.qpos, &d.qvel, &d.ctrl, &d.act, d.time, &d.applied)?;
        d.qpos.clone_from(&s.qpos);
        d.qvel.clone_from(&s.qvel);
        d.act.clone_from(&s.act);
        d.time = s.time;
        if s.reset.is_some() {
            d.ctrl.iter_mut().for_each(|u| *u = 0.0);
            d.applied = MjcfApplied::default();
        }
        Ok(s)
    }

    /// `d.sensordata` after `mj_forward`: one entry per sensor ([`MjcfTree::sensordata`]), with the applied
    /// forces in the force sensors' books.
    pub fn sensors_of(&self, d: &MjcfData) -> Result<Vec<Option<Vec<f64>>>, String> {
        use crate::mjcf_sensor::SensorKind::*;
        let acc_stage = self.sensors.iter().any(|s| matches!(s.kind, Accelerometer | Force | Torque | Touch | FrameLinAcc | FrameAngAcc));
        let forward = if acc_stage { Some(self.forward(d)?) } else { None };
        self.sensordata_applied(&d.qpos, &d.qvel, &d.ctrl, &d.act, d.time, forward.as_ref(), &d.applied)
    }
}

/// One `<keyframe><key>`: a named state to reset to (`mj_resetDataKeyframe`). What the file leaves out is
/// what `mjCKey::Compile` fills in — `qpos0`, zero velocity, activation and control, `time` zero — and a
/// length that does not match the model is refused, as MuJoCo refuses it.
#[derive(Clone, Debug, PartialEq)]
pub struct MjcfKeyframe {
    pub name: String,
    pub time: f64,
    pub qpos: Vec<f64>,
    pub qvel: Vec<f64>,
    pub act: Vec<f64>,
    pub ctrl: Vec<f64>,
    /// the mocap bodies' positions and quaternions, as the file states them; empty when it does not, which
    /// means the mocap bodies' own poses
    pub mpos: Vec<f64>,
    pub mquat: Vec<f64>,
}

/// A keyframe's quaternion as MuJoCo stores it: normalised TWICE — `mjuu_normvec` when the keys are copied
/// in (by division, and a zero quaternion left as it is), then `mj_normalizeQuat`'s `mju_normalize4` (by
/// the reciprocal, and a zero quaternion made the identity). A file that writes `0 0 0 0` for a free base
/// (pal_talos, pal_tiago) gets `1 0 0 0`.
fn normalize_key_quat(q: &mut [f64]) {
    normvec(q);
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if norm < 1e-15 {
        q.copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    } else if (norm - 1.0).abs() > 1e-15 {
        let inv = 1.0 / norm;
        q.iter_mut().for_each(|x| *x *= inv);
    }
}

/// `mjuu_normvec`: divide by the norm unless it is below `mjEPS` (1e-14) or within `mjEPS` of one.
fn normvec(v: &mut [f64]) {
    let nrm = v.iter().map(|x| x * x).sum::<f64>();
    if nrm < 1e-14 {
        return;
    }
    let nrm = nrm.sqrt();
    if (nrm - 1.0).abs() > 1e-14 {
        v.iter_mut().for_each(|x| *x /= nrm);
    }
}

/// `mju_mulQuat`, on `(w, x, y, z)`.
fn quat_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// `mju_mulQuatAxis`: `q·(0, axis)`.
fn quat_mul_axis(q: &[f64; 4], axis: &[f64; 3]) -> [f64; 4] {
    [
        -q[1] * axis[0] - q[2] * axis[1] - q[3] * axis[2],
        q[0] * axis[0] + q[2] * axis[2] - q[3] * axis[1],
        q[0] * axis[1] + q[3] * axis[0] - q[1] * axis[2],
        q[0] * axis[2] + q[1] * axis[1] - q[2] * axis[0],
    ]
}

/// `mju_negQuat`: the conjugate.
fn quat_neg(q: &[f64; 4]) -> [f64; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// `mju_derivQuat`: `0.5·q·(0, vel)`.
fn quat_deriv(q: &[f64; 4], v: &[f64; 3]) -> [f64; 4] {
    [
        0.5 * (-v[0] * q[1] - v[1] * q[2] - v[2] * q[3]),
        0.5 * (v[0] * q[0] + v[1] * q[3] - v[2] * q[2]),
        0.5 * (-v[0] * q[3] + v[1] * q[0] + v[2] * q[1]),
        0.5 * (v[0] * q[2] - v[1] * q[1] + v[2] * q[0]),
    ]
}

/// `mjMAXVAL`: past it, `mju_isBad` calls a number bad.
const MJ_MAXVAL: f64 = 1e10;

/// The per-geom facts [`MjcfTree::collide`] filters by, resolved from names to indices once.
#[derive(Clone, Debug)]
struct CollideStatic {
    /// `(weld body, its parent's weld, the weld's dof count)` per geom — `filter_body_pair`'s inputs
    weld: Vec<(usize, usize, usize)>,
    /// `<contact><exclude>` as geom-body index pairs, both orders
    excludes: std::collections::HashSet<(usize, usize)>,
    body: Vec<usize>,
    /// explicit `<contact><pair>`s by geom index pair, both orders
    pairs: HashMap<(usize, usize), usize>,
    /// `geom_rbound`
    rbound: Vec<f64>,
}

/// The three `*_invweight0`, as [`MjcfTree`] keeps them.
#[derive(Clone, Debug)]
struct Invweight0 {
    dof: Vec<f64>,
    body: BTreeMap<String, crate::mujoco_contact::InvWeight>,
    tendon: Vec<f64>,
}

/// Past this `|cos(pitch)|` a step re-charts ([`MjcfTree::charted`]): the Euler basis's condition number
/// is `1/|cos(pitch)|`, so this bounds what the chart costs in digits to one bit.
const CHART_COS: f64 = 0.5;

/// One contact [`MjcfTree::collide`] found: the two geoms (indices into [`MjcfTree::geoms`], lower first,
/// as MuJoCo orders them) and the contact as `mj_setContact` completes it.
#[derive(Clone, Debug)]
pub struct MjcfContact {
    pub geom: [usize; 2],
    pub record: crate::mujoco_collision::ContactRecord,
}

/// What `mj_collision` decided at one state; see [`MjcfTree::collide`].
#[derive(Clone, Debug, Default)]
pub struct MjcfCollision {
    /// every contact detected, INCLUDING those in the gap band (`record.exclude`), which the solver skips
    pub contacts: Vec<MjcfContact>,
    /// geom pairs this port cannot collide (a height field): a state with any is not the whole answer
    pub refused: Vec<[usize; 2]>,
    /// for each pair considered, what happened to it — the gate that dropped it, or what the collider found;
    /// filled only by [`MjcfTree::collide_explained`]
    pub why: BTreeMap<(usize, usize), String>,
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

    /// **MuJoCo's `qpos0`, in MuJoCo's layout**: what `mj_resetData` puts back. A hinge or slide at its `ref`,
    /// a ball at the identity, a free joint at the pose its `<body>` states.
    pub fn qpos0(&self) -> Vec<f64> {
        let mut out = Vec::new();
        for j in &self.joints {
            match j.kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => out.push(j.reference),
                MjcfJointKind::Ball => out.extend([1.0, 0.0, 0.0, 0.0]),
                MjcfJointKind::Free => {
                    let pose = self.free_base_pose.get(&j.body).copied().unwrap_or_else(Iso::identity);
                    let r = pose.rotation.quaternion();
                    out.extend(pose.translation.vector.iter().copied());
                    out.extend([r.w, r.i, r.j, r.k]);
                }
            }
        }
        out
    }

    /// **This model with every free and ball joint RE-CHARTED at `qpos`**, so that its Euler angles there
    /// are zero.
    ///
    /// ⛔⛔ This port gives a free or ball joint three hinges, yaw-pitch-roll, and the map from their rates to
    /// MuJoCo's angular velocity is SINGULAR at a pitch of ±90° — gimbal lock — and ill-conditioned near it
    /// (`1/|cos(pitch)|`). A robot that falls flat on its face goes straight through it. The cure is a moving
    /// chart: put a constant rotation `C` ahead of the yaw hinge, `R = C·Rz·Ry·Rx`, and choose `C` as the
    /// current rotation, so the angles are zero and the map is a permutation. `C` is constant, so the
    /// body-frame angular velocity is the same function of the Euler rates and nothing downstream changes.
    ///
    /// Everything in MuJoCo's coordinates — `qpos`, `qvel`, `qacc`, forces, the mass matrix — is the same in
    /// any chart up to rounding. The `*_invweight0` are inherited from this model, not recomputed.
    /// [`MjcfTree::step_mujoco`] re-charts on its own once a pitch passes 60°; a caller of
    /// [`MjcfTree::forward_mujoco`] near the singularity should call this first and pass `q_from_qpos` of the
    /// result.
    pub fn charted(&self, qpos: &[f64]) -> Result<MjcfTree, String> {
        let _ = self.invweight0();
        let qposadr = self.qposadr();
        let mut t = self.clone();
        for (ji, j) in self.joints.iter().enumerate() {
            let (padr, hinge, reference) = match j.kind {
                MjcfJointKind::Free => (qposadr[ji] + 3, j.first + 3, self.free_base_pose.get(&j.body).map_or_else(UnitQuaternion::identity, |p| p.rotation)),
                MjcfJointKind::Ball => (qposadr[ji], j.first, UnitQuaternion::identity()),
                _ => continue,
            };
            let w = qpos.get(padr..padr + 4).ok_or("charted: qpos is too short")?;
            let r = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(w[0], w[1], w[2], w[3]));
            let old = self.chart.get(&ji).copied().unwrap_or_else(UnitQuaternion::identity);
            let origin = self.tree.joints[hinge].origin;
            t.tree.joints[hinge].origin = Iso::from_parts(origin.translation, origin.rotation * old.inverse() * r);
            t.chart.insert(ji, r);
            let (roll, pitch, yaw) = (r.inverse() * reference).euler_angles();
            t.reference_q[hinge..hinge + 3].copy_from_slice(&[yaw, pitch, roll]);
        }
        Ok(t)
    }

    /// `Some` re-charted copy when a free or ball joint's pitch at `qpos` is past [`CHART_COS`].
    fn chart_if_needed(&self, qpos: &[f64]) -> Result<Option<MjcfTree>, String> {
        let qposadr = self.qposadr();
        let near = self.joints.iter().enumerate().any(|(ji, j)| {
            let padr = match j.kind {
                MjcfJointKind::Free => qposadr[ji] + 3,
                MjcfJointKind::Ball => qposadr[ji],
                _ => return false,
            };
            qpos.get(padr..padr + 4).is_some_and(|w| ypr_in(w, self.chart.get(&ji)).1.cos().abs() < CHART_COS)
        });
        if near {
            self.charted(qpos).map(Some)
        } else {
            Ok(None)
        }
    }

    /// **Map MuJoCo's `qpos` onto this tree's `q`.** `qposadr` gives each MuJoCo joint's address, in the order
    /// of [`MjcfTree::joints`]. Hinge and slide copy through; a ball's quaternion becomes `(yaw, pitch, roll)`
    /// on its three hinges; a free joint copies its translation and does the same with its quaternion.
    pub fn q_from_qpos(&self, qpos: &[f64], qposadr: &[usize]) -> Result<Vec<f64>, String> {
        if qposadr.len() != self.joints.len() {
            return Err(format!("{} qpos addresses for {} joints", qposadr.len(), self.joints.len()));
        }
        let mut q = vec![0.0; self.tree.dof()];
        for (ji, (j, &adr)) in self.joints.iter().zip(qposadr).enumerate() {
            let need = adr + j.kind.qpos_width();
            if need > qpos.len() {
                return Err(format!("joint '{}' reads qpos[{adr}..{need}] of {}", j.name, qpos.len()));
            }
            match j.kind {
                MjcfJointKind::Hinge | MjcfJointKind::Slide => q[j.first] = qpos[adr],
                MjcfJointKind::Ball => {
                    let (yaw, pitch, roll) = ypr_in(&qpos[adr..adr + 4], self.chart.get(&ji));
                    q[j.first] = yaw;
                    q[j.first + 1] = pitch;
                    q[j.first + 2] = roll;
                }
                MjcfJointKind::Free => {
                    q[j.first..j.first + 3].copy_from_slice(&qpos[adr..adr + 3]);
                    let (yaw, pitch, roll) = ypr_in(&qpos[adr + 3..adr + 7], self.chart.get(&ji));
                    q[j.first + 3] = yaw;
                    q[j.first + 4] = pitch;
                    q[j.first + 5] = roll;
                }
            }
        }
        Ok(q)
    }
}

/// `(yaw, pitch, roll)` of a MuJoCo `(w, x, y, z)` quaternion, in the order the three hinges take them,
/// measured from `chart` (the identity when `None`).
fn ypr_in(wxyz: &[f64], chart: Option<&UnitQuaternion<f64>>) -> (f64, f64, f64) {
    let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]));
    let q = match chart {
        Some(c) => c.inverse() * q,
        None => q,
    };
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
///
/// ⭐ MuJoCo publishes `com` and `quat` as `mesh_pos` and `mesh_quat`, so these are directly comparable
/// to the reference — and they sit UPSTREAM of every body's `ipos` and inertia, which is where a residual
/// in the mesh integral first becomes visible.
#[derive(Clone, Copy, Debug)]
pub struct MeshData {
    /// volume (or area, for a shell): what the density multiplies
    pub volume: f64,
    /// centre of mass in the mesh frame (`pos_`, published as `mesh_pos`)
    pub com: Vector3<f64>,
    /// principal frame (`quat_`, published as `mesh_quat`), MuJoCo `(w x y z)`
    pub quat: [f64; 4],
    /// the equivalent inertia box half-sizes (`boxsz_`)
    pub boxsz: [f64; 3],
    /// axis-aligned bounds in the mesh frame after centring at the CoM and rotating into the principal frame
    /// (`aamm_`), which is what `fitaabb` fits
    pub aamm: [f64; 6],
}

/// What [`Walk::geom_spec`] resolves: MuJoCo's own `geom_type`, `geom_size` and the geom's frame in its body
/// (`geom_pos`, `geom_quat`), with a mesh's own frame — the centre of mass and principal axes
/// `mjCMesh::Process` left it in — already accumulated (`mjuu_frameaccum`), as `mjCGeom::Compile` does.
struct GeomSpec {
    ty: String,
    size: Vec<f64>,
    mj: MjPose,
    mesh: Option<String>,
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
    /// as written: `ApplyTransformations` normalises it itself (`mjuu_normvec`)
    refquat: [f64; 4],
    inertia: MeshInertia,
}

/// Volume (or area, for a shell), centre of mass and inertia per unit density about that centre, computed
/// exactly as `mjCMesh::Process` does: face centroid → one pass for the CoM → one pass for the second moments
/// about it. Both passes take pyramids from the apex to each face; `legacy` takes their volumes absolute.
pub(crate) fn mesh_inertia_mujoco(mesh: &crate::TriMesh3, method: MeshInertia) -> Result<(f64, Vector3<f64>, Matrix3<f64>), String> {
    const MINVAL: f64 = 1e-15;
    // `triangle()`: unit normal, centre, area — faces below MINVAL in |cross| are ignored (area 0)
    let tri = |a: Vector3<f64>, b: Vector3<f64>, c: Vector3<f64>| -> (Vector3<f64>, Vector3<f64>, f64) {
        let center = Vector3::new((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0, (a.z + b.z + c.z) / 3.0);
        let n = uu_cross(&(b - a), &(c - a));
        let len = uu_dot3(&n, &n).sqrt();
        if len < MINVAL {
            return (Vector3::zeros(), center, 0.0);
        }
        (Vector3::new(n.x / len, n.y / len, n.z / len), center, 0.5 * len)
    };
    let v = |i: usize| mesh.verts[i];
    // `ComputeFaceCentroid`: area-weighted centroid of face centres
    let (mut facecen, mut total_area) = (Vector3::zeros(), 0.0);
    for t in &mesh.tris {
        let (_, c, area) = tri(v(t[0]), v(t[1]), v(t[2]));
        facecen = Vector3::new(area.mul_add(c.x, facecen.x), area.mul_add(c.y, facecen.y), area.mul_add(c.z, facecen.z));
        total_area += area;
    }
    if total_area < MINVAL {
        return Err("mesh surface area is too small".into());
    }
    facecen = Vector3::new(facecen.x / total_area, facecen.y / total_area, facecen.z / total_area);
    // `ComputeVolume` / `ComputeSurfaceArea`: volume (or area) and CoM from pyramids at the face centroid
    let (mut vol, mut com) = (0.0, Vector3::zeros());
    for t in &mesh.tris {
        let (n, c, area) = tri(v(t[0]), v(t[1]), v(t[2]));
        let mut volume = match method {
            MeshInertia::Shell => area,
            _ => uu_dot3(&(c - facecen), &n) * area / 3.0,
        };
        if method == MeshInertia::Legacy {
            volume = volume.abs();
        }
        vol += volume;
        // `CoM += volume * (center * 3.0 / 4.0 + facecen / 4.0)`: the sum's terms are quotients, so only the
        // accumulation fuses
        let x = |k: usize| c[k] * 3.0 / 4.0 + facecen[k] / 4.0;
        com = Vector3::new(volume.mul_add(x(0), com.x), volume.mul_add(x(1), com.y), volume.mul_add(x(2), com.z));
    }
    if vol < MINVAL {
        return Err(if vol < 0.0 { "mesh volume is negative (misoriented triangles)".into() } else { "mesh volume is too small".into() });
    }
    com = Vector3::new(com.x / vol, com.y / vol, com.z / vol);
    // `ComputeInertia`: products of inertia about the CoM, volume recomputed with the CoM as apex
    let (mut total, mut p) = (0.0, [0.0f64; 6]);
    let k = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
    let cdiv = if method == MeshInertia::Shell { 12.0 } else { 20.0 };
    for t in &mesh.tris {
        let (d, e, f) = (v(t[0]) - com, v(t[1]) - com, v(t[2]) - com);
        let (n, c, area) = tri(d, e, f);
        let mut volume = match method {
            MeshInertia::Shell => area,
            _ => uu_dot3(&c, &n) * area / 3.0,
        };
        if method == MeshInertia::Legacy {
            volume = volume.abs();
        }
        total += volume;
        for (j, &(a, b)) in k.iter().enumerate() {
            // `2*(DD + EE + FF) + DE + ED + DF + FD + EF + FE`, each later product fused into the running sum
            let sq = f[a].mul_add(f[b], d[a].mul_add(d[b], e[a] * e[b]));
            let mut sum = 2.0f64.mul_add(sq, d[a] * e[b]);
            sum = d[b].mul_add(e[a], sum);
            sum = d[a].mul_add(f[b], sum);
            sum = d[b].mul_add(f[a], sum);
            sum = e[a].mul_add(f[b], sum);
            sum = e[b].mul_add(f[a], sum);
            p[j] = (volume / cdiv).mul_add(sum, p[j]);
        }
    }
    let inertia = Matrix3::new(p[1] + p[2], -p[3], -p[4], -p[3], p[0] + p[2], -p[5], -p[4], -p[5], p[0] + p[1]);
    Ok((total, com, inertia))
}

// ⛔ MuJoCo's compiler is C++ that clang builds with `-ffp-contract=on` on the arm64 wheel, so the helpers
// below fuse where clang fuses (the rule is in `mujoco_ccd`: the LEFT product of `a*b + c*d`, each later
// product of a chain into the running sum, `s += a*b`). A stored mesh vertex is a double rounded to `f32`:
// one ulp in the centre of mass can move a whole binade of vertices to the neighbouring float.

/// `mjuu_dot3`, contracted.
fn uu_dot3(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.z.mul_add(b.z, a.x.mul_add(b.x, a.y * b.y))
}

/// `mjuu_crossvec` (`b × c`), contracted.
fn uu_cross(b: &Vector3<f64>, c: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(b.y.mul_add(c.z, -(b.z * c.y)), b.z.mul_add(c.x, -(b.x * c.z)), b.x.mul_add(c.y, -(b.y * c.x)))
}

/// `mjuu_normvec`: the squared norm below `mjEPS` (1e-14) leaves the vector alone and returns 0, and a norm
/// within `mjEPS` of 1 leaves it alone too; otherwise each component is DIVIDED by the norm, which is
/// returned. The sum of squares is one fused accumulation.
fn uu_normvec(v: &mut [f64]) -> f64 {
    let mut n2 = 0.0f64;
    for x in v.iter() {
        n2 = x.mul_add(*x, n2);
    }
    if n2 < 1e-14 {
        return 0.0;
    }
    let n = n2.sqrt();
    if (n - 1.0).abs() > 1e-14 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
    n
}

fn uu_normvec4(q: &mut [f64; 4]) {
    uu_normvec(q);
}

/// `mjuu_quat2mat` (its products are separate statements, so nothing fuses).
fn uu_quat2mat(q: &[f64; 4]) -> Matrix3<f64> {
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
}

/// `mjuu_mulquat`, contracted, normalised as it normalises.
fn uu_mulquat(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let mut r = [
        (-a[3]).mul_add(b[3], (-a[2]).mul_add(b[2], a[0].mul_add(b[0], -(a[1] * b[1])))),
        (-a[3]).mul_add(b[2], a[2].mul_add(b[3], a[0].mul_add(b[1], a[1] * b[0]))),
        a[3].mul_add(b[1], a[2].mul_add(b[0], a[0].mul_add(b[2], -(a[1] * b[3])))),
        a[3].mul_add(b[0], (-a[2]).mul_add(b[1], a[0].mul_add(b[3], a[1] * b[2]))),
    ];
    uu_normvec4(&mut r);
    r
}

/// `mjuu_mulmat` (3×3), contracted.
fn uu_mulmat(a: &Matrix3<f64>, b: &Matrix3<f64>) -> Matrix3<f64> {
    Matrix3::from_fn(|i, j| a[(i, 2)].mul_add(b[(2, j)], a[(i, 0)].mul_add(b[(0, j)], a[(i, 1)] * b[(1, j)])))
}

/// `mjuu_mulvecmat`: `mat · vec`, contracted.
fn uu_mulvecmat(v: &Vector3<f64>, m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::from_fn(|k, _| m[(k, 2)].mul_add(v.z, m[(k, 0)].mul_add(v.x, m[(k, 1)] * v.y)))
}

/// `mjuu_mulvecmatT`: `matᵀ · vec`, contracted.
fn uu_mulvecmat_t(v: &Vector3<f64>, m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::from_fn(|k, _| m[(2, k)].mul_add(v.z, m[(0, k)].mul_add(v.x, m[(1, k)] * v.y)))
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
    let (quat2mat, normalize, mulquat) = (uu_quat2mat, uu_normvec4, uu_mulquat);
    let mut quat = [1.0, 0.0, 0.0, 0.0];
    let mut eigval = [0.0; 3];
    for _ in 0..500 {
        let ev = quat2mat(&quat);
        let d = uu_mulmat(&uu_mulmat(&ev.transpose(), mat), &ev);
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
        let t = if tau >= 0.0 { 1.0 / (tau + tau.mul_add(tau, 1.0).sqrt()) } else { -1.0 / (-tau + tau.mul_add(tau, 1.0).sqrt()) };
        let c = 1.0 / t.mul_add(t, 1.0).sqrt();
        if c > 1.0 - EPS {
            break;
        }
        let mut tmp = [0.0; 4];
        let half = (-0.5f64).mul_add(c, 0.5).sqrt();
        tmp[rotk + 1] = if tau >= 0.0 { -half } else { half };
        if rotk == 1 {
            tmp[rotk + 1] = -tmp[rotk + 1];
        }
        tmp[0] = (-tmp[rotk + 1]).mul_add(tmp[rotk + 1], 1.0).sqrt();
        normalize(&mut tmp);
        quat = mulquat(&quat, &tmp);
        normalize(&mut quat);
    }
    // bubble sort into decreasing order (0, 1, 0), rotating the frame by a quarter turn about the third axis
    for j in 0..3 {
        let j1 = j % 2;
        if eigval[j1] + EPS < eigval[j1 + 1] {
            eigval.swap(j1, j1 + 1);
            // ⛔ MuJoCo's literal, which is NOT the double nearest 1/√2 (that is 0.7071067811865476)
            #[allow(clippy::approx_constant)]
            let mut tmp = [0.707106781186548, 0.0, 0.0, 0.0];
            tmp[(j1 + 2) % 3 + 1] = tmp[0];
            quat = mulquat(&quat, &tmp);
            normalize(&mut quat);
        }
    }
    (eigval, quat)
}

/// Rotation matrix of a MuJoCo `(w x y z)` quaternion, columns = axes: `mjuu_quat2mat`.
#[cfg(test)]
pub(crate) fn quat_to_rotation(q: &[f64; 4]) -> Matrix3<f64> {
    uu_quat2mat(q)
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
                    // ⛔⛔ **MuJoCo DROPS a face with more than 100 vertices.** Not an error, not a
                    // fallback — the face is silently absent from the mesh. Measured by bisection on
                    // synthetic OBJ files: a 100-gon yields its 98 triangles and a 101-gon yields none,
                    // and the cap is on the VERTEX COUNT, not on the line (a 100-gon written with 6-digit
                    // indices, 701 characters, still triangulates).
                    //
                    // It matters once: `hello_robot_stretch_3/link_SG3_gripper_body` holds one 215-gon
                    // among 23,103 faces. Fanning it gave this port 23,776 triangles against MuJoCo's
                    // 23,575 — a DIFFERENT SOLID over the same 14,083 vertices — which moved that mesh's
                    // centre of mass by 2.7e-3, the body's `ipos` by 2.5e-3 (the worst in Menagerie), its
                    // `qfrc_gravcomp` by 3.6e-4, and its constrained acceleration by 1.4e-2.
                    101.. => {}
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
    /// the obstacle this point lies on, set on the FIRST of an arc's two tangent points. A comparison
    /// against MuJoCo needs the obstacle's IDENTITY: two paths that wrap the same NUMBER of obstacles can
    /// be wrapping different ones, and a count cannot tell those apart.
    obstacle: Option<String>,
}

/// **`mju_wrap`**: where a cable leaving `x0` for `x1` first touches an obstacle and where it leaves it.
///
/// Returns the two tangent points in the world and the length between them, or `None` when MuJoCo would
/// not wrap. The construction is two-dimensional. A SPHERE works in the plane of the centre and both
/// endpoints; a CYLINDER works in its own xy-plane with an infinite axis, and the two points get their
/// heights by interpolating `z` along the 2-D path, so the arc is a helix.
///
/// ⭐⭐ **Ported line by line from MuJoCo's own source** (`mju_wrap`, `wrap_circle`, `wrap_inside`,
/// `is_intersect` and `length_circle` in `engine_util_misc.c`, identical in 3.3.7 and 3.13.0), not
/// fitted to its outputs. Four earlier rules here were measured against MuJoCo's answers and each was a
/// partial view of the code below:
///   * "the side site chooses by direction, and only when the infinite line cuts the circle" is
///     `good = dot(unit(t0 + t1), side)` everywhere, plus a penalty on any candidate whose two straight
///     runs CROSS each other. Where the line misses the circle, one candidate always crosses, which is
///     what the gate was standing in for;
///   * with no side site the candidate is the one whose tangent points lie CLOSER TOGETHER, not the
///     shorter total path;
///   * a side site INSIDE the obstacle (its 3-D distance from the centre, even for a cylinder) is a
///     different wrap altogether, `wrap_inside`: the cable passes through ONE point of the circle, found by
///     a Newton solve, and the arc between the two tangent points is zero. Treating it as an ordinary
///     circle wrap gave the wrong arc, which is why forcing the wrap there once made lengths worse;
///   * a chosen candidate whose straight runs cross is no wrap at all.
fn wrap_obstacle(x0: Vector3<f64>, x1: Vector3<f64>, pose: &Iso, radius: f64, cylinder: bool, side: Option<Vector3<f64>>) -> Option<([Vector3<f64>; 2], f64)> {
    const MINVAL: f64 = 1e-15; // mjMINVAL
    let c = pose.translation.vector;
    let xmat = pose.rotation.to_rotation_matrix().into_inner();
    // map sites to the wrap object's local frame
    let p0 = xmat.transpose() * (x0 - c);
    let p1 = xmat.transpose() * (x1 - c);
    if p0.norm() < MINVAL || p1.norm() < MINVAL {
        return None;
    }
    // the 2-D frame: a sphere's first axis is p0 and its plane is (p0, centre, p1); a cylinder's is xy
    let (ax0, ax1) = if cylinder {
        (Vector3::x(), Vector3::y())
    } else {
        let ax0 = normalize3(p0);
        let cr = p0.cross(&p1);
        let normal = if cr.norm() < MINVAL {
            // (p0, p1) parallel: a normal from p0 and a vector that is 0 at p0's largest component
            let a = ax0.map(f64::abs);
            let i = if a[1] > a[0] && a[1] > a[2] {
                1
            } else if a[2] > a[0] && a[2] > a[1] {
                2
            } else {
                0
            };
            let mut t = Vector3::new(1.0, 1.0, 1.0);
            t[i] = 0.0;
            normalize3(ax0.cross(&t))
        } else {
            cr / cr.norm()
        };
        (ax0, normalize3(normal.cross(&ax0)))
    };
    let d = [p0.dot(&ax0), p0.dot(&ax1), p1.dot(&ax0), p1.dot(&ax1)];
    // the side site: same projection, then rescaled onto the circle
    let local_side = side.map(|s| xmat.transpose() * (s - c));
    let sd = local_side.map(|s| {
        let v = normalize2([s.dot(&ax0), s.dot(&ax1)]);
        [v[0] * radius, v[1] * radius]
    });
    let (pnt, mut wlen) = match local_side {
        Some(s) if s.norm() < radius => wrap_inside(&d, radius)?,
        _ => wrap_circle(&d, sd, radius)?,
    };
    // back to 3-D in the local frame
    let mut res = [ax0 * pnt[0] + ax1 * pnt[1], ax0 * pnt[2] + ax1 * pnt[3]];
    if cylinder {
        let l0 = ((p0[0] - res[0][0]).powi(2) + (p0[1] - res[0][1]).powi(2)).sqrt();
        let l1 = ((p1[0] - res[1][0]).powi(2) + (p1[1] - res[1][1]).powi(2)).sqrt();
        res[0][2] = p0[2] + (p1[2] - p0[2]) * l0 / (l0 + wlen + l1);
        res[1][2] = p0[2] + (p1[2] - p0[2]) * (l0 + wlen) / (l0 + wlen + l1);
        let height = (res[1][2] - res[0][2]).abs();
        wlen = (wlen * wlen + height * height).sqrt();
    }
    Some(([xmat * res[0] + c, xmat * res[1] + c], wlen))
}

/// `mju_quatIntegrate`: rotate `quat` by `scale·vel`, an angular velocity in the body's own frame — the
/// axis-angle quaternion multiplied on the RIGHT, after `quat` is normalised (`mju_normalize4`, which leaves a
/// quaternion already within `mjMINVAL` of unit length untouched).
fn quat_integrate(quat: &mut [f64], vel: &[f64], scale: f64) {
    let v = Vector3::new(vel[0], vel[1], vel[2]);
    let n = v.norm();
    let axis = normalize3(v);
    let angle = scale * n;
    // ⛔ ONE `sincos` call, as clang compiles `mju_axisAngle2Quat` (`__sincos_stret`); two separate calls
    // differ in the last bit
    let qrot = if angle == 0.0 {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        let (s, c) = crate::mujoco_kinematics::sincos(angle * 0.5);
        [c, axis[0] * s, axis[1] * s, axis[2] * s]
    };
    let norm = (quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]).sqrt();
    if norm < 1e-15 {
        quat.copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
    } else if (norm - 1.0).abs() > 1e-15 {
        let k = 1.0 / norm;
        for x in quat.iter_mut() {
            *x *= k;
        }
    }
    let (a, b) = ([quat[0], quat[1], quat[2], quat[3]], qrot);
    quat[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
    quat[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
    quat[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
    quat[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

/// `mju_normalize3`: a vector shorter than `mjMINVAL` becomes the x axis rather than NaN.
fn normalize3(v: Vector3<f64>) -> Vector3<f64> {
    let n = v.norm();
    if n < 1e-15 {
        Vector3::x()
    } else {
        v * (1.0 / n)
    }
}

/// `mju_normalize` in two dimensions, with the same fallback.
fn normalize2(v: [f64; 2]) -> [f64; 2] {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n < 1e-15 {
        [1.0, 0.0]
    } else {
        let k = 1.0 / n;
        [v[0] * k, v[1] * k]
    }
}

/// `is_intersect`: do the 2-D segments p1→p2 and p3→p4 cross?
fn segments_cross(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> bool {
    let det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1]);
    if det.abs() < 1e-15 {
        return false;
    }
    let a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det;
    let b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det;
    (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)
}

/// `length_circle`: the arc from `p0` to `p1`, going the way candidate `ind` goes round.
fn arc_length(p0: [f64; 2], p1: [f64; 2], ind: usize, radius: f64) -> f64 {
    let (n0, n1) = (normalize2(p0), normalize2(p1));
    let mut angle = (n0[0] * n1[0] + n0[1] * n1[1]).acos();
    let cross = p0[1] * p1[0] - p0[0] * p1[1];
    if (cross > 0.0 && ind == 1) || (cross < 0.0 && ind == 0) {
        angle = 2.0 * std::f64::consts::PI - angle;
    }
    radius * angle
}

/// `wrap_circle`: the two tangent points and the arc, for endpoints `end` = (x0, y0, x1, y1) outside the
/// circle, or `None` for no wrap.
fn wrap_circle(end: &[f64; 4], side: Option<[f64; 2]>, radius: f64) -> Option<([f64; 4], f64)> {
    let sqlen0 = end[0] * end[0] + end[1] * end[1];
    let sqlen1 = end[2] * end[2] + end[3] * end[3];
    let sqrad = radius * radius;
    // either point inside the circle, or the circle too small
    if sqlen0 < sqrad || sqlen1 < sqrad || radius < 1e-15 {
        return None;
    }
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if dd < 1e-15 {
        return None;
    }
    // the nearest point of the SEGMENT to the centre
    let a = (-(dif[0] * end[0] + dif[1] * end[1]) / dd).clamp(0.0, 1.0);
    let near = [a * dif[0] + end[0], a * dif[1] + end[1]];
    // no wrap if the segment clears the circle, unless a side site sits across from where it passes
    if near[0] * near[0] + near[1] * near[1] > sqrad && side.is_none_or(|s| s[0] * near[0] + s[1] * near[1] >= 0.0) {
        return None;
    }
    let (sqrt0, sqrt1) = ((sqlen0 - sqrad).sqrt(), (sqlen1 - sqrad).sqrt());
    let (e0, e1) = ([end[0], end[1]], [end[2], end[3]]);
    let mut sol = [[[0.0; 2]; 2]; 2];
    let mut good = [0.0; 2];
    for (i, sgn) in [1.0, -1.0].into_iter().enumerate() {
        sol[i][0] = [(end[0] * sqrad + sgn * radius * end[1] * sqrt0) / sqlen0, (end[1] * sqrad - sgn * radius * end[0] * sqrt0) / sqlen0];
        sol[i][1] = [(end[2] * sqrad - sgn * radius * end[3] * sqrt1) / sqlen1, (end[3] * sqrad + sgn * radius * end[2] * sqrt1) / sqlen1];
        good[i] = match side {
            // close to the side site, by direction only
            Some(s) => {
                let m = normalize2([sol[i][0][0] + sol[i][1][0], sol[i][0][1] + sol[i][1][1]]);
                m[0] * s[0] + m[1] * s[1]
            }
            // tangent points closer together
            None => -((sol[i][0][0] - sol[i][1][0]).powi(2) + (sol[i][0][1] - sol[i][1][1]).powi(2)),
        };
        // a candidate whose two straight runs cross each other is penalised
        if segments_cross(e0, sol[i][0], e1, sol[i][1]) {
            good[i] = -10000.0;
        }
    }
    let i = if good[0] > good[1] { 0 } else { 1 };
    // and if the chosen one still crosses, there is no wrap
    if segments_cross(e0, sol[i][0], e1, sol[i][1]) {
        return None;
    }
    Some(([sol[i][0][0], sol[i][0][1], sol[i][1][0], sol[i][1][1]], arc_length(sol[i][0], sol[i][1], i, radius)))
}

/// `wrap_inside`: a side site INSIDE the obstacle. The cable passes through a single point of the circle,
/// where `asin(A z) + asin(B z) − 2 asin(z) + G = 0` (Newton from just below 1), so both returned points are
/// that point and the arc is zero. `None` for no wrap; MuJoCo's numerical-failure exits keep the default,
/// the circle point in the direction of the endpoints' midpoint.
fn wrap_inside(end: &[f64; 4], radius: f64) -> Option<([f64; 4], f64)> {
    const MAXITER: usize = 20;
    const ZINIT: f64 = 1.0 - 1e-7;
    const TOLERANCE: f64 = 1e-6;
    const MINVAL: f64 = 1e-15;
    let len0 = (end[0] * end[0] + end[1] * end[1]).sqrt();
    let len1 = (end[2] * end[2] + end[3] * end[3]).sqrt();
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0] * dif[0] + dif[1] * dif[1];
    if len0 <= radius || len1 <= radius || radius < MINVAL || len0 < MINVAL || len1 < MINVAL {
        return None;
    }
    // the segment passing through the circle: no wrap
    if dd > MINVAL {
        let a = -(dif[0] * end[0] + dif[1] * end[1]) / dd;
        if a > 0.0 && a < 1.0 {
            let t = [end[0] + a * dif[0], end[1] + a * dif[1]];
            if (t[0] * t[0] + t[1] * t[1]).sqrt() <= radius {
                return None;
            }
        }
    }
    let mid = normalize2([0.5 * (end[0] + end[2]), 0.5 * (end[1] + end[3])]);
    let fallback = Some(([mid[0] * radius, mid[1] * radius, mid[0] * radius, mid[1] * radius], 0.0));
    let (a, b) = (radius / len0, radius / len1);
    let cos_g = (len0 * len0 + len1 * len1 - dd) / (2.0 * len0 * len1);
    if cos_g < -1.0 + MINVAL {
        return None;
    } else if cos_g > 1.0 - MINVAL {
        return fallback;
    }
    let g = cos_g.acos();
    let fz = |z: f64| (a * z).asin() + (b * z).asin() - 2.0 * z.asin() + g;
    let mut z = ZINIT;
    let mut f = fz(z);
    if f > 0.0 {
        return fallback;
    }
    let mut iter = 0;
    while iter < MAXITER && f.abs() > TOLERANCE {
        let df = a / (1.0 - z * z * a * a).sqrt().max(MINVAL) + b / (1.0 - z * z * b * b).sqrt().max(MINVAL) - 2.0 / (1.0 - z * z).sqrt().max(MINVAL);
        if df > -MINVAL {
            return fallback;
        }
        let z1 = z - f / df;
        if z1 > z {
            return fallback;
        }
        z = z1;
        f = fz(z);
        if f > TOLERANCE {
            return fallback;
        }
        iter += 1;
    }
    if iter >= MAXITER {
        return fallback;
    }
    // rotate from whichever endpoint the turn is measured from
    let (vec, ang) = if end[0] * end[3] - end[1] * end[2] > 0.0 { ([end[0], end[1]], z.asin() - (a * z).asin()) } else { ([end[2], end[3]], z.asin() - (b * z).asin()) };
    let v = normalize2(vec);
    let (sn, cs) = crate::mujoco_kinematics::sincos(ang);
    let p = [radius * (cs * v[0] - sn * v[1]), radius * (sn * v[0] + cs * v[1])];
    Some(([p[0], p[1], p[0], p[1]], 0.0))
}

/// One geom's contribution to its body, as `mjCGeom` holds it: `mass_`, the geom's frame in the body (the
/// mesh frame already folded in) and its principal inertia in that frame (`SetInertia`).
struct GeomMass {
    mass: f64,
    pose: MjPose,
    inertia: [f64; 3],
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
    /// The orientation attribute that decides an element's frame, as MuJoCo's reader and compiler decide it.
    ///
    /// ⛔ MuJoCo keeps `quat` and ONE alternative (`axisangle`, `euler`, `xyaxes` or `zaxis`) in separate
    /// slots, each inherited through the class chain on its own, and the alternative is resolved AFTER the
    /// quaternion — so a class's `euler` beats an element's own `quat`. `anybotics_anymal_b`'s shank boxes
    /// state `quat="1 0 1 0"` under a class with `euler="0 1.57079632679 0"`, and MuJoCo places them by the
    /// Euler angles (6.9e-12 apart). Only on ONE element is more than one specifier refused
    /// (`ReadAlternative`), which is why an element's own list is returned whole when it has several.
    fn orientation_attrs(&self, el: &El, kind: &str, childclass: Option<&str>) -> Vec<(String, String)> {
        const ALT: [&str; 4] = ["axisangle", "euler", "xyaxes", "zaxis"];
        let own: Vec<(String, String)> = ["quat", "axisangle", "euler", "xyaxes", "zaxis"].iter().filter_map(|k| el.attr(k).map(|v| ((*k).to_string(), v.to_string()))).collect();
        if own.len() > 1 {
            return own;
        }
        // the nearest source — the element, then its classes from the most specific up — of any of `keys`
        let nearest = |keys: &[&str]| -> Option<(String, String)> {
            if let Some((k, v)) = keys.iter().find_map(|k| el.attr(k).map(|v| (*k, v))) {
                return Some((k.to_string(), v.to_string()));
            }
            let mut class: Option<&str> = el.attr("class").or(childclass).or(Some(MAIN));
            let mut guard = 0;
            while let Some(c) = class {
                if let Some((k, v)) = self.attrs.get(&(c.to_string(), kind.to_string())).and_then(|list| list.iter().rev().find(|(kk, _)| keys.contains(&kk.as_str()))) {
                    return Some((k.clone(), v.clone()));
                }
                class = self.parent.get(c).and_then(|p| p.as_deref());
                guard += 1;
                if guard > 64 {
                    break;
                }
            }
            None
        };
        nearest(&ALT).or_else(|| nearest(&["quat"])).into_iter().collect()
    }

    fn known(&self, class: &str) -> bool {
        self.parent.contains_key(class)
    }
}

// ---------------------------------------------------------------------------------------------
// Orientation, with every MJCF form
// ---------------------------------------------------------------------------------------------

/// A frame as MuJoCo's compiler holds it — `pos` and a `(w x y z)` quaternion — computed in MuJoCo's own
/// arithmetic (`ResolveOrientation`, `mjuu_*`, contracted as clang contracts them), so its bits are MuJoCo's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct MjPose {
    pos: [f64; 3],
    quat: [f64; 4],
}

impl MjPose {
    /// The same frame for the rest of the loader. The quaternion is taken as is: renormalising it would
    /// change its bits.
    fn iso(&self) -> Iso {
        Iso::from_parts(Translation3::new(self.pos[0], self.pos[1], self.pos[2]), UnitQuaternion::new_unchecked(nalgebra::Quaternion::new(self.quat[0], self.quat[1], self.quat[2], self.quat[3])))
    }

    /// `mjuu_frameaccum`: this frame followed by a child frame expressed in it.
    fn accum(&mut self, childpos: &[f64; 3], childquat: &[f64; 4]) {
        let v = uu_mulvecmat(&Vector3::from(*childpos), &uu_quat2mat(&self.quat));
        self.pos = [self.pos[0] + v.x, self.pos[1] + v.y, self.pos[2] + v.z];
        self.quat = uu_mulquat(&self.quat, childquat);
    }
}

/// `mjuu_rotVecQuat`, contracted.
fn uu_rot_vec_quat(v: &[f64; 3], q: &[f64; 4]) -> [f64; 3] {
    if v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0 {
        return [0.0; 3];
    }
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return *v;
    }
    let t = [
        (-q[3]).mul_add(v[1], q[0].mul_add(v[0], q[2] * v[2])),
        (-q[1]).mul_add(v[2], q[0].mul_add(v[1], q[3] * v[0])),
        (-q[2]).mul_add(v[0], q[0].mul_add(v[2], q[1] * v[1])),
    ];
    [
        2.0f64.mul_add(q[2].mul_add(t[2], -(q[3] * t[1])), v[0]),
        2.0f64.mul_add(q[3].mul_add(t[0], -(q[1] * t[2])), v[1]),
        2.0f64.mul_add(q[1].mul_add(t[1], -(q[2] * t[0])), v[2]),
    ]
}

/// `mjuu_z2quat`: the rotation taking +z to `vec` (which the caller has normalised).
fn uu_z2quat(vec: &[f64; 3]) -> [f64; 4] {
    let mut q = [0.0, 0.0f64.mul_add(vec[2], -vec[1]), 1.0f64.mul_add(vec[0], -(0.0 * vec[2])), 0.0f64.mul_add(vec[1], -(0.0 * vec[0]))];
    let s = uu_normvec(&mut q[1..4]);
    if s < 1e-10 {
        q[1] = 1.0;
        q[2] = 0.0;
        q[3] = 0.0;
    }
    let ang = s.atan2(vec[2]);
    let (sn, cs) = crate::mujoco_kinematics::sincos(ang / 2.0);
    q[0] = cs;
    for x in &mut q[1..4] {
        *x *= sn;
    }
    q
}

/// `mjuu_frame2quat`: the quaternion of the frame whose columns are `x`, `y`, `z`.
fn uu_frame2quat(x: &[f64; 3], y: &[f64; 3], z: &[f64; 3]) -> [f64; 4] {
    let m = [x, y, z]; // m[c][r]
    let mut q = [0.0; 4];
    if m[0][0] + m[1][1] + m[2][2] > 0.0 {
        q[0] = 0.5 * (1.0 + m[0][0] + m[1][1] + m[2][2]).sqrt();
        q[1] = 0.25 * (m[1][2] - m[2][1]) / q[0];
        q[2] = 0.25 * (m[2][0] - m[0][2]) / q[0];
        q[3] = 0.25 * (m[0][1] - m[1][0]) / q[0];
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        q[1] = 0.5 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        q[0] = 0.25 * (m[1][2] - m[2][1]) / q[1];
        q[2] = 0.25 * (m[1][0] + m[0][1]) / q[1];
        q[3] = 0.25 * (m[2][0] + m[0][2]) / q[1];
    } else if m[1][1] > m[2][2] {
        q[2] = 0.5 * (1.0 - m[0][0] + m[1][1] - m[2][2]).sqrt();
        q[0] = 0.25 * (m[2][0] - m[0][2]) / q[2];
        q[1] = 0.25 * (m[1][0] + m[0][1]) / q[2];
        q[3] = 0.25 * (m[2][1] + m[1][2]) / q[2];
    } else {
        q[3] = 0.5 * (1.0 - m[0][0] - m[1][1] + m[2][2]).sqrt();
        q[0] = 0.25 * (m[0][1] - m[1][0]) / q[3];
        q[1] = 0.25 * (m[2][0] + m[0][2]) / q[3];
        q[2] = 0.25 * (m[2][1] + m[1][2]) / q[3];
    }
    uu_normvec(&mut q);
    q
}

/// Orientation from `quat` / `axisangle` / `euler` / `xyaxes` / `zaxis`, reading each through the defaults,
/// as `mjuu_normvec` (a stated quaternion) and `ResolveOrientation` (the alternatives) compute it.
fn orientation_mj(get: &dyn Fn(&str) -> Option<String>, c: &Compiler) -> Result<[f64; 4], String> {
    orientation_mj_as(get, c, true)
}

/// [`orientation_mj`], leaving a stated `quat` unnormalised when `normalize` is false: sites and frames
/// normalise only after accumulating their enclosing `<frame>`.
fn orientation_mj_as(get: &dyn Fn(&str) -> Option<String>, c: &Compiler, normalize: bool) -> Result<[f64; 4], String> {
    // `angle="degree"` converts as `x / 180.0 * mjPI`, which is not always the same double as `x * (π/180)`
    let degree = c.deg != 1.0;
    let rad = |x: f64| if degree { x / 180.0 * std::f64::consts::PI } else { x };
    let mut given = 0;
    let mut quat = [1.0, 0.0, 0.0, 0.0];
    if let Some(q) = get("quat") {
        let v = floats(&q)?;
        if v.len() != 4 {
            return Err("quat needs 4 numbers (w x y z)".into());
        }
        quat = [v[0], v[1], v[2], v[3]];
        if normalize {
            uu_normvec(&mut quat);
        }
        given += 1;
    }
    if let Some(aa) = get("axisangle") {
        let v = floats(&aa)?;
        if v.len() != 4 {
            return Err("axisangle needs 4 numbers".into());
        }
        let mut a = [v[0], v[1], v[2], rad(v[3])];
        if uu_normvec(&mut a[..3]) < 1e-14 {
            return Err("axisangle too small".into());
        }
        let (sn, cs) = crate::mujoco_kinematics::sincos(a[3] / 2.0);
        quat = [cs, sn * a[0], sn * a[1], sn * a[2]];
        given += 1;
    }
    if let Some(e) = get("euler") {
        let v = floats(&e)?;
        if v.len() != 3 {
            return Err("euler needs 3 numbers".into());
        }
        let mut q = [1.0, 0.0, 0.0, 0.0];
        for (ch, &ang) in c.eulerseq.chars().zip(&v) {
            let (sn, cs) = crate::mujoco_kinematics::sincos(rad(ang) / 2.0);
            let mut qrot = [cs, 0.0, 0.0, 0.0];
            let axis = match ch.to_ascii_lowercase() {
                'x' => 1,
                'y' => 2,
                'z' => 3,
                _ => return Err("euler sequence can only contain x, y, z, X, Y, Z".into()),
            };
            qrot[axis] = sn;
            // lower-case: moving axes, post-multiply; upper-case: fixed axes, pre-multiply
            q = if ch.is_ascii_lowercase() { uu_mulquat(&q, &qrot) } else { uu_mulquat(&qrot, &q) };
        }
        uu_normvec(&mut q);
        quat = q;
        given += 1;
    }
    if let Some(xy) = get("xyaxes") {
        let v = floats(&xy)?;
        if v.len() != 6 {
            return Err("xyaxes needs 6 numbers".into());
        }
        let (mut x, mut y) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
        if uu_normvec(&mut x) < 1e-14 {
            return Err("xaxis too small".into());
        }
        let d = x[2].mul_add(y[2], x[0].mul_add(y[0], x[1] * y[1]));
        for k in 0..3 {
            y[k] = (-x[k]).mul_add(d, y[k]);
        }
        if uu_normvec(&mut y) < 1e-14 {
            return Err("yaxis too small".into());
        }
        let zv = uu_cross(&Vector3::from(x), &Vector3::from(y));
        let mut z = [zv.x, zv.y, zv.z];
        if uu_normvec(&mut z) < 1e-14 {
            return Err("cross(xaxis, yaxis) too small".into());
        }
        quat = uu_frame2quat(&x, &y, &z);
        given += 1;
    }
    if let Some(za) = get("zaxis") {
        let v = vec3(&za)?;
        let mut z = [v.x, v.y, v.z];
        if uu_normvec(&mut z) < 1e-14 {
            return Err("zaxis too small".into());
        }
        quat = uu_z2quat(&z);
        given += 1;
    }
    if given > 1 {
        return Err("more than one of quat/axisangle/euler/xyaxes/zaxis given — MuJoCo refuses this too".into());
    }
    Ok(quat)
}

/// `fromto` (geoms and sites): the frame at the segment's midpoint with +z along it, and the half-length.
/// ⛔ MuJoCo's +z runs from the SECOND endpoint to the FIRST (`vec = fromto[0..3] − fromto[3..6]`, then
/// `mjuu_z2quat`). Taking it the other way puts the frame 180° out — invisible on a symmetric capsule until
/// something reads `geom_xmat`: 77 of Menagerie's 11,912 geoms, every one of them a `fromto` capsule.
fn fromto_mj(v: &[f64], kind: &str) -> Result<(MjPose, f64), String> {
    if v.len() != 6 {
        return Err(format!("{kind} fromto needs 6 numbers"));
    }
    let mut vec = [v[0] - v[3], v[1] - v[4], v[2] - v[5]];
    let half = uu_normvec(&mut vec) / 2.0;
    if half < 1e-14 {
        return Err(format!("{kind} fromto endpoints coincide"));
    }
    let pos = [(v[0] + v[3]) / 2.0, (v[1] + v[4]) / 2.0, (v[2] + v[5]) / 2.0];
    Ok((MjPose { pos, quat: uu_z2quat(&vec) }, half))
}

/// An element's own frame in MuJoCo's arithmetic: `fromto` for geoms and sites (pos and the orientation
/// attributes are then ignored), else `pos` and the orientation.
fn pose_mj(el: &El, kind: &str, defaults: &Defaults, childclass: Option<&str>, c: &Compiler) -> Result<MjPose, String> {
    pose_mj_as(el, kind, defaults, childclass, c, true)
}

fn pose_mj_as(el: &El, kind: &str, defaults: &Defaults, childclass: Option<&str>, c: &Compiler, normalize: bool) -> Result<MjPose, String> {
    let get = |k: &str| defaults.get(el, kind, k, childclass).map(|s| s.to_string());
    if let Some(ft) = get("fromto").filter(|_| kind == "geom" || kind == "site") {
        return Ok(fromto_mj(&floats(&ft)?, kind)?.0);
    }
    let p = get("pos").map(|s| vec3(&s)).transpose()?.unwrap_or_else(Vector3::zeros);
    let orient = defaults.orientation_attrs(el, kind, childclass);
    let get_orient = |k: &str| orient.iter().find(|(kk, _)| kk == k).map(|(_, v)| v.clone());
    Ok(MjPose { pos: [p.x, p.y, p.z], quat: orientation_mj_as(&get_orient, c, normalize)? })
}

/// `kFrameEps`: MuJoCo's compiler treats two frames closer than this, component by component, as the SAME
/// frame (`IsSamePose`), and its kinematics then copy one instead of computing the other.
const FRAME_EPS: f64 = 1e-6;

fn same_vec(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < FRAME_EPS)
}

/// `IsSameQuat`, double cover included.
fn same_quat(a: &[f64; 4], b: &[f64; 4]) -> bool {
    (0..4).all(|k| (a[k] - b[k]).abs() < FRAME_EPS) || (0..4).all(|k| (a[k] + b[k]).abs() < FRAME_EPS)
}

const UNIT_QUAT: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

/// ⛔⛔ **`mjtSameFrame`: a frame within 1e-6 of another IS that other one at run time.** The compiler tags
/// every geom and site (`geom_sameframe`, `site_sameframe`) and `mj_local2Global` then COPIES the body's
/// frame (`BODY`), its orientation (`BODYROT`), its inertial frame (`INERTIA`) or the inertial orientation
/// (`INERTIAROT`) instead of composing the element's own pose — so an element 9e-8 m from its body's
/// inertial frame sits exactly on it. This returns the local pose the kinematics effectively use; `inertial`
/// is the body's stored inertial frame and `inertial_rt` the one its own `body_sameframe` leaves at run time.
fn snap_to_body(p: MjPose, inertial: &MjPose, inertial_rt: &MjPose) -> MjPose {
    use crate::mujoco_kinematics::SameFrame;
    match sameframe_of(&p, inertial) {
        SameFrame::Body => MjPose { pos: [0.0; 3], quat: UNIT_QUAT },
        SameFrame::BodyRot => MjPose { pos: p.pos, quat: UNIT_QUAT },
        SameFrame::Inertia => *inertial_rt,
        SameFrame::InertiaRot => MjPose { pos: p.pos, quat: inertial_rt.quat },
        SameFrame::None => p,
    }
}

/// The `geom_sameframe` / `site_sameframe` tag `mjCModel` gives an element at pose `p` in a body whose
/// stored inertial frame is `inertial`.
fn sameframe_of(p: &MjPose, inertial: &MjPose) -> crate::mujoco_kinematics::SameFrame {
    use crate::mujoco_kinematics::SameFrame;
    if same_vec(&p.pos, &[0.0; 3]) && same_quat(&p.quat, &UNIT_QUAT) {
        SameFrame::Body
    } else if same_quat(&p.quat, &UNIT_QUAT) {
        SameFrame::BodyRot
    } else if same_vec(&p.pos, &inertial.pos) && same_quat(&p.quat, &inertial.quat) {
        SameFrame::Inertia
    } else if same_quat(&p.quat, &inertial.quat) {
        SameFrame::InertiaRot
    } else {
        SameFrame::None
    }
}

/// `body_sameframe`: the body's inertial frame as its kinematics place it — the body frame itself when the
/// stored `ipos`/`iquat` are within 1e-6 of null, its orientation when only `iquat` is.
fn inertial_at_runtime(inertial: &MjPose) -> MjPose {
    if same_vec(&inertial.pos, &[0.0; 3]) && same_quat(&inertial.quat, &UNIT_QUAT) {
        MjPose { pos: [0.0; 3], quat: UNIT_QUAT }
    } else if same_quat(&inertial.quat, &UNIT_QUAT) {
        MjPose { pos: inertial.pos, quat: UNIT_QUAT }
    } else {
        *inertial
    }
}

/// A body's inertia as `mjCBody` stores it: mass, the inertial frame (`body_ipos`, `body_iquat`) and the
/// principal inertia in that frame (`body_inertia`).
type BodyInertial = (f64, MjPose, [f64; 3]);

/// `<inertial>` as `mjCBody::Compile` reads it: `pos` as written, the orientation normalised or resolved
/// from an alternative, and a `fullinertia` diagonalised by `mjuu_eig3`, whose frame REPLACES the orientation.
///
/// ⛔ MuJoCo does not keep the tensor you wrote: it stores the eigenvalues beside the eigenframe, and
/// `mjuu_eig3` stops on an ABSOLUTE 1e-12, which on an off-diagonal term of 8.3e-7 is 2.3e-7 RELATIVE.
/// Keeping the exact input instead leaves the mass matrix out by 1.1e-7 relative at the root of
/// `franka_emika_panda`.
fn inertial_of(el: &El, c: &Compiler) -> Result<BodyInertial, String> {
    let mass = el.attr("mass").ok_or("inertial needs mass")?.trim().parse::<f64>().map_err(|e| e.to_string())?;
    let com = el.attr("pos").map(vec3).transpose()?.unwrap_or_else(Vector3::zeros);
    let get = |k: &str| el.attr(k).map(|s| s.to_string());
    let mut iquat = orientation_mj(&get, c)?;
    let principal = if let Some(d) = el.attr("diaginertia") {
        let v = floats(d)?;
        if v.len() != 3 {
            return Err("diaginertia needs 3 numbers".into());
        }
        [v[0], v[1], v[2]]
    } else if let Some(f) = el.attr("fullinertia") {
        let v = floats(f)?;
        if v.len() != 6 {
            return Err("fullinertia needs 6 numbers (xx yy zz xy xz yz)".into());
        }
        let (eig, q) = eig3_mujoco(&Matrix3::new(v[0], v[3], v[4], v[3], v[1], v[5], v[4], v[5], v[2]));
        if eig[2] < 1e-14 {
            return Err("error 'inertia must have positive eigenvalues' in fullinertia".into());
        }
        iquat = q;
        eig
    } else if mass == 0.0 {
        // MuJoCo accepts `<inertial pos="0 0 0" mass="0"/>` (Menagerie's rby1 uses it for its world body)
        [0.0; 3]
    } else {
        return Err("inertial needs diaginertia or fullinertia".into());
    };
    Ok((mass, MjPose { pos: [com.x, com.y, com.z], quat: iquat }, principal))
}

/// `mjuu_globalinertia`: a principal inertia in a frame, as the six entries of the tensor in the parent
/// (`xx yy zz xy xz yz`), contracted.
fn uu_globalinertia(local: &[f64; 3], quat: &[f64; 4]) -> [f64; 6] {
    let m = uu_quat2mat(quat);
    let mm = |k: usize| m[(k / 3, k % 3)];
    let t = [mm(0) * local[0], mm(3) * local[0], mm(6) * local[0], mm(1) * local[1], mm(4) * local[1], mm(7) * local[1], mm(2) * local[2], mm(5) * local[2], mm(8) * local[2]];
    let row = |a: usize, b: usize, c: usize, x: usize, y: usize, z: usize| mm(c).mul_add(t[z], mm(a).mul_add(t[x], mm(b) * t[y]));
    [row(0, 1, 2, 0, 3, 6), row(3, 4, 5, 1, 4, 7), row(6, 7, 8, 2, 5, 8), row(0, 1, 2, 1, 4, 7), row(0, 1, 2, 2, 5, 8), row(3, 4, 5, 2, 5, 8)]
}

/// `mjuu_offcenter`: the parallel-axis term of a point mass at `v`, contracted.
fn uu_offcenter(mass: f64, v: &[f64; 3]) -> [f64; 6] {
    [
        mass * v[1].mul_add(v[1], v[2] * v[2]),
        mass * v[0].mul_add(v[0], v[2] * v[2]),
        mass * v[0].mul_add(v[0], v[1] * v[1]),
        -mass * v[0] * v[1],
        -mass * v[0] * v[2],
        -mass * v[1] * v[2],
    ]
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
    mesh_file: HashMap<String, crate::TriMesh3>,
    /// the final vertices in `f64`, CoM-centred in the principal frame: what `MakePolygonNormals` reads
    mesh_final: HashMap<String, Vec<Vector3<f64>>>,
    /// meshes referenced by a collidable mesh geom; their hulls are built once the walk is done
    collidable_meshes: BTreeSet<String>,
    /// the enclosing bodies, innermost last (empty at the worldbody)
    body_stack: Vec<String>,
    /// MuJoCo's body ids: the world is 0 and the rest are numbered on first entry, depth-first in file order
    body_ids: HashMap<String, usize>,
    /// geoms with the key MuJoCo sorts them by — `(body id, order within the file)` — and the name the MJCF
    /// gave them, if any; the auto-generated `geom{i}` needs the FINAL index, so it is assigned after sorting
    geom_records: Vec<(usize, usize, Option<String>, MjcfGeom, crate::mujoco_kinematics::KinGeom)>,
    /// per body: its stored inertial frame and the one its kinematics use (`body_sameframe`), which the
    /// sameframe tags of its geoms and sites are taken against
    inertial_mj: HashMap<String, (MjPose, MjPose)>,
    /// MuJoCo's compiled kinematic data per body id (0 = world), for [`crate::mujoco_kinematics`]; a joint's
    /// `qposadr` holds its index in `out.joints` until the walk is done
    kin_bodies: Vec<crate::mujoco_kinematics::KinBody>,
    resolve: &'a dyn Fn(&str) -> Option<Vec<u8>>,
}

const REFUSED: [&str; 4] = ["replicate", "attach", "composite", "flexcomp"];

impl Walk<'_> {
    /// Visit the children of a body-like element (`worldbody`, `body`, `frame`).
    ///
    /// `parent` is the tree joint the enclosing frame rides on (`-1` = world), `carry` the fixed transform from
    /// that joint's frame to the element's own frame.
    ///
    /// `frame` is the accumulated `<frame>` the children sit in, relative to the body (`None` directly in it):
    /// MuJoCo folds it into each child's own pose (`mjuu_frameaccumChild`), so it is carried apart from
    /// `carry` and composed in MuJoCo's arithmetic.
    fn children(&mut self, el: &El, parent: isize, carry: Iso, childclass: Option<&str>, frame: Option<MjPose>) -> Result<(), String> {
        for ch in &el.children {
            match ch.name.as_str() {
                "body" => self.body(ch, parent, carry, childclass, frame)?,
                "geom" => self.record_geom(ch, parent, carry, childclass, frame)?,
                "frame" => {
                    // `mjCFrame::Compile`: its own orientation, the enclosing frame accumulated, THEN normalised
                    let mut f = pose_mj_as(ch, "frame", self.defaults, childclass, self.c, false)?;
                    if let Some(mut outer) = frame {
                        outer.accum(&f.pos, &f.quat);
                        f = outer;
                    }
                    uu_normvec(&mut f.quat);
                    let cc = ch.attr("childclass").or(childclass);
                    self.children(ch, parent, carry, cc, Some(f))?;
                }
                "site" => {
                    let name = ch.attr("name").map(|s| s.to_string());
                    // `mjCSite::Compile`: its own frame, the enclosing `<frame>`, then normalised; placed where
                    // `site_sameframe` puts it
                    let mut sm = pose_mj_as(ch, "site", self.defaults, childclass, self.c, false)?;
                    if let Some(mut outer) = frame {
                        outer.accum(&sm.pos, &sm.quat);
                        sm = outer;
                    }
                    uu_normvec(&mut sm.quat);
                    let sp = self.snap_in_current_body(sm).iso();
                    if let Some(name) = name {
                        // which BODY the site sits on — a site-based `connect` needs it for the two
                        // `body_invweight0` values its rows regularise against
                        if let Some(b) = self.body_stack.last() {
                            self.out.site_body.insert(name.clone(), b.clone());
                        }
                        // `mjCSite`: a sphere of 0.005 unless stated; `fromto` sets the half-length
                        let get = |k: &str| self.defaults.get(ch, "site", k, childclass).map(|s| s.to_string());
                        let kind = match get("type").as_deref() {
                            None | Some("sphere") => crate::mujoco_collision::GeomType::Sphere,
                            Some("capsule") => crate::mujoco_collision::GeomType::Capsule,
                            Some("ellipsoid") => crate::mujoco_collision::GeomType::Ellipsoid,
                            Some("cylinder") => crate::mujoco_collision::GeomType::Cylinder,
                            Some("box") => crate::mujoco_collision::GeomType::Box,
                            Some(other) => return Err(format!("site '{name}': unknown type '{other}'")),
                        };
                        let mut size = [0.005, 0.005, 0.005];
                        if let Some(v) = get("size") {
                            for (i, x) in floats(&v)?.iter().take(3).enumerate() {
                                size[i] = *x;
                            }
                        }
                        if let Some(ft) = get("fromto") {
                            let v = floats(&ft)?;
                            if v.len() == 6 {
                                size[1] = 0.5 * ((v[0] - v[3]).powi(2) + (v[1] - v[4]).powi(2) + (v[2] - v[5]).powi(2)).sqrt();
                            }
                        }
                        self.out.site_shapes.insert(name.clone(), (kind, size));
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
    fn record_geom(&mut self, g: &El, parent: isize, carry: Iso, childclass: Option<&str>, frame: Option<MjPose>) -> Result<(), String> {
        use crate::mujoco_collision::{GeomParams, GeomType};
        let body = self.body_stack.last().cloned().unwrap_or_else(|| "world".to_string());
        let mut spec = self.geom_spec(g, childclass, &body)?;
        if let Some(mut outer) = frame {
            outer.accum(&spec.mj.pos, &spec.mj.quat);
            spec.mj = outer;
        }
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
        let (stored, _) = self.body_stack.last().and_then(|b| self.inertial_mj.get(b)).copied().unwrap_or((MjPose { pos: [0.0; 3], quat: UNIT_QUAT }, MjPose { pos: [0.0; 3], quat: UNIT_QUAT }));
        let kin = crate::mujoco_kinematics::KinGeom { body: body_id, pos: spec.mj.pos, quat: spec.mj.quat, sameframe: sameframe_of(&spec.mj, &stored) };
        self.geom_records.push((body_id, index, g.attr("name").map(|s| s.to_string()), MjcfGeom {
            name: String::new(),
            body,
            joint: (parent >= 0).then_some(parent as usize),
            pose: carry * self.snap_in_current_body(spec.mj).iso(),
            kind,
            size: [spec.size.first().copied().unwrap_or(0.0), spec.size.get(1).copied().unwrap_or(0.0), spec.size.get(2).copied().unwrap_or(0.0)],
            mesh: spec.mesh,
            params,
        }, kin));
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

    fn body(&mut self, b: &El, parent: isize, carry: Iso, childclass: Option<&str>, frame: Option<MjPose>) -> Result<(), String> {
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
        let body_id = self.body_ids[&name];
        let parent_id = self.body_stack.last().and_then(|p| self.body_ids.get(p)).copied().unwrap_or(0);
        if self.kin_bodies.len() <= body_id {
            self.kin_bodies.resize(body_id + 1, crate::mujoco_kinematics::KinBody::default());
        }
        let joints: Vec<&El> = b.children.iter().filter(|c| c.name == "joint" || c.name == "freejoint").collect();
        let has_free = joints.iter().any(|j| j.name == "freejoint" || self.defaults.get(j, "joint", "type", childclass) == Some("free"));
        if has_free && (parent >= 0 || joints.len() > 1) {
            return Err(format!("body '{name}': a free joint must be the only joint of a child of the world"));
        }
        // the body frame relative to the enclosing frame — dropped entirely for a free body, whose pose IS qpos
        let body_own = pose_mj(b, "body", self.defaults, None, self.c)?;
        // an enclosing `<frame>` is folded into the body's pose last (`mjCBody::Compile`), after the
        // inertial frame has been settled — so a body MuJoCo cannot weigh copies its OWN pose, not this one
        let body_mj = match frame {
            Some(mut outer) => {
                outer.accum(&body_own.pos, &body_own.quat);
                outer
            }
            None => body_own,
        };
        let stated_pose = body_mj.iso();
        self.kin_bodies[body_id].parent = parent_id;
        self.kin_bodies[body_id].pos = body_mj.pos;
        self.kin_bodies[body_id].quat = body_mj.quat;
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
            let springdamper = match pair(get("springdamper"), [0.0, 0.0])? {
                [0.0, 0.0] => None,
                [tc, dr] if tc <= 0.0 || dr <= 0.0 => return Err(format!("joint '{jname}': when defined, springdamper values must be positive")),
                v => Some(v),
            };
            {
                // `mjCJoint::Compile`: a free or ball joint's axis is +z, every axis is `mjuu_normvec`ed, a
                // free joint's position is the origin, and `qpos0` is the reference
                use crate::mujoco_kinematics::{KinJoint, KinJointKind};
                let kk = match kind {
                    MjcfJointKind::Free => KinJointKind::Free,
                    MjcfJointKind::Ball => KinJointKind::Ball,
                    MjcfJointKind::Hinge => KinJointKind::Hinge,
                    MjcfJointKind::Slide => KinJointKind::Slide,
                };
                let mut ax = if matches!(kk, KinJointKind::Free | KinJointKind::Ball) { [0.0, 0.0, 1.0] } else { [axis.x, axis.y, axis.z] };
                uu_normvec(&mut ax);
                let jpos = if kk == KinJointKind::Free { [0.0; 3] } else { [anchor.x, anchor.y, anchor.z] };
                let qpos0 = if matches!(kk, KinJointKind::Hinge | KinJointKind::Slide) { reference } else { 0.0 };
                let index = self.out.joints.len();
                self.kin_bodies[body_id].joints.push(KinJoint { kind: kk, pos: jpos, axis: ax, qposadr: index, qpos0 });
            }
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
                springdamper,
            });
            // after the motion, the body frame sits at −anchor from the joint's frame
            pre = Iso::from_parts(Translation3::from(-anchor), UnitQuaternion::identity());
        }

        // inertia: a jointed body owns its last joint's link; a jointless one welds into the ancestor it rides on.
        // `inertiafromgeom`: auto = the <inertial> if stated, else the geoms; true = always the geoms; false = never
        let stated = b.child("inertial");
        // the stated `<inertial>`, if any; then `InertiaFromGeom` when `inertiafromgeom` is true, or is auto
        // and nothing was stated — which replaces the stated values only if some geom has mass
        let mut inertial: Option<BodyInertial> = stated.map(|el| inertial_of(el, self.c).map_err(|e| format!("body '{name}': {e}"))).transpose()?;
        if stated.is_none() {
            self.out.no_inertial.push(name.clone());
        }
        let from_geoms = match self.c.inertiafromgeom {
            InertiaFromGeom::True => true,
            InertiaFromGeom::Auto => stated.is_none(),
            InertiaFromGeom::False => false,
        };
        if let Some(x) = if from_geoms { self.inertia_from_geoms(b, childclass, &name)? } else { None } {
            self.out.inferred_from_geoms.push(name.clone());
            inertial = Some(x);
        }
        // ⛔⛔ **a body MuJoCo could not weigh keeps its OWN placement as its inertial frame.** With no
        // `<inertial>` and nothing to infer from — a camera mount, a site holder, an attachment frame, or
        // geoms that all carry `mass="0"` — the compiler leaves `body_ipos` and `body_iquat` holding the
        // body's `pos` and `quat`, which describe where the body sits in its PARENT and are then read as an
        // offset inside the body itself. `xipos` lands somewhere with no physical meaning, and every value
        // taken there follows it: on `hello_robot_stretch` the camera's `body_invweight0` is 4.4x what the
        // body frame gives, on `franka_emika_panda` the flange's is 1.56x. It fires BEFORE `boundmass`, so a
        // floored mass still sits at `pos`.
        let (mut mass, ipose, mut principal) = inertial.unwrap_or((0.0, body_own, [0.0; 3]));
        // mjCBody::Compile: floors on the mass and on each principal inertia, in the stored frame
        mass = mass.max(self.c.boundmass);
        for d in &mut principal {
            *d = d.max(self.c.boundinertia);
        }
        let iframe = uu_quat2mat(&ipose.quat);
        // the dynamics see the inertial frame where the kinematics put it (`body_sameframe`)
        let irt = inertial_at_runtime(&ipose);
        let rrt = uu_quat2mat(&irt.quat);
        let li = LinkInertia {
            mass,
            com: Vector3::from(irt.pos),
            inertia: rrt * Matrix3::from_diagonal(&Vector3::from(principal)) * rrt.transpose(),
        };
        self.inertial_mj.insert(name.clone(), (ipose, irt));
        {
            use crate::mujoco_kinematics::SameFrame;
            let kb = &mut self.kin_bodies[body_id];
            kb.ipos = ipose.pos;
            kb.iquat = ipose.quat;
            kb.sameframe = if same_vec(&ipose.pos, &[0.0; 3]) && same_quat(&ipose.quat, &UNIT_QUAT) {
                SameFrame::Body
            } else if same_quat(&ipose.quat, &UNIT_QUAT) {
                SameFrame::BodyRot
            } else {
                SameFrame::None
            };
        }
        self.out.body_inertial_runtime.insert(name.clone(), (Vector3::from(irt.pos), rrt));
        if ride >= 0 {
            let idx = ride as usize;
            let moved = crate::dynamics::transform_inertia(&li, &pre);
            self.out.tree.inertia[idx] = crate::dynamics::combine_inertia(&self.out.tree.inertia[idx], &moved);
            if !joints.is_empty() {
                self.out.tree.link_names.insert(name.clone(), idx);
            }
        }
        self.out.body_ipos.insert(name.clone(), Vector3::from(ipose.pos));
        self.out.body_mass.insert(name.clone(), li.mass);
        // MuJoCo keeps a body's inertia as a DIAGONAL plus the frame it is diagonal in; the fluid model
        // reads both, and these are the two halves as the compiler computed them — not a second
        // diagonalisation of the tensor they were reassembled into
        self.out.body_iinertia.insert(name.clone(), principal);
        self.out.body_iquat.insert(name.clone(), iframe);
        self.out.body_iquat_wxyz.insert(name.clone(), ipose.quat);
        // `<body gravcomp>` is a plain body attribute: no default class carries it, and it does not inherit
        if let Some(v) = b.attr("gravcomp") {
            let g = v.trim().parse::<f64>().map_err(|e| format!("body '{name}' gravcomp: {e}"))?;
            if g != 0.0 {
                self.out.body_gravcomp.insert(name.clone(), g);
            }
        }
        self.place("body", name.clone(), ride, pre)?;
        self.body_stack.push(name);
        let r = self.children(b, ride, pre, childclass, None);
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
        // `mjCGeom::Compile`: the geom's own frame; `fromto` sets it and the half-length
        let mut mj = pose_mj(g, "geom", self.defaults, childclass, self.c)?;
        let mesh_name = get("mesh");
        if let Some(ft) = get("fromto") {
            if mesh_name.is_some() {
                return Err(format!("body '{body}': fromto cannot be used with a mesh geom"));
            }
            // MuJoCo puts the half-length into size[1] (capsule, cylinder) or size[2] with size[1] = size[0]
            // (box, ellipsoid)
            let (_, half) = fromto_mj(&floats(&ft)?, "geom")?;
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
        if let Some(mname) = &mesh_name {
            if ty != "mesh" && ty != "sphere" && ty != "capsule" && ty != "cylinder" && ty != "ellipsoid" && ty != "box" {
                return Err(format!("body '{body}': invalid geom type '{ty}' in fitting mesh '{mname}'"));
            }
            let md = self.mesh_data(mname)?;
            // **Mesh fitting** (`mjCMesh::FitGeom`). A geom of a primitive type that names a `mesh` is sized
            // from that mesh — from its equivalent inertia box, or its bounding box under `fitaabb`, times
            // `fitscale` — and placed at the mesh's centre of mass in its principal frame. Menagerie uses this
            // for collision capsules fitted to visual meshes, and a geom with a mesh and no type is a fitted
            // SPHERE.
            let mut center = [0.0f64; 3];
            if ty != "mesh" {
                let fitscale: f64 = get("fitscale").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(1.0);
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
                        _ => sz = b,
                    }
                } else {
                    let a = md.aamm;
                    center = [(a[0] + a[3]) / 2.0, (a[1] + a[4]) / 2.0, (a[2] + a[5]) / 2.0];
                    let half = [a[3] - center[0], a[4] - center[1], a[5] - center[2]];
                    match ty.as_str() {
                        "sphere" => sz[0] = half[0].max(half[1]).max(half[2]),
                        "capsule" | "cylinder" => {
                            sz[0] = half[0].max(half[1]);
                            sz[1] = half[2];
                            if ty == "capsule" {
                                sz[1] -= sz[0];
                            }
                        }
                        _ => sz = half,
                    }
                }
                size = sz.iter().map(|v| v * fitscale).collect();
            }
            // rotate the centre into the geom frame, add the mesh's own position, and accumulate the mesh frame
            // into the geom frame
            let rc = uu_rot_vec_quat(&center, &md.quat);
            mj.accum(&[rc[0] + md.com.x, rc[1] + md.com.y, rc[2] + md.com.z], &md.quat);
            // a mesh geom takes MuJoCo's `geom_size` from the mesh's own bounds
            if ty == "mesh" {
                size = (0..3).map(|k| md.aamm[k].abs().max(md.aamm[k + 3].abs())).collect();
            }
        } else if ty == "mesh" {
            return Err(format!("body '{body}': a mesh geom needs a `mesh` attribute"));
        }
        Ok(GeomSpec { ty, size, mj, mesh: mesh_name })
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
        let GeomSpec { ty, size, mj, mesh: mesh_name } = self.geom_spec(g, childclass, body)?;
        let need = |k: usize| -> Result<(), String> {
            if size.len() < k {
                Err(format!("body '{body}': geom type '{ty}' needs {k} size value(s), got {}", size.len()))
            } else {
                Ok(())
            }
        };
        let pi = std::f64::consts::PI;
        // `mjCGeom::GetVolume` (volume inertia), contracted as clang contracts it
        let mut boxsz = [0.0f64; 3];
        let volume = match ty.as_str() {
            "sphere" => {
                need(1)?;
                4.0 * pi * size[0] * size[0] * size[0] / 3.0
            }
            "capsule" => {
                need(2)?;
                let (r, h) = (size[0], 2.0 * size[1]);
                pi * (r * r).mul_add(h, 4.0 * r * r * r / 3.0)
            }
            "cylinder" => {
                need(2)?;
                let (r, h) = (size[0], 2.0 * size[1]);
                pi * r * r * h
            }
            "ellipsoid" => {
                need(3)?;
                4.0 * pi * size[0] * size[1] * size[2] / 3.0
            }
            "box" => {
                need(3)?;
                size[0] * size[1] * size[2] * 8.0
            }
            "mesh" => {
                let name = mesh_name.clone().ok_or_else(|| format!("body '{body}': a mesh geom needs a `mesh` attribute"))?;
                let md = self.mesh_data(&name)?;
                boxsz = md.boxsz;
                md.volume
            }
            "plane" | "hfield" | "sdf" => return Ok(None),
            other => return Err(format!("body '{body}': geom type '{other}' is not one this loader can weigh")),
        };
        // an explicit `mass` is taken as is (on a geom with volume); otherwise density × volume, default 1000
        let mass = match get("mass").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()? {
            Some(0.0) => return Ok(None),
            Some(m) if volume > 1e-14 => m,
            Some(_) => return Ok(None),
            None => {
                let density = get("density").map(|s| s.trim().parse::<f64>().map_err(|e| e.to_string())).transpose()?.unwrap_or(1000.0);
                if density == 0.0 {
                    return Ok(None);
                }
                density * volume
            }
        };
        // `InertiaFromGeom` weighs only geoms with `mass_ > mjEPS`
        if mass.is_nan() || mass <= 1e-14 {
            return Ok(None);
        }
        // `mjCGeom::SetInertia` (volume inertia), contracted
        let inertia = match ty.as_str() {
            "sphere" => {
                let i = 2.0 * mass * size[0] * size[0] / 5.0;
                [i, i, i]
            }
            "capsule" => {
                let (r, h) = (size[0], 2.0 * size[1]);
                let sphere_mass = mass * 4.0 * r / 4.0f64.mul_add(r, 3.0 * h);
                let cylinder_mass = mass - sphere_mass;
                let side = cylinder_mass * (3.0 * r).mul_add(r, h * h) / 12.0;
                let mut i = [side, side, cylinder_mass * r * r / 2.0];
                let sphere_inertia = 2.0 * sphere_mass * r * r / 5.0;
                let shift = sphere_mass * h * 3.0f64.mul_add(r, 2.0 * h) / 8.0;
                i[0] += sphere_inertia + shift;
                i[1] += sphere_inertia + shift;
                i[2] += sphere_inertia;
                i
            }
            "cylinder" => {
                let (r, h) = (size[0], 2.0 * size[1]);
                let side = mass * (3.0 * r).mul_add(r, h * h) / 12.0;
                [side, side, mass * r * r / 2.0]
            }
            "ellipsoid" | "box" => {
                let d = if ty == "box" { 3.0 } else { 5.0 };
                let (s00, s11, s22) = (size[0] * size[0], size[1] * size[1], size[2] * size[2]);
                [mass * (s11 + s22) / d, mass * (s00 + s22) / d, mass * (s00 + s11) / d]
            }
            // the mesh's equivalent inertia box, in the mesh's principal frame (already the geom's)
            _ => {
                let b = boxsz;
                [mass * b[1].mul_add(b[1], b[2] * b[2]) / 3.0, mass * b[0].mul_add(b[0], b[2] * b[2]) / 3.0, mass * b[0].mul_add(b[0], b[1] * b[1]) / 3.0]
            }
        };
        Ok(Some(GeomMass { mass, pose: mj, inertia }))
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
        self.mesh_file.insert(name.to_string(), mesh.clone());
        // a mirroring `scale` (odd number of negative factors) would turn every face inside out, so
        // `mjCMesh::Process` swaps each face's last two vertices first — after the hull, before the inertia
        if asset.scale.x * asset.scale.y * asset.scale.z <= 0.0 {
            for t in &mut mesh.tris {
                t.swap(1, 2);
            }
        }
        // mjCMesh::ApplyTransformations — refpos, then the inverse refquat rotation, then scale
        let rq = (asset.refquat != [1.0, 0.0, 0.0, 0.0]).then(|| {
            let mut q = asset.refquat;
            uu_normvec4(&mut q);
            uu_quat2mat(&q)
        });
        for v in &mut mesh.verts {
            let p = *v - asset.refpos;
            let p = rq.as_ref().map_or(p, |m| uu_mulvecmat_t(&p, m));
            *v = Vector3::new(p.x * asset.scale.x, p.y * asset.scale.y, p.z * asset.scale.z);
        }
        self.mesh_raw.insert(name.to_string(), mesh.clone());
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
        // the mesh is re-expressed centred at its CoM in its principal frame (`mjCMesh::Rotate`: the matrix of
        // the CONJUGATE quaternion, applied by `mjuu_mulvecmat`); the bounds are taken there
        let rneg = uu_quat2mat(&[quat[0], -quat[1], -quat[2], -quat[3]]);
        let final_f64: Vec<Vector3<f64>> = mesh.verts.iter().map(|v| uu_mulvecmat(&(*v - com), &rneg)).collect();
        let mut aamm = [f64::INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in &final_f64 {
            for k in 0..3 {
                aamm[k] = aamm[k].min(p[k]);
                aamm[k + 3] = aamm[k + 3].max(p[k]);
            }
        }
        let d = MeshData { volume, com, quat, boxsz, aamm };
        // what `mjCMesh::Process` stores back into its float vertex array: CoM-centred, in the principal frame
        let stored = crate::TriMesh3 { verts: final_f64.iter().map(|v| v.map(|x| x as f32 as f64)).collect(), tris: mesh.tris.clone() };
        self.mesh_stored.insert(name.to_string(), stored);
        self.mesh_final.insert(name.to_string(), final_f64);
        self.mesh_cache.insert(name.to_string(), d);
        Ok(d)
    }

    /// `mjCBody::InertiaFromGeom`: the body's inertia from the geoms MuJoCo would weigh, in the body frame.
    #[allow(clippy::type_complexity)]
    fn inertia_from_geoms(&mut self, b: &El, childclass: Option<&str>, body: &str) -> Result<Option<BodyInertial>, String> {
        // every geom of the body, including those inside `<frame>`s, in file order, each with its frame folded in
        type Found<'e> = Vec<(&'e El, Option<MjPose>, Option<&'e str>)>;
        fn collect<'e>(el: &'e El, frame: Option<MjPose>, w: &Walk, childclass: Option<&'e str>, out: &mut Found<'e>) -> Result<(), String> {
            for ch in &el.children {
                match ch.name.as_str() {
                    "geom" => out.push((ch, frame, childclass)),
                    "frame" => {
                        let mut f = pose_mj_as(ch, "frame", w.defaults, childclass, w.c, false)?;
                        if let Some(mut outer) = frame {
                            outer.accum(&f.pos, &f.quat);
                            f = outer;
                        }
                        uu_normvec(&mut f.quat);
                        collect(ch, Some(f), w, ch.attr("childclass").or(childclass), out)?;
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        let mut geoms = Vec::new();
        collect(b, None, self, childclass, &mut geoms)?;
        let mut sel = Vec::new();
        for (g, frame, cc) in geoms {
            if let Some(mut gm) = self.geom_mass(g, cc, body)? {
                if let Some(mut outer) = frame {
                    outer.accum(&gm.pose.pos, &gm.pose.quat);
                    gm.pose = outer;
                }
                sel.push(gm);
            }
        }
        match sel.as_slice() {
            [] => Ok(None),
            // ⛔ one geom is COPIED — its frame and principal inertia become the body's, with no second
            // diagonalisation (which, on a fly's 1e-12 tarsus, stops at iteration zero and keeps the diagonal
            // of a rotated tensor: a stored inertia 27% off)
            [one] => Ok(Some((one.mass, one.pose, one.inertia))),
            _ => {
                let (mut mass, mut com) = (0.0, [0.0f64; 3]);
                for p in &sel {
                    mass += p.mass;
                    for (k, c) in com.iter_mut().enumerate() {
                        *c = p.mass.mul_add(p.pose.pos[k], *c);
                    }
                }
                if mass < 1e-14 {
                    return Err(format!("body '{body}': body mass is too small, cannot compute center of mass"));
                }
                let ipos = [com[0] / mass, com[1] / mass, com[2] / mass];
                let mut toti = [0.0f64; 6];
                for p in &sel {
                    let dpos = [p.pose.pos[0] - ipos[0], p.pose.pos[1] - ipos[1], p.pose.pos[2] - ipos[2]];
                    let (i0, i1) = (uu_globalinertia(&p.inertia, &p.pose.quat), uu_offcenter(p.mass, &dpos));
                    for j in 0..6 {
                        toti[j] = toti[j] + i0[j] + i1[j];
                    }
                }
                let (eig, quat) = eig3_mujoco(&Matrix3::new(toti[0], toti[3], toti[4], toti[3], toti[1], toti[5], toti[4], toti[5], toti[2]));
                if eig[2] < 1e-14 {
                    return Err(format!("body '{body}': error 'inertia must have positive eigenvalues' in alternative for principal axes"));
                }
                Ok(Some((mass, MjPose { pos: ipos, quat }, eig)))
            }
        }
    }

    /// An element's pose in the body being walked, snapped as `mjtSameFrame` snaps it at run time.
    fn snap_in_current_body(&self, p: MjPose) -> MjPose {
        const NULL: MjPose = MjPose { pos: [0.0; 3], quat: UNIT_QUAT };
        let (stored, rt) = self.body_stack.last().and_then(|b| self.inertial_mj.get(b)).copied().unwrap_or((NULL, NULL));
        snap_to_body(p, &stored, &rt)
    }

    fn body_count(&self) -> usize {
        self.out.body_frames.len() + self.out.world_fixed.keys().filter(|k| k.starts_with("body:")).count() + 1
    }
}

/// **`<actuator>`**, resolved the way MuJoCo's compiler resolves it: the shortcut tag picks a gain and bias
/// TYPE, the attributes fill in the parameters, and everything downstream sees only `gain·ctrl + bias`.
///
/// Refuses, rather than approximating, anything this port does not carry: a slider-crank transmission, a
/// joint target that is not a hinge or a slide, a `user` or `dcmotor` activation law, and the tags not yet
/// resolved (`damper`, `cylinder`, `intvelocity`, `pid`, `dcmotor`, `orientation`, a `<position>` with a
/// `timeconst`). A silently dropped actuator is a robot that does not move for reasons nobody can see.
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
    gainprm: [f64; 10],
    bias: crate::mujoco_actuator::ActBias,
    biasprm: [f64; 10],
    dynamics: crate::mujoco_actuator::ActDyn,
    dynprm: [f64; 3],
    lengthrange: [f64; 2],
    gear: f64,
    /// the whole `gear` vector: a site transmission reads all six, as a wrench in the site's frame
    gear6: [f64; 6],
    ctrlrange: [f64; 2],
    ctrllimited: Option<bool>,
    forcerange: [f64; 2],
    forcelimited: Option<bool>,
    actrange: [f64; 2],
    actlimited: Option<bool>,
    actearly: bool,
    /// `actdim`, where the file states it; MuJoCo's default is one activation per stateful actuator
    actdim: Option<i64>,
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
            gainprm: {
                let mut p = [0.0; 10];
                p[0] = 1.0;
                p
            },
            bias: crate::mujoco_actuator::ActBias::None,
            biasprm: [0.0; 10],
            dynamics: crate::mujoco_actuator::ActDyn::None,
            dynprm: [1.0, 0.0, 0.0],
            lengthrange: [0.0; 2],
            gear: 1.0,
            gear6: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrlrange: [0.0; 2],
            ctrllimited: None,
            forcerange: [0.0; 2],
            forcelimited: None,
            actrange: [0.0; 2],
            actlimited: None,
            actearly: false,
            actdim: None,
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
        let vecn = |k: &str, p: &mut [f64]| -> Result<(), ActErr> {
            if let Some(v) = at(k) {
                let f = floats(v).map_err(Bad)?;
                for (i, x) in f.iter().take(p.len()).enumerate() {
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
            Some("muscle") => self.gain = crate::mujoco_actuator::ActGain::Muscle,
            Some(o) => self.unsupported = Some(format!("gaintype '{o}'")),
        }
        match at("biastype") {
            None => {}
            Some("none") => self.bias = crate::mujoco_actuator::ActBias::None,
            Some("affine") => self.bias = crate::mujoco_actuator::ActBias::Affine,
            Some("muscle") => self.bias = crate::mujoco_actuator::ActBias::Muscle,
            Some(o) => self.unsupported = Some(format!("biastype '{o}'")),
        }
        match at("dyntype") {
            None => {}
            Some("none") => self.dynamics = crate::mujoco_actuator::ActDyn::None,
            Some("integrator") => self.dynamics = crate::mujoco_actuator::ActDyn::Integrator,
            Some("filter") => self.dynamics = crate::mujoco_actuator::ActDyn::Filter,
            Some("filterexact") => self.dynamics = crate::mujoco_actuator::ActDyn::FilterExact,
            Some("muscle") => self.dynamics = crate::mujoco_actuator::ActDyn::Muscle,
            Some(o) => self.unsupported = Some(format!("dyntype '{o}'")),
        }
        match at("actearly") {
            None => {}
            Some("true") => self.actearly = true,
            Some("false") => self.actearly = false,
            Some(o) => return Err(Bad(format!("actuator actearly '{o}' is not true/false"))),
        }
        if let Some(v) = at("actdim") {
            self.actdim = Some(v.trim().parse::<i64>().map_err(|e| Bad(format!("actuator actdim: {e}")))?);
        }
        vecn("gainprm", &mut self.gainprm)?;
        vecn("biasprm", &mut self.biasprm)?;
        vecn("dynprm", &mut self.dynprm)?;
        vecn("lengthrange", &mut self.lengthrange)?;
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
        rng("actrange", "actlimited", &mut self.actrange, &mut self.actlimited)?;
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
                self.biasprm = [0.0; 10];
                self.gainprm[0] = kv;
                self.biasprm[2] = -kv;
                self.gain = ActGain::Fixed;
                self.bias = ActBias::Affine;
            }
            // ⛔⛔ the `<muscle>` shortcut does NOT write into `gainprm`/`biasprm` the way `<position>`
            // writes `kp`: it REPLACES both vectors wholesale with MuJoCo's muscle defaults and then
            // overwrites the named entries. A class that set `gainprm` for a motor and an element that
            // says `<muscle/>` must end at the muscle defaults, not at a blend of the two.
            //
            // ⛔ and `force = -1` is the DEFAULT, not a sentinel for "unset": it means "scale me by
            // `scale/acc0`", so a port that treats a negative force as an error refuses most muscles.
            "muscle" => {
                let mut prm = [0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2, 0.0];
                if let Some(v) = at("range") {
                    for (i, x) in floats(v).map_err(Bad)?.iter().take(2).enumerate() {
                        prm[i] = *x;
                    }
                }
                for (k, i) in [("force", 2), ("scale", 3), ("lmin", 4), ("lmax", 5), ("vmax", 6), ("fpmax", 7), ("fvmax", 8)] {
                    if let Some(v) = num(k)? {
                        prm[i] = v;
                    }
                }
                self.gainprm = prm;
                self.biasprm = prm;
                self.dynprm = [0.01, 0.04, 0.0];
                if let Some(v) = at("timeconst") {
                    for (i, x) in floats(v).map_err(Bad)?.iter().take(2).enumerate() {
                        self.dynprm[i] = *x;
                    }
                }
                if let Some(v) = num("tausmooth")? {
                    self.dynprm[2] = v;
                }
                self.gain = ActGain::Muscle;
                self.bias = ActBias::Muscle;
                self.dynamics = crate::mujoco_actuator::ActDyn::Muscle;
            }
            // `mjs_setToAdhesion`: a fixed gain (the element's `gain`, else what the record already holds),
            // no bias, and a control range that is ALWAYS limited and may not go negative
            "adhesion" => {
                let gain = num("gain")?.unwrap_or(self.gainprm[0]);
                self.gainprm[0] = gain;
                self.ctrllimited = Some(true);
                self.gain = ActGain::Fixed;
                self.bias = ActBias::None;
                if gain < 0.0 {
                    return Err(Bad("adhesion gain cannot be negative".into()));
                }
                if self.ctrlrange[0] < 0.0 || self.ctrlrange[1] < 0.0 {
                    return Err(Bad("adhesion control range cannot be negative".into()));
                }
            }
            other => self.unsupported = Some(format!("<{other}>")),
        }
        Ok(())
    }
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
    // `<extension><plugin plugin="..."><instance name="..."><config key value/>`: named plugin configurations
    // an actuator can point at with `instance=`
    let mut instances: PluginInstances = HashMap::new();
    for ext in root.children_named("extension") {
        for pl in ext.children_named("plugin") {
            let plugin = pl.attr("plugin").unwrap_or("").to_string();
            for inst in pl.children_named("instance") {
                let config = inst.children_named("config").filter_map(|c| Some((c.attr("key")?.to_string(), c.attr("value")?.to_string()))).collect();
                instances.insert(inst.attr("name").unwrap_or("").to_string(), (plugin.clone(), config));
            }
        }
    }
    for section in root.children_named("actuator") {
        for el in &section.children {
            let name = el.attr("name").map(|s| s.to_string()).unwrap_or_else(|| format!("actuator{}", out.len() + unsupported.len()));
            match one_actuator(el, defaults, c, &by_name, &ten_by_name, &instances, section.attr("childclass"), &name) {
                Ok(a) => out.push(a),
                Err(Unsupported(why)) => unsupported.push((name, why)),
                Err(Bad(why)) => return Err(format!("actuator '{name}': {why}")),
            }
        }
    }
    Ok((out, unsupported))
}

/// Plugin instances by name: the plugin's name and its `config` pairs.
type PluginInstances = HashMap<String, (String, HashMap<String, String>)>;

/// **`mujoco.pid`**, as `PidConfig::FromModel` reads it: `kp`, `ki`, `kd` (absent is zero), `imax` (a
/// FORCE, turned into the integral's clamp by dividing by `ki`) and `slewmax`. From a named instance, or
/// from `<config>` children of the actuator element itself.
fn plugin_pid(el: &El, instances: &PluginInstances) -> Result<crate::mujoco_actuator::Pid, ActErr> {
    let (plugin, config) = match el.attr("instance") {
        Some(i) => instances.get(i).cloned().ok_or_else(|| Bad(format!("no plugin instance named '{i}'")))?,
        None => (el.attr("plugin").unwrap_or("").to_string(), el.children_named("config").filter_map(|c| Some((c.attr("key")?.to_string(), c.attr("value")?.to_string()))).collect()),
    };
    if plugin != "mujoco.pid" {
        return Err(Unsupported(format!("the '{plugin}' actuator plugin")));
    }
    let num = |k: &str| -> Result<Option<f64>, ActErr> {
        match config.get(k).map(|v| v.trim()).filter(|v| !v.is_empty()) {
            Some(v) => v.parse::<f64>().map(Some).map_err(|e| Bad(format!("pid {k}: {e}"))),
            None => Ok(None),
        }
    };
    let (kp, ki, kd) = (num("kp")?.unwrap_or(0.0), num("ki")?.unwrap_or(0.0), num("kd")?.unwrap_or(0.0));
    let i_max = num("imax")?.filter(|_| ki != 0.0).map(|m| m / ki);
    let slew_max = num("slewmax")?;
    if i_max.is_some_and(|m| m < 0.0) {
        return Err(Bad("pid: negative imax".into()));
    }
    if slew_max.is_some_and(|m| m < 0.0) {
        return Err(Bad("pid: slewmax must be non-negative".into()));
    }
    Ok(crate::mujoco_actuator::Pid { kp, ki, kd, i_max, slew_max })
}

#[allow(clippy::too_many_arguments)]
fn one_actuator(el: &El, defaults: &Defaults, c: &Compiler, by_name: &HashMap<&str, &MjcfJoint>, ten_by_name: &HashMap<&str, (usize, &MjcfTendon)>, instances: &PluginInstances, childclass: Option<&str>, name: &str) -> Result<crate::mujoco_actuator::Actuator, ActErr> {
    use crate::mujoco_actuator::Actuator;
    let tag = el.name.as_str();
    // an actuator `<plugin>` takes the general attribute table (transmission, ranges, `actdim`); the plugin
    // computes its force
    let pid = if tag == "plugin" { Some(plugin_pid(el, instances)?) } else { None };
    let tag = if tag == "plugin" { "general" } else { tag };
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
        // an ADHESION actuator: its moment comes from the contacts on this body, at every state
        "body" => {
            dynamic = Some(crate::mujoco_actuator::DynTransmission::Body { body: target.clone() });
            (None, Vec::new())
        }
        _ => return Err(Unsupported(format!("a {kind} transmission"))),
    };
    // `limited="auto"` (`islimited`): limited exactly when the range is a range, `lo < hi`; and with autolimits
    // off MuJoCo refuses any nonzero range with no `limited` (`checklimited`).
    //
    // ⛔ "a stated range limits" is NOT the rule. `actrange="1 -1"` under auto is stated, and MuJoCo leaves
    // the actuator UNLIMITED and compiles it; reading it as limited refuses the model as an invalid range.
    let limit = |r: [f64; 2], lim: Option<bool>, what: &str| -> Result<Option<[f64; 2]>, ActErr> {
        let on = match lim {
            Some(v) => v,
            None => {
                if r != [0.0, 0.0] && !c.autolimits {
                    return Err(Bad(format!("{what}range is given but {what}limited is not, and autolimits is off")));
                }
                r[0] < r[1]
            }
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
    // `mjCActuator::Compile`: a limited range must be a range. This is what refuses an `<adhesion>` with no
    // `ctrlrange`, since that shortcut always limits the control and the default range is [0, 0].
    if forcerange.is_some_and(|r| r[0] >= r[1]) {
        return Err(Bad("invalid force range for actuator".into()));
    }
    if ctrlrange.is_some_and(|r| r[0] >= r[1]) {
        return Err(Bad("invalid control range for actuator".into()));
    }
    // the activation's range and count, by `mjCActuator::Compile`'s rules
    let actrange = limit(rec.actrange, rec.actlimited, "act")?;
    let stateful = rec.dynamics != crate::mujoco_actuator::ActDyn::None;
    if actrange.is_some_and(|r| r[0] >= r[1]) {
        return Err(Bad("invalid actrange for actuator".into()));
    }
    if actrange.is_some() && !stateful {
        return Err(Bad("actrange specified but dyntype is 'none' in actuator".into()));
    }
    match (&pid, rec.actdim) {
        // `Pid::Create`: the plugin's own activations, which the file has to state
        (Some(p), n) => {
            if stateful {
                return Err(Unsupported("a PID plugin whose setpoint has an activation law".into()));
            }
            if n != Some(p.actdim() as i64) {
                return Err(Bad(format!("actuator has actdim {}, expected {}; add actdim=\"{}\" to the actuator plugin element", n.unwrap_or(0), p.actdim(), p.actdim())));
            }
        }
        (None, Some(n)) if n > 1 => return Err(Bad("actdim > 1 is only allowed for dyntype 'user' and 'dcmotor'".into())),
        (None, Some(1)) if !stateful => return Err(Bad("invalid actdim 1 in stateless actuator".into())),
        (None, Some(0)) if stateful => return Err(Bad("invalid actdim 0 in stateful actuator".into())),
        _ => {}
    }
    Ok(Actuator {
        name: name.to_string(),
        moment,
        dynamic,
        gear: rec.gear,
        gain: rec.gain,
        gainprm: rec.gainprm,
        bias: rec.bias,
        biasprm: rec.biasprm,
        dynamics: rec.dynamics,
        dynprm: rec.dynprm,
        lengthrange: rec.lengthrange,
        // ⛔ filled in by the loader once the tree exists: it is `‖M⁻¹·moment‖` at `qpos0`, so it cannot
        // be resolved while reading the file, exactly like `dampratio`.
        acc0: 0.0,
        ctrlrange,
        forcerange,
        actrange,
        actearly: rec.actearly,
        pid,
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
                    [v[0], v[1], v[2], v[3]]
                }
                None => [1.0, 0.0, 0.0, 0.0],
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
            site_body: BTreeMap::new(),
            world_fixed: BTreeMap::new(),
            site_shapes: BTreeMap::new(),
            no_inertial: Vec::new(),
            inferred_from_geoms: Vec::new(),
            angle_scale: c.deg,
            timestep: 0.002,
            gravity: Vector3::new(0.0, 0.0, -9.81),
            reference_q: Vec::new(),
            free_base_pose: BTreeMap::new(),
            chart: BTreeMap::new(),
            sensors: Vec::new(),
            keyframes: Vec::new(),
            magnetic: Vector3::new(0.0, -0.5, 0.0),
            invweight0_cache: std::sync::OnceLock::new(),
            collide_cache: std::sync::OnceLock::new(),
            density: 0.0,
            viscosity: 0.0,
            wind: Vector3::zeros(),
            cone: crate::mujoco_contact::Cone::Pyramidal,
            impratio: 1.0,
            integrator: MjcfIntegrator::Euler,
            eulerdamp: true,
            noslip_iterations: 0,
            noslip_tolerance: 1e-6,
            meaninertia: 1.0,
            body_iinertia: BTreeMap::new(),
            body_iquat: BTreeMap::new(),
            body_iquat_wxyz: BTreeMap::new(),
            body_inertial_runtime: BTreeMap::new(),
            mj_kin: Default::default(),
            body_mass: BTreeMap::new(),
            body_gravcomp: BTreeMap::new(),
            tendons: Vec::new(),
            tendons_unsupported: Vec::new(),
            body_parent: BTreeMap::new(),
            mesh_hulls: BTreeMap::new(),
            mesh_raw: BTreeMap::new(),
            mesh_file: BTreeMap::new(),
            mesh_props: BTreeMap::new(),
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
        mesh_file: HashMap::new(),
        mesh_final: HashMap::new(),
        collidable_meshes: BTreeSet::new(),
        body_stack: Vec::new(),
        body_ids: HashMap::new(),
        geom_records: Vec::new(),
        inertial_mj: HashMap::new(),
        kin_bodies: vec![crate::mujoco_kinematics::KinBody::default()],
        resolve,
    };
    for world in worlds {
        walk.children(world, -1, Iso::identity(), world.attr("childclass"), None)?;
    }
    let mut out = walk.out;
    // ⛔ MuJoCo orders geoms BODY-MAJOR — every geom of body 0 (the world), then of body 1, and so on —
    // not in file order. A scene that `<include>`s a robot and then declares its own floor puts that floor
    // LAST in the file and FIRST in `geom_xpos`, because the floor belongs to the world body. Recording in
    // file order gives the right set with the wrong indices, which every by-index comparison then reads as a
    // pose error on almost every geom.
    walk.geom_records.sort_by_key(|(body, seq, _, _, _)| (*body, *seq));
    let mut kin_geoms = Vec::with_capacity(walk.geom_records.len());
    out.geoms = walk
        .geom_records
        .into_iter()
        .enumerate()
        .map(|(i, (_, _, name, mut g, k))| {
            g.name = name.unwrap_or_else(|| format!("geom{i}"));
            kin_geoms.push(k);
            g
        })
        .collect();
    // MuJoCo's own kinematic data; a joint's placeholder (its index among the joints) becomes its qposadr
    let qposadr = out.qposadr();
    let mut kin_bodies = walk.kin_bodies;
    for b in &mut kin_bodies {
        for j in &mut b.joints {
            j.qposadr = qposadr[j.qposadr];
        }
    }
    out.mj_kin = crate::mujoco_kinematics::MjKinematics { bodies: kin_bodies, geoms: kin_geoms };
    for name in &walk.collidable_meshes {
        let stored = &walk.mesh_stored[name];
        let cap = walk.meshes.get(name).and_then(|a| a.maxhullvert);
        let hull = crate::mujoco_hull::MeshHull::from_frames(&walk.mesh_file[name].verts, stored, &walk.mesh_final[name], cap).ok_or_else(|| format!("mesh '{name}': no 3-D convex hull (MuJoCo refuses such a collision mesh)"))?;
        out.mesh_hulls.insert(name.clone(), hull);
        out.mesh_raw.insert(name.clone(), walk.mesh_raw[name].clone());
        out.mesh_file.insert(name.clone(), walk.mesh_file[name].clone());
    }
    for (name, d) in &walk.mesh_cache {
        out.mesh_props.insert(name.clone(), *d);
    }
    for el in root.children_named("option") {
        if let Some(v) = el.attr("timestep") {
            out.timestep = v.trim().parse::<f64>().map_err(|e| format!("<option timestep>: {e}"))?;
        }
        if let Some(v) = el.attr("gravity") {
            out.gravity = vec3(v).map_err(|e| format!("<option gravity>: {e}"))?;
        }
        if let Some(v) = el.attr("magnetic") {
            out.magnetic = vec3(v).map_err(|e| format!("<option magnetic>: {e}"))?;
        }
        if let Some(v) = el.attr("wind") {
            out.wind = vec3(v).map_err(|e| format!("<option wind>: {e}"))?;
        }
        if let Some(v) = el.attr("cone") {
            out.cone = match v.trim() {
                "pyramidal" => crate::mujoco_contact::Cone::Pyramidal,
                "elliptic" => crate::mujoco_contact::Cone::Elliptic,
                other => return Err(format!("<option cone>: '{other}' is neither pyramidal nor elliptic")),
            };
        }
        if let Some(v) = el.attr("impratio") {
            out.impratio = v.trim().parse::<f64>().map_err(|e| format!("<option impratio>: {e}"))?;
        }
        if let Some(v) = el.attr("noslip_iterations") {
            out.noslip_iterations = v.trim().parse::<usize>().map_err(|e| format!("<option noslip_iterations>: {e}"))?;
        }
        if let Some(v) = el.attr("noslip_tolerance") {
            out.noslip_tolerance = v.trim().parse::<f64>().map_err(|e| format!("<option noslip_tolerance>: {e}"))?;
        }
        for flag in el.children_named("flag") {
            if let Some(v) = flag.attr("eulerdamp") {
                out.eulerdamp = match v.trim() {
                    "enable" => true,
                    "disable" => false,
                    other => return Err(format!("<flag eulerdamp>: '{other}' is neither enable nor disable")),
                };
            }
        }
        if let Some(v) = el.attr("integrator") {
            out.integrator = match v.trim() {
                "Euler" => MjcfIntegrator::Euler,
                "RK4" => MjcfIntegrator::Rk4,
                "implicit" => MjcfIntegrator::Implicit,
                "implicitfast" => MjcfIntegrator::ImplicitFast,
                other => return Err(format!("<option integrator>: '{other}' is not one of Euler, RK4, implicit, implicitfast")),
            };
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
    // ⛔ **`actuator_acc0`**, the other thing `mj_setConst` cannot do while reading the file: the
    // acceleration a unit force on each transmission produces at `qpos0`, `‖M⁻¹·moment‖`. A muscle whose
    // `force` is negative — MuJoCo's DEFAULT — reads it as `scale/acc0` and is its peak force, so getting
    // it wrong scales the whole muscle rather than shifting it. MuJoCo computes it for EVERY actuator,
    // muscle or not, and so does this — the moment of a spatial tendon is only known once the tree can be
    // posed, so it cannot be done any earlier either.
    if !out.actuators.is_empty() {
        let nv = out.tree.joints.len();
        if let Some(chol) = out.mass_matrix(&out.reference_q).cholesky() {
            // ⛔ WITHOUT the contact set: `mj_setConst` never calls `mj_collision`, so an adhesion actuator's
            // `acc0` is zero whatever touches it at the reference pose
            let state = out.actuator_state_with(&out.reference_q, &vec![0.0; nv], false);
            // ⛔⛔ `acc0` is a NORM of an acceleration, so it is BASIS-DEPENDENT. MuJoCo takes it in its
            // own coordinates, and this port's free base is a different six. `M⁻¹·moment` maps across as
            // `T·(M_ours⁻¹·moment_ours)`, and only where `T` is orthogonal do the two norms agree — which
            // at the reference pose is exactly the case of an UNROTATED base, because there `T` is a
            // permutation. Every Menagerie model happens to be that case, so the map changes nothing
            // measurable here — it is carried because the identity it relies on is an accident of the
            // corpus, not of the definition, and a model with a tilted base would be silently wrong.
            let t = out.free_basis(&out.reference_q);
            let acc0: Vec<f64> = state
                .iter()
                .map(|st| {
                    let mut m = nalgebra::DVector::zeros(nv);
                    for (d, v) in &st.moment {
                        m[*d] += v;
                    }
                    (&t * chol.solve(&m)).norm()
                })
                .collect();
            for (a, v) in out.actuators.iter_mut().zip(acc0) {
                a.acc0 = v;
            }
        }
    }
    // `mjCModel::AutoSpringDamper`, from MuJoCo's source: a joint that states `springdamper` gets its
    // stiffness and damping from its average inverse weight at `qpos0`, the same way a `solref` time constant
    // becomes a stiffness — `inertia = ndim / Σ dof_invweight0`, `stiffness = inertia / (τ²ζ²)`,
    // `damping = 2·inertia / τ` — replacing whatever the file stated. flybody's halteres are sprung this way.
    if out.joints.iter().any(|j| j.springdamper.is_some()) {
        let invweight = out.dof_invweight0();
        for j in out.joints.iter_mut() {
            let Some([timeconst, dampratio]) = j.springdamper else { continue };
            let ndim = j.kind.dofs();
            let inertia = ndim as f64 / (0..ndim).map(|k| invweight[j.first + k]).sum::<f64>().max(1e-15);
            j.stiffness = inertia / (timeconst * timeconst * dampratio * dampratio).max(1e-15);
            j.damping = 2.0 * inertia / timeconst.max(1e-15);
        }
    }
    // `stat.meaninertia` (`mj_setConst`): the mean diagonal of M at qpos0, in MuJoCo's basis
    if !out.tree.joints.is_empty() {
        let tinv = out.free_basis(&out.reference_q).try_inverse();
        if let Some(tinv) = tinv {
            let mm = tinv.transpose() * out.mass_matrix(&out.reference_q) * &tinv;
            out.meaninertia = mm.diagonal().sum() / mm.nrows() as f64;
        }
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
                if el.name != "joint" && el.name != "connect" && el.name != "weld" && el.name != "tendon" {
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
                    // ⭐ the SITE form: `<connect site1="a" site2="b"/>` says the two sites coincide, and
                    // each side already carries its own point — there is no anchor to resolve at `qpos0`
                    // and no second copy to derive. `toddlerbot` is written this way.
                    if let (Some(s1), Some(s2)) = (el.attr("site1"), el.attr("site2")) {
                        let at = |n: &str| -> Option<(Option<usize>, Vector3<f64>, String)> {
                            match out.site_frames.get(n) {
                                Some((ride, off)) => Some((Some(*ride), off.translation.vector, out.site_body.get(n).cloned().unwrap_or_default())),
                                None => out.world_fixed.get(&format!("site:{n}")).map(|i| (None, i.translation.vector, String::new())),
                            }
                        };
                        let (Some((r1, l1, b1)), Some((r2, l2, b2))) = (at(s1), at(s2)) else {
                            out.equalities_unsupported.push((name, format!("a connect between sites '{s1}' and '{s2}', one of which is not in the model")));
                            continue;
                        };
                        let diag_a = biw0(Some(b1.as_str())) + biw0(Some(b2.as_str()));
                        out.equalities.push(MjcfEquality { name, kind: EqualityKind::Connect { side1: (r1, l1), side2: (r2, l2), diag_a, bodies: [b1, b2] }, solref, solimp });
                        continue;
                    }
                    let Some(anchor) = el.attr("anchor") else {
                        out.equalities_unsupported.push((name, "a connect with neither an anchor nor two sites".into()));
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
                    let body = |b: Option<&str>| b.filter(|b| *b != "world").unwrap_or("").to_string();
                    let bodies = [body(el.attr("body1")), body(el.attr("body2"))];
                    out.equalities.push(MjcfEquality { name, kind: EqualityKind::Connect { side1: local(ride1, world1), side2: local(ride2, world2), diag_a, bodies }, solref, solimp });
                    continue;
                }
                if el.name == "weld" {
                    let torquescale = get("torquescale").map(|v| v.trim().parse::<f64>().map_err(|e| format!("equality '{name}' torquescale: {e}"))).transpose()?.unwrap_or(1.0);
                    let iw = |b: &str| invw.get(b).copied().unwrap_or(crate::mujoco_contact::InvWeight { tran: 0.0, rot: 0.0 });
                    // the SITE form: the two sites' own positions and orientations, no relative pose
                    if let (Some(s1), Some(s2)) = (el.attr("site1"), el.attr("site2")) {
                        let at = |n: &str| -> Option<(Option<usize>, Iso, String)> {
                            match out.site_frames.get(n) {
                                Some((ride, off)) => Some((Some(*ride), *off, out.site_body.get(n).cloned().unwrap_or_default())),
                                None => out.world_fixed.get(&format!("site:{n}")).map(|i| (None, *i, String::new())),
                            }
                        };
                        let (Some((r1, o1, b1)), Some((r2, o2, b2))) = (at(s1), at(s2)) else {
                            out.equalities_unsupported.push((name, format!("a weld between sites '{s1}' and '{s2}', one of which is not in the model")));
                            continue;
                        };
                        let (w1, w2) = (iw(&b1), iw(&b2));
                        out.equalities.push(MjcfEquality {
                            name,
                            kind: EqualityKind::Weld { side1: (r1, o1.translation.vector), side2: (r2, o2.translation.vector), rot1: (r1, o1.rotation), rot2: (r2, o2.rotation), relpose: [1.0, 0.0, 0.0, 0.0], torquescale, diag: [w1.tran + w2.tran, w1.rot + w2.rot], bodies: [b1, b2] },
                            solref,
                            solimp,
                        });
                        continue;
                    }
                    // the BODY form: `anchor` is in body2's frame; `relpose` (pos, quat) is body2's pose in
                    // body1's, and when its quaternion is zero (the default) both are taken at `qpos0`
                    let body_at = |b: Option<&str>| -> Result<(Option<usize>, Iso, Iso, String), String> {
                        let Some(b) = b.filter(|b| *b != "world") else { return Ok((None, Iso::identity(), Iso::identity(), String::new())) };
                        if let Some((ride, pre)) = out.body_frames.get(b) {
                            return Ok((Some(*ride), *pre, frames0[*ride] * pre, b.to_string()));
                        }
                        match out.world_fixed.get(&format!("body:{b}")) {
                            Some(iso) => Ok((None, *iso, *iso, b.to_string())),
                            None => Err(format!("equality '{name}': no body named '{b}'")),
                        }
                    };
                    let Some(b1) = el.attr("body1") else {
                        out.equalities_unsupported.push((name, "a weld with neither body1 nor two sites".into()));
                        continue;
                    };
                    let (ride1, pre1, x1, n1) = body_at(Some(b1))?;
                    let (ride2, pre2, x2, n2) = body_at(el.attr("body2"))?;
                    let anchor = get("anchor").map(|v| vec3(&v)).transpose()?.unwrap_or_else(Vector3::zeros);
                    let mut relpose = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    if let Some(v) = get("relpose") {
                        for (i, x) in floats(&v)?.iter().take(7).enumerate() {
                            relpose[i] = *x;
                        }
                    }
                    let (anchor1, rel) = if relpose[3..7].iter().any(|x| *x != 0.0) {
                        // `mj_setConst`: a stated relative pose is kept, its quaternion normalised
                        let mut q = [relpose[3], relpose[4], relpose[5], relpose[6]];
                        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
                        if n < 1e-15 {
                            q = [1.0, 0.0, 0.0, 0.0];
                        } else if (n - 1.0).abs() > 1e-15 {
                            q.iter_mut().for_each(|x| *x *= 1.0 / n);
                        }
                        (Vector3::new(relpose[0], relpose[1], relpose[2]), q)
                    } else {
                        // at `qpos0`: body2's anchor seen from body1, and `neg(xquat1)·xquat2`
                        let a1 = (x1.inverse() * (x2 * Point3::from(anchor))).coords;
                        let (qa, qb) = (x1.rotation, x2.rotation);
                        (a1, quat_mul(&quat_neg(&[qa.w, qa.i, qa.j, qa.k]), &[qb.w, qb.i, qb.j, qb.k]))
                    };
                    let local = |ride: Option<usize>, pre: Iso, p: Vector3<f64>| (ride, (pre * Point3::from(p)).coords);
                    let (w1, w2) = (iw(&n1), iw(&n2));
                    out.equalities.push(MjcfEquality {
                        name,
                        kind: EqualityKind::Weld {
                            side1: local(ride1, pre1, anchor1),
                            side2: local(ride2, pre2, anchor),
                            rot1: (ride1, pre1.rotation),
                            rot2: (ride2, pre2.rotation),
                            relpose: rel,
                            torquescale,
                            diag: [w1.tran + w2.tran, w1.rot + w2.rot],
                            bodies: [n1, n2],
                        },
                        solref,
                        solimp,
                    });
                    continue;
                }
                if el.name == "tendon" {
                    let find = |n: &str| out.tendons.iter().position(|t| t.name == n);
                    let t1 = el.attr("tendon1").ok_or_else(|| format!("equality '{name}': needs tendon1"))?;
                    let Some(i1) = find(t1) else {
                        out.equalities_unsupported.push((name, format!("tendon '{t1}' is not one this loader carries")));
                        continue;
                    };
                    let i2 = match el.attr("tendon2") {
                        Some(t2) => match find(t2) {
                            Some(i) => Some(i),
                            None => {
                                out.equalities_unsupported.push((name, format!("tendon '{t2}' is not one this loader carries")));
                                continue;
                            }
                        },
                        None => None,
                    };
                    let mut polycoef = [0.0, 1.0, 0.0, 0.0, 0.0];
                    if let Some(v) = get("polycoef") {
                        for (i, x) in floats(&v)?.iter().take(5).enumerate() {
                            polycoef[i] = *x;
                        }
                    }
                    // `tendon_length0`: each tendon's length at `qpos0`
                    let l0 = out.ten_length(&out.reference_q);
                    out.equalities.push(MjcfEquality { name, kind: EqualityKind::Tendon { tendon1: i1, tendon2: i2, length0: (l0[i1], i2.map_or(0.0, |i| l0[i])), polycoef }, solref, solimp });
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
    out.sensors = crate::mjcf_sensor::parse_sensors(&root, &out)?;
    // `<keyframe>`: sizes against the finished model
    {
        let nq: usize = out.joints.iter().map(|j| j.kind.qpos_width()).sum();
        let (nv, na, nu) = (out.tree.joints.len(), out.na(), out.actuators.len());
        let qpos0 = out.qpos0();
        for section in root.children_named("keyframe") {
            for (k, el) in section.children_named("key").enumerate() {
                let name = el.attr("name").unwrap_or("").to_string();
                let vals = |attr: &str, n: usize, dflt: &dyn Fn() -> Vec<f64>, check: bool| -> Result<Vec<f64>, String> {
                    match el.attr(attr) {
                        None => Ok(dflt()),
                        Some(v) => {
                            let x = floats(v)?;
                            if check && x.len() != n {
                                return Err(format!("keyframe '{name}' ({k}): invalid {attr} size, expected {n}, got {}", x.len()));
                            }
                            Ok(x)
                        }
                    }
                };
                let time = el.attr("time").map(|t| t.trim().parse::<f64>().map_err(|e| format!("keyframe '{name}' time: {e}"))).transpose()?.unwrap_or(0.0);
                // `ctrl` and `act` are only checked when every actuator is carried: a refused one shortens both
                let actuators_whole = out.actuators_unsupported.is_empty();
                let mut qpos = vals("qpos", nq, &|| qpos0.clone(), true)?;
                for (j, adr) in out.joints.iter().zip(out.qposadr()) {
                    match j.kind {
                        MjcfJointKind::Ball => normalize_key_quat(&mut qpos[adr..adr + 4]),
                        MjcfJointKind::Free => normalize_key_quat(&mut qpos[adr + 3..adr + 7]),
                        _ => {}
                    }
                }
                let mut mquat = vals("mquat", 0, &Vec::new, false)?;
                for q in mquat.chunks_mut(4) {
                    if q.len() == 4 {
                        normvec(q);
                    }
                }
                out.keyframes.push(MjcfKeyframe {
                    time,
                    qpos,
                    qvel: vals("qvel", nv, &|| vec![0.0; nv], true)?,
                    act: vals("act", na, &|| vec![0.0; na], actuators_whole)?,
                    ctrl: vals("ctrl", nu, &|| vec![0.0; nu], actuators_whole)?,
                    mpos: vals("mpos", 0, &Vec::new, false)?,
                    mquat,
                    name,
                });
            }
        }
        // `<size nkey>` allocates keys the file never states: each is the reset state
        let nkey = root.children_named("size").filter_map(|el| el.attr("nkey")).filter_map(|v| v.trim().parse::<usize>().ok()).max().unwrap_or(0);
        while out.keyframes.len() < nkey {
            out.keyframes.push(MjcfKeyframe { name: String::new(), time: 0.0, qpos: qpos0.clone(), qvel: vec![0.0; nv], act: vec![0.0; na], ctrl: vec![0.0; nu], mpos: Vec::new(), mquat: Vec::new() });
        }
    }
    // the loader read inverse weights while the model was still being built; the first caller after it
    // computes them from the finished one
    out.invweight0_cache = std::sync::OnceLock::new();
    out.collide_cache = std::sync::OnceLock::new();
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

    /// ⛔ A geom with its own `quat` inside a class that states `euler` is NOT "two orientations given" —
    /// that refusal once kept `anybotics_anymal_b` out while MuJoCo compiles it. ⛔⛔ And the class's `euler`
    /// WINS: MuJoCo keeps `quat` and the alternative in separate slots, inherits each on its own, and
    /// resolves the alternative after the quaternion. (This test used to assert the element's `quat` won —
    /// an assumption never run through MuJoCo; MuJoCo 3.13.0 turns BOTH geoms below to +x.)
    #[test]
    fn a_classs_alternative_orientation_beats_an_elements_own_quat() {
        let xml = r#"<mujoco>
            <default><default class="turned"><geom type="cylinder" size="0.05 0.05" euler="0 90 0"/></default></default>
            <worldbody><body name="b"><joint name="j" type="hinge" axis="0 0 1"/>
              <geom name="own" class="turned" quat="1 0 0 0"/>
              <geom name="inherited" class="turned"/>
            </body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).expect("loads: the two are alternatives, not a conflict");
        let axis = |name: &str| t.geoms.iter().find(|g| g.name == name).unwrap().pose.rotation.to_rotation_matrix() * Vector3::z();
        assert!((axis("own") - Vector3::x()).norm() < 1e-12, "the geom's own quat beat its class's euler: {:?}", axis("own"));
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

    /// **MuJoCo's wrap rules, from its source, against its own numbers.** Each tendon here exercises one
    /// branch of `mju_wrap` that the Menagerie sweep had only been able to approximate from outputs:
    ///   * `inside`: a side site INSIDE the obstacle is `wrap_inside`, a cable through ONE point of the
    ///     circle with no arc, not a circle wrap;
    ///   * `across`: a side site across from the run, with the infinite line clear of the circle. MuJoCo
    ///     still wraps, and picks the way round by direction with crossing runs penalised;
    ///   * `tall`: a side site inside the cylinder's CIRCLE but far along its axis. "Inside" is the 3-D
    ///     distance from the centre, so this is an ordinary circle test, and here it does not wrap at all;
    ///   * `ball_inside`: the inside wrap on a sphere, in the plane of the centre and both endpoints;
    ///   * `crossing`: the side site's DIRECTION prefers the candidate whose two straight runs cross each
    ///     other, so the penalty hands the wrap to the other way round.
    ///
    /// ⛔ One line of `wrap_circle` is not exercised: "no wrap if the CHOSEN candidate's runs still cross".
    /// In 2,000,000 random 2-D geometries, with and without a side site, both candidates never crossed
    /// at once, so deleting that check changes nothing this test (or Menagerie) can see.
    #[test]
    fn mujocos_wrap_rules_inside_across_and_along_the_axis() {
        let t = tree_from_mjcf_str(WRAP_RULES_MODEL).unwrap();
        assert!(t.tendons_unsupported.is_empty(), "{:?}", t.tendons_unsupported);
        let q = [0.07];
        let (len, vel) = (t.ten_length(&q), t.ten_velocity(&q, &[1.0]));
        // name, ten_length, ten_velocity at q̇ = 1, and MuJoCo's wrap_xpos pair (None: no wrap)
        let want: [(&str, f64, f64, Option<[[f64; 3]; 2]>); 5] = [
            ("inside", 0.9092707159093799, 0.9538181602238763, Some([[0.0028113626194876985, 0.10602107912979178, -0.019707759182168463]; 2])),
            (
                "across",
                1.095556113756721,
                0.7852796289662715,
                Some([[-0.06146384977322341, -0.06761687651599636, -0.0501413698334621], [0.06719017030120919, -0.06735076384761232, -0.06229949760136277]]),
            ),
            ("tall", 0.8771544903835357, 0.991843523048708, None),
            ("ball_inside", 0.9111720267678524, 0.947923827835781, Some([[-0.002799374752766072, 0.09993816332085413, 1.0021276778534032]; 2])),
            ("crossing", 1.0619512100538142, 0.7786545888708268, Some([[-0.07309461601156907, 0.06824351331900542, 2.0], [0.06274528119551333, 0.07786545888708267, 2.0]])),
        ];
        for (i, (name, wl, wv, wp)) in want.iter().enumerate() {
            assert_eq!(&t.tendons[i].name, name);
            assert!((len[i] - wl).abs() < 1e-12, "{name}: length {} vs MuJoCo {wl}", len[i]);
            assert!((vel[i] - wv).abs() < 1e-12, "{name}: velocity {} vs MuJoCo {wv}", vel[i]);
            let wraps = t.tendon_wraps(i, &q);
            match wp {
                None => assert!(wraps.is_empty(), "{name}: MuJoCo does not wrap, this port wrapped {wraps:?}"),
                Some(pts) => {
                    assert_eq!(wraps.len(), 1, "{name}: {wraps:?}");
                    for (k, w) in pts.iter().enumerate() {
                        let d = (wraps[0].1[k] - Vector3::from(*w)).norm();
                        assert!(d < 1e-12, "{name}: wrap point {k} is {:?}, MuJoCo {w:?} ({d:.2e} apart)", wraps[0].1[k]);
                    }
                }
            }
        }
    }

    /// **`<adhesion>`, against MuJoCo's own numbers.** The pad touches the floor twice: sphere `a` is an
    /// ACTIVE contact and sphere `b` sits in its GAP band (MuJoCo's `exclude = 1`, no constraint rows). Both
    /// count, so the moment is minus the mean of the two normal Jacobians. A third contact, on another body,
    /// must not. Under both cones the answer is the same: a pyramidal contact's rows average back to the
    /// normal. And `acc0` is ZERO, because `mj_setConst` never collides.
    #[test]
    fn an_adhesion_actuator_averages_every_contact_on_its_body_active_or_in_the_gap() {
        for cone in ["elliptic", "pyramidal"] {
            let t = tree_from_mjcf_str(&ADHESION_MODEL.replace(r#"cone="elliptic""#, &format!(r#"cone="{cone}""#))).unwrap();
            assert!(t.actuators_unsupported.is_empty(), "{:?}", t.actuators_unsupported);
            assert_eq!(t.actuators.len(), 2);
            // mjs_setToAdhesion: fixed gain, no bias, ALWAYS limited
            assert_eq!((t.actuators[0].gainprm[0], t.actuators[1].gainprm[0]), (5.0, 1.0));
            assert_eq!((t.actuators[0].ctrlrange, t.actuators[1].ctrlrange), (Some([0.0, 1.0]), Some([0.0, 2.0])));
            assert_eq!((t.actuators[0].acc0, t.actuators[1].acc0), (0.0, 0.0), "mj_setConst runs no collision");
            let (q, qd, ctrl) = ([0.001, 0.05, -0.03, 0.0], [0.3, -0.2, 0.5, 0.1], [0.7, 1.5]);
            let st = t.actuator_state(&q, &qd);
            let want_moment = [[-1.0, -0.016649820522149587, -0.028975390251305563, 0.0], [0.0, 0.0, 0.0, -1.0]];
            let want_vel = [-0.31115773102122285, -0.1];
            for (u, (wm, wv)) in want_moment.iter().zip(want_vel).enumerate() {
                let mut dense = [0.0; 4];
                for (d, v) in &st[u].moment {
                    dense[*d] += v;
                }
                for k in 0..4 {
                    assert!((dense[k] - wm[k]).abs() < 1e-12, "{cone}: actuator {u} moment[{k}] {} vs MuJoCo {}", dense[k], wm[k]);
                }
                assert_eq!(st[u].length, 0.0);
                assert!((st[u].velocity - wv).abs() < 1e-12, "{cone}: actuator {u} velocity {} vs MuJoCo {wv}", st[u].velocity);
            }
            let f = t.qfrc_actuator(&q, &qd, &ctrl);
            for (k, w) in [-3.5, -0.058274371827523555, -0.10141386587956946, -1.5].iter().enumerate() {
                assert!((f[k] - w).abs() < 1e-12, "{cone}: qfrc_actuator[{k}] {} vs MuJoCo {w}", f[k]);
            }
        }
        // ⛔ the shortcut ALWAYS limits the control, and the default range is [0, 0], so an adhesion actuator
        // with no ctrlrange is refused. MuJoCo 3.13.0: "invalid control range for actuator".
        let why = match tree_from_mjcf_str(&ADHESION_MODEL.replace("</actuator>", r#"<adhesion name="unranged" body="other" gain="2"/></actuator>"#)) {
            Err(e) => e,
            Ok(t) => format!("loaded {} actuators, refused {:?}", t.actuators.len(), t.actuators_unsupported),
        };
        assert!(why.contains("invalid control range"), "an adhesion actuator with no ctrlrange must be refused: {why}");
    }

    /// **`springdamper`, against MuJoCo's own numbers.** A joint that states a time constant and a damping
    /// ratio gets its stiffness and damping COMPUTED from its average inverse weight at `qpos0`
    /// (`mjCModel::AutoSpringDamper`), overriding the `stiffness="99" damping="99"` the hinge also states. A
    /// ball joint averages over its three dofs (which `dof_invweight0` has already made one shared value, so
    /// that average cannot be told from taking the first). flybody's two halteres are sprung this way.
    #[test]
    fn a_springdamper_joint_gets_the_stiffness_and_damping_mujoco_computes() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <body name="a" pos="0 0 0.3">
    <joint name="hinge" type="hinge" axis="0.3 1 0.2" springdamper="0.05 0.7" stiffness="99" damping="99"/>
    <geom type="capsule" size="0.03 0.2" pos="0.1 0 0" euler="0 1.2 0" mass="0.8"/>
    <body name="b" pos="0.3 0 0">
      <joint name="slide" type="slide" axis="1 0.2 0" springdamper="0.2 1.5"/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.4"/>
      <body name="c" pos="0.1 0.05 0">
        <joint name="ball" type="ball" springdamper="0.1 0.3"/>
        <geom type="ellipsoid" size="0.06 0.03 0.02" pos="0.04 0 0" mass="0.2"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>"#,
        )
        .unwrap();
        let want = [("hinge", 50.45954575355362, 2.4725177419241273), ("slide", 6.6095777627584535, 5.948619986482609), ("ball", 0.1380264945560344, 0.0024844769020086195)];
        for (name, k, c) in want {
            let j = t.joints.iter().find(|j| j.name == name).unwrap();
            assert!((j.stiffness - k).abs() < 1e-9 * k, "{name}: stiffness {} vs MuJoCo {k}", j.stiffness);
            assert!((j.damping - c).abs() < 1e-9 * c, "{name}: damping {} vs MuJoCo {c}", j.damping);
        }
        let err = tree_from_mjcf_str(r#"<mujoco><worldbody><body><joint name="j" type="hinge" springdamper="0.1 0"/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"#).err().unwrap_or_default();
        assert!(err.contains("springdamper values must be positive"), "{err}");
    }

    /// **`forward_mujoco`, the library's one-call `mj_forward`, against MuJoCo's own converged answer.** A box
    /// on a free joint resting tilted on the floor (two corner contacts) beside an arm pushed past its
    /// joint limit, moving, under both cones and `impratio = 3`, all read from the file. MuJoCo is run with
    /// `iterations="1000" tolerance="0"` so that its answer is the optimum and not an iterate.
    #[test]
    fn forward_mujoco_matches_mujocos_converged_qacc_under_both_cones() {
        let qpos = [0.01, -0.02, 0.049, 0.9998249759398523, 0.010000249809360396, -0.015000374714040593, 0.005000124904680198, 0.31];
        let qvel = [0.2, -0.1, -0.05, 0.3, -0.2, 0.1, 0.4];
        let want: [(&str, usize, [f64; 7]); 2] = [
            ("pyramidal", 9, [-4.5258058715528815, 3.620304256307129, -1.6638898721399942, -24.83205204979471, 245.69038295979536, -73.91773656072378, -64.08928673223734]),
            ("elliptic", 7, [-4.94571432905561, 3.5708644345652583, -1.9503002825102176, -18.58818661642026, 244.0181614948763, -79.17526063139837, -64.08928673223734]),
        ];
        for (cone, nefc, qacc) in want {
            let t = tree_from_mjcf_str(&FORWARD_MODEL.replace("CONE", cone)).unwrap();
            let q = t.q_from_qpos(&qpos, &[0, 7]).unwrap();
            let f = t.forward_mujoco(&q, &qvel, &[], &[]).unwrap();
            assert_eq!(f.nefc, nefc, "{cone}: rows");
            assert_eq!(f.collision.contacts.len(), 2, "{cone}: contacts");
            for k in 0..7 {
                assert!((f.qacc[k] - qacc[k]).abs() < 1e-8 * qacc[k].abs().max(1.0), "{cone}: qacc[{k}] {} vs MuJoCo {}", f.qacc[k], qacc[k]);
            }
        }
    }

    /// **`step_mujoco`, the library's `mj_step` (Euler), against MuJoCo's own trajectory.** A free box on the
    /// floor, an arm past its joint limit and a ball-jointed pendulum, moving: once with dof damping (MuJoCo
    /// then integrates the velocity implicitly in it, `(M + h·D)⁻¹`) and once without. One step is held to
    /// 1e-10 and twenty-five to 1e-7; MuJoCo runs `iterations="1000" tolerance="0"` so each of its steps is
    /// the converged one this port takes.
    #[test]
    fn step_mujoco_follows_mujocos_euler_trajectory_with_and_without_damping() {
        let qpos0 = [0.01, -0.02, 0.049, 0.9998249759398523, 0.010000249809360396, -0.015000374714040593, 0.005000124904680198, 0.31, 0.9822618062296308, 0.10023079655404396, -0.15034619483106593, 0.05011539827702198];
        let qvel0 = [0.2, -0.1, -0.05, 0.3, -0.2, 0.1, 0.4, 0.5, -0.3, 0.2];
        type Snap = ([f64; 12], [f64; 10]);
        let want: [(&str, Snap, Snap); 2] = [
            (
                "0.2",
                (
                    [0.010380217142683779, -0.020185716542261742, 0.04889219879886996, 0.9998268823411698, 0.010262461713718413, -0.014710489975223043, 0.004948607096941901, 0.3105504420547829, 0.9821754855744429, 0.10067103163600079, -0.1505514440942976, 0.05030826579273555],
                    [0.19010857134188877, -0.09285827113087017, -0.05390060056502018, 0.2628236267671603, 0.28803632298975357, -0.058350521262796405, 0.2752210273914292, 0.45978886481975273, -0.21731779597310744, 0.14815702303299097],
                ),
                (
                    [0.013258487049700713, -0.020169147196851943, 0.04925187689690385, 0.9999919866484078, 0.0016613894442337517, -0.0008634559997705678, 0.0035384838307690497, 0.3044215407714665, 0.9835869936613889, 0.09924292950081777, -0.1411042732925926, 0.052887152530803255],
                    [0.011299785223229634, 0.014808428339528975, 0.021940691234653982, -0.3054497078983967, 0.21471685401162136, -0.0016131745722577867, -0.1344479267160266, -0.33179869800999484, 0.7932692266127152, 0.025934366118544976],
                ),
            ),
            (
                "0",
                (
                    [0.010380217142683779, -0.02018571654226174, 0.04889219879886996, 0.9998268823411698, 0.010262461713718411, -0.014710489975223043, 0.004948607096941901, 0.3105447326523489, 0.9821693824625731, 0.1006702029158415, -0.15057551022428187, 0.050357026507414186],
                    [0.19010857134188877, -0.09285827113086964, -0.05390060056502038, 0.2628236267671595, 0.28803632298975274, -0.058350521262796745, 0.2723663261744362, 0.465711471404695, -0.2369457611948599, 0.19889543632290546],
                ),
                (
                    [0.013258487049700713, -0.02016914719685193, 0.04925187689690385, 0.9999919866484078, 0.0016613894442337532, -0.0008634559997705658, 0.0035384838307690545, 0.3043323205075922, 0.9837553010328234, 0.10009915764353215, -0.13802299888637334, 0.05617221828703629],
                    [0.011299785223229625, 0.014808428339528988, 0.021940691234653975, -0.30544970789839737, 0.21471685401162147, -0.0016131745722573932, -0.13234429671280784, -0.36300009987506565, 1.2673169892733536, 0.16826335790464692],
                ),
            ),
        ];
        let near = |got: &[f64], want: &[f64], tol: f64, what: &str| {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() < tol * w.abs().max(1.0), "{what}[{k}] {g} vs MuJoCo {w}");
            }
        };
        for (damp, one, twentyfive) in want {
            let t = tree_from_mjcf_str(&STEP_MODEL.replace("DAMP", damp)).unwrap();
            let (mut qpos, mut qvel) = (qpos0.to_vec(), qvel0.to_vec());
            for k in 1..=25 {
                let st = t.step_mujoco(&qpos, &qvel, &[], &[]).unwrap();
                (qpos, qvel) = (st.qpos, st.qvel);
                if k == 1 {
                    near(&qpos, &one.0, 1e-10, &format!("damping {damp}, step 1: qpos"));
                    near(&qvel, &one.1, 1e-10, &format!("damping {damp}, step 1: qvel"));
                }
            }
            near(&qpos, &twentyfive.0, 1e-7, &format!("damping {damp}, step 25: qpos"));
            near(&qvel, &twentyfive.1, 1e-7, &format!("damping {damp}, step 25: qvel"));
        }
        // ⛔ `<flag eulerdamp="disable">`, which every Menagerie file written for MJX sets: the damping is still
        // a force, but the velocity is integrated EXPLICITLY in it
        let t = tree_from_mjcf_str(&STEP_MODEL.replace("DAMP", "0.2").replace("<worldbody>", r#"<option><flag eulerdamp="disable"/></option><worldbody>"#)).unwrap();
        assert!(!t.eulerdamp);
        let st = t.step_mujoco(&qpos0, &qvel0, &[], &[]).unwrap();
        near(&st.qvel, &[0.19010857134188877, -0.09285827113086964, -0.05390060056502038, 0.2628236267671595, 0.28803632298975274, -0.058350521262796745, 0.2718214265355253, 0.5751878886107167, -0.15442634068772546, -0.1480369835302928], 1e-10, "eulerdamp disabled: qvel");
    }

    /// **`implicitfast`, MuJoCo 3.13.0's rules, against its own trajectory.** Three things the integrator
    /// does that a plain `M − h·∂f/∂v` would get wrong, one in each part of the model:
    /// * the wrist servo's force is PINNED at its `forcerange` (2 N·m), so its `kv` contributes no derivative
    ///   at all (`actuatorDerivSkip`);
    /// * the grip servo pulls two SIBLING slides through one tendon, and MuJoCo's derivative keeps only the
    ///   entries the mass matrix's sparsity has a slot for, so there is no cross term between the fingers;
    /// * the tumbling box is a standalone free body, solved on its own as an unsymmetric 6×6 that includes
    ///   the gyroscopic derivative (`mjd_freeMhat`).
    #[test]
    fn step_mujoco_follows_mujocos_implicitfast_trajectory() {
        let t = tree_from_mjcf_str(IMPLICITFAST_MODEL).unwrap();
        assert_eq!(t.integrator, MjcfIntegrator::ImplicitFast);
        let (mut qpos, mut qvel) = (
            vec![0.01, -0.02, 0.3, 0.9713121577180394, 0.10013527399155045, -0.2002705479831009, 0.08010821919324036, 0.4, 0.01, -0.005],
            vec![0.1, 0.2, -0.3, 3.0, -2.0, 5.0, 0.7, 0.3, -0.2],
        );
        let ctrl = [0.9, 0.03];
        let near = |got: &[f64], want: &[f64], tol: f64, what: &str| {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() < tol * w.abs().max(1.0), "{what}[{k}] {g} vs MuJoCo {w}");
            }
        };
        for k in 1..=25 {
            let st = t.step_mujoco(&qpos, &qvel, &ctrl, &[]).unwrap();
            (qpos, qvel) = (st.qpos, st.qvel);
            if k == 1 {
                near(&qpos, &[0.01020172181685666, -0.01959891925532773, 0.29936188978505046, 0.9701961287279255, 0.10220854248728502, -0.20244375230181513, 0.08537805805904039, 0.402194902373685, 0.010624545454545454, -0.005357272727272727], 1e-10, "step 1 qpos");
                near(&qvel, &[0.10086090842833019, 0.20054037233613445, -0.3190551074747613, 3.00686581689145, -1.9727691084805241, 5.010493211551619, 1.09745118684248, 0.31227272727272726, -0.17863636363636365], 1e-10, "step 1 qvel");
            }
        }
        near(&qpos, &[0.015504119794759796, -0.009611904021463651, 0.2726344269996223, 0.933660060990064, 0.15054890592708775, -0.24409313508770888, 0.21455176261042744, 0.6121286807547449, 0.02602953323854886, -0.008837776633251056], 1e-7, "step 25 qpos");
        near(&qvel, &[0.11828805485946914, 0.21556122827087396, -0.7752658029575709, 3.1305892176094394, -1.2844060516761395, 5.23209093858203, 5.630739997332279, 0.28443201416748676, -0.031611412945476945], 1e-7, "step 25 qvel");
    }

    /// **`RK4`, against MuJoCo's own trajectory.** A damped double pendulum (whose accelerations depend
    /// strongly on position, so a stage evaluated at the wrong position shows) beside a spinning free body
    /// (so every stage composes a quaternion). One step to 1e-13, twenty-five to 1e-10.
    #[test]
    fn step_mujoco_follows_mujocos_rk4_trajectory() {
        let t = tree_from_mjcf_str(RK4_MODEL).unwrap();
        assert_eq!(t.integrator, MjcfIntegrator::Rk4);
        let (mut qpos, mut qvel) = (
            vec![0.4, -0.7, 1.0, 0.0, 2.0, 0.9535826651341416, 0.10037712264569912, -0.20075424529139824, 0.20075424529139824],
            vec![0.5, -1.2, 0.1, 0.0, 0.3, 2.0, -3.0, 4.0],
        );
        let near = |got: &[f64], want: &[f64], tol: f64, what: &str| {
            for (k, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() < tol * w.abs().max(1.0), "{what}[{k}] {g} vs MuJoCo {w}");
            }
        };
        for k in 1..=25 {
            let st = t.step_mujoco(&qpos, &qvel, &[], &[]).unwrap();
            (qpos, qvel) = (st.qpos, st.qvel);
            if k == 1 {
                near(&qpos, &[0.4065548904268078, -0.7127477564085901, 1.001, 0.0, 2.0025095, 0.9452088759680639, 0.10895365820470451, -0.21480300140460598, 0.22038364672219776], 1e-13, "step 1 qpos");
                near(&qvel, &[0.8100800075771147, -1.3462424410623726, 0.1, 0.0, 0.20189999999999997, 2.0560849137108663, -2.9319656411257933, 4.036092615011649], 1e-13, "step 1 qvel");
            }
        }
        near(&qpos, &[1.1322821854482021, -0.29929360309755704, 1.0249999999999972, 0.0, 1.7684374999999999, 0.5643603657602463, 0.2701423967690588, -0.36015299916286625, 0.6919611840420108], 1e-10, "step 25 qpos");
        near(&qvel, &[2.387657516718123, 11.472761626118137, 0.1, 0.0, -2.1525000000000007, 2.9765787970938073, -0.6147526956155133, 4.711319057619501], 1e-10, "step 25 qvel");
    }

    const RK4_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option integrator="RK4" timestep="0.01"/><worldbody>
  <body name="l1" pos="0 0 1">
    <joint name="h1" type="hinge" axis="0 1 0" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="0.6"/>
    <body name="l2" pos="0.3 0 0">
      <joint name="h2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0.05" size="0.02" mass="0.4"/>
    </body>
  </body>
  <body name="spinner" pos="1 0 2">
    <freejoint/>
    <geom type="box" size="0.1 0.05 0.03" mass="0.5"/>
  </body>
</worldbody></mujoco>"#;

    const IMPLICITFAST_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option integrator="implicitfast"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="tumbler" pos="0 0 0.3">
    <freejoint/>
    <geom type="box" size="0.12 0.05 0.03" mass="0.9"/>
    <geom type="sphere" size="0.02" pos="0.1 0.03 0.01" mass="0.2"/>
  </body>
  <body name="hand" pos="0.6 0 0.4">
    <joint name="wrist" type="hinge" axis="0 1 0" damping="0.3"/>
    <geom type="capsule" size="0.03 0.1" fromto="0 0 0 0.2 0 0" mass="0.5"/>
    <body name="f1" pos="0.2 0.03 0">
      <joint name="s1" type="slide" axis="0 1 0" damping="0.5"/>
      <geom type="box" size="0.01 0.01 0.04" mass="0.05"/>
    </body>
    <body name="f2" pos="0.2 -0.03 0">
      <joint name="s2" type="slide" axis="0 -1 0" damping="0.5"/>
      <geom type="box" size="0.01 0.01 0.04" mass="0.05"/>
    </body>
  </body>
</worldbody>
<tendon><fixed name="grip"><joint joint="s1" coef="0.5"/><joint joint="s2" coef="0.5"/></fixed></tendon>
<actuator>
  <position name="wrist_servo" joint="wrist" kp="400" kv="20" forcerange="-2 2"/>
  <position name="grip_servo" tendon="grip" kp="50" kv="8" ctrlrange="0 0.04"/>
</actuator></mujoco>"#;

    const STEP_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option impratio="3" cone="elliptic"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="box" pos="0 0 0.049">
    <freejoint/>
    <geom type="box" size="0.1 0.07 0.05" mass="1.3"/>
  </body>
  <body name="arm" pos="0.5 0 0.3">
    <joint name="hinge" type="hinge" axis="0 1 0" range="-0.3 0.3" damping="DAMP"/>
    <geom type="capsule" size="0.03 0.15" fromto="0 0 0 0.25 0 -0.2" mass="0.4"/>
  </body>
  <body name="bob" pos="-0.6 0 0.8">
    <joint name="swivel" type="ball" damping="DAMP"/>
    <geom type="capsule" size="0.02 0.1" fromto="0 0 0 0.1 0.05 -0.25" mass="0.3"/>
  </body>
</worldbody></mujoco>"#;

    /// **A connect row on a moving free base, against MuJoCo.** A free pelvis carries a loop closed by a
    /// `<connect>` through a ball joint, all moving. The row's `J̇·q̇` is built in this port's Euler basis,
    /// and posing it in MuJoCo's needs the basis term `J·Ṫ·q̇` ([`MjcfTree::joint_constraint_rows_mujoco`]);
    /// without it this state fails (the base's first acceleration alone is 2e-4 off).
    #[test]
    fn a_connect_on_a_moving_free_base_matches_mujoco() {
        let t = tree_from_mjcf_str(LOOP_MODEL).unwrap();
        let qpos = [0.0, 0.0, 1.0, 0.9759716969200002, 0.10061563885773199, -0.15092345828659798, 0.12073876662927838, 0.05, -0.04, 1.0, 0.0, 0.0, 0.0];
        let qvel = [0.3, -0.2, 0.1, 1.5, -2.0, 2.5, 0.8, -0.6, 0.4, -0.3, 0.2];
        let q = t.q_from_qpos(&qpos, &t.qposadr()).unwrap();
        let f = t.forward_mujoco(&q, &qvel, &[], &[]).unwrap();
        assert_eq!(f.nefc, 7, "3 connect rows and one pyramidal contact");
        let want = [-3.2148317422921644, -0.08016359738568923, -9.760127323568188, -34.24292916867317, 61.004861351132476, 197.3940511854931, -306.4759751575595, -34.697554052837724, -3514.6476773385607, 515.191190718718, 156.86010838363552];
        for (k, w) in want.iter().enumerate() {
            assert!((f.qacc[k] - w).abs() < 1e-8 * w.abs().max(1.0), "qacc[{k}] {} vs MuJoCo {w}", f.qacc[k]);
        }
    }

    /// **A mesh's hull polygons as `mjCMesh::Process` builds them, against MuJoCo.** A prism with an interior
    /// vertex, compiled three ways: as is, MIRRORED (`scale="-1 1 1"`) and mirrored with a non-uniform scale.
    /// The polygons merge in the file's frame and take their normals from the final one without being
    /// reoriented, so on both mirrored meshes MuJoCo's normals point INWARD — and must here too.
    #[test]
    fn mesh_polygons_follow_mujocos_frames_including_mirrored_meshes() {
        const PRISM: &str = "v 0 0 0\nv 0.3 0 0\nv 0.05 0.2 0\nv 0 0 0.12\nv 0.3 0 0.12\nv 0.05 0.2 0.12\nv 0.1 0.05 0.06\nf 1 3 2\nf 4 5 6\nf 1 2 5\nf 1 5 4\nf 2 3 6\nf 2 6 5\nf 3 1 4\nf 3 4 6\n";
        type Want = [(&'static [usize], [f64; 3]); 5];
        let cases: [(&str, Want); 3] = [
            ("1 1 1", [
                (&[0, 1, 2], [-1.0, -7.777655979001052e-18, 4.704439586427497e-17]),
                (&[0, 1, 3, 4], [-5.2268167869603613e-17, 0.9175895474279813, 0.3975291466684079]),
                (&[0, 2, 3, 5], [-2.479304043496177e-18, 0.16311176581727824, -0.9866075977063877]),
                (&[1, 2, 4, 5], [-3.0752644307847943e-17, -0.9648515504601946, 0.2627955204613625]),
                (&[3, 4, 5], [1.0, 4.350176607880381e-17, -5.677453113971596e-17]),
            ]),
            ("-1 1 1", [
                (&[0, 1, 2], [1.0, -4.350176607880383e-17, -5.677453113971597e-17]),
                (&[0, 1, 3, 4], [3.2779995430006977e-17, -0.9175895474279813, 0.3975291466684078]),
                (&[0, 2, 3, 5], [1.1395669270223995e-17, -0.16311176581727818, -0.9866075977063877]),
                (&[1, 2, 4, 5], [9.475362177739566e-18, 0.9648515504601946, 0.2627955204613625]),
                (&[3, 4, 5], [-1.0, 7.777655979001045e-18, 4.7044395864274966e-17]),
            ]),
            ("0.5 -2 1.5", [
                (&[0, 1, 2], [0.0, 1.0, 0.0]),
                (&[0, 1, 3, 4], [0.13811967072676465, -7.137494834774114e-18, 0.9904155474134734]),
                (&[0, 2, 3, 5], [0.9798711225213051, -4.02406304746538e-17, -0.19963111793715238]),
                (&[1, 2, 4, 5], [-0.9865294523907788, 1.538772676468377e-17, -0.1635837386953549]),
                (&[3, 4, 5], [0.0, -1.0, 0.0]),
            ]),
        ];
        for (scale, want) in cases {
            let xml = format!(r#"<mujoco><asset><mesh name="p" file="prism.obj" scale="{scale}"/></asset><worldbody><body><freejoint/><geom type="mesh" mesh="p"/></body></worldbody></mujoco>"#);
            let t = tree_from_mjcf(&xml, &|p: &str| (p == "prism.obj").then(|| PRISM.as_bytes().to_vec())).unwrap();
            let hull = &t.mesh_hulls["p"];
            assert_eq!(hull.polygons.len(), 5, "scale {scale}: the interior vertex is on no face and the three quads merge");
            for (verts, n) in want {
                let p = hull
                    .polygons
                    .iter()
                    .find(|p| {
                        let mut v = p.verts.clone();
                        v.sort_unstable();
                        v == verts
                    })
                    .unwrap_or_else(|| panic!("scale {scale}: no polygon on {verts:?}"));
                for k in 0..3 {
                    assert!((p.normal[k] - n[k]).abs() < 1e-9, "scale {scale}, polygon {verts:?}: normal {:?} vs MuJoCo {n:?}", p.normal);
                }
            }
        }
    }

    /// **What the mesh compiler stores, bit for bit against MuJoCo** — `mesh_pos`, `mesh_quat` and the `f32`
    /// `mesh_vert` — for the prism as is, mirrored (`mjCMesh::Process` swaps each face's winding first),
    /// mirrored with a non-uniform scale, moved by `refpos` and an unnormalised `refquat`, and mirrored AND
    /// rotated. Products are fused where clang fuses MuJoCo's compiler on arm64, `mjuu_normvec` leaves a norm
    /// within 1e-14 of one alone, and `mjuu_eig3`'s quarter turn is its literal `0.707106781186548`. Undoing
    /// any of these breaks a bit here: the winding swap, the literal, the threshold, the fused face centroid,
    /// CoM, cross product, `mjuu_mulmat`, and either rotation. The fused inertia accumulation does not change
    /// this prism, but it does change 346 of the corpus's 2,339 frames (`examples/menagerie_mesh_store`).
    /// Expected values are MuJoCo 3.13.0's, each case from its own file (MuJoCo caches a mesh asset by path,
    /// so reusing one file measures the cache).
    #[test]
    fn the_mesh_compiler_stores_mujocos_frame_and_vertices_bit_for_bit() {
        const PRISM: &str = "v 0 0 0\nv 0.3 0 0\nv 0.05 0.2 0\nv 0 0 0.12\nv 0.3 0 0.12\nv 0.05 0.2 0.12\nv 0.1 0.05 0.06\nf 1 3 2\nf 4 5 6\nf 1 2 5\nf 1 5 4\nf 2 3 6\nf 2 6 5\nf 3 1 4\nf 3 4 6\n";
        #[allow(clippy::type_complexity)]
        let cases: [(&str, [f64; 3], [f64; 4], [u32; 21]); 5] = [
            (r#"scale="1 1 1""#, [0.11666667088866235, 0.06666666766007741, 0.0599999986588955], [0.1435361039704115, 0.6923852878686818, -0.1435361039704115, 0.6923852878686818], [0xbd75c28f, 0x3ddc43b8, 0xbda4f77d, 0xbd75c28f, 0xbc3fd1b1, 0x3e476670, 0xbd75c28f, 0xbdc44982, 0xbde9d562, 0x3d75c28f, 0x3ddc43b8, 0xbda4f77d, 0x3d75c28f, 0xbc3fd1b1, 0x3e476670, 0x3d75c28f, 0xbdc44982, 0xbde9d562, 0xa3111111, 0x3cb38ebc, 0xbc0e02de]),
            (r#"scale="-1 1 1""#, [-0.11666667088866235, 0.06666666766007741, 0.0599999986588955], [-0.14353610397041147, 0.6923852878686818, 0.14353610397041147, 0.6923852878686818], [0xbd75c28f, 0x3ddc43b8, 0x3da4f77d, 0xbd75c28f, 0xbc3fd1b1, 0xbe476670, 0xbd75c28f, 0xbdc44982, 0x3de9d562, 0x3d75c28f, 0x3ddc43b8, 0x3da4f77d, 0x3d75c28f, 0xbc3fd1b1, 0xbe476670, 0x3d75c28f, 0xbdc44982, 0x3de9d562, 0xa2ddddde, 0x3cb38ebc, 0x3c0e02de]),
            (r#"scale="0.5 -2 1.5""#, [0.05833333544433117, -0.1333333353201548, 0.08999999798834324], [0.7054104385769813, 0.7054104385769813, -0.04895010874994723, -0.04895010874994723], [0xbd9c099b, 0xbdb851eb, 0xbdfdf2e3, 0x3d9437d8, 0xbdb851eb, 0xbe143088, 0x3b7a3868, 0xbdb851eb, 0x3e8994fd, 0xbd9c099b, 0x3db851eb, 0xbdfdf2e3, 0x3d9437d8, 0x3db851eb, 0xbe143088, 0x3b7a3868, 0x3db851eb, 0x3e8994fd, 0xbc52a811, 0x00000000, 0xbd0282a0]),
            (r#"refquat="1 -1 0.3 0" refpos="0.01 0.02 -0.03""#, [0.05824561799328291, -0.11473684199118726, 0.07140351099236919], [0.5233267598406292, 0.6080047538712383, 0.4345389905553699, 0.4094327629682389], [0x3d75c28a, 0xbddc43ba, 0xbda4f77d, 0x3d75c290, 0x3c3fd1a6, 0x3e476670, 0x3d75c293, 0x3dc44981, 0xbde9d562, 0xbd75c294, 0xbddc43b7, 0xbda4f77d, 0xbd75c28e, 0x3c3fd1bc, 0x3e476670, 0xbd75c28b, 0x3dc44983, 0xbde9d562, 0xb17e90f7, 0xbcb38ebc, 0xbc0e02de]),
            (r#"scale="1 1 -1" refquat="0.9 0.1 0.2 0.3""#, [0.10649123216668764, 0.010175436735153203, -0.10105263218283654], [0.30406589296972475, 0.7519565609609825, -0.38723438184744163, 0.4383546471471347], [0x3d75c28f, 0x3ddc43b8, 0xbda4f77d, 0x3d75c28f, 0xbc3fd1b1, 0x3e476670, 0x3d75c28f, 0xbdc44982, 0xbde9d562, 0xbd75c28f, 0x3ddc43b8, 0xbda4f77d, 0xbd75c28f, 0xbc3fd1b1, 0x3e476670, 0xbd75c28f, 0xbdc44982, 0xbde9d562, 0x29d97476, 0x3cb38ebc, 0xbc0e02de]),
        ];
        for (attrs, pos, quat, vert) in cases {
            let xml = format!(r#"<mujoco><asset><mesh name="p" file="prism.obj" {attrs}/></asset><worldbody><body><freejoint/><geom type="mesh" mesh="p"/></body></worldbody></mujoco>"#);
            let t = tree_from_mjcf(&xml, &|p: &str| (p == "prism.obj").then(|| PRISM.as_bytes().to_vec())).unwrap();
            let d = &t.mesh_props["p"];
            assert_eq!([d.com.x, d.com.y, d.com.z], pos, "{attrs}: mesh_pos");
            assert_eq!(d.quat, quat, "{attrs}: mesh_quat");
            let got: Vec<u32> = t.mesh_hulls["p"].verts.iter().flat_map(|v| [v.x as f32, v.y as f32, v.z as f32]).map(f32::to_bits).collect();
            assert_eq!(got, vert, "{attrs}: mesh_vert");
        }
    }

    /// **Body inertia and element frames exactly as MuJoCo's compiler leaves them.** A capsule so small that
    /// `mjuu_eig3`'s absolute 1e-12 would stop a second diagonalisation at once (a single geom is COPIED:
    /// its frame and principal inertia become the body's); five geoms weighed together, oriented by `xyaxes`,
    /// `axisangle`, `zaxis` and `euler` (under `angle="degree"`, `eulerseq="zyx"`), with a stated density and
    /// a stated mass; a `fullinertia`; and two geoms within 1e-6 of their body's inertial frame and of
    /// the body frame, which MuJoCo's kinematics place EXACTLY there (`geom_sameframe`). Mass, `body_inertia`,
    /// `body_iquat` and `body_ipos` are compared bit for bit (the capsule's quaternion carries a `-0.0`).
    ///
    /// Undone, each of these fails here: the single-geom copy, `x / 180.0 * π`, the sameframe snap, the
    /// fused parallel-axis term and centre-of-mass sum, the Euler composition order, and a `fullinertia`'s
    /// eigenframe replacing the stated orientation. The fused `mjuu_globalinertia` is pinned by the corpus
    /// instead (unfused, 58 of Menagerie's 4,626 bodies change). ⚠ The fused `xyaxes` dot product and
    /// orthogonalisation and `axisangle`'s `mjuu_normvec` are ported but pinned by nothing: neither this
    /// model nor any Menagerie body moves a bit when they are unfused.
    #[test]
    fn body_inertia_and_frames_match_mujocos_compiler_bit_for_bit() {
        const MODEL: &str = r#"<mujoco>
  <compiler angle="degree" eulerseq="zyx"/>
  <worldbody>
    <body name="tiny" pos="0.1 0 1">
      <joint type="hinge" axis="0.3 1 0.2"/>
      <geom type="capsule" size="0.0005" fromto="0.0003 -0.0011 0.0002 -0.0009 0.0021 0.0013"/>
    </body>
    <body name="pair" pos="-0.2 0.1 1" euler="10 -20 35">
      <joint type="ball"/>
      <geom type="box" size="0.03 0.05 0.07" pos="0.01 0.02 -0.03" xyaxes="0.31 0.97 0.13 -0.83 0.29 0.41"/>
      <geom type="sphere" size="0.04" pos="-0.05 0.01 0.02" density="700"/>
      <geom type="cylinder" size="0.02 0.06" pos="0.02 -0.04 0.05" axisangle="1.3 2.1 3.7 47" mass="0.3"/>
      <geom type="ellipsoid" size="0.02 0.03 0.015" zaxis="0.2 -0.5 0.8"/>
      <geom type="capsule" size="0.01 0.03" pos="0.03 0.03 0.03" euler="17 -41 73"/>
    </body>
    <body name="full" pos="0 0.4 1">
      <joint type="slide" axis="0 0 1"/>
      <inertial pos="0.01 0.02 0.03" mass="1.7" fullinertia="0.021 0.017 0.013 0.0013 -0.0021 0.0009"/>
      <geom type="box" size="0.05 0.05 0.05" contype="0" conaffinity="0"/>
    </body>
    <body name="snap" pos="0 -0.4 1">
      <joint type="hinge" axis="1 0 0"/>
      <inertial pos="0.02 0.03 0.04" quat="0.9 0.1 0.2 0.3" mass="0.5" diaginertia="0.001 0.002 0.0025"/>
      <geom name="near_inertial" type="box" size="0.01 0.02 0.03" pos="0.0200004 0.0299996 0.0400003" quat="0.9 0.1 0.2 0.3" contype="0" conaffinity="0"/>
      <geom name="near_body" type="sphere" size="0.01" pos="0.0000006 -0.0000004 0.0000002" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"#;
        let want: [(&str, [f64; 11]); 4] = [
            ("tiny", [3.343386009704378e-06, 5.297288492289184e-12, 5.297288492289184e-12, 4.0483328182308983e-13, 0.5889040235618007, 0.7567440281846111, 0.28377901056922916, -0.0, -0.00030000000000000003, 0.0004999999999999999, 0.00075]),
            ("pair", [1.3883952591438333, 0.0054157787576827035, 0.004444959626535254, 0.002163181855674452, 0.7535891875991336, 0.23131982243392565, -0.0790689882846118, 0.610198796438066, 0.004111408683855843, 0.005306650499568431, -0.004145572778944047]),
            ("full", [1.7, 0.02175572069593266, 0.017060856629607813, 0.012183422674459531, 0.9787378415042712, 0.13658524460465873, 0.1186397093384105, 0.09665054543013765, 0.01, 0.02, 0.03]),
            ("snap", [0.5, 0.001, 0.002, 0.0025, 0.9233805168766387, 0.10259783520851541, 0.20519567041703082, 0.3077935056255462, 0.02, 0.03, 0.04]),
        ];
        let t = tree_from_mjcf_str(MODEL).unwrap();
        for (body, w) in want {
            let (i, q, p) = (t.body_iinertia[body], t.body_iquat_wxyz[body], t.body_ipos[body]);
            let got = [t.body_mass[body], i[0], i[1], i[2], q[0], q[1], q[2], q[3], p[0], p[1], p[2]];
            let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&w), "{body}: {got:?}\n   MuJoCo {w:?}");
        }
        // geom_xpos at qpos0: the near-inertial box sits ON xipos, the near-body sphere ON xpos
        let frames = crate::tree_frames(&t.tree, &vec![0.0; t.tree.joints.len()]);
        for (name, xpos) in [("near_inertial", [0.02, -0.37, 1.04]), ("near_body", [0.0, -0.4, 1.0])] {
            let g = t.geoms.iter().find(|g| g.name == name).unwrap();
            let world = g.joint.map_or(g.pose, |j| frames[j] * g.pose);
            let off = (world.translation.vector - Vector3::from(xpos)).norm();
            assert!(off < 1e-12, "{name}: {:?} vs MuJoCo {xpos:?} ({off:.1e} off)", world.translation.vector);
        }
    }

    /// **`mj_kinematics`, bit for bit** ([`MjcfTree::geom_frames_mujoco`]): a free base whose `qpos`
    /// quaternion is not unit length, a ball joint at an offset, a hinge with a `ref`, a slide with a `ref`,
    /// and frames stated every way — `euler` (the pair of angles `sin`/`cos` must come from ONE
    /// `__sincos_stret` call to match), an unnormalised `quat`, `fromto`, `axisangle`, `xyaxes`, `zaxis`, and
    /// a class `euler` that beats the element's own `quat`. Expected: MuJoCo 3.13.0's `geom_xpos`,
    /// `geom_xmat` after `mj_kinematics` at this `qpos`.
    #[test]
    fn geom_frames_match_mujocos_kinematics_bit_for_bit() {
        const MODEL: &str = r#"<mujoco>
  <default>
    <default class="tilted">
      <geom euler="0.3 1.57079632679 -0.2"/>
    </default>
  </default>
  <worldbody>
    <body name="base" pos="0.1 0.2 0.8">
      <freejoint/>
      <geom type="box" size="0.1 0.05 0.02" pos="0.01 -0.02 0.03" euler="-1.397059993 3.139694278 -0.001741181"/>
      <body name="arm" pos="0.1234567 -0.0321987 0.0543219" euler="-1.397059993 3.139694278 -0.001741181">
        <joint name="shoulder" type="ball" pos="0.01 0.02 -0.01"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0.05 -0.03"/>
        <body name="fore" pos="0.2 0.05 -0.03" axisangle="0.3 -1 0.2 0.7">
          <joint name="elbow" type="hinge" axis="0.2 0.9 -0.1" pos="0.005 0 0" ref="0.4"/>
          <geom class="tilted" type="box" size="0.02 0.03 0.08" quat="1 0 1 0"/>
          <body name="tip" pos="0.0123457 -0.0234568 0.16789" xyaxes="0.31 0.97 0.13 -0.83 0.29 0.41">
            <joint name="slide" type="slide" axis="1 1 0.3" ref="0.02"/>
            <geom type="sphere" size="0.015" zaxis="0.2 -0.5 0.8"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>"#;
        let t = tree_from_mjcf_str(MODEL).unwrap();
        let qpos = [0.3, -0.2, 1.1, 0.9, 0.2, -0.3, 0.5, 0.7, 0.1, -0.4, 0.3, 1.3, 0.11];
        let want: [[f64; 12]; 4] = [
            [0.31285714285714283, -0.22033613445378153, 1.1286554621848741, 0.4447426323016755, -0.8499085837596758, -0.28259403791481397, 0.6841457508697124, 0.5259956262170982, -0.5052456756543302, 0.578055864584293, 0.031368781489010955, 0.8153940255898375],
            [0.3085959510102222, -0.13392457947568095, 1.2925641574957125, 0.0666397393551405, -0.7477583884274337, 0.6606182995308918, -0.9487862447474189, -0.25240505636206584, -0.18999039265435044, 0.30881030895481915, -0.6141246453768733, -0.726283080502334],
            [0.23830341655439918, -0.11875122357818646, 1.371547304660976, -0.16836836064385963, -0.9678677492365428, -0.1867728971823524, 0.9674924599959009, -0.1259869958231279, -0.21928432852017726, 0.18870727327048392, -0.21762191266334718, 0.9576169736087426],
            [0.15693883366716896, -0.19463005451941134, 1.5655321866442131, -0.9742445822748731, -0.17989415059448038, -0.13596171700133045, 0.044693908937482305, -0.7450394066341359, 0.6655214024102788, -0.22102024434867382, 0.6423039600875481, 0.733890778279648],
        ];
        let got = t.geom_frames_mujoco(&qpos);
        for (g, ((p, m), w)) in got.iter().zip(&want).enumerate() {
            let ours = [p[0], p[1], p[2], m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(2, 0)], m[(2, 1)], m[(2, 2)]];
            assert_eq!(ours.map(f64::to_bits), w.map(f64::to_bits), "geom {g}: {ours:?}\n  MuJoCo {w:?}");
        }
    }

    /// **The `mujoco.pid` actuator plugin, against MuJoCo's `mj_step`.** One PID with all its parts (an
    /// integral that reaches its `imax` clamp, a slew limiter, a force pinned at its `forcerange`) and one
    /// bare PD, fifteen steps from `t = 0`
    /// under each integrator. The slew limiter is idle at `t = 0` and live after it — and inside RK4's
    /// stages, which run at `t + h/2` — so the step's `time` is carried from one call to the next.
    #[test]
    fn a_pid_plugin_steps_like_mujoco() {
        type Want = ([f64; 2], [f64; 2], [f64; 2]);
        let cases: [(&str, Want, Want); 3] = [
            ("Euler", ([0.1965807539017862, -0.3592294124671274], [-0.6838492196427609, 8.154117506574524], [0.053500000000000006, 0.9]), ([0.22301724345770013, 0.5825004217957128], [4.100974848278261, 2.127321181409043], [0.08, 0.9])),
            ("implicitfast", ([0.1965807539017862, -0.3592294124671274], [-0.6838492196427609, 8.154117506574524], [0.053500000000000006, 0.9]), ([0.22301724345770013, 0.5825004217957128], [4.100974848278261, 2.127321181409043], [0.08, 0.9])),
            ("RK4", ([0.19968409888977318, -0.3815842647163792], [-0.5912666901467641, 7.400497689910254], [0.05155670829153824, 0.4083333333333333]), ([0.20047795416661005, 0.6010920520467117], [3.576285876095827, 3.5372361500089604], [0.07416701628118336, 0.5483333333333335])),
        ];
        let ctrl = [0.9, 0.6];
        for (integrator, one, fifteen) in cases {
            let t = tree_from_mjcf_str(&PID_MODEL.replace("INTEGRATOR", integrator)).unwrap();
            assert!(t.actuators_unsupported.is_empty(), "{:?}", t.actuators_unsupported);
            assert_eq!((t.na(), t.act_adr()), (2, vec![(0, 2), (2, 0)]));
            let (mut qpos, mut qvel, mut act, mut time) = (vec![0.2, -0.4], vec![0.5, -0.3], vec![0.05, 0.3], 0.0);
            for k in 1..=15 {
                let st = t.step_mujoco_at(&qpos, &qvel, &ctrl, &act, time).unwrap();
                (qpos, qvel, act, time) = (st.qpos, st.qvel, st.act, st.time);
                let want = match k {
                    1 => one,
                    15 => fifteen,
                    _ => continue,
                };
                for (what, got, w) in [("qpos", &qpos, want.0), ("qvel", &qvel, want.1), ("act", &act, want.2)] {
                    for (i, (g, w)) in got.iter().zip(w).enumerate() {
                        assert!((g - w).abs() < 1e-10 * w.abs().max(1.0), "{integrator} step {k}: {what}[{i}] {g} vs MuJoCo {w}");
                    }
                }
            }
            assert!((time - 0.075).abs() < 1e-15);
        }
    }

    const PID_MODEL: &str = r#"<mujoco><option integrator="INTEGRATOR" timestep="0.005"/>
<extension>
  <plugin plugin="mujoco.pid">
    <instance name="full"><config key="kp" value="3.0"/><config key="ki" value="5.0"/><config key="kd" value="0.2"/><config key="imax" value="0.4"/><config key="slewmax" value="2.0"/></instance>
    <instance name="pd"><config key="kp" value="4.0"/><config key="kd" value="0.1"/></instance>
  </plugin>
</extension>
<worldbody>
  <body name="a" pos="0 0 1">
    <joint name="j1" type="hinge" axis="0 1 0" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.8" contype="0" conaffinity="0"/>
    <body name="b" pos="0.3 0 0">
      <joint name="j2" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3" contype="0" conaffinity="0"/>
    </body>
  </body>
</worldbody>
<actuator>
  <plugin name="p1" plugin="mujoco.pid" instance="full" joint="j1" ctrlrange="-1 1" forcerange="-1.5 1.5" actdim="2"/>
  <plugin name="p2" plugin="mujoco.pid" instance="pd" joint="j2" actdim="0"/>
</actuator></mujoco>"#;

    /// **`integrator="implicit"`, against MuJoCo's `mj_step`.** A free base (tilted) with a hinge chain ending
    /// in a ball, a side hinge, dof and tendon damping, a position servo with `kv`, a velocity servo and a
    /// motor, all moving fast enough that the Coriolis derivative matters — twenty steps, into contact
    /// with the floor. `implicit` adds `−∂qfrc_bias/∂qvel` to what `implicitfast` carries, in the full
    /// (ancestor-masked) matrix, and solves it by LU.
    #[test]
    fn step_mujoco_follows_mujocos_implicit_trajectory() {
        let t = tree_from_mjcf_str(IMPLICIT_MODEL).unwrap();
        assert_eq!(t.integrator, MjcfIntegrator::Implicit);
        let (mut qpos, mut qvel) = (
            vec![0.0, 0.0, 0.5, 0.9515485246437885, 0.03813457647485015, 0.18930785741199999, -0.2392983377447303, 0.3, -0.4, 1.0, 0.0, 0.0, 0.0, 0.2],
            (0..12).map(|i| -1.5 + 3.5 * i as f64 / 11.0).collect::<Vec<f64>>(),
        );
        for _ in 0..20 {
            let st = t.step_mujoco(&qpos, &qvel, &[0.5, -0.3, 0.2], &[]).unwrap();
            (qpos, qvel) = (st.qpos, st.qvel);
        }
        let want_qpos = [-0.3405147484973204, -0.19650511248718286, 0.16797693728332794, 0.9832630306029653, -0.13005171029432228, 0.01749333719522763, -0.12638966909034788, 0.43942769192544673, -0.48655390988463093, 0.935787762721399, 0.14698639953731069, -0.19668091904177035, 0.25300766307924444, 0.7141884821951416];
        let want_qvel = [-1.85441665084804, -0.8288573874391787, -2.4242877047945446, -2.369768681674996, -3.673276855448845, 1.241405801723756, 0.01996409374927284, -0.008407786848053411, 4.786566164547825, 0.44758485686847677, -3.4805910730411433, 3.295366847110663];
        for (k, (g, w)) in qpos.iter().zip(&want_qpos).chain(qvel.iter().zip(&want_qvel)).enumerate() {
            assert!((g - w).abs() < 1e-8 * w.abs().max(1.0), "[{k}] {g} vs MuJoCo {w}");
        }
    }

    const IMPLICIT_MODEL: &str = r#"<mujoco><option integrator="implicit" timestep="0.01"/><worldbody>
  <geom type="plane" size="2 2 0.1"/>
  <body name="base" pos="0 0 0.5" euler="10 20 -30">
    <freejoint/>
    <geom type="box" size="0.12 0.08 0.05" mass="2"/>
    <body name="l1" pos="0.12 0 0">
      <joint name="a" type="hinge" axis="0 1 0" damping="0.3"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" mass="0.5"/>
      <body name="l2" pos="0.25 0 0">
        <joint name="b" type="hinge" axis="0 0 1" damping="0.1"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3"/>
        <body name="l3" pos="0.2 0 0">
          <joint name="c" type="ball" damping="0.05"/>
          <geom type="box" size="0.04 0.03 0.02" pos="0.04 0 0" mass="0.2"/>
        </body>
      </body>
    </body>
    <body name="side" pos="-0.12 0 0">
      <joint name="d" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 -0.2 0 0" size="0.02" mass="0.4"/>
    </body>
  </body>
</worldbody>
<tendon><fixed name="t" damping="0.2"><joint joint="a" coef="1"/><joint joint="d" coef="0.5"/></fixed></tendon>
<actuator><position joint="a" kp="20" kv="2"/><velocity joint="b" kv="1.5"/><motor joint="d"/></actuator></mujoco>"#;

    /// **`<equality tendon>`, against MuJoCo.** A fixed tendon coupled by a quadratic to a SPATIAL one (each
    /// length measured from its `tendon_length0`), with an impedance wide enough to read the residual, and a
    /// single fixed tendon held at an offset. MuJoCo's `qacc`, then ten `mj_step`s.
    #[test]
    fn tendon_equalities_match_mujoco() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><worldbody>
  <site name="s0" pos="0 0.1 1.2"/>
  <body pos="0 0 1">
    <joint name="a" type="hinge" axis="0 1 0" damping="0.1"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.6"/>
    <site name="s1" pos="0.2 0 0.05"/>
    <body pos="0.3 0 0">
      <joint name="b" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025" mass="0.4"/>
      <body pos="0.25 0 0">
        <joint name="c" type="hinge" axis="0 0 1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02" mass="0.2"/>
        <site name="s2" pos="0.15 0 0"/>
      </body>
    </body>
  </body>
</worldbody>
<tendon>
  <fixed name="t1"><joint joint="a" coef="1"/><joint joint="b" coef="-0.5"/></fixed>
  <spatial name="t2"><site site="s0"/><site site="s1"/><site site="s2"/></spatial>
  <fixed name="t3"><joint joint="c" coef="2"/></fixed>
</tendon>
<equality>
  <tendon tendon1="t1" tendon2="t2" polycoef="0.01 1.5 0.3 0 0" solimp="0.8 0.95 0.2 0.5 2"/>
  <tendon tendon1="t3" polycoef="0.05 0 0 0 0"/>
</equality></mujoco>"#,
        )
        .unwrap();
        assert!(t.equalities_unsupported.is_empty(), "{:?}", t.equalities_unsupported);
        let (qpos, qvel) = ([0.3, -0.4, 0.2], [0.5, -0.7, 1.1]);
        let f = t.forward_mujoco(&qpos, &qvel, &[], &[]).unwrap();
        assert_eq!(f.nefc, 2);
        for (g, w) in f.qacc.iter().zip([-519.8919287145774, 1317.997475738774, -552.0591726357393]) {
            assert!((g - w).abs() < 1e-9 * w.abs().max(1.0), "qacc {g} vs MuJoCo {w}");
        }
        let (mut qp, mut qv) = (qpos.to_vec(), qvel.to_vec());
        for _ in 0..10 {
            let st = t.step_mujoco(&qp, &qv, &[], &[]).unwrap();
            (qp, qv) = (st.qpos, st.qvel);
        }
        let want = [0.2487084794027434, -0.2531629138270305, 0.15446362859124024, -3.3791463501446177, 9.789883711854351, -3.2969856210514323];
        for (k, (g, w)) in qp.iter().chain(&qv).zip(want).enumerate() {
            assert!((g - w).abs() < 1e-9 * w.abs().max(1.0), "after 10 steps [{k}] {g} vs MuJoCo {w}");
        }
    }

    /// **`<weld>` in every form, against MuJoCo.** An arm's forearm welded to a free box at their `qpos0`
    /// pose (the relative pose computed by `mj_setConst`), a free ball welded to the world with a STATED
    /// `relpose`, `anchor` and `torquescale` (and an impedance wide enough not to saturate, so it is the norm of
    /// all SIX residuals that sets it), two sites welded (the site form, no relative pose), and a free
    /// plate welded to that ball — all moving, all violated, so every row's residual, Jacobian and `J̇·v`
    /// is at work. MuJoCo's `qacc`, then ten `mj_step`s.
    #[test]
    fn welds_match_mujoco_in_every_form() {
        let t = tree_from_mjcf_str(WELD_MODEL).unwrap();
        assert!(t.equalities_unsupported.is_empty(), "{:?}", t.equalities_unsupported);
        let qpos = [0.3, -0.4, 0.36, 0.05, 1.02, 0.988851329954785, 0.03176851580115925, -0.07211804743841453, 0.12634395863718664, -0.3, 0.2, 0.8, 1.0, 0.0, 0.0, 0.0, 0.1, -0.3, 0.9, 1.0, 0.0, 0.0, 0.0];
        let qvel: Vec<f64> = (0..20).map(|i| -0.5 + 1.1 * i as f64 / 19.0).collect();
        let want_qacc = [-243.5613129785541, 175.19689224779685, -64.21618664252793, -59.891339299056646, -192.99830610320288, 231.44795796803703, -186.06991261861762, -429.69126965063094, 181.3729527397196, 17.70717444609069, -438.2736547531831, -1281.6237549776133, 245.5629362236282, -360.1148279329517, 14.872050535256632, -15.386382187592138, -10.669628658440594, -825.2298687242633, -249.2533065366348, -321.4013647256903];
        let f = t.forward_mujoco(&t.q_from_qpos(&qpos, &t.qposadr()).unwrap(), &qvel, &[], &[]).unwrap();
        assert_eq!(f.nefc, 24, "four welds, six rows each");
        for (k, (g, w)) in f.qacc.iter().zip(&want_qacc).enumerate() {
            assert!((g - w).abs() < 1e-8 * w.abs().max(1.0), "qacc[{k}] {g} vs MuJoCo {w}");
        }
        // the arm's force/torque site: its subtree carries a forearm and a site, both WELDED to the box, so
        // the weld forces (and the weld torque) are external to it and have to be booked there
        let sd = t.sensordata_with(&qpos, &qvel, &[], &[], 0.0, Some(&f)).unwrap();
        let want_ft = [-4.09190892897531, -19.113241571051187, -9.411669965971562, 1.3232670554206547, 1.5076594245261965, -6.269422499047928];
        let got_ft: Vec<f64> = sd.iter().flat_map(|v| v.clone().unwrap()).collect();
        for (k, (g, w)) in got_ft.iter().zip(&want_ft).enumerate() {
            assert!((g - w).abs() < 1e-8 * w.abs().max(1.0), "force/torque[{k}] {g} vs MuJoCo {w}");
        }
        let want_qpos = [0.268921680993056, -0.3851630903566696, 0.3449304137451297, 0.03728200373013553, 0.9909312312370034, 0.9906027883598434, 0.04775811477844495, -0.07956378428834306, 0.10047329195477585, -0.27207424879023845, 0.20145983923463553, 0.7307455503773158, 0.9972164995524224, -0.07059219092423387, 0.012840217004554833, -0.02027620349586223, 0.10836651729053051, -0.29477339048667645, 0.9060866336088069, 0.9976170169658066, -0.06581562634075785, -0.0064668503550642485, -0.019666485097860672];
        let want_qvel = [-1.427933147837055, 1.2834241322394402, -0.8391306861360185, -0.6684397481327558, -1.8213811346601152, 1.9057364020673693, -0.9693224013008912, -3.454641029203958, 2.1915248314252276, -0.045835635270237804, -5.356642261068582, -8.384663628116838, 1.5450426077691637, -2.9785118593934636, 0.43524506664423257, 0.2596229476985236, 0.19062666510784068, -10.166319285734494, -1.244162978646895, -3.4086564730112263];
        let (mut qp, mut qv) = (qpos.to_vec(), qvel.clone());
        for _ in 0..10 {
            let st = t.step_mujoco(&qp, &qv, &[], &[]).unwrap();
            (qp, qv) = (st.qpos, st.qvel);
        }
        for (k, (g, w)) in qp.iter().zip(&want_qpos).chain(qv.iter().zip(&want_qvel)).enumerate() {
            assert!((g - w).abs() < 1e-8 * w.abs().max(1.0), "after 10 steps [{k}] {g} vs MuJoCo {w}");
        }
    }

    /// **Applied forces through [`MjcfData`], against MuJoCo.** `qfrc_applied` on every dof (a hinge chain, a
    /// ball, a free cart and its pole) and `xfrc_applied` on four bodies — a forearm whose centre of mass is
    /// off its joint, a ball-jointed wrist, the cart (in contact with the floor) and its pole, whose force is
    /// inside the pole's force/torque sensor's subtree and has to be booked as external there. At a nonzero
    /// `time`, with control. MuJoCo's `qacc` and `sensordata` after `mj_forward`, then ten `mj_step`s — with
    /// Euler and with RK4 — and a step that trips `mj_checkAcc`'s reset.
    #[test]
    fn applied_forces_match_mujoco() {
        let t = tree_from_mjcf_str(APPLIED_MODEL).unwrap();
        let mut d = t.make_data();
        assert_eq!((d.qpos.len(), d.qvel.len(), d.ctrl.len()), (14, 12, 2));
        d.qpos = vec![0.3, -0.4, 0.9535826651341416, 0.10037712264569912, -0.20075424529139824, 0.20075424529139824, 0.5, 0.3, 0.048, 0.9846387287763668, 0.02009466793421157, -0.030142001901317352, 0.17080467744079836, 0.25];
        d.qvel = (0..12).map(|i| -0.5 + 1.1 * i as f64 / 11.0).collect();
        d.ctrl = vec![0.7, -0.3];
        d.time = 0.4;
        d.applied.qfrc = vec![0.3, -0.33, 0.0036, -0.0039000000000000003, 0.0042, -0.44999999999999996, 0.48, -0.51, 0.54, -0.57, 0.6, -0.63];
        for (b, w) in [("forearm", [0.5, -1.2, 2.0, 0.1, -0.3, 0.05]), ("pole", [-0.8, 0.4, 1.5, 0.02, 0.07, -0.04]), ("cart", [2.0, 1.0, 3.0, -0.2, 0.1, 0.3]), ("wrist", [0.1, 0.2, -0.3, 0.001, 0.002, 0.003])] {
            d.applied.xfrc.insert(b.into(), w);
        }
        let start = d.clone();
        let close = |g: f64, w: f64| (g - w).abs() < 1e-9 * w.abs().max(1.0);
        let f = t.forward(&d).unwrap();
        assert_eq!(f.nefc, 8, "the cart's two floor contacts, four pyramid rows each");
        let want_qacc = [34.61785414447881, -58.180860886726585, 147.72327329326586, 13.620987249747657, 277.8590548108485, -4.8285227520966085, -5.166179969997382, 3.9577985038036396, 139.0575069196684, 73.07430349407136, 88.70336443197453, -254.85636972066575];
        for (k, (g, w)) in f.qacc.iter().zip(want_qacc).enumerate() {
            assert!(close(*g, w), "qacc[{k}] {g} vs MuJoCo {w}");
        }
        let want_sd = [3.91058863544872, 2.6782742110655082, 1.527506089513932, -0.7960862894467228, 0.8268481433609536, 0.060056668908081694, 2.9616140360354546, 1.2141548091008696, 14.298893079021887];
        let sd: Vec<f64> = t.sensors_of(&d).unwrap().into_iter().flat_map(Option::unwrap).collect();
        for (k, (g, w)) in sd.iter().zip(want_sd).enumerate() {
            assert!(close(*g, w), "sensordata[{k}] {g} vs MuJoCo {w}");
        }
        for _ in 0..10 {
            t.step(&mut d).unwrap();
        }
        assert!(close(d.time, 0.42000000000000004), "time {}", d.time);
        let want_qpos = [0.2975782145810774, -0.42073104093243013, 0.9457973005163347, 0.10750620517908743, -0.20098973631975023, 0.23132878782832683, 0.4991394675828127, 0.30106582543953003, 0.05272961681775667, 0.9815655986301296, 0.03980880238877929, -0.015625728266028784, 0.18627955189553497, 0.1967821756686385];
        let want_qvel = [0.18716491140324654, -1.5497901169886688, 2.7956726492082216, 0.18734933037690799, 5.361859000949678, -0.07107857783953643, 0.02022177255762923, 0.2626206661210189, 4.089138211485138, 1.6543906130245478, 2.3225081046119898, -5.614102286188027];
        for (k, (g, w)) in d.qpos.iter().zip(want_qpos).chain(d.qvel.iter().zip(want_qvel)).enumerate() {
            assert!(close(*g, w), "after 10 steps [{k}] {g} vs MuJoCo {w}");
        }
        // RK4 evaluates the applied forces at every stage
        let rk4 = tree_from_mjcf_str(&APPLIED_MODEL.replace(r#"<option timestep="0.002"/>"#, r#"<option timestep="0.002" integrator="RK4"/>"#)).unwrap();
        let mut r = rk4.make_data();
        (r.qpos, r.qvel, r.ctrl, r.time) = (start.qpos.clone(), start.qvel.clone(), start.ctrl.clone(), start.time);
        r.applied = start.applied.clone();
        for _ in 0..10 {
            rk4.step(&mut r).unwrap();
        }
        let want_qpos = [0.2968954302796029, -0.41958008346219877, 0.9465541060335043, 0.10664808644757628, -0.2012173283738647, 0.22841430946914915, 0.4992263435105761, 0.3011600536235026, 0.05265921763560146, 0.9817872351811727, 0.038341486236605334, -0.016553927448475756, 0.18533678197947512, 0.2024472724924038];
        let want_qvel = [0.1875143558510835, -1.5491549568263236, 2.8008445085791993, 0.19610932672892872, 5.357547700222157, -0.07002357393198831, 0.02077607459101893, 0.2623904515304731, 4.115106407089137, 1.6555435459934438, 2.3239965047591666, -5.639620847689321];
        for (k, (g, w)) in r.qpos.iter().zip(want_qpos).chain(r.qvel.iter().zip(want_qvel)).enumerate() {
            assert!(close(*g, w), "after 10 RK4 steps [{k}] {g} vs MuJoCo {w}");
        }
        // a torque that sends `qacc` past `mjMAXVAL`: `mj_checkAcc` resets the data, which zeroes the control
        // and the applied forces too, and the step then runs from the reset state
        let mut d = start;
        d.applied.xfrc.insert("wrist".into(), [0.0, 0.0, 0.0, 1e9, 0.0, 0.0]);
        assert!(t.step(&mut d).unwrap().reset.is_some());
        assert_eq!((d.ctrl.as_slice(), &d.applied), (&[0.0, 0.0][..], &MjcfApplied::default()));
        let want_qpos = [0.00012147139779340167, 5.159596715055786e-21, 0.9999999767253738, 2.4154066618325593e-19, -0.00021575275657405364, 2.451964668848554e-20, 0.5, 0.3, 0.04996076, 0.984807753012208, 1.5294600818987096e-21, 2.69685078355321e-22, 0.17364817766693033, 3.858664933547006e-21];
        let want_qvel = [0.060735698896700836, 2.579798357527893e-18, 2.4154066805717887e-16, -0.21575275824790857, 2.4519646878714075e-17, 6.1044065919280735e-21, 6.170573241758271e-21, -0.019620000000000002, -1.488190197243287e-18, 4.440892098500626e-19, 7.272835269408394e-21, 1.9293324667735027e-18];
        assert!(close(d.time, 0.002), "time after the reset step {}", d.time);
        for (k, (g, w)) in d.qpos.iter().zip(want_qpos).chain(d.qvel.iter().zip(want_qvel)).enumerate() {
            assert!(close(*g, w), "after the reset step [{k}] {g} vs MuJoCo {w}");
        }
    }

    const APPLIED_MODEL: &str = r#"<mujoco><option timestep="0.002"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="arm" pos="0 0 1.2">
    <joint name="j1" type="hinge" axis="0 1 0" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.8" contype="0" conaffinity="0"/>
    <body name="forearm" pos="0.3 0 0">
      <joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="box" pos="0.1 0.02 0" size="0.1 0.03 0.02" mass="0.3" contype="0" conaffinity="0"/>
      <body name="wrist" pos="0.2 0 0">
        <joint name="j3" type="ball"/>
        <geom type="box" pos="0.05 0 0.01" size="0.05 0.02 0.01" mass="0.1" contype="0" conaffinity="0"/>
      </body>
    </body>
  </body>
  <body name="cart" pos="0.5 0.3 0.05" euler="0 0 20">
    <freejoint/>
    <geom type="box" size="0.1 0.06 0.05" mass="1.2"/>
    <body name="pole" pos="0.02 0 0.05">
      <joint name="p" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.02" mass="0.2" contype="0" conaffinity="0"/>
      <site name="pole_ft" pos="0 0 0.05"/>
    </body>
  </body>
</worldbody>
<actuator><motor joint="j1" gear="2"/><motor joint="p"/></actuator>
<sensor><force name="f" site="pole_ft"/><torque name="t" site="pole_ft"/><accelerometer name="acc" site="pole_ft"/></sensor></mujoco>"#;

    const WELD_MODEL: &str = r#"<mujoco><option timestep="0.002"/><worldbody>
  <body name="arm" pos="0 0 1">
    <joint name="j1" type="hinge" axis="0 1 0" damping="0.05"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.8" contype="0" conaffinity="0"/>
    <site name="arm_ft" pos="0.05 0 0"/>
    <site name="tool" pos="0.3 0 0" euler="0 20 10"/>
    <body name="forearm" pos="0.3 0 0">
      <joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3" contype="0" conaffinity="0"/>
    </body>
  </body>
  <body name="box" pos="0.35 0.05 1.02" euler="5 -10 15">
    <freejoint/>
    <geom type="box" size="0.05 0.04 0.03" mass="0.4" contype="0" conaffinity="0"/>
    <site name="grip" pos="0.01 0 0.02" euler="0 15 0"/>
  </body>
  <body name="ball" pos="-0.3 0.2 0.8">
    <freejoint/>
    <geom type="sphere" size="0.05" mass="0.2" contype="0" conaffinity="0"/>
  </body>
  <body name="plate" pos="0.1 -0.3 0.9">
    <freejoint/>
    <geom type="box" size="0.08 0.08 0.01" mass="0.3" contype="0" conaffinity="0"/>
  </body>
</worldbody>
<equality>
  <weld name="grasp" body1="forearm" body2="box" solref="0.02 1" solimp="0.9 0.95 0.001"/>
  <weld name="tether" body1="ball" relpose="0.1 0 -0.2 0.9 0.1 0.2 -0.1" anchor="0.02 0.01 0" torquescale="0.5" solimp="0.5 0.95 5 0.5 2"/>
  <weld name="sites" site1="tool" site2="grip" torquescale="2"/>
  <weld name="plate_to_ball" body1="plate" body2="ball" anchor="0 0 0.05"/>
</equality>
<sensor><force name="f" site="arm_ft"/><torque name="t" site="arm_ft"/></sensor></mujoco>"#;

    /// **Keyframes as `mjCKey::Compile` completes them, against MuJoCo.** A zero quaternion on a free base
    /// becomes the identity and a (0, 0, 3, 4) ball quaternion is normalised; a partial key takes `qpos0`
    /// (the hinge's `ref` in DEGREES, 0.3° here) and zeros for the rest; `<size nkey="3">` adds an unnamed
    /// reset key.
    #[test]
    fn keyframes_are_completed_the_way_mujoco_completes_them() {
        let t = tree_from_mjcf_str(r#"<mujoco><size nkey="3"/><worldbody><body name="b" pos="0 0 1"><freejoint/><geom size="0.1"/><body pos="0.2 0 0"><joint name="h" type="hinge" ref="0.3"/><geom size="0.05"/><body pos="0.1 0 0"><joint type="ball"/><geom size="0.02"/></body></body></body></worldbody>
<actuator><motor joint="h"/></actuator>
<keyframe><key name="zero_quat" time="1.5" qpos="0 0 1 0 0 0 0 0.7 0 0 3 4" ctrl="0.25"/><key name="partial" qvel="1 2 3 4 5 6 7 8 9 10"/></keyframe></mujoco>"#)
        .unwrap();
        let qpos0 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.005235987755982988, 1.0, 0.0, 0.0, 0.0];
        let want = [
            ("zero_quat", 1.5, vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.6, 0.8], vec![0.0; 10], vec![0.25]),
            ("partial", 0.0, qpos0.to_vec(), (1..=10).map(f64::from).collect(), vec![0.0]),
            ("", 0.0, qpos0.to_vec(), vec![0.0; 10], vec![0.0]),
        ];
        assert_eq!(t.keyframes.len(), 3);
        for (k, (name, time, qpos, qvel, ctrl)) in t.keyframes.iter().zip(want) {
            assert_eq!((k.name.as_str(), k.time), (name, time));
            for (a, b) in k.qpos.iter().zip(&qpos) {
                assert!((a - b).abs() < 1e-15, "'{name}' qpos {:?} vs MuJoCo {qpos:?}", k.qpos);
            }
            assert_eq!((&k.qvel, &k.ctrl), (&qvel, &ctrl), "'{name}'");
        }
        // `mj_resetDataKeyframe`: the key's state, and the applied forces cleared
        let mut d = t.make_data();
        d.applied.qfrc = vec![1.0; 10];
        t.reset_data_keyframe(&mut d, 0).unwrap();
        let k = &t.keyframes[0];
        assert_eq!((d.time, &d.qpos, &d.qvel, &d.act, &d.ctrl, &d.applied), (1.5, &k.qpos, &k.qvel, &k.act, &k.ctrl, &MjcfApplied::default()));
        assert!(t.reset_data_keyframe(&mut d, 3).is_err());
        // and a key whose qpos has the wrong length is refused, as MuJoCo refuses it
        assert!(tree_from_mjcf_str(r#"<mujoco><worldbody><body><joint type="hinge"/><geom size="0.1"/></body></worldbody><keyframe><key qpos="0 0"/></keyframe></mujoco>"#).is_err());
    }

    /// **A tendon limit on a SPINNING free body, against MuJoCo.** A limited spatial tendon from a site on a
    /// free body to a world site, stretched past its range with the body turning fast. MuJoCo carries `J̇·v`
    /// on connect and weld rows only; a limit row's reference must not pick up the Euler basis's `J·Ṫ·q̇`.
    #[test]
    fn a_tendon_limit_on_a_spinning_free_body_matches_mujoco() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><worldbody>
  <site name="anchor" pos="0 0 0.2"/>
  <body name="object" pos="0 0 0.15">
    <freejoint/>
    <geom type="box" size="0.015 0.02 0.01" mass="0.1" contype="0" conaffinity="0"/>
    <site name="hook" pos="0.015 0.015 0.015"/>
  </body>
</worldbody>
<tendon><spatial limited="true" range="0 0.02"><site site="hook"/><site site="anchor"/></spatial></tendon></mujoco>"#,
        )
        .unwrap();
        let qpos = [0.01, -0.02, 0.14, 0.9233805168766386, 0.2051956704170308, -0.3077935056255462, 0.1025978352085154];
        let qvel = [0.1, -0.2, 0.05, 2.0, -1.5, 3.0];
        let want = [-9.371005398606533, 12.476824330716129, 26.01401492381582, 2.6999999999983264, -1406.432459922388, 731.7048791596427];
        let f = t.forward_mujoco(&t.q_from_qpos(&qpos, &t.qposadr()).unwrap(), &qvel, &[], &[]).unwrap();
        assert_eq!(f.nefc, 1, "the limit is active");
        for (k, w) in want.iter().enumerate() {
            assert!((f.qacc[k] - w).abs() < 1e-9 * w.abs().max(1.0), "qacc[{k}] {} vs MuJoCo {w}", f.qacc[k]);
        }
    }

    /// **Through gimbal lock, against MuJoCo.** A free base that STARTS at a pitch of exactly 90° — where
    /// this port's Euler-rate basis is singular and a step used to refuse the state — with a hinge and a ball
    /// below it, the ball also at 90°, all turning. Twenty steps under each integrator, compared to MuJoCo's
    /// `mj_step`: the step re-charts ([`MjcfTree::charted`]) and MuJoCo's coordinates never notice.
    #[test]
    fn step_mujoco_goes_through_gimbal_lock_like_mujoco() {
        let s2 = 0.5f64.sqrt();
        let qpos0 = [0.1, -0.2, 1.0, s2, 0.0, s2, 0.0, 0.3, s2, 0.0, s2, 0.0];
        let qvel0 = [0.2, -0.1, 0.3, 0.3, -0.2, 0.5, -0.4, 0.2, 0.1, -0.3];
        let euler = (
            [0.10804389152599277, -0.20400533150524294, 1.0037106268004348, 0.7104691552668625, 0.01107271155054746, 0.7036350567098751, 0.002946428719789748, 0.2897345040312094, 0.7077805769911099, -0.0010178021614747686, 0.7064305112655508, -0.0012456588557764846],
            [0.2017194966296767, -0.10016500356322214, -0.0946796272682277, 0.28410230573061823, -0.2674104105858261, 0.4950443358779727, -0.16597216460081382, 1.0743437232637642e-05, -0.08065675153217357, -0.013717550126385319],
        );
        let rk4 = (
            [0.10804317070726957, -0.20400535321891186, 1.004105200470931, 0.7104568853373144, 0.011072016327269287, 0.7036474590184834, 0.0029458357187373777, 0.28960431037385725, 0.7077850172906829, -0.0010282339392657057, 0.7064260285365054, -0.0012562801975344068],
            [0.20174708241614164, -0.10016699878878345, -0.09469263035333003, 0.28394732154847185, -0.2684216479337131, 0.49496807397106923, -0.16277928025239366, 3.786330660829462e-05, -0.08221567265601067, -0.011215533340992323],
        );
        for (integrator, (want_qpos, want_qvel)) in [("Euler", euler), ("implicitfast", euler), ("RK4", rk4)] {
            let t = tree_from_mjcf_str(&GIMBAL_MODEL.replace("INTEGRATOR", integrator)).unwrap();
            // in the loaded chart the start IS the singularity, to rounding
            assert!(t.q_from_qpos(&qpos0, &t.qposadr()).unwrap()[4].cos().abs() < 1e-7, "{integrator}: the test starts at gimbal lock");
            let (mut qpos, mut qvel) = (qpos0.to_vec(), qvel0.to_vec());
            for _ in 0..20 {
                let st = t.step_mujoco(&qpos, &qvel, &[], &[]).unwrap();
                (qpos, qvel) = (st.qpos, st.qvel);
            }
            for (k, (g, w)) in qpos.iter().zip(&want_qpos).chain(qvel.iter().zip(&want_qvel)).enumerate() {
                assert!((g - w).abs() < 1e-10 * w.abs().max(1.0), "{integrator}: [{k}] {g} vs MuJoCo {w}");
            }
        }
    }

    const GIMBAL_MODEL: &str = r#"<mujoco><option integrator="INTEGRATOR"/><worldbody>
  <body name="base" pos="0 0 1">
    <freejoint/>
    <geom type="box" size="0.2 0.1 0.05" mass="3" contype="0" conaffinity="0"/>
    <body name="arm" pos="0.2 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.2"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.03" mass="0.4" contype="0" conaffinity="0"/>
      <body name="wrist" pos="0.25 0 0">
        <joint name="ball" type="ball" damping="0.05"/>
        <geom type="box" size="0.05 0.03 0.02" pos="0.05 0 0" mass="0.2" contype="0" conaffinity="0"/>
      </body>
    </body>
  </body>
</worldbody></mujoco>"#;

    /// **`qpos0` and the auto-reset, against MuJoCo.** A free base posed by `euler` (in degrees, the compiler
    /// default), a hinge with a `ref` (also degrees) and a ball. A NaN velocity trips `mj_checkVel` and a
    /// velocity of 9e9 — under the 1e10 limit, so it passes — makes `qacc` bad and trips `mj_checkAcc`; either
    /// way MuJoCo steps from `mj_resetData`'s state with the control ZEROED (the motor's 3.0 is gone).
    #[test]
    fn step_mujoco_resets_the_way_mj_step_does() {
        let t = tree_from_mjcf_str(RESET_MODEL).unwrap();
        let want_qpos0 = [0.1, -0.2, 0.6, 0.943714364147489, 0.12767944069578063, 0.14487812541736916, 0.2685358227515692, 0.006981317007977318, 1.0, 0.0, 0.0, 0.0];
        for (g, w) in t.qpos0().iter().zip(&want_qpos0) {
            assert!((g - w).abs() < 1e-15, "qpos0 {:?} vs MuJoCo {want_qpos0:?}", t.qpos0());
        }
        let want_qpos = [0.1, -0.2, 0.59996076, 0.943714364147489, 0.12767944069578063, 0.14487812541736916, 0.2685358227515692, 0.006981317007977318, 1.0, 0.0, 0.0, 0.0];
        let want_qvel = [0.0, 0.0, -0.019620000000000002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let mut qpos = want_qpos0.to_vec();
        qpos[7] = 0.3;
        for (qvel6, reset) in [(f64::NAN, MjcfReset::BadQvel(6)), (9e9, MjcfReset::BadQacc(0))] {
            let mut qvel = vec![0.0; 10];
            qvel[6] = qvel6;
            let st = t.step_mujoco(&qpos, &qvel, &[3.0], &[]).unwrap();
            match (st.reset, reset) {
                (Some(MjcfReset::BadQvel(i)), MjcfReset::BadQvel(j)) => assert_eq!(i, j),
                (Some(MjcfReset::BadQacc(_)), MjcfReset::BadQacc(_)) => {}
                other => panic!("reset {other:?}"),
            }
            for (g, w) in st.qpos.iter().zip(&want_qpos).chain(st.qvel.iter().zip(&want_qvel)) {
                assert!((g - w).abs() < 1e-12, "after the reset: qpos {:?} qvel {:?}", st.qpos, st.qvel);
            }
        }
        // and a state that is merely large is not reset
        let st = t.step_mujoco(&qpos, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], &[3.0], &[]).unwrap();
        assert_eq!(st.reset, None);
    }

    const RESET_MODEL: &str = r#"<mujoco><worldbody>
  <geom type="plane" size="2 2 0.1"/>
  <body name="base" pos="0.1 -0.2 0.6" euler="10 20 30">
    <freejoint/>
    <geom type="box" size="0.1 0.08 0.05" mass="2"/>
    <body name="link" pos="0.12 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" ref="0.4" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.3"/>
      <body name="tip" pos="0.2 0 0">
        <joint name="ball" type="ball"/>
        <geom type="sphere" size="0.03" mass="0.1"/>
      </body>
    </body>
  </body>
</worldbody>
<actuator><motor name="m" joint="hinge" gear="2"/></actuator></mujoco>"#;

    /// **Activations, against MuJoCo's `mj_step`.** One actuator of every law this port carries on a
    /// three-hinge arm: an integrator servo whose `actrange` clamp binds, a filtered motor, an `actearly`
    /// exact filter with an affine (velocity-dependent) gain, and a muscle. Ten steps under each integrator
    /// the file can ask for: `Euler` (with implicit dof damping), `implicitfast` (where the muscle gain's
    /// velocity derivative and the activation-as-input enter the modified mass matrix) and `RK4` (where the
    /// activations ride along as the last block of the state).
    #[test]
    fn step_mujoco_integrates_every_activation_law_like_mujoco() {
        let cases: [(&str, [f64; 3], [f64; 3], [f64; 4]); 3] = [
            ("Euler", [0.10938355310581413, -0.26232040273315615, 0.13019692942416222], [-2.356487648621272, 5.265759975936574, -4.762078101750861], [0.5, 0.90724665183, -0.6300119574461944, 0.7952200727116999]),
            ("implicitfast", [0.11307784334355178, -0.27450581665350465, 0.1350899542153339], [-2.6122162969576195, 5.942100227423588, -4.501561061158574], [0.5, 0.90724665183, -0.6300119574461944, 0.7952200727116999]),
            ("RK4", [0.11815451674406732, -0.28230838325995716, 0.13782354707999936], [-2.673430809497219, 6.043041690289047, -4.797248178380324], [0.5, 0.8746043834987527, -0.6040614148142172, 0.7910749661574509]),
        ];
        let ctrl = [1.5, 1.5, -0.8, 0.8];
        let near = |got: &[f64], want: &[f64], tol: f64, what: &str| {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!((g - w).abs() <= tol * w.abs().max(1.0), "{what}[{i}]: {g} vs MuJoCo {w}");
            }
        };
        for (integrator, want_qpos, want_qvel, want_act) in cases {
            let t = tree_from_mjcf_str(&ACT_MODEL.replace("INTEGRATOR", integrator)).unwrap();
            assert!(t.actuators_unsupported.is_empty(), "{:?}", t.actuators_unsupported);
            assert_eq!(t.na(), 4);
            let (mut qpos, mut qvel, mut act) = (vec![0.3, -0.7, 0.2], vec![0.4, -0.9, 1.3], vec![0.45, -0.2, 0.1, 0.6]);
            for _ in 0..10 {
                let st = t.step_mujoco(&qpos, &qvel, &ctrl, &act).unwrap();
                (qpos, qvel, act) = (st.qpos, st.qvel, st.act);
            }
            near(&qpos, &want_qpos, 1e-10, &format!("{integrator}: qpos"));
            near(&qvel, &want_qvel, 1e-10, &format!("{integrator}: qvel"));
            near(&act, &want_act, 1e-12, &format!("{integrator}: act"));
        }
    }

    /// **What MuJoCo refuses, and what it quietly leaves unlimited.** Each verdict is MuJoCo 3.13.0's on the
    /// same one-actuator model.
    #[test]
    fn activation_ranges_and_counts_compile_by_mujocos_rules() {
        let model = |a: &str| format!(r#"<mujoco><worldbody><body><joint name="j" type="hinge"/><geom size="0.1" mass="1"/></body></worldbody><actuator>{a}</actuator></mujoco>"#);
        for (a, why) in [
            (r#"<general joint="j" actrange="-1 1"/>"#, "actrange specified but dyntype is 'none'"),
            (r#"<general joint="j" dyntype="integrator" actdim="2"/>"#, "actdim > 1"),
            (r#"<general joint="j" actdim="1"/>"#, "invalid actdim 1"),
            (r#"<general joint="j" dyntype="filter" actdim="0"/>"#, "invalid actdim 0"),
            (r#"<general joint="j" dyntype="filter" actlimited="true"/>"#, "invalid actrange"),
        ] {
            let e = tree_from_mjcf_str(&model(a)).err().unwrap_or_else(|| panic!("MuJoCo refuses {a}"));
            assert!(e.contains(why), "{a}: {e}");
        }
        // under `auto` a reversed range is not a range: MuJoCo compiles it UNLIMITED
        let t = tree_from_mjcf_str(&model(r#"<general joint="j" dyntype="filter" actrange="1 -1"/>"#)).unwrap();
        assert_eq!((t.na(), t.actuators[0].actrange), (1, None));
        let t = tree_from_mjcf_str(&model(r#"<general joint="j" dyntype="filter" actrange="-1 1"/>"#)).unwrap();
        assert_eq!(t.actuators[0].actrange, Some([-1.0, 1.0]));
    }

    const ACT_MODEL: &str = r#"<mujoco><option integrator="INTEGRATOR" timestep="0.005"/><worldbody>
  <body name="upper" pos="0 0 1">
    <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="0.3"/>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1.1"/>
    <body name="fore" pos="0.3 0 0">
      <joint name="elbow" type="hinge" axis="0 1 0" range="-2.5 0.5"/>
      <geom type="capsule" fromto="0 0 0 0.25 0 0.05" size="0.025" mass="0.7"/>
      <body name="wrist" pos="0.25 0 0.05">
        <joint name="twist" type="hinge" axis="1 0 0"/>
        <geom type="box" size="0.08 0.06 0.04" mass="0.5"/>
      </body>
    </body>
  </body>
</worldbody>
<actuator>
  <general name="integ" joint="shoulder" dyntype="integrator" gainprm="30" biastype="affine" biasprm="0 -30 -2" actlimited="true" actrange="-0.5 0.5"/>
  <general name="filt" joint="elbow" dyntype="filter" dynprm="0.05" gainprm="2"/>
  <general name="exact" joint="twist" dyntype="filterexact" dynprm="0.03" actearly="true" gaintype="affine" gainprm="0.3 0 0.02"/>
  <muscle name="flexor" joint="elbow" lengthrange="-2.5 0.5" force="40" ctrllimited="true" ctrlrange="0 1"/>
</actuator></mujoco>"#;

    /// **`noslip`, against MuJoCo.** A block resting on a 0.2 rad slope and two arms on dry-friction hinges,
    /// all nearly at rest. MuJoCo's noslip pass moves `qacc` by 5.29: the arm whose gravity torque (0.49) is
    /// under its `frictionloss` (1.0) is held exactly instead of creeping, while the one over its loss (0.3)
    /// still falls at the clamp. Five variants cover every branch: pyramidal at condim 3 and 6 (paired edges),
    /// and elliptic at condim 3, 4 and 6 (`mju_QCQP2`, `mju_QCQP3` and the dense `mju_QCQP`).
    #[test]
    fn noslip_matches_mujoco_on_every_cone_branch() {
        let qpos = [0.0, 0.0, 0.0492, 0.9950041652780258, 0.0, 0.09983341664682815, 0.0, 0.0, 0.0];
        let qvel = [0.001, 0.0, -0.0005, 0.0, 0.002, 0.0, 0.0, 0.0];
        let cases: [(&str, &str, usize, [f64; 8]); 5] = [
            ("pyramidal", "3", 18, [0.8139417833716127, -2.565346625497493e-06, 4.4578210156693885, -5.107428800551414e-05, 0.3129934991356733, 5.521138936649959e-05, 1.4210854715202004e-14, 20.52981157659209]),
            ("pyramidal", "6", 42, [0.8114809475341013, -4.107825191113079e-14, 4.5716665678900075, 5.711283716849816e-13, -0.19730704391441561, 1.1076650644710803e-14, 1.4210854715202004e-14, 20.52981157659209]),
            ("elliptic", "3", 14, [0.8046735836031695, -1.8565471483531046e-06, 4.459699772773551, -3.696257761390441e-05, 0.12040403547146435, 3.6717659464763824e-05, 1.4210854715202004e-14, 20.52981157659209]),
            ("elliptic", "4", 18, [0.804675179098578, -1.2471948007535838e-16, 4.459699449350618, -2.967275923143441e-14, 0.12038484799091274, 3.203228665470154e-15, 1.4210854715202004e-14, 20.52981157659209]),
            ("elliptic", "6", 26, [0.7887481882927976, 4.870127041270169e-16, 4.462928010222404, -1.3593574431771503e-14, -0.21052631578946962, -1.4233155935978153e-15, 1.4210854715202004e-14, 20.52981157659209]),
        ];
        for (cone, condim, nefc, want) in cases {
            let t = tree_from_mjcf_str(&NOSLIP_MODEL.replace("CONE", cone).replace("CONDIM", condim)).unwrap();
            assert_eq!(t.noslip_iterations, 4);
            let q = t.q_from_qpos(&qpos, &t.qposadr()).unwrap();
            let f = t.forward_mujoco(&q, &qvel, &[], &[]).unwrap();
            assert_eq!(f.nefc, nefc, "{cone} condim {condim}: rows");
            for (k, w) in want.iter().enumerate() {
                assert!((f.qacc[k] - w).abs() < 1e-8 * w.abs().max(1.0), "{cone} condim {condim}: qacc[{k}] {} vs MuJoCo {w}", f.qacc[k]);
            }
        }
    }

    const NOSLIP_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option cone="CONE" noslip_iterations="4"/><worldbody>
  <body name="ramp" euler="0 0.2 0"><geom name="slope" type="box" size="1 1 0.05" pos="0 0 -0.05" friction="0.8 0.02 0.001"/></body>
  <body name="block" pos="0 0 0.0495" euler="0 0.2 0">
    <freejoint/>
    <geom type="box" size="0.1 0.07 0.05" mass="1.2" condim="CONDIM"/>
  </body>
  <body name="arm" pos="0.6 0 0.4">
    <joint name="shoulder" type="hinge" axis="0 1 0" frictionloss="1.0"/>
    <geom type="capsule" fromto="0 0 0 0.2 0 -0.1" size="0.03" mass="0.5"/>
  </body>
  <body name="arm2" pos="0.6 0.5 0.4">
    <joint name="elbow" type="hinge" axis="0 1 0" frictionloss="0.3"/>
    <geom type="capsule" fromto="0 0 0 0.2 0 -0.1" size="0.03" mass="0.5"/>
  </body>
</worldbody></mujoco>"#;

    const LOOP_MODEL: &str = r#"<mujoco><compiler angle="radian"/><worldbody>
  <body name="pelvis" pos="0 0 1">
    <freejoint/>
    <geom type="box" size="0.1 0.08 0.05" mass="2"/>
    <body name="a" pos="0.1 0 0">
      <joint name="ha" type="hinge" axis="0 1 0" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.02" mass="0.3"/>
      <site name="tip_a" pos="0 0 -0.3"/>
    </body>
    <body name="b" pos="-0.1 0 0">
      <joint name="hb" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 -0.28" size="0.02" mass="0.3"/>
      <body name="c" pos="0.1 0 -0.28">
        <joint name="hc" type="ball"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 -0.02" size="0.015" mass="0.1"/>
      </body>
    </body>
  </body>
</worldbody>
<equality><connect body1="c" body2="a" anchor="0.1 0 -0.02"/></equality>
</mujoco>"#;

    const FORWARD_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option impratio="3" cone="CONE"/><worldbody>
  <geom name="floor" type="plane" size="2 2 0.1"/>
  <body name="box" pos="0 0 0.049">
    <freejoint/>
    <geom type="box" size="0.1 0.07 0.05" mass="1.3"/>
  </body>
  <body name="arm" pos="0.5 0 0.3">
    <joint name="hinge" type="hinge" axis="0 1 0" range="-0.3 0.3" damping="0.2"/>
    <geom type="capsule" size="0.03 0.15" fromto="0 0 0 0.25 0 -0.2" mass="0.4"/>
  </body>
</worldbody></mujoco>"#;

    const ADHESION_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0" cone="elliptic"/><worldbody>
  <geom name="floor" type="plane" size="1 1 0.1"/>
  <body name="pad" pos="0 0 0.1">
    <joint name="z" type="slide" axis="0 0 1"/>
    <joint name="tilt" type="hinge" axis="1 0 0"/>
    <joint name="roll" type="hinge" axis="0 1 0" pos="0.03 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom name="a" type="sphere" size="0.05" pos="0.1 0 -0.045" margin="0.02"/>
    <geom name="b" type="sphere" size="0.05" pos="-0.1 0.03 -0.02" margin="0.02" gap="0.015"/>
  </body>
  <body name="other" pos="0.5 0 0.1">
    <joint name="z2" type="slide" axis="0 0 1"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <geom name="c" type="sphere" size="0.05" pos="0 0 -0.052"/>
  </body>
</worldbody>
<actuator><adhesion name="stick" body="pad" ctrlrange="0 1" gain="5"/><adhesion name="idle" body="other" ctrlrange="0 2"/></actuator>
</mujoco>"#;

    const WRAP_RULES_MODEL: &str = r#"<mujoco><compiler angle="radian"/><option gravity="0 0 0"/><worldbody>
  <geom name="drum" type="cylinder" size="0.1 0.3" euler="0.2 -0.1 0.3"/>
  <geom name="ball" type="sphere" size="0.1" pos="0 0 1"/>
  <geom name="post" type="cylinder" size="0.1 0.3" pos="0 0 2"/>
  <site name="c0" pos="-0.4 -0.2819 2"/>
  <site name="left" pos="-0.4973 -0.0521 2"/>
  <site name="a0" pos="-0.4 0.2 0.05"/>
  <site name="b0" pos="-0.4 0.2 1.05"/>
  <site name="in" pos="0.03 0.02 0"/>
  <site name="tall" pos="0.02 0.02 0.25"/>
  <site name="below" pos="0 -0.3 0"/>
  <site name="bin" pos="0.02 -0.03 1.01"/>
  <body name="cart" pos="0 0 0"><joint name="x" type="slide" axis="1 0 0"/>
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
    <site name="a1" pos="0.4 0.25 -0.05"/>
    <site name="b1" pos="0.4 0.25 0.95"/>
    <site name="c1" pos="0.33 -0.1939 2"/>
  </body>
</worldbody>
<tendon>
  <spatial name="inside"><site site="a0"/><geom geom="drum" sidesite="in"/><site site="a1"/></spatial>
  <spatial name="across"><site site="a0"/><geom geom="drum" sidesite="below"/><site site="a1"/></spatial>
  <spatial name="tall"><site site="a0"/><geom geom="drum" sidesite="tall"/><site site="a1"/></spatial>
  <spatial name="ball_inside"><site site="b0"/><geom geom="ball" sidesite="bin"/><site site="b1"/></spatial>
  <spatial name="crossing"><site site="c0"/><geom geom="post" sidesite="left"/><site site="c1"/></spatial>
</tendon></mujoco>"#;

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
        let got = moving.qacc_smooth_mujoco(&qm, &v, &[], &[]).unwrap();
        let want = [-0.40776163054058184, 1.0160972487730715, -9.159571401180923, -11.697022186283133, 4.64153544898943, -8.757526118453406, -1.8570604740614982];
        for (i, w) in want.iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-9, "moving qacc_smooth[{i}] {} vs {w}", got[i]);
        }
        // without the velocity term the answer is wrong, and only while moving — the check that it is wired
        let still = moving.qacc_smooth_mujoco(&qm, &[0.0; 7], &[], &[]).unwrap();
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
        let got = t.qacc_smooth_mujoco(&q, &[0.0; 4], &[], &[]).unwrap();
        for (i, w) in want.iter().enumerate() {
            assert!((got[i] - w).abs() < 1e-9, "qacc_smooth[{i}] {} vs {w}", got[i]);
        }
    }

    /// **A body with TWO joints still hangs off its parent.** MuJoCo's contact filter asks whether one
    /// geom's weld body is the other's PARENT weld; the shortcut "the tree dof a geom rides on is its body"
    /// answers that with the body's own previous dof the moment a body has more than one joint, and the pair
    /// is collided when MuJoCo filters it.
    ///
    /// On `hello_robot_stretch` the rubber tips have two joints each and collided with the fingers they hang
    /// off — a contact MuJoCo never produces, in every sampled state.
    ///
    /// ⛔⛔ And the question is asked of the WELD's parent, not the body's: the two coincide for a body
    /// that has joints and diverge for one that does not, which is why reading it the wrong way survived a
    /// whole corpus sweep. It cost 31 states of contact-count disagreement on its own.
    #[test]
    fn a_body_with_two_joints_is_still_its_parents_child() {
        let t = tree_from_mjcf_str(
            r#"<mujoco><compiler angle="radian"/><worldbody>
<body name="upper"><joint name="a" type="hinge" axis="0 0 1"/><inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
  <geom name="g_upper" type="box" size="0.05 0.05 0.05"/>
  <body name="tip" pos="0.06 0 0">
    <joint name="b1" type="slide" axis="1 0 0"/><joint name="b2" type="slide" axis="0 1 0"/>
    <inertial pos="0 0 0" mass="0.1" diaginertia="0.001 0.001 0.001"/>
    <geom name="g_tip" type="box" size="0.05 0.05 0.05"/>
    <body name="cap" pos="0.02 0 0"><inertial pos="0 0 0" mass="0.05" diaginertia="0.0001 0.0001 0.0001"/>
      <geom name="g_cap" type="sphere" size="0.02"/></body>
  </body>
</body></worldbody></mujoco>"#,
        )
        .unwrap();
        let w = t.body_welds();
        let (upper, tip, cap) = (w["upper"], w["tip"], w["cap"]);
        assert_eq!(tip.2, 2, "the tip has two dofs of its own");
        assert_eq!(tip.1, upper.0, "the tip's PARENT weld is the upper body, not the tip's own first dof");
        assert_eq!(cap.0, tip.0, "a jointless child welds into the body it hangs off");
        // ⛔⛔ the parent of the WELD, not of the BODY. The cap welds into the tip, so its parent weld is
        // the TIP'S parent — `upper` — and not the tip itself. MuJoCo: `body_weldid[body_parentid[
        // body_weldid[b]]]`, which prints `upper` for the cap on this very model.
        assert_eq!(cap.1, upper.0, "the cap welds into the tip, so its parent weld is the TIP's parent");
        let f = |a: (usize, usize, usize), b: (usize, usize, usize)| crate::mujoco_collision::filter_body_pair(a.0, a.1, a.2, b.0, b.1, b.2, true);
        assert!(f(upper, tip), "parent and child by weld");
        assert!(f(tip, cap), "the same weld body");
        // ⛔⛔ and therefore a JOINTLESS GRANDCHILD *IS* filtered against its grandparent, because welding
        // into its parent makes it that parent — which is a child. Verified against MuJoCo 3.13.0 on this
        // exact tree with the cap moved to `pos="-0.06 0 0"`, where it sits at the origin fully inside
        // `g_upper`: `d.ncon` is ZERO. An earlier reading of this test asserted the opposite and was wrong;
        // it was the reason `aloha` collided `left/wrist_link` with `left/gripper_base` at 3.7 mm of
        // penetration in every sampled state.
        assert!(f(upper, cap), "welding into the tip makes the cap a CHILD of upper, and children are filtered");
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
    /// ⛔⛔ A body MuJoCo cannot weigh keeps its OWN `pos` and `quat` as its inertial frame — verified
    /// against MuJoCo 3.13.0, which reports `body_ipos == body_pos` for an empty body, a body holding only
    /// a site or camera, and a body whose geoms all state `mass="0"`, and `body_ipos` as WRITTEN for an
    /// explicit `<inertial mass="0">`. The distinction is inferred-and-weightless, not zero mass.
    #[test]
    fn a_body_mujoco_cannot_weigh_keeps_its_own_placement_as_its_inertial_frame() {
        let xml = r#"<mujoco><worldbody><body name="a" pos="0.1 0.2 0.3">
  <joint type="hinge" axis="0 1 0"/>
  <geom type="box" size="0.05 0.05 0.05"/>
  <body name="empty" pos="0.7 -0.3 0.9" quat="0.3826834 0 0 0.9238795"/>
  <body name="site_only" pos="0.3 0 0" quat="0 1 0 0"><site name="s"/></body>
  <body name="zero_mass_geom" pos="0.4 0.1 0"><geom type="box" size="0.05 0.05 0.05" pos="0.2 0 0" mass="0"/></body>
  <body name="explicit_zero" pos="0.6 0 0"><inertial pos="0 0 0" mass="0" diaginertia="0 0 0"/><geom type="sphere" size="0.02" mass="0"/></body>
</body></worldbody></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        let ipos = |n: &str| t.body_ipos[n];
        assert!((ipos("empty") - Vector3::new(0.7, -0.3, 0.9)).norm() < 1e-15, "{:?}", ipos("empty"));
        assert!((ipos("site_only") - Vector3::new(0.3, 0.0, 0.0)).norm() < 1e-15, "{:?}", ipos("site_only"));
        assert!((ipos("zero_mass_geom") - Vector3::new(0.4, 0.1, 0.0)).norm() < 1e-15, "{:?}", ipos("zero_mass_geom"));
        // stated, so obeyed as written even at zero mass — and the weighable body is unaffected
        assert!(ipos("explicit_zero").norm() < 1e-15, "{:?}", ipos("explicit_zero"));
        assert!(ipos("a").norm() < 1e-15, "{:?}", ipos("a"));
        // the frame comes with it: `quat="0.3826834 0 0 0.9238795"` is a 135 degree turn about z
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(0.3826834, 0.0, 0.0, 0.9238795));
        let r = t.body_iquat["empty"];
        assert!((r - q.to_rotation_matrix().into_inner()).norm() < 1e-7, "{r:?}");
        assert!((t.body_iquat["a"] - Matrix3::identity()).norm() < 1e-15);
    }
    /// ⛔⛔ The `<muscle>` shortcut REPLACES both parameter vectors with MuJoCo's muscle defaults, and a
    /// `<general>` that names a class whose `<muscle>` declaration set them inherits the whole record —
    /// gain type, bias type, dynamics and all nine parameters. That second case is how `ms_human_700`
    /// writes all 700 of its muscles, so a loader that only handles the literal `<muscle>` tag carries
    /// none of them. Every number verified against MuJoCo 3.13.0.
    #[test]
    fn a_muscle_is_carried_through_the_tag_and_through_a_class() {
        let xml = r#"<mujoco>
<default><default class="m"><muscle ctrllimited="true" ctrlrange="0 1" force="300" lmax="1.7"/></default></default>
<worldbody><body><joint name="j" type="slide" axis="1 0 0"/><geom type="box" size="0.1 0.1 0.1"/></body></worldbody>
<actuator>
  <muscle name="plain" joint="j" lengthrange="0.1 0.4"/>
  <general name="viaclass" class="m" joint="j" lengthrange="0.2 0.5"/>
  <muscle name="tuned" joint="j" lengthrange="0.1 0.4" timeconst="0.02 0.05" tausmooth="0.3" range="0.6 1.2" vmax="2" fpmax="1.1" fvmax="1.4" scale="150" lmin="0.45"/>
</actuator></mujoco>"#;
        let t = tree_from_mjcf_str(xml).unwrap();
        assert!(t.actuators_unsupported.is_empty(), "{:?}", t.actuators_unsupported);
        let by: BTreeMap<&str, &crate::mujoco_actuator::Actuator> = t.actuators.iter().map(|a| (a.name.as_str(), a)).collect();
        for a in t.actuators.iter() {
            assert_eq!(a.gain, crate::mujoco_actuator::ActGain::Muscle, "{}", a.name);
            assert_eq!(a.bias, crate::mujoco_actuator::ActBias::Muscle, "{}", a.name);
            assert_eq!(a.dynamics, crate::mujoco_actuator::ActDyn::Muscle, "{}", a.name);
            assert_eq!(a.gainprm, a.biasprm, "MuJoCo writes the same nine into both");
            // a unit force on a slide joint carrying a 8 kg box: 1/8
            assert!((a.acc0 - 0.125).abs() < 1e-12, "{} acc0 {}", a.name, a.acc0);
        }
        let plain = by["plain"];
        assert_eq!(&plain.gainprm[..9], &[0.75, 1.05, -1.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.2]);
        assert_eq!(plain.dynprm, [0.01, 0.04, 0.0]);
        assert_eq!(plain.lengthrange, [0.1, 0.4]);
        assert_eq!(plain.ctrlrange, None, "no ctrlrange stated and none inherited");
        // the class's `force` and `lmax` land in the record the `<general>` inherits, defaults elsewhere
        let via = by["viaclass"];
        assert_eq!(&via.gainprm[..9], &[0.75, 1.05, 300.0, 200.0, 0.5, 1.7, 1.5, 1.3, 1.2]);
        assert_eq!(via.lengthrange, [0.2, 0.5]);
        assert_eq!(via.ctrlrange, Some([0.0, 1.0]));
        let tuned = by["tuned"];
        assert_eq!(&tuned.gainprm[..9], &[0.6, 1.2, -1.0, 150.0, 0.45, 1.6, 2.0, 1.1, 1.4]);
        assert_eq!(tuned.dynprm, [0.02, 0.05, 0.3]);
        // and the force follows the activation, not the control: a muscle at rest with ctrl 1 produces
        // only its passive force, which at the optimum is zero
        let (l, v) = (0.25, 0.0);
        assert_eq!(tuned.force(l, v, 1.0, 0.0), 0.0);
        assert!(tuned.force(l, v, 0.0, 1.0) < 0.0, "activation, not control, drives it");
    }
    /// ⛔⛔ **MuJoCo DROPS an OBJ face with more than 100 vertices** — silently, not as an error and not
    /// with a fallback. Found by bisection against MuJoCo 3.13.0 on synthetic files: a 100-gon yields its
    /// 98 triangles and a 101-gon yields none. The cap is on the VERTEX COUNT, not the line: a 100-gon
    /// written with 6-digit indices, 701 characters long, still triangulates.
    ///
    /// It decides one mesh in Menagerie — `hello_robot_stretch_3/link_SG3_gripper_body` has a 215-gon —
    /// and through it that model's centre of mass, gravity compensation and constrained acceleration.
    #[test]
    fn an_obj_face_with_more_than_a_hundred_vertices_is_dropped_as_mujoco_drops_it() {
        let ngon = |n: usize| -> String {
            let mut s = String::new();
            for k in 0..n {
                let a = std::f64::consts::TAU * (k as f64) / (n as f64);
                s.push_str(&format!("v {} {} 0\n", a.cos(), a.sin()));
            }
            // one triangle apart from the n-gon, so the count below is unambiguous
            s.push_str("v 0 0 1\nv 1 0 1\nv 0 1 1\n");
            s.push_str("f ");
            for k in 0..n {
                s.push_str(&format!("{} ", k + 1));
            }
            s.push('\n');
            s.push_str(&format!("f {} {} {}\n", n + 1, n + 2, n + 3));
            s
        };
        let tris = |n: usize| obj_as_mujoco_reads_it(&ngon(n)).expect("parses").tris.len();
        assert_eq!(tris(100), 98 + 1, "a 100-gon is fanned into 98 triangles");
        assert_eq!(tris(101), 1, "a 101-gon is DROPPED, leaving only the separate triangle");
        assert_eq!(tris(215), 1, "and so is the 215-gon that decides hello_robot_stretch_3");
        // the boundary is exactly there, and small faces are untouched
        assert_eq!(tris(99), 97 + 1);
        assert_eq!(tris(5), 3 + 1);
    }
}
