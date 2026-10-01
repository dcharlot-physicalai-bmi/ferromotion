//! **MuJoCo's smooth dynamics in MuJoCo's own coordinates and arithmetic** — `mj_comPos`, `mj_crb`,
//! `mj_factorM`, `mj_comVel` and `mj_rne` from `engine_core_smooth.c` (3.13.0), on the compiled model's own
//! arrays: the composite inertias `cinert` about each tree's centre of mass, the motion dofs `cdof`, the
//! mass matrix in MuJoCo's sparse layout (`M_rownnz`, `M_rowadr`, `M_colind`) and its `L'DL` factor.
//!
//! The rest of this crate computes the same quantities in its own coordinates (a free base as Euler angles,
//! jointless bodies welded into the link they ride on), which agree with MuJoCo's to round-off and no
//! further. That is not enough to follow MuJoCo's trajectory: its step is not continuous everywhere
//! (see `examples/menagerie_step.rs`), so a state one ulp off can land somewhere else. This module is the
//! first stage of computing MuJoCo's state to the bit.
//!
//! ⛔ Contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]): the left product of
//! `a*b + c*d` is fused, each later product of a chain is fused into the running sum, `s += a*b` is
//! `fma(a, b, s)`, and `x - p*q` is `fma(-p, q, x)`.

use crate::mujoco_kinematics::{KinJointKind, KinState, MjKinematics, SameFrame};

/// `mjMINVAL`
const MJ_MINVAL: f64 = 1e-15;

/// `mju_mulInertVec`: a 6-D motion vector through a com-based inertia (`i` = the 10 numbers of `cinert`).
fn mul_inert_vec(i: &[f64; 10], v: &[f64; 6]) -> [f64; 6] {
    [
        i[7].mul_add(v[5], (-i[8]).mul_add(v[4], i[4].mul_add(v[2], i[0].mul_add(v[0], i[3] * v[1])))),
        (-i[6]).mul_add(v[5], i[8].mul_add(v[3], i[5].mul_add(v[2], i[3].mul_add(v[0], i[1] * v[1])))),
        i[6].mul_add(v[4], (-i[7]).mul_add(v[3], i[2].mul_add(v[2], i[4].mul_add(v[0], i[5] * v[1])))),
        i[9].mul_add(v[3], i[8].mul_add(v[1], -(i[7] * v[2]))),
        i[9].mul_add(v[4], i[6].mul_add(v[2], -(i[8] * v[0]))),
        i[9].mul_add(v[5], i[7].mul_add(v[0], -(i[6] * v[1]))),
    ]
}

/// `mji_dot6`: `((v0·w0 + v2·w2) + (v1·w1 + v3·w3)) + (v4·w4 + v5·w5)`, each pair contracted.
fn dot6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    (a[0].mul_add(b[0], a[2] * b[2]) + a[1].mul_add(b[1], a[3] * b[3])) + a[4].mul_add(b[4], a[5] * b[5])
}

/// `mju_dot` at `n = 6`: four running sums, then the two left over.
fn dot_n6(a: &[f64; 6], b: &[f64; 6]) -> f64 {
    let (r0, r1, r2, r3) = (a[0].mul_add(b[0], 0.0), a[1].mul_add(b[1], 0.0), a[2].mul_add(b[2], 0.0), a[3].mul_add(b[3], 0.0));
    ((r0 + r2) + (r1 + r3)) + a[4].mul_add(b[4], a[5] * b[5])
}

/// `mju_dotSparse`: four running sums over the first `4⌊n/4⌋` entries, then the rest one by one.
pub(crate) fn dot_sparse(v: &[f64], x: &[f64], ind: &[usize]) -> f64 {
    let n = v.len();
    let (mut r0, mut r1, mut r2, mut r3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut i = 0;
    while i + 4 <= n {
        r0 = v[i].mul_add(x[ind[i]], r0);
        r1 = v[i + 1].mul_add(x[ind[i + 1]], r1);
        r2 = v[i + 2].mul_add(x[ind[i + 2]], r2);
        r3 = v[i + 3].mul_add(x[ind[i + 3]], r3);
        i += 4;
    }
    let mut r = (r0 + r2) + (r1 + r3);
    while i < n {
        r = v[i].mul_add(x[ind[i]], r);
        i += 1;
    }
    r
}

/// `mju_mulMatTVec` of a `3×nv` Jacobian (per dof, the three rows) and a 3-vector: the rows in order, a
/// zero coefficient skipped, each fused into the running column.
fn mul_mat_t_vec3(j: &[[f64; 3]], v: &[f64; 3]) -> Vec<f64> {
    let mut res = vec![0.0f64; j.len()];
    for r in 0..3 {
        if v[r] != 0.0 {
            for (x, row) in res.iter_mut().zip(j) {
                *x = row[r].mul_add(v[r], *x);
            }
        }
    }
    res
}

/// `mju_max`
fn fmax(a: f64, b: f64) -> f64 {
    if a >= b { a } else { b }
}

/// `mji_cross`, contracted.
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
}

/// `mji_crossMotion`: `vel × v` for motion vectors.
fn cross_motion(vel: &[f64; 6], v: &[f64; 6]) -> [f64; 6] {
    let mut r = [
        (-vel[2]).mul_add(v[1], vel[1] * v[2]),
        vel[2].mul_add(v[0], -(vel[0] * v[2])),
        (-vel[1]).mul_add(v[0], vel[0] * v[1]),
        (-vel[2]).mul_add(v[4], vel[1] * v[5]),
        vel[2].mul_add(v[3], -(vel[0] * v[5])),
        (-vel[1]).mul_add(v[3], vel[0] * v[4]),
    ];
    r[3] += (-vel[5]).mul_add(v[1], vel[4] * v[2]);
    r[4] += vel[5].mul_add(v[0], -(vel[3] * v[2]));
    r[5] += (-vel[4]).mul_add(v[0], vel[3] * v[1]);
    r
}

/// `mji_crossForce`: `vel ×* f` for force vectors.
fn cross_force(vel: &[f64; 6], f: &[f64; 6]) -> [f64; 6] {
    let mut r = [
        (-vel[2]).mul_add(f[1], vel[1] * f[2]),
        vel[2].mul_add(f[0], -(vel[0] * f[2])),
        (-vel[1]).mul_add(f[0], vel[0] * f[1]),
        (-vel[2]).mul_add(f[4], vel[1] * f[5]),
        vel[2].mul_add(f[3], -(vel[0] * f[5])),
        (-vel[1]).mul_add(f[3], vel[0] * f[4]),
    ];
    r[0] += (-vel[5]).mul_add(f[4], vel[4] * f[5]);
    r[1] += vel[5].mul_add(f[3], -(vel[3] * f[5]));
    r[2] += (-vel[4]).mul_add(f[3], vel[3] * f[4]);
    r
}

/// `mju_inertCom`: a body's principal inertia `inert` in frame `mat`, mass `mass`, moved by `dif` to a
/// com-based 10-vector (the rotational 3×3 as `xx yy zz xy xz yz`, then `mass·dif`, then `mass`).
fn inert_com(inert: &[f64; 3], mat: &[f64; 9], dif: &[f64; 3], mass: f64) -> [f64; 10] {
    let tmp = [mat[0] * inert[0], mat[3] * inert[0], mat[6] * inert[0], mat[1] * inert[1], mat[4] * inert[1], mat[7] * inert[1], mat[2] * inert[2], mat[5] * inert[2], mat[8] * inert[2]];
    let mut r = [0.0; 10];
    r[0] = mat[2].mul_add(tmp[6], mat[0].mul_add(tmp[0], mat[1] * tmp[3]));
    r[1] = mat[5].mul_add(tmp[7], mat[3].mul_add(tmp[1], mat[4] * tmp[4]));
    r[2] = mat[8].mul_add(tmp[8], mat[6].mul_add(tmp[2], mat[7] * tmp[5]));
    r[3] = mat[2].mul_add(tmp[7], mat[0].mul_add(tmp[1], mat[1] * tmp[4]));
    r[4] = mat[2].mul_add(tmp[8], mat[0].mul_add(tmp[2], mat[1] * tmp[5]));
    r[5] = mat[5].mul_add(tmp[8], mat[3].mul_add(tmp[2], mat[4] * tmp[5]));
    r[0] = mass.mul_add(dif[1].mul_add(dif[1], dif[2] * dif[2]), r[0]);
    r[1] = mass.mul_add(dif[0].mul_add(dif[0], dif[2] * dif[2]), r[1]);
    r[2] = mass.mul_add(dif[0].mul_add(dif[0], dif[1] * dif[1]), r[2]);
    r[3] = (-(mass * dif[0])).mul_add(dif[1], r[3]);
    r[4] = (-(mass * dif[0])).mul_add(dif[2], r[4]);
    r[5] = (-(mass * dif[1])).mul_add(dif[2], r[5]);
    r[6] = mass * dif[0];
    r[7] = mass * dif[1];
    r[8] = mass * dif[2];
    r[9] = mass;
    r
}

/// `mju_dofCom`: a hinge's axis about `offset` (a slide has none).
fn dof_com(axis: &[f64; 3], offset: Option<&[f64; 3]>) -> [f64; 6] {
    match offset {
        Some(o) => {
            let c = cross(axis, o);
            [axis[0], axis[1], axis[2], c[0], c[1], c[2]]
        }
        None => [0.0, 0.0, 0.0, axis[0], axis[1], axis[2]],
    }
}

/// `mju_mulDofVec`: `Σ dofₖ·vecₖ` — one dof scaled, several accumulated from zero (`mju_mulMatTVec`, which
/// skips a zero coefficient).
fn mul_dof_vec(dof: &[[f64; 6]], vec: &[f64]) -> [f64; 6] {
    match dof.len() {
        0 => [0.0; 6],
        1 => dof[0].map(|x| x * vec[0]),
        _ => {
            let mut r = [0.0; 6];
            for (d, &v) in dof.iter().zip(vec) {
                if v != 0.0 {
                    for k in 0..6 {
                        r[k] = d[k].mul_add(v, r[k]);
                    }
                }
            }
            r
        }
    }
}

/// `mju_clip`
fn clip(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// `mju_sigmoid`, contracted.
fn sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    x * x * x * (3.0 * x).mul_add(2.0f64.mul_add(x, -5.0), 10.0)
}

/// `mju_muscleGainLength`, contracted.
fn muscle_gain_length(length: f64, lmin: f64, lmax: f64) -> f64 {
    if lmin <= length && length <= lmax {
        let a = 0.5 * (lmin + 1.0);
        let b = 0.5 * (1.0 + lmax);
        if length <= a {
            let x = (length - lmin) / MJ_MINVAL.max(a - lmin);
            0.5 * x * x
        } else if length <= 1.0 {
            let x = (1.0 - length) / MJ_MINVAL.max(1.0 - a);
            (-(0.5 * x)).mul_add(x, 1.0)
        } else if length <= b {
            let x = (length - 1.0) / MJ_MINVAL.max(b - 1.0);
            (-(0.5 * x)).mul_add(x, 1.0)
        } else {
            let x = (lmax - length) / MJ_MINVAL.max(lmax - b);
            0.5 * x * x
        }
    } else {
        0.0
    }
}

/// `mju_muscleGain`.
fn muscle_gain(len: f64, vel: f64, lr: &[f64; 2], acc0: f64, prm: &[f64; 10]) -> f64 {
    let (range, mut force, scale, lmin, lmax, vmax, fvmax) = ([prm[0], prm[1]], prm[2], prm[3], prm[4], prm[5], prm[6], prm[8]);
    if force < 0.0 {
        force = scale / MJ_MINVAL.max(acc0);
    }
    let l0 = (lr[1] - lr[0]) / MJ_MINVAL.max(range[1] - range[0]);
    let l = range[0] + (len - lr[0]) / MJ_MINVAL.max(l0);
    let v = vel / MJ_MINVAL.max(l0 * vmax);
    let fl = muscle_gain_length(l, lmin, lmax);
    let y = fvmax - 1.0;
    let fv = if v <= -1.0 {
        0.0
    } else if v <= 0.0 {
        (v + 1.0) * (v + 1.0)
    } else if v <= y {
        fvmax - (y - v) * (y - v) / MJ_MINVAL.max(y)
    } else {
        fvmax
    };
    -force * fl * fv
}

/// `mju_muscleBias`.
fn muscle_bias(len: f64, lr: &[f64; 2], acc0: f64, prm: &[f64; 10]) -> f64 {
    let (range, mut force, scale, lmax, fpmax) = ([prm[0], prm[1]], prm[2], prm[3], prm[5], prm[7]);
    if force < 0.0 {
        force = scale / MJ_MINVAL.max(acc0);
    }
    let l0 = (lr[1] - lr[0]) / MJ_MINVAL.max(range[1] - range[0]);
    let l = range[0] + (len - lr[0]) / MJ_MINVAL.max(l0);
    let b = 0.5 * (1.0 + lmax);
    if l <= 1.0 {
        0.0
    } else if l <= b {
        let x = (l - 1.0) / MJ_MINVAL.max(b - 1.0);
        -force * fpmax * 0.5 * x * x
    } else {
        let x = (l - b) / MJ_MINVAL.max(b - 1.0);
        -force * fpmax * (0.5 + x)
    }
}

/// `mju_muscleDynamics`, contracted.
fn muscle_dynamics(ctrl: f64, act: f64, prm: &[f64; 3]) -> f64 {
    let ctrlclamp = clip(ctrl, 0.0, 1.0);
    let actclamp = clip(act, 0.0, 1.0);
    let s = 1.5f64.mul_add(actclamp, 0.5);
    let tau_act = prm[0] * s;
    let tau_deact = prm[1] / s;
    let dctrl = ctrlclamp - act;
    let tau = if prm[2] < MJ_MINVAL {
        if dctrl > 0.0 {
            tau_act
        } else {
            tau_deact
        }
    } else {
        (tau_act - tau_deact).mul_add(sigmoid(dctrl / prm[2] + 0.5), tau_deact)
    };
    dctrl / MJ_MINVAL.max(tau)
}

/// One actuator as the native pipeline carries it: a hinge or slide JOINT transmission, which is what
/// most of Menagerie drives, or a FIXED TENDON. (Spatial-tendon, site and body transmissions are listed as
/// unsupported.)
#[derive(Clone, Debug)]
pub(crate) struct NativeActuator {
    pub(crate) act: crate::mujoco_actuator::Actuator,
    /// the joint's dof and `qpos` address, for a joint transmission
    pub(crate) dofadr: usize,
    pub(crate) qposadr: usize,
    /// the fixed tendon, for a tendon transmission
    pub(crate) tendon: Option<usize>,
    /// the site and its six-component `gear` (a wrench in the site's frame), for a site transmission
    pub(crate) site: Option<(crate::mujoco_kinematics::KinGeom, [f64; 6])>,
    /// the body, for an adhesion (`mjTRN_BODY`) transmission, whose moment comes from the contact rows
    pub(crate) body: Option<usize>,
    /// index of its activation in `act`, when it has one
    pub(crate) actadr: Option<usize>,
}

/// A tendon as `mj_tendon` reads it: a fixed one's joints in file order, or a spatial one's path, and its
/// Jacobian's static sparsity; with its spring and damper.
#[derive(Clone, Debug, Default)]
pub(crate) struct NativeTendon {
    /// `(jnt_qposadr, jnt_dofadr, wrap_prm)` per joint, in file order (a fixed tendon)
    pub(crate) path: Vec<(usize, usize, f64)>,
    /// the sites, wrap geoms and pulleys, in order (a spatial tendon)
    pub(crate) spatial: Option<Vec<crate::mujoco_tendon::TendonObj>>,
    /// `ten_J_colind` (`makeTendonSparse`), sorted
    pub(crate) colind: Vec<usize>,
    pub(crate) stiffness: f64,
    pub(crate) damping: f64,
    /// `tendon_lengthspring`, resolved (`-1 -1` is the length at `qpos0`, from `mj_setConst`)
    pub(crate) lengthspring: [f64; 2],
}

/// A tendon's path as the loader hands it over.
#[derive(Clone, Debug)]
pub(crate) enum NativeTendonPath {
    /// `(dof, coef)` per joint, in file order
    Fixed(Vec<(usize, f64)>),
    Spatial(Vec<crate::mujoco_tendon::TendonObj>),
}

/// One tendon for [`SmoothInputs`]: its path, `stiffness`, `damping` and `springlength` (`None`: the length
/// at `qpos0`).
#[derive(Clone, Debug)]
pub(crate) struct TendonInput {
    pub(crate) path: NativeTendonPath,
    pub(crate) stiffness: f64,
    pub(crate) damping: f64,
    pub(crate) springlength: Option<[f64; 2]>,
}

/// What the passive and actuation stages read beyond the kinematic model, joints in MuJoCo's order.
#[derive(Clone, Debug, Default)]
pub(crate) struct SmoothInputs {
    pub(crate) body_mass: Vec<f64>,
    pub(crate) body_inertia: Vec<[f64; 3]>,
    pub(crate) body_gravcomp: Vec<f64>,
    pub(crate) jnt_armature: Vec<f64>,
    pub(crate) jnt_stiffness: Vec<f64>,
    pub(crate) jnt_springref: Vec<f64>,
    pub(crate) jnt_damping: Vec<f64>,
    pub(crate) jnt_actfrcrange: Vec<Option<[f64; 2]>>,
    pub(crate) qpos0: Vec<f64>,
    pub(crate) gravity: [f64; 3],
    pub(crate) timestep: f64,
    pub(crate) actuators: Vec<crate::mujoco_actuator::Actuator>,
    /// every tendon in MuJoCo's order; `None` for one this port cannot carry
    pub(crate) tendons: Vec<Option<TendonInput>>,
    /// per actuator: the site of a site transmission, and its `gear`
    pub(crate) actuator_sites: Vec<Option<(crate::mujoco_kinematics::KinGeom, [f64; 6])>>,
    /// per actuator: the body of an adhesion transmission
    pub(crate) actuator_bodies: Vec<Option<usize>>,
    /// `<option density viscosity wind>`, for the inertia-box fluid model
    pub(crate) density: f64,
    pub(crate) viscosity: f64,
    pub(crate) wind: [f64; 3],
    /// what this port does not yet compute natively, by name — a model with any is not compared
    pub(crate) unsupported: Vec<String>,
}

/// The compiled model's arrays the smooth dynamics read, bodies, joints and dofs in MuJoCo's order.
#[derive(Clone, Debug, Default)]
pub(crate) struct SmoothModel {
    pub(crate) kin: MjKinematics,
    pub(crate) body_rootid: Vec<usize>,
    pub(crate) body_jntadr: Vec<usize>,
    pub(crate) body_dofadr: Vec<usize>,
    pub(crate) body_dofnum: Vec<usize>,
    pub(crate) body_simple: Vec<i32>,
    pub(crate) body_mass: Vec<f64>,
    pub(crate) body_subtreemass: Vec<f64>,
    pub(crate) body_inertia: Vec<[f64; 3]>,
    pub(crate) jnt_type: Vec<KinJointKind>,
    pub(crate) jnt_dofadr: Vec<usize>,
    pub(crate) jnt_bodyid: Vec<usize>,
    pub(crate) dof_bodyid: Vec<usize>,
    pub(crate) dof_jntid: Vec<usize>,
    pub(crate) dof_parentid: Vec<i32>,
    pub(crate) dof_simplenum: Vec<i32>,
    pub(crate) dof_armature: Vec<f64>,
    pub(crate) dof_m0: Vec<f64>,
    pub(crate) m_rownnz: Vec<usize>,
    pub(crate) m_rowadr: Vec<usize>,
    pub(crate) m_colind: Vec<usize>,
    pub(crate) gravity: [f64; 3],
    pub(crate) timestep: f64,
    pub(crate) jnt_qposadr: Vec<usize>,
    pub(crate) jnt_stiffness: Vec<f64>,
    pub(crate) jnt_springref: Vec<f64>,
    pub(crate) jnt_actfrcrange: Vec<Option<[f64; 2]>>,
    pub(crate) dof_damping: Vec<f64>,
    pub(crate) body_gravcomp: Vec<f64>,
    pub(crate) actuators: Vec<NativeActuator>,
    /// every tendon, by MuJoCo's index; `None` for one this port cannot carry
    pub(crate) tendons: Vec<Option<NativeTendon>>,
    pub(crate) density: f64,
    pub(crate) viscosity: f64,
    pub(crate) wind: [f64; 3],
    pub(crate) na: usize,
    pub(crate) unsupported: Vec<String>,
    /// what the constraint rows read ([`crate::mujoco_efc`]), filled in by the tree
    pub(crate) efc: crate::mujoco_efc::EfcModel,
}

/// What the smooth stages leave in `mjData`.
#[derive(Clone, Debug)]
pub(crate) struct SmoothData {
    pub(crate) kin: KinState,
    pub(crate) subtree_com: Vec<[f64; 3]>,
    pub(crate) cinert: Vec<[f64; 10]>,
    pub(crate) cdof: Vec<[f64; 6]>,
    pub(crate) crb: Vec<[f64; 10]>,
    /// the mass matrix, `M_rowadr`/`M_colind` layout
    pub(crate) m: Vec<f64>,
    pub(crate) qld: Vec<f64>,
    pub(crate) qld_diag_inv: Vec<f64>,
    pub(crate) cvel: Vec<[f64; 6]>,
    pub(crate) cdof_dot: Vec<[f64; 6]>,
    pub(crate) qfrc_bias: Vec<f64>,
    pub(crate) qfrc_spring: Vec<f64>,
    pub(crate) qfrc_damper: Vec<f64>,
    pub(crate) qfrc_gravcomp: Vec<f64>,
    pub(crate) qfrc_fluid: Vec<f64>,
    pub(crate) qfrc_passive: Vec<f64>,
    /// `ten_length`, `ten_J` (each row's values on its tendon's `colind`) and `ten_velocity`; zero for a
    /// spatial tendon
    pub(crate) ten_length: Vec<f64>,
    pub(crate) ten_j: Vec<Vec<f64>>,
    pub(crate) ten_velocity: Vec<f64>,
    /// `actuator_moment`, row by row: `(moment_colind, values)`
    pub(crate) moment: Vec<(Vec<usize>, Vec<f64>)>,
    pub(crate) actuator_length: Vec<f64>,
    pub(crate) actuator_velocity: Vec<f64>,
    pub(crate) actuator_force: Vec<f64>,
    pub(crate) act_dot: Vec<f64>,
    pub(crate) qfrc_actuator: Vec<f64>,
    pub(crate) qfrc_smooth: Vec<f64>,
    pub(crate) qacc_smooth: Vec<f64>,
}

impl SmoothModel {
    /// The compiler's arrays from the kinematic model, each body's `body_mass` and principal `body_inertia`,
    /// each joint's armature (joints in MuJoCo's order), `qpos0` and `<option gravity>`. `dof_M0` is what
    /// `mj_setConst` makes of them: the composite inertia at `qpos0`.
    pub(crate) fn new(kin: &MjKinematics, inp: SmoothInputs) -> SmoothModel {
        let nbody = kin.bodies.len();
        let (jnt_armature, qpos0) = (&inp.jnt_armature, &inp.qpos0);
        let mut m = SmoothModel {
            kin: kin.clone(),
            body_mass: inp.body_mass.clone(),
            body_inertia: inp.body_inertia.clone(),
            body_gravcomp: inp.body_gravcomp.clone(),
            gravity: inp.gravity,
            timestep: inp.timestep,
            jnt_stiffness: inp.jnt_stiffness.clone(),
            jnt_springref: inp.jnt_springref.clone(),
            jnt_actfrcrange: inp.jnt_actfrcrange.clone(),
            unsupported: inp.unsupported.clone(),
            density: inp.density,
            viscosity: inp.viscosity,
            wind: inp.wind,
            ..Default::default()
        };
        let parent: Vec<usize> = kin.bodies.iter().map(|b| b.parent).collect();
        let width = |k: KinJointKind| match k {
            KinJointKind::Free => 6,
            KinJointKind::Ball => 3,
            _ => 1,
        };
        // rootid, joint and dof numbering, dof_parentid through each body's last dof
        let mut lastdof = vec![-1i32; nbody];
        for (i, b) in kin.bodies.iter().enumerate() {
            m.body_rootid.push(if i == 0 || parent[i] == 0 { i } else { m.body_rootid[parent[i]] });
            m.body_jntadr.push(m.jnt_type.len());
            m.body_dofadr.push(m.dof_bodyid.len());
            let mut last = if i == 0 { -1 } else { lastdof[parent[i]] };
            for j in &b.joints {
                let jid = m.jnt_type.len();
                m.jnt_type.push(j.kind);
                m.jnt_dofadr.push(m.dof_bodyid.len());
                m.jnt_bodyid.push(i);
                m.jnt_qposadr.push(j.qposadr);
                for _ in 0..width(j.kind) {
                    m.dof_damping.push(inp.jnt_damping.get(jid).copied().unwrap_or(0.0));
                    m.dof_parentid.push(last);
                    last = m.dof_bodyid.len() as i32;
                    m.dof_bodyid.push(i);
                    m.dof_jntid.push(jid);
                    m.dof_armature.push(jnt_armature.get(jid).copied().unwrap_or(0.0));
                }
            }
            lastdof[i] = last;
            m.body_dofnum.push(m.dof_bodyid.len() - m.body_dofadr[i]);
        }
        // `setFixed`: subtree masses, children into parents from the last body back
        m.body_subtreemass = m.body_mass.clone();
        for i in (1..nbody).rev() {
            let s = m.body_subtreemass[i];
            m.body_subtreemass[parent[i]] += s;
        }
        // `body_simple`, as `mjCModel::CopyTree` decides it, then `dof_simplenum`
        let null_vec = |v: &[f64; 3]| v.iter().all(|x| x.abs() < 1e-6);
        let mut simple = vec![0i32; nbody];
        for i in 0..nbody {
            let b = &kin.bodies[i];
            let p = parent[i];
            simple[i] = (b.sameframe == SameFrame::Body && (m.body_rootid[i] == i || (parent[p] == 0 && m.body_dofnum[p] == 0))) as i32;
            if i > 0 && p > 0 {
                simple[p] = 0;
            }
            let mut rotfound = false;
            for j in &b.joints {
                let aligned = j.axis.iter().filter(|a| a.abs() > 1e-14).count() == 1;
                if rotfound || !null_vec(&j.pos) || (matches!(j.kind, KinJointKind::Hinge | KinJointKind::Slide) && !aligned) {
                    simple[i] = 0;
                }
                if matches!(j.kind, KinJointKind::Ball | KinJointKind::Hinge) {
                    rotfound = true;
                }
            }
            if simple[i] != 0 && m.body_dofnum[i] != 0 {
                simple[i] = if b.joints.iter().all(|j| j.kind == KinJointKind::Slide) { 2 } else { 1 };
            }
        }
        m.body_simple = simple;
        let nv = m.dof_bodyid.len();
        let njnt = m.jnt_type.len();
        m.jnt_stiffness.resize(njnt, 0.0);
        m.jnt_springref.resize(njnt, 0.0);
        m.jnt_actfrcrange.resize(njnt, None);
        m.body_gravcomp.resize(nbody, 0.0);
        m.dof_simplenum = vec![0; nv];
        let mut count = 0;
        for i in (0..nv).rev() {
            count = if m.body_simple[m.dof_bodyid[i]] != 0 { count + 1 } else { 0 };
            m.dof_simplenum[i] = count;
        }
        // `M`: a simple dof's row is its diagonal; any other's is its ancestors, ascending, then itself
        for i in 0..nv {
            m.m_rowadr.push(m.m_colind.len());
            let mut row = vec![i];
            if m.dof_simplenum[i] == 0 {
                let mut j = m.dof_parentid[i];
                while j >= 0 {
                    row.push(j as usize);
                    j = m.dof_parentid[j as usize];
                }
            }
            row.reverse();
            m.m_rownnz.push(row.len());
            m.m_colind.extend(row);
        }
        // `mj_setM0`: at qpos0, the composite inertia seen by each dof, plus its armature
        let kin0 = kin.state(qpos0);
        let (cinert, cdof, _) = m.com_pos(&kin0);
        let crb = m.composite(&cinert);
        m.dof_m0 = (0..nv).map(|i| m.dof_armature[i] + dot_n6(&cdof[i], &mul_inert_vec(&crb[m.dof_bodyid[i]], &cdof[i]))).collect();
        // springs on a ball or free joint are not ported yet
        for (j, &k) in m.jnt_type.iter().enumerate() {
            if matches!(k, KinJointKind::Ball | KinJointKind::Free) && m.jnt_stiffness[j] != 0.0 {
                m.unsupported.push("a spring on a ball or free joint".into());
            }
        }
        // tendons: a fixed one's joints (`qpos` and dof address), a spatial one's path, and
        // `makeTendonSparse`'s sorted columns
        for t in &inp.tendons {
            let Some(t) = t else {
                m.tendons.push(None);
                continue;
            };
            let mut nt = NativeTendon { stiffness: t.stiffness, damping: t.damping, lengthspring: t.springlength.unwrap_or([0.0; 2]), ..Default::default() };
            match &t.path {
                NativeTendonPath::Fixed(path) => {
                    for &(dof, coef) in path {
                        match m.jnt_dofadr.iter().position(|&d| d == dof).filter(|&j| matches!(m.jnt_type[j], KinJointKind::Hinge | KinJointKind::Slide)) {
                            Some(j) => nt.path.push((m.jnt_qposadr[j], dof, coef)),
                            None => m.unsupported.push("a fixed tendon on a joint this port cannot place".into()),
                        }
                    }
                    nt.colind = nt.path.iter().map(|p| p.1).collect();
                    nt.colind.sort_unstable();
                    if nt.colind.windows(2).any(|w| w[0] == w[1]) {
                        m.unsupported.push("a fixed tendon that names a joint twice".into());
                    }
                }
                NativeTendonPath::Spatial(objs) => {
                    nt.colind = m.spatial_colind(objs);
                    nt.spatial = Some(objs.clone());
                }
            }
            m.tendons.push(Some(nt));
        }
        // `mj_setConst`: a `-1 -1` spring length is the tendon's length at `qpos0`
        if inp.tendons.iter().flatten().any(|t| t.springlength.is_none()) {
            let k0 = m.kin.state(qpos0);
            let (cinert0, cdof0, com0) = m.com_pos(&k0);
            let _ = cinert0;
            let (len0, _) = m.tendon(qpos0, &k0, &cdof0, &com0);
            for (i, t) in inp.tendons.iter().enumerate() {
                if let (Some(t), Some(Some(nt))) = (t, m.tendons.get_mut(i))
                    && t.springlength.is_none()
                {
                    nt.lengthspring = [len0[i], len0[i]];
                }
            }
        }
        // actuators: hinge and slide joint transmissions, and fixed tendons
        let mut na = 0;
        for (ai, a) in inp.actuators.iter().enumerate() {
            let actadr = (a.actnum() > 0).then(|| {
                na += a.actnum();
                na - a.actnum()
            });
            if let Some(site) = inp.actuator_sites.get(ai).cloned().flatten() {
                m.actuators.push(NativeActuator { act: a.clone(), dofadr: 0, qposadr: 0, tendon: None, site: Some(site), body: None, actadr });
                continue;
            }
            if let Some(body) = inp.actuator_bodies.get(ai).copied().flatten() {
                m.actuators.push(NativeActuator { act: a.clone(), dofadr: 0, qposadr: 0, tendon: None, site: None, body: Some(body), actadr });
                continue;
            }
            let spatial = match a.dynamic {
                Some(crate::mujoco_actuator::DynTransmission::SpatialTendon { index }) => Some(index),
                Some(_) => {
                    m.unsupported.push("an actuator that is not a joint, tendon or site transmission".into());
                    continue;
                }
                None => None,
            };
            if let Some(t) = a.tendon.or(spatial) {
                match m.tendons.get(t) {
                    Some(Some(_)) => m.actuators.push(NativeActuator { act: a.clone(), dofadr: 0, qposadr: 0, tendon: Some(t), site: None, body: None, actadr }),
                    _ => m.unsupported.push("a tendon transmission this port did not record".into()),
                }
                continue;
            }
            let joint = match a.moment.as_slice() {
                [(dof, _)] => m.jnt_dofadr.iter().position(|&d| d == *dof).filter(|&j| matches!(m.jnt_type[j], KinJointKind::Hinge | KinJointKind::Slide)),
                _ => None,
            };
            match joint {
                Some(j) => m.actuators.push(NativeActuator { act: a.clone(), dofadr: m.jnt_dofadr[j], qposadr: m.jnt_qposadr[j], tendon: None, site: None, body: None, actadr }),
                None => m.unsupported.push("an actuator that is not a joint, tendon or site transmission".into()),
            }
        }
        m.na = na;
        m.unsupported.sort();
        m.unsupported.dedup();
        m
    }

    /// `mj_comPos`: `cinert`, `cdof` and `subtree_com`.
    fn com_pos(&self, k: &KinState) -> (Vec<[f64; 10]>, Vec<[f64; 6]>, Vec<[f64; 3]>) {
        let nbody = self.kin.bodies.len();
        let mut sub: Vec<[f64; 3]> = (0..nbody).map(|i| k.xipos[i].map(|x| x * self.body_mass[i])).collect();
        for i in (1..nbody).rev() {
            let p = self.kin.bodies[i].parent;
            let c = sub[i];
            for a in 0..3 {
                sub[p][a] += c[a];
            }
        }
        for i in 0..nbody {
            if self.body_subtreemass[i] < MJ_MINVAL {
                sub[i] = k.xipos[i];
            } else {
                let inv = 1.0 / self.body_subtreemass[i];
                sub[i] = sub[i].map(|x| x * inv);
            }
        }
        let mut cinert = vec![[0.0; 10]; nbody];
        for i in 1..nbody {
            let r = &sub[self.body_rootid[i]];
            let off = [k.xipos[i][0] - r[0], k.xipos[i][1] - r[1], k.xipos[i][2] - r[2]];
            cinert[i] = inert_com(&self.body_inertia[i], &k.ximat[i], &off, self.body_mass[i]);
        }
        let mut cdof = vec![[0.0; 6]; self.dof_bodyid.len()];
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let i = self.jnt_bodyid[j];
            let da = self.jnt_dofadr[j];
            let r = &sub[self.body_rootid[i]];
            let a = &k.xanchor[j];
            let off = [r[0] - a[0], r[1] - a[1], r[2] - a[2]];
            let rot = |cdof: &mut Vec<[f64; 6]>, first: usize| {
                for c in 0..3 {
                    let axis = [k.xmat[i][c], k.xmat[i][c + 3], k.xmat[i][c + 6]];
                    cdof[first + c] = dof_com(&axis, Some(&off));
                }
            };
            match kind {
                KinJointKind::Free => {
                    for c in 0..3 {
                        cdof[da + c] = [0.0; 6];
                        cdof[da + c][3 + c] = 1.0;
                    }
                    rot(&mut cdof, da + 3);
                }
                KinJointKind::Ball => rot(&mut cdof, da),
                KinJointKind::Slide => cdof[da] = dof_com(&k.xaxis[j], None),
                KinJointKind::Hinge => cdof[da] = dof_com(&k.xaxis[j], Some(&off)),
            }
        }
        (cinert, cdof, sub)
    }

    /// The first half of `mj_crb`: each body's composite inertia, children into parents from the last back.
    fn composite(&self, cinert: &[[f64; 10]]) -> Vec<[f64; 10]> {
        let mut crb = cinert.to_vec();
        for i in (1..crb.len()).rev() {
            let p = self.kin.bodies[i].parent;
            if p > 0 {
                let c = crb[i];
                for a in 0..10 {
                    crb[p][a] += c[a];
                }
            }
        }
        crb
    }

    /// **`mj_kinematics` → `mj_comPos` → `mj_crb` → `mj_factorM` → `mj_comVel` → `mj_rne`** (without
    /// acceleration) at `qpos`, `qvel`.
    pub(crate) fn forward(&self, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64]) -> SmoothData {
        self.forward_at(0.0, qpos, qvel, ctrl, act)
    }

    /// [`SmoothModel::forward`] at `d->time`, which a `mujoco.pid` slew limiter reads (no previous setpoint
    /// at `t = 0`).
    pub(crate) fn forward_at(&self, time: f64, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64]) -> SmoothData {
        self.forward_with(time, qpos, qvel, ctrl, act, &std::collections::BTreeMap::new())
    }

    /// [`SmoothModel::forward_at`] with the adhesion actuators' moments, by actuator index — which MuJoCo
    /// builds from the constraint rows (`mj_transmission` runs after `mj_makeConstraint`), so the caller
    /// passes them in from a first pass; an adhesion actuator without one has no moment.
    pub(crate) fn forward_with(&self, time: f64, qpos: &[f64], qvel: &[f64], ctrl: &[f64], act: &[f64], body_moments: &std::collections::BTreeMap<usize, (Vec<usize>, Vec<f64>)>) -> SmoothData {
        let nbody = self.kin.bodies.len();
        let nv = self.dof_bodyid.len();
        let kin = self.kin.state(qpos);
        let (cinert, cdof, subtree_com) = self.com_pos(&kin);
        // `mj_crb`
        let crb = self.composite(&cinert);
        let mut m = vec![0.0; self.m_colind.len()];
        for i in 0..nv {
            let adr = self.m_rowadr[i];
            if self.dof_simplenum[i] != 0 {
                m[adr] = self.dof_m0[i];
                continue;
            }
            let mut madr = adr + self.m_rownnz[i] - 1;
            m[madr] = self.dof_armature[i];
            let buf = mul_inert_vec(&crb[self.dof_bodyid[i]], &cdof[i]);
            let mut j = i as i32;
            while j >= 0 {
                m[madr] += dot6(&cdof[j as usize], &buf);
                madr = madr.wrapping_sub(1);
                j = self.dof_parentid[j as usize];
            }
        }
        // `mj_factorI`: L'DL in place, rows from the last back
        let mut qld = m.clone();
        let mut qld_diag_inv = vec![0.0; nv];
        for k in (0..nv).rev() {
            let start = self.m_rowadr[k];
            let diag = self.m_rownnz[k] - 1;
            let end = start + diag;
            if qld[end] < MJ_MINVAL {
                qld[end] = MJ_MINVAL;
            }
            let inv_d = 1.0 / qld[end];
            qld_diag_inv[k] = inv_d;
            for adr in (start..end).rev() {
                let i = self.m_colind[adr];
                let scl = -qld[adr] * inv_d;
                let (ri, n) = (self.m_rowadr[i], self.m_rownnz[i]);
                for t in 0..n {
                    qld[ri + t] = qld[start + t].mul_add(scl, qld[ri + t]);
                }
            }
            for t in 0..diag {
                qld[start + t] *= inv_d;
            }
        }
        // `mj_comVel`
        let mut cvel = vec![[0.0; 6]; nbody];
        let mut cdof_dot = vec![[0.0; 6]; nv];
        for i in 1..nbody {
            let mut v = cvel[self.kin.bodies[i].parent];
            let (bda, n) = (self.body_dofadr[i], self.body_dofnum[i]);
            let mut j = 0;
            while j < n {
                let kind = self.jnt_type[self.dof_jntid[bda + j]];
                if kind == KinJointKind::Free {
                    let t = mul_dof_vec(&cdof[bda..bda + 3], &qvel[bda..bda + 3]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    for c in 0..3 {
                        cdof_dot[bda + c] = [0.0; 6];
                    }
                    j += 3;
                }
                if matches!(kind, KinJointKind::Free | KinJointKind::Ball) {
                    for c in 0..3 {
                        cdof_dot[bda + j + c] = cross_motion(&v, &cdof[bda + j + c]);
                    }
                    let t = mul_dof_vec(&cdof[bda + j..bda + j + 3], &qvel[bda + j..bda + j + 3]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    j += 3;
                } else {
                    cdof_dot[bda + j] = cross_motion(&v, &cdof[bda + j]);
                    let t = mul_dof_vec(&cdof[bda + j..bda + j + 1], &qvel[bda + j..bda + j + 1]);
                    for a in 0..6 {
                        v[a] += t[a];
                    }
                    j += 1;
                }
            }
            cvel[i] = v;
        }
        // `mj_rne` without acceleration
        let mut cacc = vec![[0.0; 6]; nbody];
        cacc[0] = [0.0, 0.0, 0.0, -self.gravity[0], -self.gravity[1], -self.gravity[2]];
        let mut cfrc = vec![[0.0; 6]; nbody];
        for i in 1..nbody {
            let (bda, n) = (self.body_dofadr[i], self.body_dofnum[i]);
            let t = mul_dof_vec(&cdof_dot[bda..bda + n], &qvel[bda..bda + n]);
            let p = cacc[self.kin.bodies[i].parent];
            cacc[i] = std::array::from_fn(|a| p[a] + t[a]);
            let mut f = mul_inert_vec(&cinert[i], &cacc[i]);
            let t = mul_inert_vec(&cinert[i], &cvel[i]);
            let t1 = cross_force(&cvel[i], &t);
            for a in 0..6 {
                f[a] += t1[a];
            }
            cfrc[i] = f;
        }
        cfrc[0] = [0.0; 6];
        for i in (1..nbody).rev() {
            let p = self.kin.bodies[i].parent;
            if p > 0 {
                let c = cfrc[i];
                for a in 0..6 {
                    cfrc[p][a] += c[a];
                }
            }
        }
        let qfrc_bias: Vec<f64> = (0..nv).map(|i| dot6(&cdof[i], &cfrc[self.dof_bodyid[i]])).collect();
        // `mj_tendon` (fixed tendons: `L += prm·q`, and `mju_combineSparseInc` puts each `prm` in its
        // column; spatial ones along their path) and `ten_velocity`, `mju_mulMatVecSparse`
        let (ten_length, ten_j) = self.tendon(qpos, &kin, &cdof, &subtree_com);
        let ten_velocity: Vec<f64> = self.tendons.iter().zip(&ten_j).map(|(t, j)| t.as_ref().map_or(0.0, |t| dot_sparse(j, qvel, &t.colind))).collect();
        // `mj_passive`: joint springs, dof dampers, gravity compensation
        let mut qfrc_spring = vec![0.0; nv];
        let mut qfrc_damper = vec![0.0; nv];
        let mut qfrc_gravcomp = vec![0.0; nv];
        for (j, &kind) in self.jnt_type.iter().enumerate() {
            let k = self.jnt_stiffness[j];
            if k == 0.0 || !matches!(kind, KinJointKind::Hinge | KinJointKind::Slide) {
                continue;
            }
            let x = qpos[self.jnt_qposadr[j]] - self.jnt_springref[j];
            qfrc_spring[self.jnt_dofadr[j]] = -x * k;
        }
        for i in 0..nv {
            if self.dof_damping[i] != 0.0 {
                qfrc_damper[i] = -qvel[i] * self.dof_damping[i];
            }
        }
        // tendon springs (a dead band between the two spring lengths) and dampers, through `ten_J`
        for (i, t) in self.tendons.iter().enumerate() {
            let Some(t) = t else { continue };
            if t.stiffness == 0.0 && t.damping == 0.0 {
                continue;
            }
            let (length, [lower, upper]) = (ten_length[i], t.lengthspring);
            let x = if length > upper { length - upper } else if length < lower { length - lower } else { 0.0 };
            let frc_spring = -x * t.stiffness;
            let frc_damper = -ten_velocity[i] * t.damping;
            if frc_spring != 0.0 || frc_damper != 0.0 {
                for (&k, &jv) in t.colind.iter().zip(&ten_j[i]) {
                    qfrc_spring[k] = jv.mul_add(frc_spring, qfrc_spring[k]);
                    qfrc_damper[k] = jv.mul_add(frc_damper, qfrc_damper[k]);
                }
            }
        }
        let has_gravcomp = self.body_gravcomp.iter().any(|&g| g != 0.0) && (self.gravity[0].abs() + self.gravity[1].abs() + self.gravity[2].abs()) != 0.0;
        if has_gravcomp {
            for b in 1..nbody {
                if self.body_gravcomp[b] == 0.0 {
                    continue;
                }
                let s = -(self.body_mass[b] * self.body_gravcomp[b]);
                let force = self.gravity.map(|g| g * s);
                self.apply_ft(&cdof, &subtree_com, &force, &kin.xipos[b], b, &mut qfrc_gravcomp);
            }
        }
        // `mj_fluid`: the inertia-box model on every body with mass
        let mut qfrc_fluid = vec![0.0; nv];
        let has_fluid = self.density != 0.0 || self.viscosity != 0.0;
        if has_fluid {
            for b in 0..nbody {
                if self.body_mass[b] < MJ_MINVAL {
                    continue;
                }
                self.inertia_box_fluid(b, &kin, &cvel, &cdof, &subtree_com, &mut qfrc_fluid);
            }
        }
        let mut qfrc_passive: Vec<f64> = (0..nv).map(|i| qfrc_spring[i] + qfrc_damper[i]).collect();
        if has_fluid {
            for i in 0..nv {
                qfrc_passive[i] += qfrc_fluid[i];
            }
        }
        if has_gravcomp {
            for i in 0..nv {
                qfrc_passive[i] += qfrc_gravcomp[i];
            }
        }
        // `mj_transmission` and the actuator velocities
        let nu = self.actuators.len();
        let mut actuator_length = vec![0.0; nu];
        let mut actuator_velocity = vec![0.0; nu];
        let mut moment = Vec::with_capacity(nu);
        for (i, a) in self.actuators.iter().enumerate() {
            let g = a.act.gear;
            if a.body.is_some() {
                actuator_length[i] = 0.0;
                moment.push(body_moments.get(&i).cloned().unwrap_or_default());
                let (cols, vals) = &moment[i];
                actuator_velocity[i] = dot_sparse(vals, qvel, cols);
                continue;
            }
            if let Some((site, gear)) = &a.site {
                // a site: no length; the moment is `J'·wrench`, the gear turned into the world by the site
                let b = site.body;
                let (sp, sm) = crate::mujoco_kinematics::mj_local2global((&kin.xpos[b], &kin.xquat[b], &kin.xmat[b]), &(kin.xipos[b], kin.ximat[b]), &site.pos, &site.quat, site.sameframe);
                let (jp, jr) = self.jac_at(&cdof, &subtree_com, &sp, b, true);
                let wf = crate::mujoco_efc::mat_vec3(&sm, &[gear[0], gear[1], gear[2]]);
                let wt = crate::mujoco_efc::mat_vec3(&sm, &[gear[3], gear[4], gear[5]]);
                let row = mul_mat_t_vec3(&jp, &wf);
                let tmp = mul_mat_t_vec3(&jr, &wt);
                let (mut cols, mut vals) = (Vec::new(), Vec::new());
                for c in 0..nv {
                    let v = row[c] + tmp[c];
                    if v != 0.0 {
                        cols.push(c);
                        vals.push(v);
                    }
                }
                actuator_length[i] = 0.0;
                moment.push((cols, vals));
                let (cols, vals) = &moment[i];
                actuator_velocity[i] = dot_sparse(vals, qvel, cols);
                continue;
            }
            match a.tendon {
                Some(t) => {
                    actuator_length[i] = ten_length[t] * g;
                    let cols = self.tendons[t].as_ref().map_or(Vec::new(), |t| t.colind.clone());
                    moment.push((cols, ten_j[t].iter().map(|v| v * g).collect::<Vec<f64>>()));
                }
                None => {
                    actuator_length[i] = qpos[a.qposadr] * g;
                    moment.push((vec![a.dofadr], vec![g]));
                }
            }
            let (cols, vals) = &moment[i];
            actuator_velocity[i] = dot_sparse(vals, qvel, cols);
        }
        // `mj_fwdActuation`
        let ctrl_raw = ctrl;
        let ctrl: Vec<f64> = (0..nu).map(|i| {
            let c = ctrl.get(i).copied().unwrap_or(0.0);
            match self.actuators[i].act.ctrlrange {
                Some([lo, hi]) => clip(c, lo, hi),
                None => c,
            }
        }).collect();
        let mut act_dot = vec![0.0; self.na];
        for (i, a) in self.actuators.iter().enumerate() {
            let Some(adr) = a.actadr else { continue };
            // a `mujoco.pid` plugin's slots: `Pid::ActDot`, on the control as `d->ctrl` holds it
            if let Some(p) = &a.act.pid {
                let n = a.act.actnum();
                let own: Vec<f64> = (adr..adr + n).map(|j| act.get(j).copied().unwrap_or(0.0)).collect();
                let v = p.act_dot(&a.act, actuator_length[i], ctrl_raw.get(i).copied().unwrap_or(0.0), &own, time, self.timestep);
                act_dot[adr..adr + n].copy_from_slice(&v);
                continue;
            }
            let (x, u) = (act.get(adr).copied().unwrap_or(0.0), ctrl[i]);
            use crate::mujoco_actuator::ActDyn;
            act_dot[adr] = match a.act.dynamics {
                ActDyn::None => 0.0,
                ActDyn::Integrator => u,
                ActDyn::Filter | ActDyn::FilterExact => (u - x) / MJ_MINVAL.max(a.act.dynprm[0]),
                ActDyn::Muscle => muscle_dynamics(u, x, &a.act.dynprm),
            };
        }
        let mut actuator_force = vec![0.0; nu];
        for (i, a) in self.actuators.iter().enumerate() {
            use crate::mujoco_actuator::{ActBias, ActDyn, ActGain};
            // `Pid::Compute` (MuJoCo runs the plugins after the builtin laws; the outputs are disjoint),
            // then `forcerange`
            if let Some(p) = &a.act.pid {
                let adr = a.actadr.unwrap_or(0);
                let own: Vec<f64> = (adr..adr + a.act.actnum()).map(|j| act.get(j).copied().unwrap_or(0.0)).collect();
                let mut f = p.force(&a.act, actuator_length[i], actuator_velocity[i], ctrl_raw.get(i).copied().unwrap_or(0.0), &own, time, self.timestep);
                if let Some([lo, hi]) = a.act.forcerange {
                    f = clip(f, lo, hi);
                }
                actuator_force[i] = f;
                continue;
            }
            let (p, len, vel) = (&a.act.gainprm, actuator_length[i], actuator_velocity[i]);
            let gain = match a.act.gain {
                ActGain::Fixed => p[0],
                ActGain::Affine => p[2].mul_add(vel, p[1].mul_add(len, p[0])),
                ActGain::Muscle => muscle_gain(len, vel, &a.act.lengthrange, a.act.acc0, p),
            };
            let input = match (a.act.dynamics, a.actadr) {
                (ActDyn::None, _) | (_, None) => ctrl[i],
                (_, Some(adr)) => {
                    let x = act.get(adr).copied().unwrap_or(0.0);
                    if a.act.actearly { self.next_activation(a, x, act_dot[adr]) } else { x }
                }
            };
            let mut f = gain * input;
            let b = &a.act.biasprm;
            let bias = match a.act.bias {
                ActBias::None => 0.0,
                ActBias::Affine => b[2].mul_add(vel, b[1].mul_add(len, b[0])),
                ActBias::Muscle => muscle_bias(len, &a.act.lengthrange, a.act.acc0, b),
            };
            f += bias;
            if let Some([lo, hi]) = a.act.forcerange {
                f = clip(f, lo, hi);
            }
            actuator_force[i] = f;
        }
        // `mju_mulMatTVecSparse`, then the joints' `actuatorfrcrange`
        let mut qfrc_actuator = vec![0.0; nv];
        for (i, (cols, vals)) in moment.iter().enumerate() {
            let f = actuator_force[i];
            if f != 0.0 {
                for (&c, &v) in cols.iter().zip(vals) {
                    qfrc_actuator[c] = v.mul_add(f, qfrc_actuator[c]);
                }
            }
        }
        for (j, r) in self.jnt_actfrcrange.iter().enumerate() {
            if let (Some([lo, hi]), Some(&d)) = (r, self.jnt_dofadr.get(j)) {
                qfrc_actuator[d] = clip(qfrc_actuator[d], *lo, *hi);
            }
        }
        // `mj_fwdAcceleration`
        let qfrc_smooth: Vec<f64> = (0..nv).map(|i| (qfrc_passive[i] - qfrc_bias[i]) + 0.0 + qfrc_actuator[i]).collect();
        let qacc_smooth = self.solve_ld(&qld, &qld_diag_inv, &qfrc_smooth);
        SmoothData {
            kin, subtree_com, cinert, cdof, crb, m, qld, qld_diag_inv, cvel, cdof_dot, qfrc_bias,
            qfrc_spring, qfrc_damper, qfrc_gravcomp, qfrc_fluid, qfrc_passive, ten_length, ten_j, ten_velocity, moment, actuator_length, actuator_velocity, actuator_force, act_dot, qfrc_actuator, qfrc_smooth, qacc_smooth,
        }
    }

    /// **`mj_inertiaBoxFluidModel`** for body `b`: the body's velocity at its centre of mass in its inertial
    /// frame (`mj_objectVelocity`, less the wind), the viscous and blunt-body drag of the box with the body's
    /// inertia, turned back into the world and applied at `xipos` (`mj_applyFT`, force then torque).
    #[allow(clippy::too_many_arguments)]
    fn inertia_box_fluid(&self, b: usize, kin: &KinState, cvel: &[[f64; 6]], cdof: &[[f64; 6]], subtree_com: &[[f64; 3]], qfrc: &mut [f64]) {
        use crate::mujoco_efc::{mat_t_vec3, mat_vec3};
        let inertia = &self.body_inertia[b];
        let mass = self.body_mass[b];
        let bx = |a: f64, c: f64, d: f64| (fmax(MJ_MINVAL, a + c - d) / mass * 6.0).sqrt();
        let bb = [bx(inertia[1], inertia[2], inertia[0]), bx(inertia[0], inertia[2], inertia[1]), bx(inertia[0], inertia[1], inertia[2])];
        // `mju_transformSpatial` of a motion vector to `xipos`, in the inertial frame
        let (pos, com, rot) = (&kin.xipos[b], &subtree_com[self.body_rootid[b]], &kin.ximat[b]);
        let transform = |v: &[f64; 6]| -> [f64; 6] {
            let dif = [pos[0] - com[0], pos[1] - com[1], pos[2] - com[2]];
            let c = cross(&dif, &[v[0], v[1], v[2]]);
            let lin = [v[3] - c[0], v[4] - c[1], v[5] - c[2]];
            let a = mat_t_vec3(rot, &[v[0], v[1], v[2]]);
            let l = mat_t_vec3(rot, &lin);
            [a[0], a[1], a[2], l[0], l[1], l[2]]
        };
        // `mj_objectVelocity`: zero on a body welded to the world
        let mut w = b;
        while w != 0 && self.body_dofnum[w] == 0 {
            w = self.kin.bodies[w].parent;
        }
        let mut lvel = if self.body_dofnum[w] == 0 { [0.0; 6] } else { transform(&cvel[b]) };
        let lwind = transform(&[0.0, 0.0, 0.0, self.wind[0], self.wind[1], self.wind[2]]);
        for k in 3..6 {
            lvel[k] -= lwind[k];
        }
        let mut lfrc = [0.0f64; 6];
        if self.viscosity > 0.0 {
            let diam = (bb[0] + bb[1] + bb[2]) / 3.0;
            let ca = -std::f64::consts::PI * diam * diam * diam * self.viscosity;
            let cl = -3.0 * std::f64::consts::PI * diam * self.viscosity;
            for k in 0..3 {
                lfrc[k] = lvel[k] * ca;
                lfrc[3 + k] = lvel[3 + k] * cl;
            }
        }
        if self.density > 0.0 {
            let rho = self.density;
            for (k, (p, q)) in [(1, 2), (0, 2), (0, 1)].into_iter().enumerate() {
                let v = lvel[3 + k];
                lfrc[3 + k] = (-(0.5 * rho * bb[p] * bb[q] * v.abs())).mul_add(v, lfrc[3 + k]);
            }
            for (k, (p, q)) in [(1, 2), (0, 2), (0, 1)].into_iter().enumerate() {
                let s = (bb[p] * bb[p] * bb[p]).mul_add(bb[p], bb[q] * bb[q] * bb[q] * bb[q]);
                let v = lvel[k];
                lfrc[k] -= rho * bb[k] * s * v.abs() * v / 64.0;
            }
        }
        let torque = mat_vec3(rot, &[lfrc[0], lfrc[1], lfrc[2]]);
        let force = mat_vec3(rot, &[lfrc[3], lfrc[4], lfrc[5]]);
        // `mj_applyFT`, dense: `J_p'·f` added, then `J_r'·τ`
        let (jp, jr) = self.jac_at(cdof, subtree_com, pos, b, true);
        let qf = mul_mat_t_vec3(&jp, &force);
        for (q, f) in qfrc.iter_mut().zip(&qf) {
            *q += f;
        }
        let qt = mul_mat_t_vec3(&jr, &torque);
        for (q, t) in qfrc.iter_mut().zip(&qt) {
            *q += t;
        }
    }

    /// `mj_tendon` for the fixed tendons: `ten_length` and each row of `ten_J` on its `colind`.
    pub(crate) fn tendon(&self, qpos: &[f64], k: &KinState, cdof: &[[f64; 6]], subtree_com: &[[f64; 3]]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut len = vec![0.0; self.tendons.len()];
        let mut jac = vec![Vec::new(); self.tendons.len()];
        for (i, t) in self.tendons.iter().enumerate() {
            let Some(t) = t else { continue };
            if let Some(objs) = &t.spatial {
                (len[i], jac[i]) = self.spatial_tendon(objs, &t.colind, k, cdof, subtree_com);
                continue;
            }
            let mut row = vec![0.0f64; t.colind.len()];
            for &(qadr, dof, prm) in &t.path {
                len[i] = prm.mul_add(qpos[qadr], len[i]);
                if let Some(k) = t.colind.iter().position(|&c| c == dof) {
                    row[k] = prm.mul_add(1.0, row[k]);
                }
            }
            jac[i] = row;
        }
        (len, jac)
    }

    /// `mj_nextActivation` for an actuator with one activation.
    pub(crate) fn next_activation(&self, a: &NativeActuator, act: f64, act_dot: f64) -> f64 {
        let h = self.timestep;
        let next = if a.act.dynamics == crate::mujoco_actuator::ActDyn::FilterExact {
            let tau = MJ_MINVAL.max(a.act.dynprm[0]);
            (act_dot * tau).mul_add(1.0 - (-h / tau).exp(), act)
        } else {
            act_dot.mul_add(h, act)
        };
        match a.act.actrange {
            Some([lo, hi]) => clip(next, lo, hi),
            None => next,
        }
    }

    /// `mj_applyFT` with a force and a zero torque: `J'·f` at `point` on `body`, into `qfrc`.
    #[allow(clippy::too_many_arguments)]
    fn apply_ft(&self, cdof: &[[f64; 6]], subtree_com: &[[f64; 3]], force: &[f64; 3], point: &[f64; 3], body: usize, qfrc: &mut [f64]) {
        let r = &subtree_com[self.body_rootid[body]];
        let offset = [point[0] - r[0], point[1] - r[1], point[2] - r[2]];
        // the weld body's last dof, then its ancestors
        let mut b = body;
        while b != 0 && self.body_dofnum[b] == 0 {
            b = self.kin.bodies[b].parent;
        }
        if self.body_dofnum[b] == 0 {
            return;
        }
        let mut i = (self.body_dofadr[b] + self.body_dofnum[b] - 1) as i32;
        while i >= 0 {
            let c = &cdof[i as usize];
            let t = cross(&[c[0], c[1], c[2]], &offset);
            let jacp = [c[3] + t[0], c[4] + t[1], c[5] + t[2]];
            // `mju_mulMatTVec` over the three rows, a zero force component skipped, then the (zero) torque
            let mut q = 0.0;
            for k in 0..3 {
                if force[k] != 0.0 {
                    q = jacp[k].mul_add(force[k], q);
                }
            }
            qfrc[i as usize] += q;
            qfrc[i as usize] += 0.0;
            i = self.dof_parentid[i as usize];
        }
    }

    /// `mj_factorI` on a matrix in `M`'s layout: the factor and its inverse diagonal.
    pub(crate) fn factor_i(&self, m: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let nv = self.dof_bodyid.len();
        let mut qld = m.to_vec();
        let mut diag_inv = vec![0.0; nv];
        for k in (0..nv).rev() {
            let start = self.m_rowadr[k];
            let diag = self.m_rownnz[k] - 1;
            let end = start + diag;
            if qld[end] < MJ_MINVAL {
                qld[end] = MJ_MINVAL;
            }
            let inv_d = 1.0 / qld[end];
            diag_inv[k] = inv_d;
            for adr in (start..end).rev() {
                let i = self.m_colind[adr];
                let scl = -qld[adr] * inv_d;
                let (ri, n) = (self.m_rowadr[i], self.m_rownnz[i]);
                for t in 0..n {
                    qld[ri + t] = qld[start + t].mul_add(scl, qld[ri + t]);
                }
            }
            for t in 0..diag {
                qld[start + t] *= inv_d;
            }
        }
        (qld, diag_inv)
    }

    /// `mj_solveLD` for one vector on a given factor.
    pub(crate) fn solve_ld_pub(&self, qld: &[f64], diag_inv: &[f64], y: &[f64]) -> Vec<f64> {
        self.solve_ld(qld, diag_inv, y)
    }

    /// `mj_solveM` for one vector, on the factor in `sd`.
    pub(crate) fn solve_m(&self, sd: &SmoothData, y: &[f64]) -> Vec<f64> {
        self.solve_ld(&sd.qld, &sd.qld_diag_inv, y)
    }

    /// `mj_solveLD` for one vector.
    fn solve_ld(&self, qld: &[f64], diag_inv: &[f64], y: &[f64]) -> Vec<f64> {
        let nv = y.len();
        let mut x = y.to_vec();
        for i in (0..nv).rev() {
            if self.m_rownnz[i] == 1 {
                continue;
            }
            let xi = x[i];
            if xi != 0.0 {
                let (start, end) = (self.m_rowadr[i], self.m_rowadr[i] + self.m_rownnz[i] - 1);
                for adr in start..end {
                    let c = self.m_colind[adr];
                    x[c] = (-qld[adr]).mul_add(xi, x[c]);
                }
            }
        }
        for i in 0..nv {
            x[i] *= diag_inv[i];
        }
        for i in 0..nv {
            if self.m_rownnz[i] == 1 {
                continue;
            }
            let d = self.m_rownnz[i] - 1;
            let adr = self.m_rowadr[i];
            x[i] -= dot_sparse(&qld[adr..adr + d], &x, &self.m_colind[adr..adr + d]);
        }
        x
    }
}

impl SmoothData {
    /// Every array under its `mjData` name, flattened as MuJoCo stores it.
    pub(crate) fn arrays(&self) -> Vec<(&'static str, Vec<f64>)> {
        fn flat<const N: usize>(v: &[[f64; N]]) -> Vec<f64> {
            v.iter().flatten().copied().collect()
        }
        vec![
            ("xpos", flat(&self.kin.xpos)),
            ("xquat", flat(&self.kin.xquat)),
            ("xmat", flat(&self.kin.xmat)),
            ("xipos", flat(&self.kin.xipos)),
            ("ximat", flat(&self.kin.ximat)),
            ("xanchor", flat(&self.kin.xanchor)),
            ("xaxis", flat(&self.kin.xaxis)),
            ("subtree_com", flat(&self.subtree_com)),
            ("cinert", flat(&self.cinert)),
            ("cdof", flat(&self.cdof)),
            ("crb", flat(&self.crb)),
            ("M", self.m.clone()),
            ("qLD", self.qld.clone()),
            ("qLDiagInv", self.qld_diag_inv.clone()),
            ("cvel", flat(&self.cvel)),
            ("cdof_dot", flat(&self.cdof_dot)),
            ("qfrc_bias", self.qfrc_bias.clone()),
            ("qfrc_spring", self.qfrc_spring.clone()),
            ("qfrc_damper", self.qfrc_damper.clone()),
            ("qfrc_gravcomp", self.qfrc_gravcomp.clone()),
            ("qfrc_fluid", self.qfrc_fluid.clone()),
            ("qfrc_passive", self.qfrc_passive.clone()),
            ("ten_length", self.ten_length.clone()),
            ("ten_velocity", self.ten_velocity.clone()),
            ("actuator_length", self.actuator_length.clone()),
            ("actuator_velocity", self.actuator_velocity.clone()),
            ("actuator_force", self.actuator_force.clone()),
            ("act_dot", self.act_dot.clone()),
            ("qfrc_actuator", self.qfrc_actuator.clone()),
            ("qfrc_smooth", self.qfrc_smooth.clone()),
            ("qacc_smooth", self.qacc_smooth.clone()),
        ]
    }
}

impl SmoothModel {
    /// Every compiled array under its `mjModel` name (integers as floats).
    pub(crate) fn arrays(&self) -> Vec<(&'static str, Vec<f64>)> {
        let f = |v: &[usize]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        let g = |v: &[i32]| v.iter().map(|&x| x as f64).collect::<Vec<_>>();
        vec![
            ("body_parentid", f(&self.kin.bodies.iter().map(|b| b.parent).collect::<Vec<_>>())),
            ("body_rootid", f(&self.body_rootid)),
            // -1 on a body with none, as MuJoCo stores it
            ("body_jntadr", self.body_jntadr.iter().zip(&self.kin.bodies).map(|(&a, b)| if b.joints.is_empty() { -1.0 } else { a as f64 }).collect()),
            ("body_dofadr", self.body_dofadr.iter().zip(&self.body_dofnum).map(|(&a, &n)| if n == 0 { -1.0 } else { a as f64 }).collect()),
            ("body_dofnum", f(&self.body_dofnum)),
            ("body_simple", g(&self.body_simple)),
            ("body_mass", self.body_mass.clone()),
            ("body_subtreemass", self.body_subtreemass.clone()),
            ("body_inertia", self.body_inertia.iter().flatten().copied().collect()),
            ("jnt_dofadr", f(&self.jnt_dofadr)),
            ("jnt_bodyid", f(&self.jnt_bodyid)),
            ("dof_bodyid", f(&self.dof_bodyid)),
            ("dof_jntid", f(&self.dof_jntid)),
            ("dof_parentid", g(&self.dof_parentid)),
            ("dof_simplenum", g(&self.dof_simplenum)),
            ("dof_M0", self.dof_m0.clone()),
            ("dof_armature", self.dof_armature.clone()),
            ("M_rownnz", f(&self.m_rownnz)),
            ("M_rowadr", f(&self.m_rowadr)),
            ("M_colind", f(&self.m_colind)),
            ("gravity", self.gravity.to_vec()),
        ]
    }
}
