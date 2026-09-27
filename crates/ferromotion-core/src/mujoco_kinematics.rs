//! **MuJoCo's forward kinematics (`mj_kinematics`), in MuJoCo's own arithmetic** — every body's `xpos`,
//! `xquat`, `xmat`, its inertial frame `xipos`/`ximat`, and every geom's `geom_xpos`/`geom_xmat`, from
//! `qpos` in MuJoCo's layout, bit for bit what MuJoCo 3.13.0 computes (`engine_core_smooth.c`,
//! `mj_local2Global` in `engine_core_util.c`, the `mji_*`/`mju_*` spatial helpers).
//!
//! The rest of this crate places bodies through its own kinematic tree, in its own coordinates (a free
//! base as Euler angles, jointless bodies welded into the link they ride on), which is the same pose to
//! within a few ulps. That is not enough everywhere: a collision between two nearly parallel surfaces
//! decides which corner, or which side, from the last bits of the geom poses, and MuJoCo then reports a
//! different contact. The collision pass therefore takes its geom poses from here when it has `qpos`.
//!
//! ⛔ Every helper is contracted as clang contracts MuJoCo's C on arm64 (see [`crate::mujoco_ccd`]): the left
//! product of `a*b + c*d` is fused, each later product of a chain is fused into the running sum.

/// `sin` and `cos` of ONE argument as MuJoCo's Apple build computes them.
///
/// ⛔⛔ clang turns every `sin(x)`, `cos(x)` pair into a single call to `__sincos_stret` (the arm64 wheel
/// calls it from `mju_axisAngle2Quat`, `mju_quatIntegrate`, `mj_kinematics1`, `mju_wrap` and the compiler's
/// orientation code, and never pairs `_sin` with `_cos`), and its `sin` is not always `sin`'s: at
/// −0.6985299965 the last bit differs. So a MuJoCo pair is computed here by the same call.
#[cfg(target_vendor = "apple")]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    #[repr(C)]
    struct SinCos {
        sin: f64,
        cos: f64,
    }
    unsafe extern "C" {
        fn __sincos_stret(x: f64) -> SinCos;
    }
    // SAFETY: a pure libm function of one double, returning two doubles in registers (an HFA)
    let r = unsafe { __sincos_stret(x) };
    (r.sin, r.cos)
}

/// `sin` and `cos` of one argument; off Apple platforms, the two libm calls.
#[cfg(not(target_vendor = "apple"))]
pub(crate) fn sincos(x: f64) -> (f64, f64) {
    (x.sin(), x.cos())
}

/// `mjtSameFrame`: where [`mj_local2global`] takes an element's frame from instead of composing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum SameFrame {
    #[default]
    None,
    Body,
    BodyRot,
    Inertia,
    InertiaRot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KinJointKind {
    Free,
    Ball,
    Slide,
    Hinge,
}

/// One joint as the compiled model holds it (`jnt_type`, `jnt_pos`, `jnt_axis`, `jnt_qposadr`, `qpos0`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct KinJoint {
    pub(crate) kind: KinJointKind,
    pub(crate) pos: [f64; 3],
    pub(crate) axis: [f64; 3],
    pub(crate) qposadr: usize,
    pub(crate) qpos0: f64,
}

/// One body: `body_parentid`, `body_pos`, `body_quat`, `body_ipos`, `body_iquat`, `body_sameframe`, joints.
#[derive(Clone, Debug)]
pub(crate) struct KinBody {
    pub(crate) parent: usize,
    pub(crate) pos: [f64; 3],
    pub(crate) quat: [f64; 4],
    pub(crate) ipos: [f64; 3],
    pub(crate) iquat: [f64; 4],
    pub(crate) sameframe: SameFrame,
    pub(crate) joints: Vec<KinJoint>,
}

impl Default for KinBody {
    fn default() -> Self {
        KinBody { parent: 0, pos: [0.0; 3], quat: [1.0, 0.0, 0.0, 0.0], ipos: [0.0; 3], iquat: [1.0, 0.0, 0.0, 0.0], sameframe: SameFrame::Body, joints: Vec::new() }
    }
}

/// One geom: `geom_bodyid`, `geom_pos`, `geom_quat`, `geom_sameframe`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KinGeom {
    pub(crate) body: usize,
    pub(crate) pos: [f64; 3],
    pub(crate) quat: [f64; 4],
    pub(crate) sameframe: SameFrame,
}

/// The compiled model's kinematic data, in MuJoCo's body and geom order (body 0 is the world).
#[derive(Clone, Debug, Default)]
pub(crate) struct MjKinematics {
    pub(crate) bodies: Vec<KinBody>,
    pub(crate) geoms: Vec<KinGeom>,
}

/// A frame in the world as `mjData` holds it: position and row-major rotation matrix.
pub(crate) type XFrame = ([f64; 3], [f64; 9]);

/// `mji_mulMatVec3`, contracted.
fn mul_mat_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [m[2].mul_add(v[2], m[0].mul_add(v[0], m[1] * v[1])), m[5].mul_add(v[2], m[3].mul_add(v[0], m[4] * v[1])), m[8].mul_add(v[2], m[6].mul_add(v[0], m[7] * v[1]))]
}

/// `mji_mulQuat` / `mju_mulQuat` (no normalisation), contracted.
pub(crate) fn mul_quat(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        (-a[3]).mul_add(b[3], (-a[2]).mul_add(b[2], a[0].mul_add(b[0], -(a[1] * b[1])))),
        (-a[3]).mul_add(b[2], a[2].mul_add(b[3], a[0].mul_add(b[1], a[1] * b[0]))),
        a[3].mul_add(b[1], a[2].mul_add(b[0], a[0].mul_add(b[2], -(a[1] * b[3])))),
        a[3].mul_add(b[0], (-a[2]).mul_add(b[1], a[0].mul_add(b[3], a[1] * b[2]))),
    ]
}

/// `mji_rotVecQuat`, contracted (no zero-vector shortcut, unlike `mju_rotVecQuat`).
fn rot_vec_quat(v: &[f64; 3], q: &[f64; 4]) -> [f64; 3] {
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

/// `mju_normalize4`: below `mjMINVAL` the unit quaternion; within `mjMINVAL` of 1 left alone; otherwise
/// MULTIPLIED by the reciprocal of the norm.
pub(crate) fn normalize4(q: &mut [f64; 4]) {
    let norm = q[3].mul_add(q[3], q[2].mul_add(q[2], q[0].mul_add(q[0], q[1] * q[1]))).sqrt();
    if norm < 1e-15 {
        *q = [1.0, 0.0, 0.0, 0.0];
    } else if (norm - 1.0).abs() > 1e-15 {
        let inv = 1.0 / norm;
        for x in q.iter_mut() {
            *x *= inv;
        }
    }
}

/// `mju_quat2Mat` (its products are separate statements, so nothing fuses).
pub(crate) fn quat2mat(q: &[f64; 4]) -> [f64; 9] {
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    }
    let (q00, q01, q02, q03) = (q[0] * q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3]);
    let (q11, q12, q13) = (q[1] * q[1], q[1] * q[2], q[1] * q[3]);
    let (q22, q23, q33) = (q[2] * q[2], q[2] * q[3], q[3] * q[3]);
    [
        q00 + q11 - q22 - q33,
        2.0 * (q12 - q03),
        2.0 * (q13 + q02),
        2.0 * (q12 + q03),
        q00 - q11 + q22 - q33,
        2.0 * (q23 - q01),
        2.0 * (q13 - q02),
        2.0 * (q23 + q01),
        q00 - q11 - q22 + q33,
    ]
}

/// `mji_axisAngle2Quat`.
fn axis_angle2quat(axis: &[f64; 3], angle: f64) -> [f64; 4] {
    if angle == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let (s, c) = sincos(angle * 0.5);
    [c, axis[0] * s, axis[1] * s, axis[2] * s]
}

/// `mj_local2Global`: an element's world frame from its body's, as its `sameframe` tag says.
fn mj_local2global(body: (&[f64; 3], &[f64; 4], &[f64; 9]), inertial: &XFrame, pos: &[f64; 3], quat: &[f64; 4], sf: SameFrame) -> XFrame {
    let (xpos, xquat, xmat) = body;
    let p = match sf {
        SameFrame::None | SameFrame::BodyRot | SameFrame::InertiaRot => {
            let v = mul_mat_vec3(xmat, pos);
            [v[0] + xpos[0], v[1] + xpos[1], v[2] + xpos[2]]
        }
        SameFrame::Body => *xpos,
        SameFrame::Inertia => inertial.0,
    };
    let m = match sf {
        SameFrame::None => quat2mat(&mul_quat(xquat, quat)),
        SameFrame::Body | SameFrame::BodyRot => *xmat,
        SameFrame::Inertia | SameFrame::InertiaRot => inertial.1,
    };
    (p, m)
}

impl MjKinematics {
    /// `mj_kinematics`: every geom's world frame (`geom_xpos`, `geom_xmat`) at `qpos` (MuJoCo's layout).
    pub(crate) fn geom_frames(&self, qpos: &[f64]) -> Vec<XFrame> {
        self.frames(qpos).0
    }

    /// `mj_kinematics`: every geom's world frame, and every body's inertial frame (`xipos`, `ximat`).
    pub(crate) fn frames(&self, qpos: &[f64]) -> (Vec<XFrame>, Vec<XFrame>) {
        let n = self.bodies.len();
        let mut xpos = vec![[0.0f64; 3]; n];
        let mut xquat = vec![[1.0f64, 0.0, 0.0, 0.0]; n];
        let mut xmat = vec![[1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; n];
        let mut xi: Vec<XFrame> = vec![([0.0; 3], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]); n];
        for i in 1..n {
            let b = &self.bodies[i];
            let (mut p, mut q);
            if b.joints.len() == 1 && b.joints[0].kind == KinJointKind::Free {
                let a = b.joints[0].qposadr;
                p = [qpos[a], qpos[a + 1], qpos[a + 2]];
                q = [qpos[a + 3], qpos[a + 4], qpos[a + 5], qpos[a + 6]];
                normalize4(&mut q);
            } else {
                let pid = b.parent;
                if pid != 0 {
                    let v = mul_mat_vec3(&xmat[pid], &b.pos);
                    p = [v[0] + xpos[pid][0], v[1] + xpos[pid][1], v[2] + xpos[pid][2]];
                    q = mul_quat(&xquat[pid], &b.quat);
                } else {
                    p = b.pos;
                    q = b.quat;
                }
                for j in &b.joints {
                    let xaxis = rot_vec_quat(&j.axis, &q);
                    let r = rot_vec_quat(&j.pos, &q);
                    let xanchor = [r[0] + p[0], r[1] + p[1], r[2] + p[2]];
                    match j.kind {
                        KinJointKind::Slide => {
                            let s = qpos[j.qposadr] - j.qpos0;
                            p = [xaxis[0].mul_add(s, p[0]), xaxis[1].mul_add(s, p[1]), xaxis[2].mul_add(s, p[2])];
                        }
                        KinJointKind::Ball | KinJointKind::Hinge => {
                            let qloc = if j.kind == KinJointKind::Ball {
                                let a = j.qposadr;
                                let mut l = [qpos[a], qpos[a + 1], qpos[a + 2], qpos[a + 3]];
                                normalize4(&mut l);
                                l
                            } else {
                                axis_angle2quat(&j.axis, qpos[j.qposadr] - j.qpos0)
                            };
                            q = mul_quat(&q, &qloc);
                            let v = rot_vec_quat(&j.pos, &q);
                            p = [xanchor[0] - v[0], xanchor[1] - v[1], xanchor[2] - v[2]];
                        }
                        // a free joint that is not its body's only joint: the compiler refuses the model
                        KinJointKind::Free => {}
                    }
                }
            }
            normalize4(&mut q);
            xpos[i] = p;
            xquat[i] = q;
            xmat[i] = quat2mat(&q);
        }
        for i in 1..n {
            let b = &self.bodies[i];
            let world: XFrame = ([0.0; 3], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
            xi[i] = mj_local2global((&xpos[i], &xquat[i], &xmat[i]), &world, &b.ipos, &b.iquat, b.sameframe);
        }
        let geoms = self.geoms.iter().map(|g| mj_local2global((&xpos[g.body], &xquat[g.body], &xmat[g.body]), &xi[g.body], &g.pos, &g.quat, g.sameframe)).collect();
        (geoms, xi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one argument this port was caught out by: `euler="-1.397059993 …"` halves to −0.6985299965, and
    /// there `__sincos_stret`'s sine is one ulp from `sin`'s. MuJoCo's arm64 build gets the former, so a
    /// separate `sin`/`cos` pair put `hello_robot_stretch_3`'s camera bodies 1.8e-16 off.
    #[test]
    #[cfg(target_vendor = "apple")]
    fn sincos_is_the_single_call_mujoco_makes() {
        let (s, c) = sincos(-0.6985299965);
        assert_eq!(s.to_bits(), (-0.6430926709021718f64).to_bits(), "{s:?}");
        assert_eq!(c.to_bits(), 0.7657883628208977f64.to_bits(), "{c:?}");
        assert_ne!(s.to_bits(), std::hint::black_box(-0.6985299965f64).sin().to_bits(), "the two calls agree here now: pick another argument");
    }
}
