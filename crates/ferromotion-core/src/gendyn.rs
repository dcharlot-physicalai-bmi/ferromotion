//! **Generic-scalar rigid-body dynamics** — FK, RNEA, the mass matrix, and forward dynamics over
//! ANY scalar implementing [`Real`], so forward-mode dual numbers (or any custom scalar) flow
//! *through the dynamics*. This is the Rust answer to Pinocchio's templated-scalar/CasADi pipeline:
//! instantiate at `f64` for plain numerics (bit-matching the `dynamics` module, see tests), or at a
//! dual type for exact derivatives of torques/accelerations with respect to **states or inertial
//! parameters** — the seed of gradient-based system identification and real-to-sim calibration.
//!
//! Deliberate scope: the generic path targets *differentiation*, not raw speed — forward dynamics
//! here is O(n³) (mass matrix by RNEA columns + generic Cholesky), which is the right trade at
//! calibration scale. The f64 O(n) Featherstone ABA in [`crate::aba`] remains the fast plain-number
//! path. Everything is fixed-size-array algebra over `T` — no trait-ecosystem dependency, wasm-clean.

use crate::dynamics::LinkInertia;
use crate::{JointKind, Robot};

/// The minimal scalar contract the dynamics need: field arithmetic plus `sin`/`cos` (joint
/// rotations, Rodrigues) and `sqrt` (Cholesky). Implemented for `f64` here and for dual-number
/// types downstream (the trait lives in core so consumers can implement it for their scalars).
pub trait Real:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    fn from_f64(v: f64) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn sqrt(self) -> Self;
}

impl Real for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn sin(self) -> Self {
        f64::sin(self)
    }
    fn cos(self) -> Self {
        f64::cos(self)
    }
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// 3-vector over `T`.
#[derive(Clone, Copy, Debug)]
pub struct V3<T>(pub [T; 3]);

/// Row-major 3×3 matrix over `T`.
#[derive(Clone, Copy, Debug)]
pub struct M3<T>(pub [[T; 3]; 3]);

impl<T: Real> V3<T> {
    pub fn zero() -> Self {
        V3([T::from_f64(0.0); 3])
    }
    pub fn from_f64(v: [f64; 3]) -> Self {
        V3([T::from_f64(v[0]), T::from_f64(v[1]), T::from_f64(v[2])])
    }
    pub fn add(self, o: Self) -> Self {
        V3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
    pub fn sub(self, o: Self) -> Self {
        V3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
    pub fn scale(self, s: T) -> Self {
        V3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
    pub fn dot(self, o: Self) -> T {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    pub fn cross(self, o: Self) -> Self {
        V3([
            self.0[1] * o.0[2] - self.0[2] * o.0[1],
            self.0[2] * o.0[0] - self.0[0] * o.0[2],
            self.0[0] * o.0[1] - self.0[1] * o.0[0],
        ])
    }
}

impl<T: Real> M3<T> {
    pub fn zero() -> Self {
        M3([[T::from_f64(0.0); 3]; 3])
    }
    pub fn identity() -> Self {
        let (o, l) = (T::from_f64(0.0), T::from_f64(1.0));
        M3([[l, o, o], [o, l, o], [o, o, l]])
    }
    pub fn from_f64(m: [[f64; 3]; 3]) -> Self {
        let c = |r: [f64; 3]| [T::from_f64(r[0]), T::from_f64(r[1]), T::from_f64(r[2])];
        M3([c(m[0]), c(m[1]), c(m[2])])
    }
    pub fn add(self, o: Self) -> Self {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[i][j] + o.0[i][j];
            }
        }
        r
    }
    pub fn scale(self, s: T) -> Self {
        let mut r = self;
        for row in &mut r.0 {
            for v in row.iter_mut() {
                *v = *v * s;
            }
        }
        r
    }
    pub fn transpose(self) -> Self {
        let mut r = self;
        for i in 0..3 {
            for j in 0..3 {
                r.0[i][j] = self.0[j][i];
            }
        }
        r
    }
    pub fn mul_v(self, v: V3<T>) -> V3<T> {
        V3([
            self.0[0][0] * v.0[0] + self.0[0][1] * v.0[1] + self.0[0][2] * v.0[2],
            self.0[1][0] * v.0[0] + self.0[1][1] * v.0[1] + self.0[1][2] * v.0[2],
            self.0[2][0] * v.0[0] + self.0[2][1] * v.0[1] + self.0[2][2] * v.0[2],
        ])
    }
    pub fn mul_m(self, o: Self) -> Self {
        let mut r = Self::zero();
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = T::from_f64(0.0);
                for k in 0..3 {
                    acc = acc + self.0[i][k] * o.0[k][j];
                }
                r.0[i][j] = acc;
            }
        }
        r
    }
}

/// Rotation about a (constant, unit) axis by generic angle `q` — Rodrigues' formula:
/// `R = I + sinq·K + (1−cosq)·K²` with `K = skew(axis)`.
pub fn rot_axis<T: Real>(axis: V3<T>, q: T) -> M3<T> {
    let (s, c) = (q.sin(), q.cos());
    let one = T::from_f64(1.0);
    let k = skew(axis);
    let k2 = k.mul_m(k);
    M3::identity().add(k.scale(s)).add(k2.scale(one - c))
}

pub fn skew<T: Real>(v: V3<T>) -> M3<T> {
    let o = T::from_f64(0.0);
    M3([
        [o, -v.0[2], v.0[1]],
        [v.0[2], o, -v.0[0]],
        [-v.0[1], v.0[0], o],
    ])
}

/// One joint+link of the generic model. Joint geometry AND inertial parameters are all `T`, so a
/// dual seeded in `mass`/`com`/`inertia` differentiates the dynamics with respect to that
/// parameter — the calibration path.
#[derive(Clone)]
pub struct GenLink<T> {
    pub origin_r: M3<T>,
    pub origin_p: V3<T>,
    pub axis: V3<T>,
    pub revolute: bool,
    pub mass: T,
    pub com: V3<T>,
    pub inertia: M3<T>,
}

/// A serial-chain model over generic scalars, with gravity.
#[derive(Clone)]
pub struct GenModel<T> {
    pub links: Vec<GenLink<T>>,
    pub gravity: V3<T>,
}

impl<T: Real> GenModel<T> {
    /// Lift an f64 [`Robot`] + link inertias into the generic scalar type.
    pub fn from_robot(robot: &Robot, inertia: &[LinkInertia], gravity: [f64; 3]) -> Self {
        let links = robot
            .joints
            .iter()
            .zip(inertia)
            .map(|(j, li)| {
                let rm = *j.origin.rotation.to_rotation_matrix().matrix();
                GenLink {
                    origin_r: M3::from_f64([
                        [rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]],
                        [rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]],
                        [rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]],
                    ]),
                    origin_p: V3::from_f64([
                        j.origin.translation.vector[0],
                        j.origin.translation.vector[1],
                        j.origin.translation.vector[2],
                    ]),
                    axis: V3::from_f64([j.axis[0], j.axis[1], j.axis[2]]),
                    revolute: j.kind == JointKind::Revolute,
                    mass: T::from_f64(li.mass),
                    com: V3::from_f64([li.com[0], li.com[1], li.com[2]]),
                    inertia: M3::from_f64([
                        [li.inertia[(0, 0)], li.inertia[(0, 1)], li.inertia[(0, 2)]],
                        [li.inertia[(1, 0)], li.inertia[(1, 1)], li.inertia[(1, 2)]],
                        [li.inertia[(2, 0)], li.inertia[(2, 1)], li.inertia[(2, 2)]],
                    ]),
                }
            })
            .collect();
        GenModel { links, gravity: V3::from_f64(gravity) }
    }

    pub fn dof(&self) -> usize {
        self.links.len()
    }

    /// Per-joint relative transform (rotation `frame i → i−1`, origin of frame i in frame i−1).
    fn joint_transform(&self, i: usize, qi: T) -> (M3<T>, V3<T>) {
        let l = &self.links[i];
        if l.revolute {
            (l.origin_r.mul_m(rot_axis(l.axis, qi)), l.origin_p)
        } else {
            (l.origin_r, l.origin_p.add(l.origin_r.mul_v(l.axis.scale(qi))))
        }
    }

    /// Forward kinematics: end of the chain (before any tool offset) — position only.
    pub fn fk_position(&self, q: &[T]) -> V3<T> {
        let mut r = M3::identity();
        let mut p = V3::zero();
        for (i, &qi) in q.iter().enumerate().take(self.dof()) {
            let (jr, jp) = self.joint_transform(i, qi);
            p = p.add(r.mul_v(jp));
            r = r.mul_m(jr);
        }
        p
    }

    /// Inverse dynamics via Recursive Newton–Euler — the generic mirror of
    /// [`crate::dynamics::inverse_dynamics`] (same Luh–Walker–Paul 3-vector formulation, verified
    /// to agree at `f64` to machine precision).
    pub fn rnea(&self, q: &[T], qd: &[T], qdd: &[T]) -> Vec<T> {
        let n = self.dof();
        let zero = T::from_f64(0.0);
        let two = T::from_f64(2.0);
        let mut rr = Vec::with_capacity(n);
        let mut pp = Vec::with_capacity(n);
        for i in 0..n {
            let (jr, jp) = self.joint_transform(i, q[i]);
            rr.push(jr);
            pp.push(jp);
        }
        let mut omega = vec![V3::zero(); n];
        let mut omegad = vec![V3::zero(); n];
        let mut vd = vec![V3::zero(); n];
        let mut ff = vec![V3::zero(); n];
        let mut nn = vec![V3::zero(); n];
        let (mut pw, mut pwd) = (V3::zero(), V3::zero());
        let mut pvd = self.gravity.scale(T::from_f64(-1.0));
        for i in 0..n {
            let rt = rr[i].transpose();
            let z = self.links[i].axis;
            let base = rt.mul_v(pvd.add(pwd.cross(pp[i])).add(pw.cross(pw.cross(pp[i]))));
            if self.links[i].revolute {
                omega[i] = rt.mul_v(pw).add(z.scale(qd[i]));
                omegad[i] = rt.mul_v(pwd).add(rt.mul_v(pw).cross(z.scale(qd[i]))).add(z.scale(qdd[i]));
                vd[i] = base;
            } else {
                omega[i] = rt.mul_v(pw);
                omegad[i] = rt.mul_v(pwd);
                vd[i] = base.add(omega[i].cross(z.scale(qd[i])).scale(two)).add(z.scale(qdd[i]));
            }
            let li = &self.links[i];
            let vdc = vd[i].add(omegad[i].cross(li.com)).add(omega[i].cross(omega[i].cross(li.com)));
            ff[i] = vdc.scale(li.mass);
            nn[i] = li.inertia.mul_v(omegad[i]).add(omega[i].cross(li.inertia.mul_v(omega[i])));
            pw = omega[i];
            pwd = omegad[i];
            pvd = vd[i];
        }
        let mut tau = vec![zero; n];
        let mut f_next = V3::zero();
        let mut n_next = V3::zero();
        for i in (0..n).rev() {
            let (rr_next, p_next) = if i + 1 < n { (rr[i + 1], pp[i + 1]) } else { (M3::identity(), V3::zero()) };
            let rf = rr_next.mul_v(f_next);
            let f_i = rf.add(ff[i]);
            let n_i = nn[i]
                .add(rr_next.mul_v(n_next))
                .add(self.links[i].com.cross(ff[i]))
                .add(p_next.cross(rf));
            tau[i] = if self.links[i].revolute { n_i.dot(self.links[i].axis) } else { f_i.dot(self.links[i].axis) };
            f_next = f_i;
            n_next = n_i;
        }
        tau
    }

    /// Joint-space mass matrix by RNEA columns (`q̈ = eⱼ`, no gravity/velocity), row-major `n×n`.
    pub fn mass_matrix(&self, q: &[T]) -> Vec<T> {
        let n = self.dof();
        let zero = T::from_f64(0.0);
        let zeros = vec![zero; n];
        let no_g = GenModel { links: self.links.clone(), gravity: V3::zero() };
        let mut m = vec![zero; n * n];
        for j in 0..n {
            let mut qdd = vec![zero; n];
            qdd[j] = T::from_f64(1.0);
            let col = no_g.rnea(q, &zeros, &qdd);
            for i in 0..n {
                m[i * n + j] = col[i];
            }
        }
        m
    }

    /// Forward dynamics `q̈ = M(q)⁻¹ (τ − bias)` with the bias from RNEA at `q̈ = 0`, solved by a
    /// generic Cholesky (M is SPD). O(n³) — the differentiation-first trade, stated in the module doc.
    pub fn forward_dynamics(&self, q: &[T], qd: &[T], tau: &[T]) -> Vec<T> {
        let n = self.dof();
        let zero = T::from_f64(0.0);
        let zeros = vec![zero; n];
        let bias = self.rnea(q, qd, &zeros);
        let m = self.mass_matrix(q);
        let rhs: Vec<T> = (0..n).map(|i| tau[i] - bias[i]).collect();
        cholesky_solve(&m, &rhs, n)
    }
}

/// Solve `A x = b` for SPD `A` (row-major n×n) via Cholesky, generically over `T`.
pub fn cholesky_solve<T: Real>(a: &[T], b: &[T], n: usize) -> Vec<T> {
    let zero = T::from_f64(0.0);
    let mut l = vec![zero; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum = sum - l[i * n + k] * l[j * n + k];
            }
            if i == j {
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    let mut y = vec![zero; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum = sum - l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    let mut x = vec![zero; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum = sum - l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::{forward_dynamics, inverse_dynamics};
    use crate::{Iso, Joint, Robot};
    use nalgebra::{Matrix3, Translation3, UnitQuaternion, Vector3};

    fn test_robot() -> (Robot, Vec<LinkInertia>) {
        // mixed revolute/prismatic 5-DOF chain with rotated origins and off-axis inertias
        let mk_origin = |xyz: [f64; 3], rpy: [f64; 3]| {
            Iso::from_parts(
                Translation3::new(xyz[0], xyz[1], xyz[2]),
                UnitQuaternion::from_euler_angles(rpy[0], rpy[1], rpy[2]),
            )
        };
        let joints = vec![
            Joint::revolute(mk_origin([0.0, 0.0, 0.3], [0.0, 0.0, 0.4]), Vector3::z()),
            Joint::revolute(mk_origin([0.1, 0.0, 0.2], [0.3, 0.0, 0.0]), Vector3::y()),
            Joint::prismatic(mk_origin([0.0, 0.05, 0.25], [0.0, 0.2, 0.0]), Vector3::x()),
            Joint::revolute(mk_origin([0.2, 0.0, 0.1], [0.0, 0.0, -0.3]), Vector3::y()),
            Joint::revolute(mk_origin([0.0, 0.0, 0.15], [0.1, 0.0, 0.0]), Vector3::z()),
        ];
        let inertia: Vec<LinkInertia> = (0..5)
            .map(|i| {
                let f = i as f64;
                LinkInertia {
                    mass: 1.5 + 0.3 * f,
                    com: Vector3::new(0.02 * f, -0.01, 0.05 + 0.01 * f),
                    inertia: Matrix3::new(
                        0.02 + 0.005 * f, 0.001, 0.002,
                        0.001, 0.03 + 0.002 * f, 0.0015,
                        0.002, 0.0015, 0.025,
                    ),
                }
            })
            .collect();
        (Robot { joints, ee_offset: Iso::identity() }, inertia)
    }

    /// The generic path instantiated at f64 must match the reference `dynamics` module to machine
    /// precision — RNEA, mass matrix, and forward dynamics.
    #[test]
    fn f64_instantiation_matches_the_reference_dynamics() {
        let (robot, inertia) = test_robot();
        let g = Vector3::new(0.0, 0.0, -9.81);
        let q = [0.3, -0.7, 0.12, 1.1, -0.4];
        let qd = [0.5, -0.2, 0.3, -0.8, 0.6];
        let qdd = [1.2, 0.4, -0.9, 0.3, -1.1];
        let m = GenModel::<f64>::from_robot(&robot, &inertia, [0.0, 0.0, -9.81]);

        let tau_ref = inverse_dynamics(&robot, &inertia, &q, &qd, &qdd, g);
        let tau_gen = m.rnea(&q, &qd, &qdd);
        for (a, b) in tau_ref.iter().zip(&tau_gen) {
            assert!((a - b).abs() < 1e-10, "RNEA mismatch: {a} vs {b}");
        }

        let mm_ref = crate::dynamics::mass_matrix(&robot, &inertia, &q);
        let mm_gen = m.mass_matrix(&q);
        for i in 0..5 {
            for j in 0..5 {
                assert!((mm_ref[(i, j)] - mm_gen[i * 5 + j]).abs() < 1e-10, "M mismatch at ({i},{j})");
            }
        }

        let tau_in = [0.7, -0.3, 0.5, 0.2, -0.6];
        let qdd_ref = forward_dynamics(&robot, &inertia, &q, &qd, &tau_in, g);
        let qdd_gen = m.forward_dynamics(&q, &qd, &tau_in);
        for (a, b) in qdd_ref.iter().zip(&qdd_gen) {
            assert!((a - b).abs() < 1e-8, "FD mismatch: {a} vs {b}");
        }
    }

    /// FK position at f64 matches the Robot's FK (chain end, no tool offset).
    #[test]
    fn f64_fk_matches_robot_fk() {
        let (robot, inertia) = test_robot();
        let m = GenModel::<f64>::from_robot(&robot, &inertia, [0.0, 0.0, -9.81]);
        let q = [0.3, -0.7, 0.12, 1.1, -0.4];
        let p_ref = robot.frame_pose(&q, robot.dof()).translation.vector;
        let p_gen = m.fk_position(&q);
        for k in 0..3 {
            assert!((p_ref[k] - p_gen.0[k]).abs() < 1e-12, "FK mismatch at {k}");
        }
    }
}
