//! **Geometric tracking control on SE(3)** (Lee, Leok & McClamroch, CDC 2010) — the coordinate-free,
//! almost-globally exponentially stable *feedback* controller for a quadrotor. It tracks a full
//! position+attitude trajectory directly on `SE(3)`, with no local coordinates and no small-angle
//! assumption, so it survives large attitude errors and aggressive flight where a linearized attitude
//! loop degrades. This is the closed-loop counterpart to the crate's `quadrotor` differential-flatness
//! generator (which only produces the *feedforward* trajectory) and its attitude-only `es_mpc`.
//!
//! The law: a desired thrust force `F_des = −k_x e_x − k_v e_v + mg e₃ + m a_d` sets the thrust magnitude
//! `f = F_des·(R e₃)` and the desired body-z `b₃_d = F_des/‖F_des‖`; with a heading `b₁_d` this builds the
//! desired attitude `R_d`, and the moment `M = −k_R e_R − k_Ω e_Ω + Ω×JΩ` closes the attitude loop, where
//! the attitude error `e_R = ½(R_dᵀR − RᵀR_d)^∨` lives on `SO(3)`. Verified: hover is an exact equilibrium
//! (`f = mg`, `M = 0`), the `SO(3)` error vanishes at alignment, and the closed loop converges from a
//! perturbed pose. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

fn hat(w: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -w.z, w.y, w.z, 0.0, -w.x, -w.y, w.x, 0.0)
}
fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

/// Full quadrotor state: world position `x`, velocity `v`, rotation `r` (body→world), body-frame angular
/// velocity `omega`.
#[derive(Clone, Copy, Debug)]
pub struct QuadFullState {
    pub x: Vector3<f64>,
    pub v: Vector3<f64>,
    pub r: Matrix3<f64>,
    pub omega: Vector3<f64>,
}

/// A tracking reference: position, velocity, acceleration, and a heading direction `b1` (plus the
/// reference angular rates, zero for a static/hover reference).
#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub x: Vector3<f64>,
    pub v: Vector3<f64>,
    pub a: Vector3<f64>,
    pub b1: Vector3<f64>,
    pub omega: Vector3<f64>,
    pub omega_dot: Vector3<f64>,
}

impl Reference {
    /// A static hover reference at position `x` with heading `+x`.
    pub fn hover(x: Vector3<f64>) -> Reference {
        Reference { x, v: Vector3::zeros(), a: Vector3::zeros(), b1: Vector3::x(), omega: Vector3::zeros(), omega_dot: Vector3::zeros() }
    }
}

/// The geometric SE(3) tracking controller for a quadrotor with mass `m`, inertia `j`, gravity `g`.
#[derive(Clone, Copy, Debug)]
pub struct GeometricSe3 {
    pub m: f64,
    pub j: Matrix3<f64>,
    pub g: f64,
    pub kx: f64,
    pub kv: f64,
    pub kr: f64,
    pub kw: f64,
}

impl GeometricSe3 {
    /// Compute `(thrust f, moment M)` for the current state and reference.
    pub fn control(&self, s: &QuadFullState, r: &Reference) -> (f64, Vector3<f64>) {
        let e3 = Vector3::z();
        let e_x = s.x - r.x;
        let e_v = s.v - r.v;
        // desired thrust force and body-z axis
        let f_des = -self.kx * e_x - self.kv * e_v + self.m * self.g * e3 + self.m * r.a;
        let f = f_des.dot(&(s.r * e3)); // thrust = projection onto current body-z
        let b3d = f_des.normalize();
        // desired attitude from (b3d, heading b1)
        let b2d = b3d.cross(&r.b1).normalize();
        let b1d = b2d.cross(&b3d);
        let rd = Matrix3::from_columns(&[b1d, b2d, b3d]);
        // attitude errors on SO(3)
        let e_r = 0.5 * vee(&(rd.transpose() * s.r - s.r.transpose() * rd));
        let e_om = s.omega - s.r.transpose() * rd * r.omega;
        // moment (static/slowly-varying reference: the feedforward attitude-rate terms use omega_d)
        let ff = s.omega.cross(&(self.j * s.omega))
            - self.j * (hat(&s.omega) * s.r.transpose() * rd * r.omega - s.r.transpose() * rd * r.omega_dot);
        let m = -self.kr * e_r - self.kw * e_om + ff;
        (f, m)
    }

    /// One step of the closed-loop quadrotor dynamics under `(f, M)` (semi-implicit Euler): `m v̇ = f·R e₃ −
    /// mg e₃`, `Ṙ = R Ω̂`, `J Ω̇ = M − Ω×JΩ`.
    pub fn step(&self, s: &QuadFullState, f: f64, m: &Vector3<f64>, dt: f64) -> QuadFullState {
        let e3 = Vector3::z();
        let acc = (f * (s.r * e3) - self.m * self.g * e3) / self.m;
        let jinv = self.j.try_inverse().unwrap();
        let omega_dot = jinv * (m - s.omega.cross(&(self.j * s.omega)));
        let v = s.v + acc * dt;
        let x = s.x + v * dt;
        let omega = s.omega + omega_dot * dt;
        // rotation update on SO(3): R⁺ = R·exp(Ω̂ dt), re-orthonormalized
        let r_raw = s.r * (Matrix3::identity() + hat(&omega) * dt);
        let svd = r_raw.svd(true, true);
        let r = svd.u.unwrap() * svd.v_t.unwrap();
        QuadFullState { x, v, r, omega }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> GeometricSe3 {
        GeometricSe3 {
            m: 1.0,
            j: Matrix3::from_diagonal(&Vector3::new(0.02, 0.02, 0.04)),
            g: 9.81,
            kx: 8.0,
            kv: 4.0,
            kr: 1.8,
            kw: 0.35,
        }
    }

    fn hover_state(x: Vector3<f64>) -> QuadFullState {
        QuadFullState { x, v: Vector3::zeros(), r: Matrix3::identity(), omega: Vector3::zeros() }
    }

    #[test]
    fn hover_is_an_exact_equilibrium() {
        // THE INVARIANT. At the hover reference the thrust exactly balances gravity and the moment is zero.
        let c = controller();
        let s = hover_state(Vector3::new(1.0, -2.0, 0.5));
        let r = Reference::hover(Vector3::new(1.0, -2.0, 0.5));
        let (f, m) = c.control(&s, &r);
        assert!((f - c.m * c.g).abs() < 1e-9, "thrust should equal weight mg = {}: {f}", c.m * c.g);
        assert!(m.norm() < 1e-9, "moment should vanish at hover: {m}");
    }

    #[test]
    fn the_attitude_error_vanishes_at_alignment_and_grows_with_offset() {
        let c = controller();
        // aligned ⇒ e_R = 0
        let s = hover_state(Vector3::zeros());
        let (_f, m) = c.control(&s, &Reference::hover(Vector3::zeros()));
        assert!(m.norm() < 1e-9);
        // a yawed reference heading ⇒ nonzero corrective moment about z
        let mut r = Reference::hover(Vector3::zeros());
        r.b1 = Vector3::new((0.4_f64).cos(), (0.4_f64).sin(), 0.0); // 0.4 rad yaw target
        let (_f2, m2) = c.control(&s, &r);
        assert!(m2.z.abs() > 1e-3, "a yaw offset should produce a z-moment: {m2}");
    }

    #[test]
    fn the_closed_loop_converges_from_a_perturbed_pose() {
        // THE HEADLINE. From a displaced, tilted, spinning start the geometric controller drives the full
        // SE(3) state to the hover reference — position and attitude error → 0.
        let c = controller();
        let target = Vector3::new(0.0, 0.0, 1.0);
        let r = Reference::hover(target);
        // perturbed: offset position, some velocity, a 0.5 rad tilt about x, and spin
        let tilt = Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.5_f64.cos(), -0.5_f64.sin(), 0.0, 0.5_f64.sin(), 0.5_f64.cos());
        let mut s = QuadFullState { x: Vector3::new(0.6, -0.4, 1.3), v: Vector3::new(0.2, 0.1, -0.1), r: tilt, omega: Vector3::new(0.3, -0.2, 0.1) };
        for _ in 0..16000 {
            let (f, m) = c.control(&s, &r);
            s = c.step(&s, f, &m, 5e-4);
        }
        let pos_err = (s.x - target).norm();
        let att_err = 0.5 * (Matrix3::identity() - s.r).trace(); // Ψ(R, I) ≥ 0, = 0 iff R = I
        assert!(pos_err < 1e-2, "position should converge: {pos_err}");
        assert!(att_err.abs() < 1e-3, "attitude should converge to level: Ψ = {att_err}");
        assert!(s.omega.norm() < 1e-2, "angular velocity should settle: {}", s.omega.norm());
    }
}
