//! **Lie-group variational integrator** for rigid-body attitude (Lee, Leok & McClamroch) — a
//! structure-preserving, symplectic, singularity-free integrator on `SO(3)`. Derived by discretizing
//! Hamilton's principle directly on the group, it keeps the rotation *exactly* orthogonal (no
//! re-normalization, no gimbal singularity), conserves angular momentum to machine precision, and
//! conserves energy with **no secular drift** over arbitrarily long horizons — unlike a generic
//! integrator that slowly gains or loses energy.
//!
//! Each step solves the Moser-Veselov equation `h·Π̂ = F J_d − J_d Fᵀ` for the relative rotation
//! `F ∈ SO(3)` (with the modified inertia `J_d = ½ tr(J) I − J`), then updates `R_{k+1} = R_k F`,
//! `Π_{k+1} = Fᵀ Π_k + h·M`. This is the geometric-mechanics counterpart to the momentum-based
//! [`crate::RigidBody`] integrator. Pure `nalgebra` → WASM-clean.

use nalgebra::{Matrix3, Vector3};

fn hat(v: Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

fn vee(m: &Matrix3<f64>) -> Vector3<f64> {
    Vector3::new(m[(2, 1)] - m[(1, 2)], m[(0, 2)] - m[(2, 0)], m[(1, 0)] - m[(0, 1)]) * 0.5
}

fn exp_so3(w: Vector3<f64>) -> Matrix3<f64> {
    let t = w.norm();
    if t < 1e-12 {
        Matrix3::identity() + hat(w)
    } else {
        let k = hat(w / t);
        Matrix3::identity() + t.sin() * k + (1.0 - t.cos()) * k * k
    }
}

/// A rigid body integrated with the Lie-group variational integrator. Attitude `r` (world←body),
/// body-frame angular momentum `pi`, and principal moments of inertia `inertia`.
#[derive(Clone, Debug)]
pub struct LgviBody {
    pub r: Matrix3<f64>,
    pub pi: Vector3<f64>,
    pub inertia: Vector3<f64>,
}

impl LgviBody {
    pub fn new(inertia: Vector3<f64>, omega0: Vector3<f64>) -> Self {
        Self { r: Matrix3::identity(), pi: inertia.component_mul(&omega0), inertia }
    }

    /// Modified (Moser-Veselov) inertia `J_d = ½ tr(J) I − J`.
    fn jd(&self) -> Matrix3<f64> {
        let (a, b, c) = (self.inertia.x, self.inertia.y, self.inertia.z);
        let tr = a + b + c;
        Matrix3::from_diagonal(&Vector3::new(0.5 * tr - a, 0.5 * tr - b, 0.5 * tr - c))
    }

    /// Solve `h·Π̂ = F J_d − J_d Fᵀ` for the incremental rotation vector `f` (with `F = exp(f̂)`),
    /// by Newton with a finite-difference Jacobian.
    fn solve_f(&self, h: f64) -> Vector3<f64> {
        let jd = self.jd();
        let omega = Vector3::new(self.pi.x / self.inertia.x, self.pi.y / self.inertia.y, self.pi.z / self.inertia.z);
        let mut f = h * omega; // good initial guess
        let residual = |f: Vector3<f64>| -> Vector3<f64> {
            let fm = exp_so3(f);
            vee(&(fm * jd - jd * fm.transpose())) - h * self.pi
        };
        for _ in 0..30 {
            let g = residual(f);
            if g.norm() < 1e-14 {
                break;
            }
            let eps = 1e-7;
            let mut jac = Matrix3::zeros();
            for i in 0..3 {
                let mut fp = f;
                fp[i] += eps;
                let col = (residual(fp) - g) / eps;
                jac.set_column(i, &col);
            }
            let step = jac.try_inverse().map(|inv| inv * g).unwrap_or(g);
            f -= step;
        }
        f
    }

    /// Advance one step of size `h` under an optional body-frame external moment `moment`.
    pub fn step(&mut self, h: f64, moment: Vector3<f64>) {
        let f = self.solve_f(h);
        let fm = exp_so3(f);
        self.r *= fm; // R_{k+1} = R_k F   (stays orthogonal by construction)
        self.pi = fm.transpose() * self.pi + h * moment;
    }

    /// Rotational kinetic energy `½ Πᵀ J⁻¹ Π`.
    pub fn energy(&self) -> f64 {
        0.5 * (self.pi.x * self.pi.x / self.inertia.x + self.pi.y * self.pi.y / self.inertia.y + self.pi.z * self.pi.z / self.inertia.z)
    }

    /// Spatial angular momentum `R·Π` (conserved for a free body).
    pub fn spatial_momentum(&self) -> Vector3<f64> {
        self.r * self.pi
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_body_conserves_energy_and_momentum_without_drift() {
        // An asymmetric top tumbling freely (Dzhanibekov-style): a stringent conservation test.
        let mut b = LgviBody::new(Vector3::new(1.0, 2.0, 3.0), Vector3::new(0.6, 0.05, 0.4));
        let (e0, l0) = (b.energy(), b.spatial_momentum());
        let (h, steps) = (0.01, 60_000);
        let (mut worst_e, mut worst_l, mut worst_orth) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..steps {
            b.step(h, Vector3::zeros());
            worst_e = worst_e.max((b.energy() - e0).abs() / e0);
            worst_l = worst_l.max((b.spatial_momentum() - l0).norm());
            worst_orth = worst_orth.max((b.r.transpose() * b.r - Matrix3::identity()).norm());
        }
        // Energy: bounded, no secular drift, over 60k steps.
        assert!(worst_e < 1e-6, "energy drifted: relative {worst_e:.2e}");
        // Spatial angular momentum: conserved.
        assert!(worst_l < 1e-8, "angular momentum not conserved: {worst_l:.2e}");
        // The rotation never leaves SO(3).
        assert!(worst_orth < 1e-10, "R left SO(3): ‖RᵀR−I‖ = {worst_orth:.2e}");
    }

    #[test]
    fn symmetric_top_rotates_steadily_about_its_axis() {
        // A body spinning purely about a principal axis keeps a constant angular velocity there.
        let mut b = LgviBody::new(Vector3::new(2.0, 2.0, 1.0), Vector3::new(0.0, 0.0, 1.5));
        for _ in 0..1000 {
            b.step(0.01, Vector3::zeros());
        }
        // ω stays aligned with +z at the same rate; no energy leaks into the other axes.
        let omega = Vector3::new(b.pi.x / b.inertia.x, b.pi.y / b.inertia.y, b.pi.z / b.inertia.z);
        assert!(omega.x.abs() < 1e-9 && omega.y.abs() < 1e-9, "spurious tumble: ω = {omega:?}");
        assert!((omega.z - 1.5).abs() < 1e-9, "axial rate changed: {}", omega.z);
    }
}
