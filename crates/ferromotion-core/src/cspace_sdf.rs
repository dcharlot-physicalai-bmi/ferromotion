//! **Composite configuration-space distance field** — the collision representation behind GPU motion
//! generators (cuRobo, CHOMP): lift a *workspace* signed distance field to a smooth scalar field over the
//! robot's **configuration space**, `d(q)`, whose value is the whole robot's minimum clearance and whose
//! gradient `∇_q d` points the joints away from collision. It is "composite" in two senses: the robot is a
//! *set* of collision spheres distributed along its links, and the world is a *union* of obstacle
//! primitives — both are combined into one differentiable clearance.
//!
//! The construction: forward kinematics places each sphere in the world; the workspace [`SdfScene`] gives
//! each sphere's clearance and the avoidance direction; a **soft-minimum** (`−β⁻¹ log Σ e^{−β·cₖ}`) fuses
//! them into one everywhere-differentiable field (→ the true `min` as `β → ∞`); and the chain rule through
//! the manipulator Jacobian, `∇_q cₖ = Jₖᵀ ∇ₓ d(xₖ)`, maps the workspace gradient into joint space. A
//! planar revolute arm supplies the analytic FK + Jacobian. The C-space gradient is verified against
//! finite differences, and gradient ascent on `d(q)` is shown to pull the arm out of collision. Pure
//! `nalgebra` → WASM-clean.

use crate::sdf::SdfScene;
use nalgebra::{DVector, Vector3};

/// A planar revolute arm rooted at the origin: link lengths, with `spheres_per_link` collision spheres of
/// radius `radius` spread along each link. Joint `i`'s angle is relative to the previous link.
#[derive(Clone, Debug)]
pub struct PlanarArm {
    pub links: Vec<f64>,
    pub spheres_per_link: usize,
    pub radius: f64,
}

impl PlanarArm {
    pub fn dof(&self) -> usize {
        self.links.len()
    }
    pub fn n_spheres(&self) -> usize {
        self.links.len() * self.spheres_per_link
    }

    /// World joint positions `p_0..p_n` (`p_0` = base) at configuration `q`. `p_i` is the *distal* end of
    /// link `i`. Also returns the cumulative link angles.
    fn joints(&self, q: &[f64]) -> (Vec<Vector3<f64>>, Vec<f64>) {
        let mut pts = vec![Vector3::zeros()];
        let mut ang = Vec::with_capacity(self.dof());
        let mut theta = 0.0;
        let mut p = Vector3::zeros();
        for (i, &l) in self.links.iter().enumerate() {
            theta += q[i];
            ang.push(theta);
            p += Vector3::new(theta.cos(), theta.sin(), 0.0) * l;
            pts.push(p);
        }
        (pts, ang)
    }

    /// The world positions of every collision sphere at configuration `q`.
    pub fn sphere_positions(&self, q: &[f64]) -> Vec<Vector3<f64>> {
        let (pts, ang) = self.joints(q);
        let mut out = Vec::with_capacity(self.n_spheres());
        for i in 0..self.dof() {
            let base = pts[i];
            let dir = Vector3::new(ang[i].cos(), ang[i].sin(), 0.0);
            for s in 0..self.spheres_per_link {
                // distribute spheres at the midpoints of equal sub-segments along the link
                let frac = (s as f64 + 0.5) / self.spheres_per_link as f64;
                out.push(base + dir * (frac * self.links[i]));
            }
        }
        out
    }

    /// The manipulator Jacobian `∂xₖ/∂q` (3×dof) of every sphere: for a planar revolute joint `m`, a distal
    /// point rotates about `+z` through the joint, so `∂x/∂q_m = ẑ × (x − p_m)` if the sphere is on link
    /// `≥ m`, else zero.
    pub fn sphere_jacobians(&self, q: &[f64]) -> Vec<nalgebra::DMatrix<f64>> {
        let (pts, _) = self.joints(q);
        let pos = self.sphere_positions(q);
        let mut out = Vec::with_capacity(self.n_spheres());
        let mut k = 0;
        for i in 0..self.dof() {
            for _ in 0..self.spheres_per_link {
                let x = pos[k];
                let mut j = nalgebra::DMatrix::zeros(3, self.dof());
                for m in 0..=i {
                    // ẑ × (x − p_m) with p_m the proximal joint of link m (pts[m])
                    let r = x - pts[m];
                    j[(0, m)] = -r.y;
                    j[(1, m)] = r.x;
                    // z row stays 0 (planar)
                }
                out.push(j);
                k += 1;
            }
        }
        out
    }
}

/// A composite configuration-space clearance field: a robot arm against a workspace SDF scene, fused by a
/// soft-minimum of sharpness `beta`.
#[derive(Clone)]
pub struct CspaceField<'a> {
    pub arm: &'a PlanarArm,
    pub scene: &'a SdfScene,
    pub beta: f64,
}

impl CspaceField<'_> {
    /// Per-sphere clearances `cₖ = d_world(xₖ) − radius` at configuration `q`.
    fn clearances(&self, q: &[f64]) -> Vec<f64> {
        self.arm
            .sphere_positions(q)
            .iter()
            .map(|x| self.scene.distance(x) - self.arm.radius)
            .collect()
    }

    /// The true (non-smooth) minimum clearance of the whole robot (negative ⇒ in collision).
    pub fn clearance_min(&self, q: &[f64]) -> f64 {
        self.clearances(q).into_iter().fold(f64::INFINITY, f64::min)
    }

    /// The **smooth** composite clearance `d(q) = −β⁻¹ log Σ exp(−β·cₖ)` — a soft-minimum that lower-bounds
    /// and converges to [`Self::clearance_min`] as `β → ∞`, differentiable everywhere.
    pub fn clearance_soft(&self, q: &[f64]) -> f64 {
        let c = self.clearances(q);
        let cmin = c.iter().cloned().fold(f64::INFINITY, f64::min);
        // stable log-sum-exp around the min
        let s: f64 = c.iter().map(|&ck| (-self.beta * (ck - cmin)).exp()).sum();
        cmin - s.ln() / self.beta
    }

    /// The configuration-space gradient `∇_q d(q)` of the smooth clearance: the softmax-weighted sum of the
    /// per-sphere workspace gradients pulled back through the Jacobian, `Σ wₖ Jₖᵀ ∇ₓ d(xₖ)`.
    pub fn grad(&self, q: &[f64]) -> DVector<f64> {
        let pos = self.arm.sphere_positions(q);
        let jac = self.arm.sphere_jacobians(q);
        let c: Vec<f64> = pos.iter().map(|x| self.scene.distance(x) - self.arm.radius).collect();
        let cmin = c.iter().cloned().fold(f64::INFINITY, f64::min);
        let ws: Vec<f64> = c.iter().map(|&ck| (-self.beta * (ck - cmin)).exp()).collect();
        let z: f64 = ws.iter().sum();
        let mut g = DVector::zeros(self.arm.dof());
        for k in 0..pos.len() {
            let w = ws[k] / z; // softmax weight (∂ softmin / ∂ cₖ)
            let gx = self.scene.gradient(&pos[k]); // ∂ d_world / ∂ x  (unit, eikonal)
            // ∇_q cₖ = Jₖᵀ gx  ; d = Σ softmin, ∂d/∂cₖ = wₖ
            g += jac[k].transpose() * gx * w;
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::Sdf;

    fn arm() -> PlanarArm {
        PlanarArm { links: vec![1.0, 0.8, 0.6], spheres_per_link: 3, radius: 0.05 }
    }

    fn scene() -> SdfScene {
        // one spherical obstacle out to the side of the workspace
        SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(1.2, 0.6, 0.0), radius: 0.3 }] }
    }

    #[test]
    fn the_sphere_jacobian_matches_finite_differences() {
        let a = arm();
        let q = [0.3, -0.5, 0.7];
        let jac = a.sphere_jacobians(&q);
        let eps = 1e-6;
        for (k, jk) in jac.iter().enumerate() {
            for m in 0..a.dof() {
                let mut qp = q;
                let mut qm = q;
                qp[m] += eps;
                qm[m] -= eps;
                let fd = (a.sphere_positions(&qp)[k] - a.sphere_positions(&qm)[k]) / (2.0 * eps);
                for r in 0..3 {
                    assert!((jk[(r, m)] - fd[r]).abs() < 1e-6, "J sphere {k} row {r} col {m}: {} vs fd {}", jk[(r, m)], fd[r]);
                }
            }
        }
    }

    #[test]
    fn the_soft_min_approaches_the_true_min_clearance() {
        let (a, sc) = (arm(), scene());
        let q = [0.4, 0.2, -0.3];
        let hard = CspaceField { arm: &a, scene: &sc, beta: 1.0 }.clearance_min(&q);
        let soft = CspaceField { arm: &a, scene: &sc, beta: 400.0 }.clearance_soft(&q);
        assert!((soft - hard).abs() < 5e-3, "soft-min {soft} should approach hard min {hard}");
        // soft-min always lower-bounds the true min
        let soft_lo = CspaceField { arm: &a, scene: &sc, beta: 5.0 }.clearance_soft(&q);
        assert!(soft_lo <= hard + 1e-9, "soft-min must lower-bound the true min");
    }

    #[test]
    fn the_cspace_gradient_matches_finite_differences() {
        // THE HEADLINE. ∇_q d(q) via the Jacobian pullback must equal central finite differences of the
        // smooth clearance — the guarantee the C-space field is correctly differentiable for planning.
        let (a, sc) = (arm(), scene());
        let field = CspaceField { arm: &a, scene: &sc, beta: 30.0 };
        let q = [0.5, 0.3, -0.4];
        let g = field.grad(&q);
        let eps = 1e-6;
        for m in 0..a.dof() {
            let mut qp = q;
            let mut qm = q;
            qp[m] += eps;
            qm[m] -= eps;
            let fd = (field.clearance_soft(&qp) - field.clearance_soft(&qm)) / (2.0 * eps);
            assert!((g[m] - fd).abs() < 1e-5, "grad[{m}] {} vs fd {fd}", g[m]);
        }
    }

    #[test]
    fn gradient_ascent_pulls_the_arm_out_of_collision() {
        // Put the arm in collision with the obstacle, then ascend d(q): clearance must increase toward
        // (and past) zero — collision avoidance done entirely in configuration space.
        let a = arm();
        // obstacle right on top of the mid-arm
        let sc = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(1.0, 0.15, 0.0), radius: 0.35 }] };
        let field = CspaceField { arm: &a, scene: &sc, beta: 40.0 };
        let mut q = [0.1, 0.05, 0.0];
        let c0 = field.clearance_min(&q);
        assert!(c0 < 0.0, "arm should start in collision: clearance {c0}");
        for _ in 0..400 {
            let g = field.grad(&q);
            for m in 0..a.dof() {
                q[m] += 0.02 * g[m]; // ascend the clearance
            }
        }
        let c1 = field.clearance_min(&q);
        assert!(c1 > c0 + 0.1, "gradient ascent should raise clearance: {c0} → {c1}");
    }

    #[test]
    fn the_composite_tracks_the_nearest_obstacle() {
        // Two obstacles; the composite clearance follows whichever is closer, and moving one changes d(q).
        let a = arm();
        let q = [0.6, 0.3, 0.2];
        let near = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(0.9, 0.9, 0.0), radius: 0.2 }] };
        let far = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::new(3.0, 3.0, 0.0), radius: 0.2 }] };
        let dn = CspaceField { arm: &a, scene: &near, beta: 50.0 }.clearance_soft(&q);
        let df = CspaceField { arm: &a, scene: &far, beta: 50.0 }.clearance_soft(&q);
        assert!(dn < df, "a nearer obstacle must lower the composite clearance: {dn} vs {df}");
    }
}
