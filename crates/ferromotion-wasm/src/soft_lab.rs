//! **Soft-rod lab** — the rig behind the textbook chapter on soft/continuum robots. Drives the real
//! [`ferromotion_core::CosseratRod`] (planar piecewise-constant-strain statics) so the reader can hang a
//! load on a compliant arm, watch the whole body curve, and see the tip deflection match Euler–Bernoulli
//! beam theory on-device.

use ferromotion_core::CosseratRod;
use nalgebra::Vector2;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SoftRod {
    rod: CosseratRod,
    fx: f64,
    fy: f64,
    kappa: Vec<f64>,
}

#[wasm_bindgen]
impl SoftRod {
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, l: f64, ei: f64) -> SoftRod {
        let rod = CosseratRod::new(n, l, ei);
        let kappa = vec![0.0; n];
        SoftRod { rod, fx: 0.0, fy: 0.0, kappa }
    }

    pub fn set_stiffness(&mut self, ei: f64) {
        self.rod.ei = ei.max(0.05);
    }
    pub fn set_load(&mut self, fx: f64, fy: f64) {
        self.fx = fx;
        self.fy = fy;
    }
    pub fn length(&self) -> f64 {
        self.rod.l
    }
    pub fn stiffness(&self) -> f64 {
        self.rod.ei
    }

    /// Solve the equilibrium shape under the current tip load.
    pub fn solve(&mut self) {
        self.kappa = self.rod.solve(Vector2::new(self.fx, self.fy));
    }

    /// Backbone points, interleaved `[x0,y0,x1,y1,…]`, for drawing the bent arm.
    pub fn backbone_xy(&self) -> Vec<f64> {
        let mut o = Vec::new();
        for p in self.rod.backbone(&self.kappa) {
            o.push(p.x);
            o.push(p.y);
        }
        o
    }

    pub fn tip_x(&self) -> f64 {
        self.rod.tip(&self.kappa).x
    }
    pub fn tip_y(&self) -> f64 {
        self.rod.tip(&self.kappa).y
    }

    /// Transverse tip deflection magnitude (perpendicular droop) under the load.
    pub fn deflection(&self) -> f64 {
        -self.rod.tip(&self.kappa).y
    }

    /// The Euler–Bernoulli small-deflection prediction `δ = F L³ / (3 EI)` for the current transverse
    /// load — the beam-theory answer the strain model should reproduce for small loads.
    pub fn euler_bernoulli(&self) -> f64 {
        let f = -self.fy; // transverse (downward) component
        f * self.rod.l.powi(3) / (3.0 * self.rod.ei)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_euler_bernoulli_for_a_small_load() {
        let mut rod = SoftRod::new(60, 1.0, 5.0);
        rod.set_load(0.0, -0.02);
        rod.solve();
        let rel = (rod.deflection() - rod.euler_bernoulli()).abs() / rod.euler_bernoulli();
        assert!(rel < 0.02, "deflection {} vs Euler-Bernoulli {} (rel {rel})", rod.deflection(), rod.euler_bernoulli());
    }

    #[test]
    fn backbone_starts_clamped_and_has_the_right_length() {
        let mut rod = SoftRod::new(50, 1.2, 3.0);
        rod.set_load(0.0, -0.05);
        rod.solve();
        let b = rod.backbone_xy();
        assert!(b[0].abs() < 1e-12 && b[1].abs() < 1e-12, "base should be clamped at the origin");
        // arc length preserved (inextensible): sum of segment lengths ≈ L
        let mut len = 0.0;
        for k in 0..(b.len() / 2 - 1) {
            len += ((b[2 * k + 2] - b[2 * k]).powi(2) + (b[2 * k + 3] - b[2 * k + 1]).powi(2)).sqrt();
        }
        assert!((len - 1.2).abs() < 1e-6, "arc length should be preserved: {len}");
    }
}
