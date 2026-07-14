//! ferromotion-tactile — a **differentiable optical-tactile sensor** simulator (GelSight / DIGIT
//! class), in the spirit of DOT-Sim / Taccel.
//!
//! An optical tactile sensor is an elastomer gel filmed from below: when an object presses in, the
//! gel surface deforms, and the camera reads that deformation as shading under several colored lights
//! (photometric stereo). The forward model here is: an **indenter** (a sphere) presses into the gel to
//! a depth, producing a smooth surface-height field `h(x,y)`; surface **normals** `n = (−h_x, −h_y, 1)`
//! follow; and a **photometric image** `I_c = albedo·max(0, n·L_c)` is rendered under colored
//! directional lights — the RGB tactile imprint.
//!
//! Every stage is smooth (the indentation uses a softplus contact), so the sensor is **differentiable**:
//! the height field's derivative w.r.t. the press depth is exact (`∂h/∂depth = σ(·)`), verified
//! against finite differences — enabling gradient-based tactile inference (estimate contact
//! depth/pose from an image). Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// A spherical indenter pressing into the gel.
#[derive(Clone, Copy, Debug)]
pub struct Indenter {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    /// Press depth of the sphere's lowest point below the gel plane.
    pub depth: f64,
}

/// A colored directional light `(direction, albedo)` for one image channel.
#[derive(Clone, Copy, Debug)]
pub struct Light {
    pub dir: Vector3<f64>,
    pub albedo: f64,
}

/// An optical-tactile gel: a square sensing patch of side `2·extent`, sampled on an `n×n` grid.
#[derive(Clone, Debug)]
pub struct GelSim {
    pub n: usize,
    pub extent: f64,
    /// Softplus temperature for the smooth contact (gel compliance).
    pub beta: f64,
}

impl GelSim {
    /// Grid coordinate of column/row `i`.
    fn coord(&self, i: usize) -> f64 {
        -self.extent + 2.0 * self.extent * i as f64 / (self.n - 1) as f64
    }

    fn cell(&self) -> f64 {
        2.0 * self.extent / (self.n - 1) as f64
    }

    /// Surface height field `h(x,y)` (downward displacement) and its exact `∂h/∂depth`.
    pub fn deformation(&self, ind: &Indenter) -> (Vec<f64>, Vec<f64>) {
        let (n, r) = (self.n, ind.radius);
        let (mut h, mut dh) = (vec![0.0; n * n], vec![0.0; n * n]);
        for iy in 0..n {
            for ix in 0..n {
                let (x, y) = (self.coord(ix), self.coord(iy));
                let rho2 = (x - ind.cx).powi(2) + (y - ind.cy).powi(2);
                if rho2 < r * r {
                    // Indentation = sphere-surface dip below the plane, softplus-smoothed.
                    let arg = (r * r - rho2).sqrt() - r + ind.depth;
                    let sp = self.beta * (1.0 + (arg / self.beta).exp()).ln(); // softplus
                    let sig = 1.0 / (1.0 + (-arg / self.beta).exp()); // ∂softplus/∂arg = σ; ∂arg/∂depth = 1
                    h[iy * n + ix] = sp;
                    dh[iy * n + ix] = sig;
                }
            }
        }
        (h, dh)
    }

    /// Surface normals from a height field (central differences).
    pub fn normals(&self, h: &[f64]) -> Vec<Vector3<f64>> {
        let n = self.n;
        let inv2c = 1.0 / (2.0 * self.cell());
        let mut out = vec![Vector3::new(0.0, 0.0, 1.0); n * n];
        for iy in 1..n - 1 {
            for ix in 1..n - 1 {
                let hx = (h[iy * n + ix + 1] - h[iy * n + ix - 1]) * inv2c;
                let hy = (h[(iy + 1) * n + ix] - h[(iy - 1) * n + ix]) * inv2c;
                // h is downward displacement, so the outward surface normal is (hx, hy, 1) normalized.
                out[iy * n + ix] = Vector3::new(hx, hy, 1.0).normalize();
            }
        }
        out
    }

    /// Render the RGB photometric-stereo tactile image under three colored lights.
    pub fn tactile_image(&self, ind: &Indenter, lights: &[Light; 3]) -> Vec<[f64; 3]> {
        let (h, _) = self.deformation(ind);
        let normals = self.normals(&h);
        normals
            .iter()
            .map(|nv| {
                let mut px = [0.0; 3];
                for (c, l) in lights.iter().enumerate() {
                    px[c] = l.albedo * nv.dot(&l.dir).max(0.0);
                }
                px
            })
            .collect()
    }

    /// Total deformation `Σ h` and its exact derivative `∂(Σh)/∂depth` — a differentiable contact feature.
    pub fn total_deformation(&self, ind: &Indenter) -> (f64, f64) {
        let (h, dh) = self.deformation(ind);
        (h.iter().sum(), dh.iter().sum())
    }

    /// Contact-patch area (cells in contact).
    pub fn contact_area(&self, ind: &Indenter) -> f64 {
        let (h, _) = self.deformation(ind);
        let cell = self.cell();
        h.iter().filter(|&&v| v > 1e-4).count() as f64 * cell * cell
    }
}

/// A default GelSight-like colored 3-light rig (three azimuths at ~45° elevation).
pub fn default_lights() -> [Light; 3] {
    let e = std::f64::consts::FRAC_1_SQRT_2;
    let tau = std::f64::consts::TAU;
    let mut ls = [Light { dir: Vector3::z(), albedo: 1.0 }; 3];
    for (k, l) in ls.iter_mut().enumerate() {
        let a = tau * k as f64 / 3.0;
        l.dir = Vector3::new(e * a.cos(), e * a.sin(), e);
        l.albedo = 1.0;
    }
    ls
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gel() -> GelSim {
        GelSim { n: 81, extent: 1.0, beta: 0.02 }
    }

    #[test]
    fn no_contact_region_is_flat() {
        let g = gel();
        let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.3, depth: 0.15 };
        let (h, _) = g.deformation(&ind);
        let normals = g.normals(&h);
        // A corner far from the indenter: no deformation, normal points straight up.
        let corner = normals[10 * g.n + 10];
        assert!((corner - Vector3::z()).norm() < 1e-6, "background not flat: {corner:?}");
        assert!(h[10 * g.n + 10].abs() < 1e-6, "background deformed");
    }

    #[test]
    fn deeper_press_gives_more_deformation_and_larger_patch() {
        let g = gel();
        let shallow = Indenter { cx: 0.0, cy: 0.0, radius: 0.4, depth: 0.05 };
        let deep = Indenter { cx: 0.0, cy: 0.0, radius: 0.4, depth: 0.2 };
        assert!(g.total_deformation(&deep).0 > g.total_deformation(&shallow).0, "deeper should deform more");
        assert!(g.contact_area(&deep) > g.contact_area(&shallow), "deeper should widen the contact patch");
    }

    #[test]
    fn depth_gradient_matches_finite_difference() {
        // Exact ∂(Σh)/∂depth vs central FD — the differentiable-tactile check.
        let g = gel();
        let ind = Indenter { cx: 0.05, cy: -0.1, radius: 0.35, depth: 0.12 };
        let (_, analytic) = g.total_deformation(&ind);
        let eps = 1e-6;
        let mut ip = ind;
        ip.depth += eps;
        let mut im = ind;
        im.depth -= eps;
        let fd = (g.total_deformation(&ip).0 - g.total_deformation(&im).0) / (2.0 * eps);
        let rel = (analytic - fd).abs() / fd.abs();
        eprintln!("tactile ∂Σh/∂depth: analytic={analytic:.5}, fd={fd:.5}, rel={rel:.2e}");
        assert!(analytic > 1.0, "gradient trivially small — test not exercising contact");
        assert!(rel < 1e-5, "tactile depth gradient wrong: {analytic} vs {fd}");
    }

    #[test]
    fn tactile_image_lights_up_under_contact() {
        let g = gel();
        let lights = default_lights();
        let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.35, depth: 0.15 };
        let img = g.tactile_image(&ind, &lights);
        // Background: normal is +z, every light has equal +z tilt ⇒ equal, positive channels.
        let bg = img[10 * g.n + 10];
        assert!(bg.iter().all(|&c| c > 0.0), "background dark: {bg:?}");
        // Near the contact rim the normals tilt, so the three channels differentiate (color contrast).
        let rim_spread = img
            .iter()
            .map(|px| px.iter().cloned().fold(0.0f64, f64::max) - px.iter().cloned().fold(f64::INFINITY, f64::min))
            .fold(0.0f64, f64::max);
        let bg_spread = bg.iter().cloned().fold(0.0f64, f64::max) - bg.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(rim_spread > bg_spread + 0.05, "contact produced no photometric contrast (rim {rim_spread}, bg {bg_spread})");
    }
}
