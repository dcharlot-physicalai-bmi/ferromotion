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
//!
//! # What the height field is, and is not
//!
//! `h(x, y)` here is **geometric**: the gel surface is taken to conform to the indenter, softplus-
//! smoothed, and set to exactly zero outside the indenter's footprint. It is not an elastic solution,
//! and `beta` is a smoothing length, not a material property. That is enough for the differentiable
//! inference this crate is built for, and it is not enough to stand in for a real gel. The difference
//! was measured rather than guessed, by pressing the same sphere (radius 0.3, depth 0.15) into a slab
//! solved by [`ferromotion-fem`](../ferromotion_fem), bonded to a rigid backing as a GelSight gel is:
//!
//! - **Just outside the contact edge** (ρ/a ≈ 1.03) the geometric field is **3.3× too small**.
//! - **Beyond the footprint** it is exactly zero, while the elastic surface carries roughly a fifth to
//!   a quarter of the total displacement out there.
//! - **The elastic surface BULGES UP** in a ring around the contact, because the material under the
//!   indenter has to go somewhere and the backing will not let it go down. Measured as a fraction of
//!   the press depth, against Poisson's ratio, everything else held fixed:
//!
//! | ν | 0.20 | 0.30 | 0.40 | 0.45 | 0.49 |
//! |---|---|---|---|---|---|
//! | upward bulge | 0.17% | 0.51% | 1.57% | 2.87% | **6.56%** |
//!
//! The 39× growth toward the incompressible limit is what identifies it as displaced volume rather
//! than a numerical artifact, and silicone gel sits at the right-hand end of that table. At ρ = 0.5 the
//! surface height **changes sign** across the row, from +2.50e-3 at ν = 0.20 to −3.91e-3 at ν = 0.49.
//!
//! **No choice of parameters can reproduce that**, which is the part worth being precise about. `h` is
//! a softplus, so it is strictly positive inside the footprint and set to zero outside; it cannot be
//! negative anywhere. The bulge is not badly fitted here, it is unrepresentable. Photometric stereo
//! reads *normals*, so a ring whose true slope has the opposite sign renders shading that a model
//! trained on this forward pass never sees. That is a sim-to-real gap, not a resolution question.
//!
//! Two caveats on the numbers, since they came from one experiment: the contact set was prescribed
//! from the undeformed sphere rather than solved as a free-boundary contact problem, and the slab was a
//! single mesh and thickness in a finite domain with free edges. The sign, the ordering with ν, and the
//! structural impossibility of a negative `h` do not depend on any of that.
//!
//! **[`elastic`] closes this**, and is the module to reach for when the surface matters
//! quantitatively. It treats the gel as a linear elastic layer characterised by one measured influence
//! function, *solves* the contact rather than assuming it, and produces a surface that is free to rise
//! outside the patch. Validated end to end against a 3D solve of its own computed loads: about 1–2% rms
//! through the press depths a real sensor works in. The geometric field here stays the default, because
//! it is cheaper and is what the existing gradient-based inference is built on.
//!
//! [`shear`] is the other elastic part of this crate: its Cattaneo-Mindlin partial-slip model is a real
//! contact-mechanics solution for the tangential direction.

pub mod elastic;
pub mod shear;

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
    /// Softplus smoothing length for the contact edge, in the same units as `extent`.
    ///
    /// **Not a material property**, despite reading like one. It sets how sharply `h` rolls off at the
    /// contact boundary and nothing else; it carries no modulus, no Poisson ratio, and no thickness.
    /// See the crate documentation for what the geometric height field does and does not represent.
    pub beta: f64,
}

impl GelSim {
    /// Grid coordinate of column/row `i`.
    pub(crate) fn coord(&self, i: usize) -> f64 {
        -self.extent + 2.0 * self.extent * i as f64 / (self.n - 1) as f64
    }

    pub(crate) fn cell(&self) -> f64 {
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

    /// The far background is flat. **This is a property of the model, not of a gel.**
    ///
    /// Named `no_contact_region_is_flat` before, which read as a physical claim. A real elastomer
    /// bonded to a backing is *not* flat outside the contact: it bulges upward in a ring, by 6.56% of
    /// the press depth at ν = 0.49, measured against a 3D elastic solve. See the crate docs. What this
    /// test actually pins is that the geometric field has bounded support and does not leak numerical
    /// noise into the background, which is worth keeping and is all it ever showed.
    #[test]
    fn the_geometric_field_has_bounded_support() {
        let g = gel();
        let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.3, depth: 0.15 };
        let (h, _) = g.deformation(&ind);
        let normals = g.normals(&h);
        // A corner far from the indenter: no deformation, normal points straight up.
        let corner = normals[10 * g.n + 10];
        assert!((corner - Vector3::z()).norm() < 1e-6, "background not flat: {corner:?}");
        assert!(h[10 * g.n + 10].abs() < 1e-6, "background deformed");
    }

    /// The height field can never go negative, so the elastic bulge is unrepresentable here.
    ///
    /// This is a structural limit, not a fitting error, and it is worth a test because it is the one
    /// thing no choice of `beta`, `radius` or `depth` can work around: `h` is a softplus inside the
    /// footprint and exactly zero outside, and a softplus is strictly positive. A real gel's surface
    /// rises in a ring around the contact, and photometric stereo reads normals, so that ring renders
    /// shading this forward model never produces.
    ///
    /// If someone gives the crate an elastic surface response, this test SHOULD fail, and its failure
    /// is the signal that the crate docs' sim-to-real section needs rewriting.
    #[test]
    fn the_surface_can_never_bulge_upward() {
        let g = gel();
        for &(r, d) in &[(0.3f64, 0.15f64), (0.4, 0.05), (0.2, 0.19), (0.45, 0.3)] {
            for &beta in &[0.002f64, 0.02, 0.2] {
                let gb = GelSim { beta, ..g.clone() };
                let (h, _) = gb.deformation(&Indenter { cx: 0.02, cy: -0.03, radius: r, depth: d });
                let lowest = h.iter().cloned().fold(f64::INFINITY, f64::min);
                assert!(
                    lowest >= 0.0,
                    "h must be non-negative everywhere by construction (r = {r}, depth = {d}, beta = {beta}): {lowest:e}"
                );
            }
        }
        // and the fixture is a real press, not a no-op
        let (h, _) = g.deformation(&Indenter { cx: 0.0, cy: 0.0, radius: 0.3, depth: 0.15 });
        assert!(h.iter().cloned().fold(0.0f64, f64::max) > 0.1, "fixture should actually indent");
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

pub mod servo;
pub use servo::{extract_features, TactileFeatures, TactileServo};
