//! **Tactile servoing** — closing the sense→act loop on the gel (Lepora et al.).
//!
//! The sensor in this crate turns contact into a deformation field; this module turns that field back
//! into *motion*. A real optical-tactile sensor never measures the object's pose — it measures a
//! picture of its own skin. So the controller reads **contact features** off the deformation exactly
//! as hardware would (where the contact is, how hard it is pressing) and servos the sensor to hold a
//! target contact: centred, at a chosen depth.
//!
//! Because the gel's indentation is radially symmetric about the contact, the deformation-weighted
//! centroid *is* the contact location — no calibration, no model inversion. Moving the sensor `+x`
//! carries the contact `−x` across the gel, which makes the tactile Jacobian a clean identity and the
//! servo law almost trivial:
//!
//! ```text
//!   ṡ_xy = k · (contact centroid)        ← centre the contact
//!   ṡ_z  = k_d · (depth* − depth)        ← regulate contact depth (i.e. force)
//! ```
//!
//! Pure Rust → WASM-clean.

use crate::{GelSim, Indenter};

/// Contact features read off the deformation field, as hardware would read them off the image.
#[derive(Clone, Copy, Debug)]
pub struct TactileFeatures {
    /// Contact centroid in gel coordinates.
    pub cx: f64,
    pub cy: f64,
    /// Peak indentation — the sensor's proxy for contact force.
    pub depth: f64,
    /// Total deformation (a smooth force proxy).
    pub total: f64,
}

/// Deformation below which the gel counts as untouched — the sensor's noise floor. The gel model is
/// softplus-smoothed for differentiability, so indentation is never *exactly* zero; contact is graded,
/// and a threshold defines it (the same 1e-4 floor [`GelSim::contact_area`] uses).
pub const CONTACT_FLOOR: f64 = 1e-4;

/// Extract contact features, or `None` if the gel is not touching anything.
pub fn extract_features(gel: &GelSim, ind: &Indenter) -> Option<TactileFeatures> {
    let (h, _) = gel.deformation(ind);
    let n = gel.n;
    let (mut sx, mut sy, mut peak, mut total) = (0.0, 0.0, 0.0f64, 0.0);
    for iy in 0..n {
        for ix in 0..n {
            let w = h[iy * n + ix];
            sx += w * gel.coord(ix);
            sy += w * gel.coord(iy);
            peak = peak.max(w);
            total += w;
        }
    }
    if peak < CONTACT_FLOOR {
        return None; // below the noise floor: not touching
    }
    Some(TactileFeatures { cx: sx / total, cy: sy / total, depth: peak, total })
}

/// A tactile servo: hold the contact centred at a target depth by moving the sensor.
#[derive(Clone, Copy, Debug)]
pub struct TactileServo {
    /// Lateral gain (centring).
    pub k_lateral: f64,
    /// Depth gain (force regulation).
    pub k_depth: f64,
    pub target_depth: f64,
}

impl TactileServo {
    /// Sensor velocity `(ṡx, ṡy, ṡz)` from the current contact features. Moving the sensor `+x` shifts
    /// the contact `−x` on the gel, so chasing the centroid centres the contact.
    pub fn velocity(&self, f: &TactileFeatures) -> (f64, f64, f64) {
        (self.k_lateral * f.cx, self.k_lateral * f.cy, self.k_depth * (self.target_depth - f.depth))
    }

    /// Velocity when contact is lost: creep forward until the gel finds the surface again.
    pub fn search_velocity(&self) -> (f64, f64, f64) {
        (0.0, 0.0, self.k_depth * self.target_depth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gel() -> GelSim {
        GelSim { n: 121, extent: 1.0, beta: 0.005 }
    }

    #[test]
    fn features_recover_the_true_contact_location_and_depth() {
        // The indentation is radially symmetric about the contact ⇒ its centroid *is* the contact.
        let g = gel();
        for &(cx, cy, depth) in &[(0.0, 0.0, 0.08), (0.25, -0.15, 0.06), (-0.3, 0.2, 0.1)] {
            let ind = Indenter { cx, cy, radius: 0.5, depth };
            let f = extract_features(&g, &ind).expect("should be in contact");
            assert!((f.cx - cx).abs() < 5e-3, "centroid x {} vs true {cx}", f.cx);
            assert!((f.cy - cy).abs() < 5e-3, "centroid y {} vs true {cy}", f.cy);
            // Peak indentation ≈ the press depth (softplus ≈ identity well above β).
            assert!((f.depth - depth).abs() < 5e-3, "depth feature {} vs true {depth}", f.depth);
        }
        // No contact ⇒ no features.
        assert!(extract_features(&g, &Indenter { cx: 0.0, cy: 0.0, radius: 0.5, depth: -0.05 }).is_none());
    }

    #[test]
    fn the_tactile_jacobian_is_the_expected_identity() {
        // Moving the sensor +x carries the contact −x; pressing +z deepens it 1:1.
        let g = gel();
        let (ox, oy, z0) = (0.2, -0.1, 0.0);
        let ind = |sx: f64, sy: f64, sz: f64| Indenter { cx: ox - sx, cy: oy - sy, radius: 0.5, depth: sz - z0 };
        let (sx, sy, sz) = (0.05, 0.02, 0.08);
        let eps = 1e-4;
        let feat = |a: f64, b: f64, c: f64| extract_features(&g, &ind(a, b, c)).unwrap();

        let dcx_dsx = (feat(sx + eps, sy, sz).cx - feat(sx - eps, sy, sz).cx) / (2.0 * eps);
        assert!((dcx_dsx + 1.0).abs() < 1e-2, "∂centroid_x/∂sensor_x should be −1, got {dcx_dsx}");
        let dcy_dsy = (feat(sx, sy + eps, sz).cy - feat(sx, sy - eps, sz).cy) / (2.0 * eps);
        assert!((dcy_dsy + 1.0).abs() < 1e-2, "∂centroid_y/∂sensor_y should be −1, got {dcy_dsy}");
        let dd_dsz = (feat(sx, sy, sz + eps).depth - feat(sx, sy, sz - eps).depth) / (2.0 * eps);
        assert!((dd_dsz - 1.0).abs() < 1e-2, "∂depth/∂sensor_z should be +1, got {dd_dsz}");
    }

    #[test]
    fn servo_centres_the_contact_and_regulates_depth() {
        // The real gel simulation is in the loop: deformation → features → sensor motion.
        let g = gel();
        let (ox, oy, z0) = (0.28, -0.17, 0.0); // where the object actually is (unknown to the servo)
        let servo = TactileServo { k_lateral: 4.0, k_depth: 4.0, target_depth: 0.06 };
        let (mut sx, mut sy, mut sz): (f64, f64, f64) = (0.0, 0.0, 0.02); // off-centre and barely touching
        let dt = 0.01;
        let start_err = ((sx - ox).powi(2) + (sy - oy).powi(2)).sqrt();
        for _ in 0..3000 {
            let ind = Indenter { cx: ox - sx, cy: oy - sy, radius: 0.5, depth: sz - z0 };
            let v = match extract_features(&g, &ind) {
                Some(f) => servo.velocity(&f),
                None => servo.search_velocity(), // lost contact — press back in
            };
            sx += v.0 * dt;
            sy += v.1 * dt;
            sz += v.2 * dt;
        }
        // Centring the contact means the sensor found the object laterally …
        assert!((sx - ox).abs() < 5e-3 && (sy - oy).abs() < 5e-3, "did not centre: sensor ({sx}, {sy}) vs object ({ox}, {oy})");
        assert!(start_err > 0.2, "test started too close to be meaningful");
        // … and the contact settled at the commanded depth.
        let f = extract_features(&g, &Indenter { cx: ox - sx, cy: oy - sy, radius: 0.5, depth: sz - z0 }).unwrap();
        assert!((f.depth - servo.target_depth).abs() < 5e-3, "depth not regulated: {} vs {}", f.depth, servo.target_depth);
    }
}
