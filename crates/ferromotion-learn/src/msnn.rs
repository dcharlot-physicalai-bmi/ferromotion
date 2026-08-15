//! **Model-structured network (Neural Compositional Module)** — an interpretable, data-efficient
//! alternative to a black-box MLP, from the physics-informed-ML "model-structured" school. Rather than one
//! opaque nonlinear map, it is a blend of **local linear models** gated by **membership functions** over a
//! scheduling variable: `f(x) = Σᵢ φᵢ(x) · (aᵢ x + bᵢ)`. The `φᵢ` are triangular bumps centered on a grid,
//! normalized to a **partition of unity** (`Σᵢ φᵢ(x) = 1` everywhere), so the output is a smooth interpolation
//! between neighboring local lines. Two properties fall out: it is **linear in its parameters** `(aᵢ, bᵢ)`, so
//! it fits by ordinary least squares — no gradient descent, no local minima — and it is **interpretable**: the
//! learned slope `aᵢ` is the function's local slope near center `i`, a number an engineer can read and sanity-
//! check. (The full model-structured toolkit adds neural FIR blocks over past inputs for dynamics and
//! physics-based initialization; this is its interpretable spatial core.)
//!
//! Verified: the memberships partition unity exactly; the model recovers an affine function to machine
//! precision; it fits a nonlinear curve with a handful of local models; and each learned local slope matches
//! the true derivative at its center.

/// A blend of local linear models with partition-of-unity triangular memberships over one scheduling input.
pub struct Msnn {
    centers: Vec<f64>,
    h: f64,
    a: Vec<f64>, // local slopes
    b: Vec<f64>, // local intercepts
    // Cells whose weighted samples do not identify a slope (fewer than two distinct weighted `x`). Tracked
    // because the interpretability claim above is conditional on this being empty: see `degenerate_cells`.
    degenerate: Vec<bool>,
}

impl Msnn {
    fn raw_memberships(centers: &[f64], h: f64, x: f64) -> Vec<f64> {
        let k = centers.len();
        let mut phi = vec![0.0; k];
        for (i, &c) in centers.iter().enumerate() {
            let t = 1.0 - (x - c).abs() / h;
            let mut v = t.max(0.0);
            // clamp the end tents so the partition of unity extends past the domain edges
            if (i == 0 && x < c) || (i == k - 1 && x > c) {
                v = 1.0;
            }
            phi[i] = v;
        }
        phi
    }

    /// The normalized membership weights at `x` (they sum to exactly 1 — the partition of unity).
    pub fn membership(&self, x: f64) -> Vec<f64> {
        let mut phi = Self::raw_memberships(&self.centers, self.h, x);
        let s: f64 = phi.iter().sum();
        if s > 0.0 {
            for p in &mut phi {
                *p /= s;
            }
        }
        phi
    }

    /// Fit `centers` local linear models on `[lo, hi]`. Each local model `i` is an *independent*
    /// membership-weighted linear regression, so its slope `aᵢ` is a well-defined, readable local slope; the
    /// prediction blends them by the partition-of-unity memberships.
    pub fn fit(xs: &[f64], ys: &[f64], centers: usize, lo: f64, hi: f64) -> Msnn {
        let c: Vec<f64> = (0..centers).map(|i| lo + (hi - lo) * i as f64 / (centers - 1).max(1) as f64).collect();
        let h = if centers > 1 { (hi - lo) / (centers - 1) as f64 } else { hi - lo };
        let mut model = Msnn {
            centers: c,
            h,
            a: vec![0.0; centers],
            b: vec![0.0; centers],
            degenerate: vec![false; centers],
        };

        for i in 0..centers {
            // weighted normal equations for a line minimizing Σ φᵢ(x)[(a x + b) − y]²
            let (mut sw, mut swx, mut swxx, mut swy, mut swxy) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for (&x, &y) in xs.iter().zip(ys.iter()) {
                let w = Self::raw_memberships(&model.centers, h, x)[i];
                sw += w;
                swx += w * x;
                swxx += w * x * x;
                swy += w * y;
                swxy += w * x * y;
            }
            // Solve [[swxx, swx],[swx, sw]] [a;b] = [swxy; swy].
            //
            // **The degenerate branch used to add its 1e-9 to the DETERMINANT (2026-08-14).** That is not a
            // ridge — a ridge perturbs the normal-matrix diagonal. Adding to the determinant of a
            // rank-deficient system leaves both numerators at ~0 while making the denominator ~1e-9, so the
            // cell returned a = 0, b = 0: a local model that predicts exactly zero, with no error and no
            // diagnostic. Because φ is a partition of unity of *overlapping* tents, one zeroed cell also
            // corrupts predictions where a neighbouring well-conditioned cell fits perfectly — at a sample
            // covered 0.25/0.75 by a good and a zeroed cell, the good fit is diluted 4:1.
            //
            // Fitting `y = 2x + 3` at five points with six centers put four of six cells on this branch and
            // made four of the five training samples come back wrong, `predict(1.0)` returning 0.0 against a
            // truth of 5.0. The suite did not catch it because the one test that reaches this path asserts
            // only on memberships.
            //
            // The fix is the weighted mean, not a Tikhonov diagonal: with fewer than two distinct weighted
            // `x` the slope is genuinely unidentifiable, so reporting the level and no slope is the honest
            // answer, and it is scale-free — an absolute 1e-9 means nothing against `swxx ~ 1e6`.
            let det0 = swxx * sw - swx * swx;
            if det0.abs() < 1e-12 {
                model.degenerate[i] = true;
                model.a[i] = 0.0;
                model.b[i] = if sw > 0.0 { swy / sw } else { 0.0 };
            } else {
                model.a[i] = (swxy * sw - swx * swy) / det0;
                model.b[i] = (swxx * swy - swx * swxy) / det0;
            }
        }
        model
    }

    /// Predict `f(x)` = blend of the local linear models.
    pub fn predict(&self, x: f64) -> f64 {
        let phi = self.membership(x);
        (0..self.centers.len()).map(|i| phi[i] * (self.a[i] * x + self.b[i])).sum()
    }

    pub fn n_centers(&self) -> usize {
        self.centers.len()
    }
    pub fn center(&self, i: usize) -> f64 {
        self.centers[i]
    }
    /// The learned local slope of model `i` — the function's derivative near `center(i)`.
    ///
    /// **Readable only if cell `i` is not degenerate.** A slope needs two distinct weighted `x` values in the
    /// cell to be identifiable at all; with fewer, this returns `0.0`, which is indistinguishable from a
    /// genuinely flat region. Check [`Msnn::is_degenerate`] before presenting the number as a derivative —
    /// silently returning a readable-looking value for an unidentifiable parameter is the dangerous part.
    pub fn local_slope(&self, i: usize) -> f64 {
        self.a[i]
    }
    /// The local line `aᵢ x + bᵢ` evaluated at `x` (for drawing the individual pieces).
    pub fn local_line(&self, i: usize, x: f64) -> f64 {
        self.a[i] * x + self.b[i]
    }

    /// Whether cell `i`'s weighted samples identify a slope at all (fewer than two distinct weighted `x`
    /// values means they do not). A degenerate cell reports the weighted mean of its `y` and a slope of
    /// `0.0`, which is a level, not a derivative.
    pub fn is_degenerate(&self, i: usize) -> bool {
        self.degenerate[i]
    }

    /// How many cells did not identify a slope. **Non-zero means the interpretability claim does not hold
    /// for this fit**: some `local_slope` readouts are placeholders rather than derivatives, and the usual
    /// cause is too few samples for the number of centers. Zero is the condition under which the module's
    /// headline property — that `aᵢ` is a readable local derivative — is actually true.
    pub fn degenerate_cells(&self) -> usize {
        self.degenerate.iter().filter(|d| **d).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memberships_partition_unity() {
        // THE STRUCTURAL PROPERTY. The membership weights sum to 1 at every point, inside and outside the grid.
        let m = Msnn::fit(&[0.0, 1.0], &[0.0, 1.0], 6, -1.0, 1.0);
        for &x in &[-2.0, -0.7, 0.0, 0.33, 0.9, 1.5] {
            let s: f64 = m.membership(x).iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "Σφ should be 1 at x={x}, got {s}");
        }
        // 2 samples across 6 centers makes every cell degenerate. Assert it: this test was the only one
        // reaching that branch and it checked nothing about the model, so the old code silently built a
        // model predicting 0.0 everywhere and the suite stayed green.
        assert_eq!(m.degenerate_cells(), 6, "2 samples over 6 centers cannot identify any slope");
    }

    #[test]
    fn a_rank_deficient_cell_reports_a_level_not_a_fabricated_zero() {
        // The degenerate branch used to add 1e-9 to the DETERMINANT, not to the normal-matrix diagonal, so
        // both numerators stayed ~0 against a ~1e-9 denominator and the cell returned a = 0, b = 0.
        // y = 2x + 3 at five points over six centers puts four cells on that branch; because φ is a
        // partition of unity of overlapping tents, 4 of the 5 training points then came back wrong —
        // predict(1.0) was 0.0 against a truth of 5.0.
        let (xs, ys) = ([-1.0, -0.5, 0.0, 0.5, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = Msnn::fit(&xs, &ys, 6, -1.0, 1.0);
        assert!(m.degenerate_cells() > 0, "this fixture must exercise the degenerate branch");
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            assert!((m.predict(x) - y).abs() < 1e-9, "sample x={x}: got {}, want {y}", m.predict(x));
        }
        // A degenerate slope must be flagged, not presented as a derivative: 0.0 is indistinguishable from
        // a genuinely flat region, which is what made the silent version dangerous.
        for i in 0..m.n_centers() {
            if m.is_degenerate(i) {
                assert_eq!(m.local_slope(i), 0.0, "a degenerate cell reports no slope");
            }
        }
    }

    #[test]
    fn recovers_an_affine_function_exactly() {
        // A linear target y = 2x + 1 is represented exactly: every local slope is 2.
        let xs: Vec<f64> = (0..40).map(|i| -1.0 + 2.0 * i as f64 / 39.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let m = Msnn::fit(&xs, &ys, 6, -1.0, 1.0);
        for &x in &[-0.8, -0.2, 0.5, 0.95] {
            assert!((m.predict(x) - (2.0 * x + 1.0)).abs() < 1e-9, "affine recovery at {x}");
        }
        for i in 0..m.n_centers() {
            assert!((m.local_slope(i) - 2.0).abs() < 1e-6, "every local slope should be 2");
        }
    }

    #[test]
    fn fits_a_nonlinear_curve_with_readable_local_slopes() {
        // THE HEADLINE. Fit sin(2x) with a few local models; the fit is close AND each learned local slope
        // matches the true derivative 2cos(2x) at its center — interpretability.
        let xs: Vec<f64> = (0..80).map(|i| -1.5 + 3.0 * i as f64 / 79.0).collect();
        let ys: Vec<f64> = xs.iter().map(|x| (2.0 * x).sin()).collect();
        let m = Msnn::fit(&xs, &ys, 10, -1.5, 1.5);
        // fit quality
        let mse: f64 = xs.iter().zip(&ys).map(|(&x, &y)| (m.predict(x) - y).powi(2)).sum::<f64>() / xs.len() as f64;
        assert!(mse < 1e-3, "should fit sin(2x): mse {mse}");
        // interpretability: local slope ≈ true derivative 2cos(2c) at interior centers
        for i in 2..m.n_centers() - 2 {
            let c = m.center(i);
            let truth = 2.0 * (2.0 * c).cos();
            assert!((m.local_slope(i) - truth).abs() < 0.6, "slope at c={c}: {} vs {truth}", m.local_slope(i));
        }
    }
}
