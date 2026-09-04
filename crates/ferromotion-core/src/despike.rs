//! **Median & Hampel despiking filters** — nonlinear robust filters that remove **impulse / salt-and-pepper
//! spikes** (encoder glitches, LiDAR dropouts, single-sample sensor faults) while **preserving step edges**,
//! which linear smoothers ([`crate::Biquad`] IIR, moving average) and even polynomial ones
//! ([`crate::SavGol`]) blur. The **median filter** replaces each sample with the median of a sliding window;
//! the **Hampel filter** is its outlier-aware cousin — it replaces a sample with the local median *only*
//! when it lies more than `n_sigma` robust deviations (scaled median-absolute-deviation) from it, so clean
//! data passes through untouched and only genuine spikes are corrected.
//!
//! These are the standard first stage of a robot's sensor-conditioning pipeline, ahead of the linear
//! smoothing/derivative filters. Edges are handled with shrinking one-sided windows. Verified: an isolated
//! spike is removed while a step edge is kept sharp (where a moving average smears it); a constant passes
//! unchanged; and Hampel corrects flagged outliers while leaving inliers exactly as they were. Pure Rust →
//! WASM-clean.

fn median(vals: &mut [f64]) -> f64 {
    if vals.is_empty() {
        return f64::NAN; // a window with nothing real in it has no median
    }
    vals.sort_by(f64::total_cmp);
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        0.5 * (vals[n / 2 - 1] + vals[n / 2])
    }
}

/// The window centred on `i`, clipped at the ends, **with non-finite samples dropped**: a dropout has no
/// value to contribute to a median, and it is the fault these filters exist to remove.
fn window(signal: &[f64], i: usize, half: usize) -> Vec<f64> {
    let lo = i.saturating_sub(half);
    let hi = (i + half).min(signal.len() - 1);
    signal[lo..=hi].iter().copied().filter(|v| v.is_finite()).collect()
}

/// Sliding-window **median filter** with half-window `half` (window size `2·half + 1`), edges shrinking.
///
/// Non-finite samples are treated as the dropouts they are: they are excluded from every window and so
/// replaced by the local median of the real samples. A sample whose whole window is non-finite has
/// nothing to be reconstructed from and comes back `NaN`.
pub fn median_filter(signal: &[f64], half: usize) -> Vec<f64> {
    (0..signal.len())
        .map(|i| {
            let mut w = window(signal, i, half);
            median(&mut w)
        })
        .collect()
}

/// **Hampel filter**: replace a sample with its local median only when it exceeds `n_sigma` scaled MADs from
/// it (MAD scaled by `1.4826` for consistency with the Gaussian σ). Clean samples pass through unchanged.
/// Returns the filtered signal and a per-sample outlier mask.
///
/// A non-finite sample is flagged and replaced unconditionally: there is no value to measure against a
/// MAD, and a dropout is exactly what the mask is meant to report.
pub fn hampel_filter(signal: &[f64], half: usize, n_sigma: f64) -> (Vec<f64>, Vec<bool>) {
    let mut out = signal.to_vec();
    let mut flagged = vec![false; signal.len()];
    for i in 0..signal.len() {
        let w = window(signal, i, half);
        let mut wm = w.clone();
        let med = median(&mut wm);
        // scaled median absolute deviation
        let mut devs: Vec<f64> = w.iter().map(|v| (v - med).abs()).collect();
        let mad = 1.4826 * median(&mut devs);
        // A non-finite sample is an outlier by definition — there is no value to compare against a MAD.
        if !signal[i].is_finite() || (mad > 0.0 && (signal[i] - med).abs() > n_sigma * mad) {
            out[i] = med;
            flagged[i] = true;
        }
    }
    (out, flagged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_filter_removes_an_isolated_spike() {
        // THE ORACLE. A single spike in an otherwise constant signal is erased by the median.
        let mut sig = vec![1.0; 11];
        sig[5] = 100.0; // impulse
        let f = median_filter(&sig, 2);
        assert!(f.iter().all(|&v| (v - 1.0).abs() < 1e-12), "spike should be removed: {f:?}");
    }

    #[test]
    fn the_median_filter_preserves_a_step_edge() {
        // THE HEADLINE. A step from 0 to 10 stays a sharp step (median is edge-preserving), whereas a moving
        // average would smear it across the window.
        let sig: Vec<f64> = (0..20).map(|i| if i < 10 { 0.0 } else { 10.0 }).collect();
        let f = median_filter(&sig, 2);
        // still 0 before the edge and 10 after, with no intermediate ramp at the transition interior
        assert!((f[7] - 0.0).abs() < 1e-12 && (f[12] - 10.0).abs() < 1e-12, "step should be preserved: {f:?}");
        assert!(f.iter().all(|&v| v == 0.0 || v == 10.0), "no intermediate values introduced");
    }

    #[test]
    fn a_constant_passes_through_unchanged() {
        let sig = vec![3.5; 15];
        assert!(median_filter(&sig, 3).iter().all(|&v| (v - 3.5).abs() < 1e-12));
    }

    #[test]
    fn hampel_corrects_outliers_and_leaves_inliers() {
        // THE APPLICATION. A gently-varying signal with two injected outliers: Hampel replaces exactly those
        // and leaves the clean samples bit-for-bit unchanged.
        let mut sig: Vec<f64> = (0..30).map(|i| (i as f64 * 0.2).sin()).collect();
        let clean = sig.clone();
        sig[10] += 5.0;
        sig[22] -= 4.0;
        let (out, flagged) = hampel_filter(&sig, 3, 3.0);
        assert!(flagged[10] && flagged[22], "the injected outliers should be flagged");
        assert_eq!(flagged.iter().filter(|&&b| b).count(), 2, "only the two outliers flagged");
        // inliers untouched
        for i in 0..30 {
            if i != 10 && i != 22 {
                assert!((out[i] - clean[i]).abs() < 1e-12, "inlier {i} changed");
            }
        }
        // the corrected samples are close to the clean signal (within the local variation)
        assert!((out[10] - clean[10]).abs() < 0.5 && (out[22] - clean[22]).abs() < 0.5, "outliers corrected toward the trend");
    }

    /// **A non-finite sample is the sensor fault this filter exists to remove.**
    ///
    /// The module doc names LiDAR dropouts and single-sample sensor faults as the target, and a dropout
    /// is reported as a non-finite value by every sensor stack that has one. Sorting the window with
    /// `partial_cmp().unwrap()` panicked on it instead, so the filter could not survive its own purpose.
    #[test]
    fn a_non_finite_sample_is_despiked_like_any_other_fault() {
        let mut sig = vec![1.0; 11];
        sig[5] = f64::NAN; // a dropout
        sig[8] = f64::INFINITY; // a saturated return
        let f = median_filter(&sig, 2);
        assert!(f.iter().all(|v| (v - 1.0).abs() < 1e-12), "both faults should be replaced by the local median: {f:?}");
        let (h, flagged) = hampel_filter(&sig, 2, 3.0);
        assert!(h.iter().all(|v| (v - 1.0).abs() < 1e-12), "Hampel must replace them too: {h:?}");
        assert!(flagged[5] && flagged[8], "and flag them as outliers: {flagged:?}");
        assert_eq!(flagged.iter().filter(|&&b| b).count(), 2, "nothing else may be flagged: {flagged:?}");
        let all_bad = median_filter(&[f64::NAN; 5], 1);
        assert!(all_bad.iter().all(|v| !v.is_finite()), "a window with nothing real yields nothing real: {all_bad:?}");

        // ADJACENT dropouts, so a window is half or more invalid. A single dropout sorts to the end of an
        // odd window and leaves a real median untouched, so only this case shows that EXCLUDING the
        // non-finite samples matters rather than merely ordering them last.
        let burst = [1.0, f64::NAN, f64::NAN, 1.0, 1.0, 1.0, 1.0];
        let fb = median_filter(&burst, 1);
        assert!(fb.iter().all(|v| (v - 1.0).abs() < 1e-12), "a burst of dropouts must still reconstruct: {fb:?}");
        let (hb, fl) = hampel_filter(&burst, 1, 3.0);
        assert!(hb.iter().all(|v| (v - 1.0).abs() < 1e-12), "Hampel must reconstruct the burst too: {hb:?}");
        assert!(fl[1] && fl[2], "both dropouts flagged: {fl:?}");
    }
}
