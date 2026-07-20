//! **Digital IIR filtering — second-order Butterworth biquads**. A causal, `O(1)`-per-sample recursive
//! filter for real-time signal conditioning: smoothing accelerometer / force-torque / encoder streams,
//! anti-aliasing before down-sampling, and band-limiting control references. A 2nd-order **Butterworth**
//! section is maximally flat in the passband; its coefficients come from the analog prototype via the
//! **bilinear transform** at a chosen cutoff and sample rate, and it runs in the numerically-friendly
//! **transposed direct-form II** (two state variables). Unlike the crate's [`crate::fft`] (a batch
//! frequency-domain transform) or [`crate::SavGol`] (a windowed polynomial smoother), this processes samples
//! one at a time with fixed cost — the right tool inside a control loop.
//!
//! Verified: unity DC gain (a constant passes unchanged at steady state); a low-frequency sinusoid passes
//! while a high-frequency one is strongly attenuated (and the high-pass does the reverse); and a step
//! settles to its final value. Pure Rust → WASM-clean.

use std::f64::consts::PI;

/// A biquad (2nd-order IIR) section in transposed direct-form II. Denominator normalized (`a0 = 1`).
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// A 2nd-order **Butterworth low-pass** at cutoff `fc` (Hz) for sample rate `fs` (Hz).
    pub fn butterworth_lowpass(fc: f64, fs: f64) -> Self {
        let k = (PI * fc / fs).tan();
        let k2 = k * k;
        let norm = 1.0 / (1.0 + std::f64::consts::SQRT_2 * k + k2);
        let b0 = k2 * norm;
        Biquad {
            b0,
            b1: 2.0 * b0,
            b2: b0,
            a1: 2.0 * (k2 - 1.0) * norm,
            a2: (1.0 - std::f64::consts::SQRT_2 * k + k2) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// A 2nd-order **Butterworth high-pass** at cutoff `fc` (Hz) for sample rate `fs` (Hz).
    pub fn butterworth_highpass(fc: f64, fs: f64) -> Self {
        let k = (PI * fc / fs).tan();
        let k2 = k * k;
        let norm = 1.0 / (1.0 + std::f64::consts::SQRT_2 * k + k2);
        Biquad {
            b0: norm,
            b1: -2.0 * norm,
            b2: norm,
            a1: 2.0 * (k2 - 1.0) * norm,
            a2: (1.0 - std::f64::consts::SQRT_2 * k + k2) * norm,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Process one input sample, returning the filtered output (transposed direct-form II).
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    /// Reset the filter's internal state to zero.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// The DC (zero-frequency) gain, `Σb / (1 + Σa)`.
    pub fn dc_gain(&self) -> f64 {
        (self.b0 + self.b1 + self.b2) / (1.0 + self.a1 + self.a2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // steady-state amplitude of the filter's response to a sinusoid at frequency `f`
    fn amplitude(mut bq: Biquad, f: f64, fs: f64) -> f64 {
        let n = 4000;
        let mut peak = 0.0f64;
        for k in 0..n {
            let y = bq.process((2.0 * PI * f * k as f64 / fs).sin());
            if k > n / 2 {
                peak = peak.max(y.abs());
            }
        }
        peak
    }

    #[test]
    fn the_low_pass_has_unity_dc_gain() {
        // THE ORACLE. A constant input passes unchanged at steady state (DC gain = 1).
        let lp = Biquad::butterworth_lowpass(5.0, 100.0);
        assert!((lp.dc_gain() - 1.0).abs() < 1e-12, "DC gain {}", lp.dc_gain());
        let mut f = lp;
        let mut y = 0.0;
        for _ in 0..500 {
            y = f.process(3.0);
        }
        assert!((y - 3.0).abs() < 1e-6, "constant should pass: {y}");
    }

    #[test]
    fn the_low_pass_passes_low_and_blocks_high() {
        // THE HEADLINE. With fc = 10 Hz at fs = 200 Hz, a 2 Hz tone passes (≈ unit amplitude) and a 60 Hz
        // tone is strongly attenuated.
        let fs = 200.0;
        let lp = || Biquad::butterworth_lowpass(10.0, fs);
        let low = amplitude(lp(), 2.0, fs);
        let high = amplitude(lp(), 60.0, fs);
        assert!(low > 0.95, "2 Hz should pass: {low}");
        assert!(high < 0.05, "60 Hz should be blocked: {high}");
    }

    #[test]
    fn the_high_pass_blocks_low_and_passes_high() {
        let fs = 200.0;
        let hp = || Biquad::butterworth_highpass(20.0, fs);
        let low = amplitude(hp(), 2.0, fs);
        let high = amplitude(hp(), 80.0, fs);
        assert!(low < 0.05, "2 Hz should be blocked by the high-pass: {low}");
        assert!(high > 0.9, "80 Hz should pass the high-pass: {high}");
        assert!(Biquad::butterworth_highpass(20.0, fs).dc_gain().abs() < 1e-9, "high-pass DC gain ≈ 0");
    }

    #[test]
    fn a_step_settles_to_the_step_value() {
        let mut lp = Biquad::butterworth_lowpass(8.0, 100.0);
        let mut y = 0.0;
        for _ in 0..1000 {
            y = lp.process(2.5);
        }
        assert!((y - 2.5).abs() < 1e-6, "step should settle at 2.5: {y}");
    }
}
