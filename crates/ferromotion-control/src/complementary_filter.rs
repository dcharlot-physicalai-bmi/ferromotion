//! Attitude complementary filter — fuse a gyroscope with an accelerometer to estimate tilt.
//!
//! A gyro is accurate over the short term but its rate has a slowly-varying bias, so integrating it
//! open-loop lets the angle drift without bound. An accelerometer, held roughly static, points along
//! gravity and gives a drift-free (but noisy) tilt reference. The classic first-order complementary
//! filter blends the two — a high-pass on the integrated gyro plus a low-pass on the accel angle:
//!
//! `angle ← α·(angle + gyro·dt) + (1−α)·accel_angle`.
//!
//! **That form is wrong across ±π, and this module blends on the CIRCLE instead (2026-08-15).** The accel
//! reference `atan2(ay, az)` is confined to `(−π, π]` while the gyro-integrated prediction accumulates without
//! bound, so as the true angle passes `π` the reference jumps by `2π` and a plain linear interpolation is pulled
//! toward the midpoint of two angles that are *physically adjacent but numerically `2π` apart*. Measured on a
//! body rolling at 1 rad/s (`α = 0.95`, `dt = 0.01`, noiseless accel): at `t = 3.20 s` the true roll is
//! `−3.0832` and the old form returned **`+1.5355`**; the worst error was **3.0642 rad = 175.6°** at
//! `t = 3.28 s` — *the filter reporting a body as essentially level while it is inverted* — and it took ≈2.9 s
//! of continued rolling to re-converge. Every crossing of `±π` repeated it.
//!
//! The correct form applies the correction to the *wrapped innovation*:
//!
//! `angle ← wrap( (angle + gyro·dt) + (1−α)·wrap(accel_angle − (angle + gyro·dt)) )`
//!
//! which is algebraically identical to the classic form whenever the two agree to within `±π`, and finite
//! everywhere. With it, the same sweep tracks to `0.00°`.
//!
//! `α ∈ (0,1)` sets the blend/time-constant `τ = α·dt/(1−α)`: large `α` trusts the gyro (smooth, but
//! a constant gyro bias `b` leaves a bounded steady offset ≈ `b·τ` instead of an unbounded ramp),
//! small `α` trusts the accel (fast bias rejection, more measurement noise passes through). Pure
//! `f64` arithmetic, no state beyond the running estimate — WASM-clean.

/// Wrap an angle into `(−π, π]`.
///
/// `rem_euclid` rather than a `while` loop: it is branch-free, exact for the large arguments an unbounded
/// gyro integration produces, and does not spin when the input is many turns out.
fn wrap_pi(a: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    (a + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
}

/// First-order complementary attitude filter. Holds the running 1-axis tilt estimate plus a
/// roll/pitch pair for the 3-axis [`update_rp`](ComplementaryFilter::update_rp) variant.
#[derive(Clone, Debug)]
pub struct ComplementaryFilter {
    /// Blend factor in `(0,1)`: weight on the integrated-gyro (high-pass) branch.
    pub alpha: f64,
    angle: f64,
    roll: f64,
    pitch: f64,
}

impl ComplementaryFilter {
    /// New filter with blend factor `alpha` and all angle states at zero.
    pub fn new(alpha: f64) -> Self {
        Self { alpha, angle: 0.0, roll: 0.0, pitch: 0.0 }
    }

    /// Seed the 1-axis estimate (and the roll state) — e.g. initialise straight from the first accel
    /// reading so the filter starts on the true tilt instead of at zero.
    pub fn with_angle(alpha: f64, angle0: f64) -> Self {
        Self { alpha, angle: angle0, roll: angle0, pitch: 0.0 }
    }

    /// Reset all angle states to zero (keeps `alpha`).
    pub fn reset(&mut self) {
        self.angle = 0.0;
        self.roll = 0.0;
        self.pitch = 0.0;
    }

    /// Current 1-axis tilt estimate (radians).
    pub fn angle(&self) -> f64 {
        self.angle
    }

    /// Current fused roll and pitch (radians).
    pub fn roll_pitch(&self) -> (f64, f64) {
        (self.roll, self.pitch)
    }

    /// One 1-axis step. `gyro_rate` is the measured angular rate (rad/s) about the tilt axis,
    /// `accel_angle` the tilt inferred from the accelerometer (rad). Returns the fused angle.
    /// **Returns a WRAPPED angle in `(−π, π]`.** The accel reference is inherently wrapped, so a fused
    /// estimate is only meaningful modulo `2π`; a caller needing a continuous unwrapped angle must accumulate
    /// its own turn count. See the module note on why the innovation is wrapped.
    pub fn update(&mut self, dt: f64, gyro_rate: f64, accel_angle: f64) -> f64 {
        let predicted = self.angle + gyro_rate * dt;
        self.angle = wrap_pi(predicted + (1.0 - self.alpha) * wrap_pi(accel_angle - predicted));
        self.angle
    }

    /// One 3-axis step. `gyro_xyz` is the body-rate vector `[ωx, ωy, ωz]` (rad/s) and `accel_xyz`
    /// the measured specific force `[ax, ay, az]` (any units — only the direction is used). Roll is
    /// integrated on `ωx`, pitch on `ωy`; the accelerometer reference is
    /// `roll = atan2(ay, az)`, `pitch = atan2(−ax, √(ay²+az²))`. Returns the fused `(roll, pitch)`.
    pub fn update_rp(&mut self, dt: f64, gyro_xyz: [f64; 3], accel_xyz: [f64; 3]) -> (f64, f64) {
        let [ax, ay, az] = accel_xyz;
        let roll_acc = ay.atan2(az);
        let pitch_acc = (-ax).atan2((ay * ay + az * az).sqrt());
        let roll_pred = self.roll + gyro_xyz[0] * dt;
        let pitch_pred = self.pitch + gyro_xyz[1] * dt;
        // Roll is the channel that crosses ±π; pitch cannot, because atan2(−ax, √(ay²+az²)) is confined to
        // ±π/2. Wrapping both is still correct — the pitch innovation is simply never large enough to differ.
        self.roll = wrap_pi(roll_pred + (1.0 - self.alpha) * wrap_pi(roll_acc - roll_pred));
        self.pitch = wrap_pi(pitch_pred + (1.0 - self.alpha) * wrap_pi(pitch_acc - pitch_pred));
        (self.roll, self.pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG → standard-normal samples (Box–Muller). No `rand`, fully reproducible.
    struct Lcg {
        s: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { s: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.s = self.s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.s
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn gauss(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-12);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    #[test]
    fn fuses_biased_gyro_and_noisy_accel_beating_both() {
        // Known time-varying tilt θ(t) = A·sin(ω t); true rate θ̇ = A·ω·cos(ω t).
        // Gyro reads θ̇ + BIAS (unbounded drift when integrated open-loop).
        // Accel reads θ + Gaussian noise (drift-free but noisy).
        let dt = 0.01_f64;
        let amp = 0.5;
        let w = 1.2;
        let bias = 0.12; // rad/s constant gyro bias — the drift source
        let sigma = 0.06; // accel angle noise std (rad)

        // τ = α·dt/(1−α): α=0.95 → τ≈0.19 s. Small enough that the steady bias offset (≈ b·τ) is
        // well under the accel noise, large enough to average that noise down.
        let mut cf = ComplementaryFilter::new(0.95);
        let mut rng = Lcg::new(0xA11C_E0FF_1234_5678);

        let mut gyro_int = 0.0_f64; // pure open-loop gyro integration (baseline a)
        let (mut se_f, mut se_gyro, mut se_acc, mut count) = (0.0, 0.0, 0.0, 0u32);

        let n = 4000; // 40 s — long enough for open-loop gyro drift to blow up
        for k in 0..n {
            let t = k as f64 * dt;
            let theta = amp * (w * t).sin();
            let rate = amp * w * (w * t).cos();

            let gyro = rate + bias;
            let accel_angle = theta + sigma * rng.gauss();

            gyro_int += gyro * dt; // baseline (a): drifts
            let est = cf.update(dt, gyro, accel_angle); // fused

            if k >= 500 {
                // skip startup transient
                se_f += (est - theta).powi(2);
                se_gyro += (gyro_int - theta).powi(2);
                se_acc += (accel_angle - theta).powi(2); // baseline (b): raw accel
                count += 1;
            }
        }
        let rmse_f = (se_f / count as f64).sqrt();
        let rmse_gyro = (se_gyro / count as f64).sqrt();
        let rmse_acc = (se_acc / count as f64).sqrt();

        // Sanity on the baselines: raw accel RMSE ≈ σ; open-loop gyro has drifted far past it.
        assert!(rmse_acc > 0.5 * sigma && rmse_acc < 1.6 * sigma, "raw accel RMSE ≈ σ, got {rmse_acc}");
        assert!(rmse_gyro > 5.0 * sigma, "gyro should have drifted badly, RMSE {rmse_gyro}");

        // The meaningful guarantee: the fused estimate beats BOTH baselines, and clearly.
        assert!(rmse_f < 0.6 * rmse_acc, "fused {rmse_f} not clearly better than raw accel {rmse_acc}");
        assert!(rmse_f < 0.2 * rmse_gyro, "fused {rmse_f} not clearly better than drifting gyro {rmse_gyro}");
    }

    #[test]
    fn roll_pitch_variant_tracks_a_tumbling_attitude() {
        // True roll φ(t) and pitch θ(t) sinusoids; gyro = derivative + bias, accel = the static
        // gravity direction in the body frame + noise. Assert the fused roll/pitch beat both the
        // open-loop-gyro and the raw-accel baselines.
        let dt = 0.01_f64;
        let g = 9.81;
        let (a_phi, w_phi) = (0.4, 1.0);
        let (a_th, w_th) = (0.3, 1.5);
        let bias = 0.10; // rad/s on each rate
        let sigma = 0.04 * g; // accel component noise (matched to g scale)

        let mut cf = ComplementaryFilter::new(0.95);
        let mut rng = Lcg::new(0x00C0_FFEE_D00D_1111);

        let (mut roll_int, mut pitch_int) = (0.0_f64, 0.0_f64);
        let (mut se_f, mut se_gyro, mut se_acc, mut count) = (0.0, 0.0, 0.0, 0u32);

        let n = 3000;
        for k in 0..n {
            let t = k as f64 * dt;
            let phi = a_phi * (w_phi * t).sin();
            let theta = a_th * (w_th * t).sin();
            let phid = a_phi * w_phi * (w_phi * t).cos();
            let thd = a_th * w_th * (w_th * t).cos();

            // Static specific force = body-frame gravity direction for (φ,θ).
            let ax = -g * theta.sin();
            let ay = g * phi.sin() * theta.cos();
            let az = g * phi.cos() * theta.cos();
            let accel = [ax + sigma * rng.gauss(), ay + sigma * rng.gauss(), az + sigma * rng.gauss()];
            let gyro = [phid + bias, thd + bias, 0.0];

            // Baseline (a): open-loop gyro integration.
            roll_int += gyro[0] * dt;
            pitch_int += gyro[1] * dt;
            // Baseline (b): raw accel-only inversion.
            let roll_acc = accel[1].atan2(accel[2]);
            let pitch_acc = (-accel[0]).atan2((accel[1] * accel[1] + accel[2] * accel[2]).sqrt());

            let (rf, pf) = cf.update_rp(dt, gyro, accel);

            if k >= 400 {
                se_f += (rf - phi).powi(2) + (pf - theta).powi(2);
                se_gyro += (roll_int - phi).powi(2) + (pitch_int - theta).powi(2);
                se_acc += (roll_acc - phi).powi(2) + (pitch_acc - theta).powi(2);
                count += 1;
            }
        }
        // RMSE over the two angles jointly.
        let rmse_f = (se_f / (2.0 * count as f64)).sqrt();
        let rmse_gyro = (se_gyro / (2.0 * count as f64)).sqrt();
        let rmse_acc = (se_acc / (2.0 * count as f64)).sqrt();

        assert!(rmse_f < rmse_acc, "fused {rmse_f} not better than raw accel {rmse_acc}");
        assert!(rmse_f < rmse_gyro, "fused {rmse_f} not better than drifting gyro {rmse_gyro}");
        assert!(rmse_f < 0.05, "fused roll/pitch RMSE {rmse_f} unexpectedly large");
    }

    #[test]
    fn a_roll_crossing_pi_does_not_report_the_body_as_level() {
        // The defect: the accel reference atan2(ay, az) is confined to (−π, π] while the gyro-integrated
        // prediction accumulates unbounded, so a plain linear blend across ±π is pulled toward the midpoint of
        // two angles that are physically adjacent but numerically 2π apart. Measured with the old form: at
        // t = 3.20 s the true roll is −3.0832 and it returned +1.5355; worst error 3.0642 rad = 175.6°.
        let wrap = |a: f64| {
            let two_pi = 2.0 * std::f64::consts::PI;
            (a + std::f64::consts::PI).rem_euclid(two_pi) - std::f64::consts::PI
        };
        let (alpha, dt, wx) = (0.95, 0.01, 1.0);
        let mut f = ComplementaryFilter::new(alpha);
        let mut worst = 0.0f64;
        let mut crossings = 0;
        let mut prev_true = 0.0f64;
        for i in 0..800 {
            let t = (i + 1) as f64 * dt;
            let true_roll = wrap(wx * t);
            if true_roll * prev_true < 0.0 && prev_true.abs() > 1.0 {
                crossings += 1; // sign flip near ±π, not near 0
            }
            prev_true = true_roll;
            // gravity in the body frame, so atan2(ay, az) is exactly the true roll
            let (ay, az) = (true_roll.sin(), true_roll.cos());
            let (roll, _) = f.update_rp(dt, [wx, 0.0, 0.0], [0.0, ay, az]);
            if t > 0.5 {
                worst = worst.max(wrap(roll - true_roll).abs());
            }
        }
        assert!(crossings >= 1, "the sweep must actually cross ±π to be a test, got {crossings}");
        assert!(
            worst < 1e-9,
            "worst roll error {worst} rad = {}° across a ±π crossing; the linear blend gave 3.0642 rad (175.6°)",
            worst.to_degrees()
        );
        // and the returned angle stays in range rather than winding up
        assert!(f.roll_pitch().0.abs() <= std::f64::consts::PI + 1e-12);
    }

    #[test]
    fn alpha_one_is_pure_integration_alpha_zero_is_pure_accel() {
        // Degenerate blends: α=1 ignores the accel entirely, α=0 ignores the gyro entirely.
        //
        // **Inputs stay inside `(−π, π]`, which the contract requires (2026-08-15).** This test used to feed
        // `acc = 0.5·k`, reaching 9.5 rad — a value `atan2(ay, az)` cannot return. Once the blend became
        // circular (see the module note; the old linear form was out by 175.6° across ±π) that input no longer
        // means anything: 9.5 rad and 9.5 − 2π are the same attitude, so "echo the accel angle" has two
        // different right answers. The property being checked is real; the input was not.
        let mut gyro_only = ComplementaryFilter::new(1.0);
        let mut accel_only = ComplementaryFilter::new(0.0);
        let dt = 0.1;
        let (mut integ, mut last_acc) = (0.0, 0.0);
        for k in 0..20 {
            let rate = 0.3;
            let acc = 0.15 * (k as f64 - 10.0); // sweeps −1.5 .. +1.35 rad, all inside (−π, π]
            integ += rate * dt;
            last_acc = acc;
            let g = gyro_only.update(dt, rate, acc);
            let a = accel_only.update(dt, rate, acc);
            assert!(integ.abs() < std::f64::consts::PI, "keep the integration inside one turn too");
            assert!((g - integ).abs() < 1e-12, "α=1 should be pure gyro integration, got {g} want {integ}");
            assert!((a - acc).abs() < 1e-12, "α=0 should echo the accel angle, got {a} want {acc}");
        }
        assert!((gyro_only.angle() - integ).abs() < 1e-12);
        assert!((accel_only.angle() - last_acc).abs() < 1e-12);
    }
}
