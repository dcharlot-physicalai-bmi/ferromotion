//! **N-step capturability** — how many steps a robot needs to stop, in closed form.
//!
//! The instantaneous capture point already lives in [`zmp`](crate::zmp) and the divergent component of
//! motion in [`dcm`](crate::dcm). What those give is a *point*: where to put a foot to come to rest in one
//! step. What they do not give is the question a balance controller actually has to answer, which is whether
//! the robot can stop **at all** from where it is, and if so in how many steps.
//!
//! Koolen's answer is a recursion, and it collapses to a closed form. Write the linear inverted pendulum's
//! divergent component as `ξ`, measured from the current centre of pressure. During a step of duration `T`
//! with the pressure held, the offset grows by exactly `e^{ωT}`; then a foot lands up to `l_max` away, which
//! subtracts from the offset. Being `N`-step capturable means surviving that `N` times and landing inside
//! the foot at the end:
//!
//! ```text
//! d₀   = r_foot                        (the offset a stationary support can already hold)
//! d_N  = (d_{N−1} + l_max) · e^{−ωT}
//! ```
//!
//! so
//!
//! ```text
//! d_N  = r_foot·e^{−NωT} + l_max·(e^{−ωT} − e^{−(N+1)ωT}) / (1 − e^{−ωT})
//! d_∞  = l_max / (e^{ωT} − 1)
//! ```
//!
//! One sign is worth stating explicitly because the intuition runs the wrong way: `ω = √(g/z₀)` *decreases*
//! with centre-of-mass height, so a **taller** robot diverges more slowly and captures a **larger** region.
//! A long pendulum topples lazily. The hardware lever that hurts is a low, quick body, not a tall one.
//!
//! Three things fall out that are worth having as numbers rather than intuitions. The regions **nest** and
//! **saturate**: there is a finite `d_∞` no number of steps can exceed, so beyond it a fall is not a control
//! failure but a kinematic fact. Faster stepping helps and there is a *rate* for how much. And every term is
//! a hardware parameter — foot size, leg length, step time — so this is where a certificate meets a bill of
//! materials.
//!
//! [`capturable_in`] and [`steps_to_capture`] answer the question directly; the tests answer it a second way
//! by simulating the pendulum with a greedy stepping policy and checking the boundary is where the formula
//! says it is.

/// The linear-inverted-pendulum balance parameters that set the capture regions.
#[derive(Clone, Copy, Debug)]
pub struct CaptureParams {
    /// Pendulum rate `ω = √(g/z₀)`.
    pub omega: f64,
    /// How far the centre of pressure can be shifted within the stance foot, in metres. This is the
    /// zero-step region: an offset inside it can be held without stepping at all.
    pub foot_radius: f64,
    /// Maximum step length, in metres.
    pub step_length: f64,
    /// Step duration, in seconds. Shorter is strictly better here, which is the formal version of "take
    /// quicker steps when you are falling", and the closed form says by exactly how much.
    pub step_time: f64,
}

impl CaptureParams {
    /// Balance parameters from a centre-of-mass height under gravity `g`.
    pub fn from_height(com_height: f64, g: f64, foot_radius: f64, step_length: f64, step_time: f64) -> Option<CaptureParams> {
        if com_height <= 0.0 || g <= 0.0 || foot_radius < 0.0 || step_length < 0.0 || step_time <= 0.0 {
            return None;
        }
        Some(CaptureParams { omega: (g / com_height).sqrt(), foot_radius, step_length, step_time })
    }

    /// The **`N`-step capture boundary** `d_N`: the largest divergent-component offset from the current
    /// centre of pressure from which the robot can still come to rest within `N` steps.
    ///
    /// `n = 0` returns the foot radius, since holding the pressure under the divergent component is what
    /// stops it without stepping.
    pub fn boundary(&self, n: usize) -> f64 {
        let a = (-self.omega * self.step_time).exp();
        if (1.0 - a).abs() < 1e-15 {
            return self.foot_radius + n as f64 * self.step_length; // degenerate: no divergence to fight
        }
        let an = a.powi(n as i32);
        self.foot_radius * an + self.step_length * (a - a * an) / (1.0 - a)
    }

    /// The **saturation boundary** `d_∞ = l_max/(e^{ωT} − 1)`: no number of steps captures an offset beyond
    /// this. Past it the robot is falling for reasons no controller reaches, and the honest response is to
    /// change the hardware or the step time, not the policy.
    pub fn boundary_limit(&self) -> f64 {
        let e = (self.omega * self.step_time).exp();
        if e <= 1.0 + 1e-15 {
            return f64::INFINITY;
        }
        self.step_length / (e - 1.0)
    }

    /// Whether an offset is capturable within `n` steps.
    pub fn capturable_in(&self, offset: f64, n: usize) -> bool {
        offset.abs() <= self.boundary(n) + 1e-12
    }

    /// The **fewest steps** that capture this offset, or `None` if it lies beyond the saturation boundary
    /// and no number of steps suffices. Bounded search, since the boundaries saturate.
    pub fn steps_to_capture(&self, offset: f64) -> Option<usize> {
        let o = offset.abs();
        if o > self.boundary_limit() + 1e-12 {
            return None;
        }
        (0..1000).find(|&n| o <= self.boundary(n) + 1e-12)
    }

    /// The **capture point** for the current state, relative to the centre of pressure: where the divergent
    /// component will be, hence where a foot must land to stop in one step.
    pub fn capture_point(&self, position: f64, velocity: f64) -> f64 {
        position + velocity / self.omega
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn humanoid() -> CaptureParams {
        // roughly a 0.9 m centre of mass, a 10 cm usable foot, a 60 cm step, a 0.35 s swing
        CaptureParams::from_height(0.9, 9.81, 0.10, 0.60, 0.35).unwrap()
    }

    /// The recursion and the closed form must agree, since the closed form is the thing the rest of the
    /// module is built on and a summation slip in it would be invisible.
    #[test]
    fn the_closed_form_matches_the_recursion_it_came_from() {
        let p = humanoid();
        let a = (-p.omega * p.step_time).exp();
        let mut d = p.foot_radius; // d_0
        for n in 0..12 {
            assert!((p.boundary(n) - d).abs() < 1e-12, "closed form and recursion disagree at n = {n}: {} vs {d}", p.boundary(n));
            d = (d + p.step_length) * a;
        }
    }

    /// The regions **nest and saturate**: more steps never hurt, and there is a finite ceiling no number of
    /// steps passes. The ceiling is the part that matters, because it says some falls are kinematic.
    #[test]
    fn the_capture_regions_nest_and_saturate() {
        let p = humanoid();
        let limit = p.boundary_limit();
        let mut prev = -1.0;
        for n in 0..40 {
            let d = p.boundary(n);
            // non-decreasing, not strictly increasing: the whole point is that these saturate, so far enough
            // out consecutive boundaries coincide to the last bit
            assert!(d >= prev, "regions must never shrink with a larger step budget");
            assert!(d < limit + 1e-12, "no budget may exceed the saturation boundary: d_{n} = {d} vs {limit}");
            if n < 6 {
                assert!(d > prev, "an extra step must still buy something at n = {n}");
            }
            prev = d;
        }
        eprintln!("humanoid: d_0 = {:.4} m, d_1 = {:.4}, d_2 = {:.4}, d_5 = {:.4}, d_inf = {limit:.4}", p.boundary(0), p.boundary(1), p.boundary(2), p.boundary(5));
        assert!((p.boundary(200) - limit).abs() < 1e-9, "the sequence must converge to the limit");
        // and an offset past the ceiling is reported as uncapturable rather than needing many steps
        assert!(p.steps_to_capture(limit * 1.01).is_none());
        assert_eq!(p.steps_to_capture(p.boundary(0) * 0.5), Some(0));
        assert_eq!(p.steps_to_capture(p.boundary(1) * 0.999), Some(1));
    }

    /// **The formula against a simulation.** Roll the pendulum forward with a greedy stepper — always step
    /// as far as allowed towards the divergent component — and count the steps it actually needs. The
    /// measured count must match `steps_to_capture`, and an offset just past a boundary must need one more
    /// step than one just inside it. This is the check that the algebra describes the robot.
    #[test]
    fn simulating_a_greedy_stepper_reproduces_the_predicted_step_counts() {
        let p = humanoid();
        let a = (p.omega * p.step_time).exp();

        // Simulate the exact LIP divergent-component dynamics between steps: xi grows by e^{omega T}
        // measured from the held centre of pressure, then the next foot lands up to l_max away.
        let greedy_steps = |offset: f64| -> Option<usize> {
            let mut xi = offset.abs();
            for n in 0..40 {
                if xi <= p.foot_radius + 1e-9 {
                    return Some(n); // the pressure can hold it: at rest without another step
                }
                // grow through the swing, then place the foot as far towards xi as the leg allows
                xi = xi * a - p.step_length;
                if xi < -p.foot_radius {
                    // overshot past the foot on the far side, which is still a stop
                    return Some(n + 1);
                }
            }
            None
        };

        for n in 0..6 {
            let d = p.boundary(n);
            // just inside the boundary: capturable in n
            let inside = greedy_steps(d * (1.0 - 1e-9));
            assert_eq!(inside, Some(n), "an offset just inside d_{n} = {d:.5} should take {n} steps, took {inside:?}");
            // just outside: needs one more
            let outside = greedy_steps(d * (1.0 + 1e-6));
            assert_eq!(outside, Some(n + 1), "an offset just outside d_{n} should take {} steps, took {outside:?}", n + 1);
            eprintln!("   d_{n} = {d:.5} m: inside -> {inside:?} steps, outside -> {outside:?} steps");
        }
        // and the formula's own verdict agrees with the simulation across a sweep
        for k in 1..60 {
            let off = k as f64 * 0.02;
            assert_eq!(p.steps_to_capture(off), greedy_steps(off), "disagreement at offset {off:.3} m");
        }
    }

    /// **Where the certificate meets the bill of materials.** Every term is hardware. Faster stepping, a
    /// longer stride, and a *higher* centre of mass all enlarge the regions — the last because `ω = √(g/z₀)`
    /// falls with height, so a taller body diverges more slowly.
    #[test]
    fn the_capture_region_responds_to_hardware_the_way_it_should() {
        let base = humanoid();
        let quicker = CaptureParams { step_time: base.step_time * 0.5, ..base };
        let longer = CaptureParams { step_length: base.step_length * 1.5, ..base };
        let taller = CaptureParams::from_height(1.4, 9.81, base.foot_radius, base.step_length, base.step_time).unwrap();

        eprintln!("d_inf: baseline {:.4} m, half the step time {:.4}, 1.5x stride {:.4}, taller CoM {:.4}", base.boundary_limit(), quicker.boundary_limit(), longer.boundary_limit(), taller.boundary_limit());
        assert!(quicker.boundary_limit() > base.boundary_limit(), "quicker steps must capture more");
        assert!(longer.boundary_limit() > base.boundary_limit(), "a longer stride must capture more");
        // Taller captures MORE, not less: omega = sqrt(g/z) falls with height, so the divergence is slower
        // and the swing has time to catch it. The opposite reading is the natural guess and it is wrong.
        assert!(taller.boundary_limit() > base.boundary_limit(), "a taller robot diverges more slowly and captures more");
        // the stride enters linearly, which makes the trade explicit rather than a matter of tuning
        assert!((longer.boundary_limit() / base.boundary_limit() - 1.5).abs() < 1e-12);
    }

    /// The capture point is the divergent component, and holding the pressure there brings the pendulum to
    /// rest. Stepped with the **exact** linear-inverted-pendulum flow map rather than a finite-difference
    /// integrator, and for a specific reason: the capture point lies on an *unstable manifold*, where the
    /// divergent coefficient is zero only exactly. Any integrator injects a small divergent component and
    /// then amplifies it by `e^{ωt}` — over 1 s that is a factor of 27, over 3 s a factor of 2e4 — so an
    /// Euler simulation drifts off the capture point for purely numerical reasons and says nothing about the
    /// control law. Using the true flow map keeps the test about the physics.
    #[test]
    fn placing_the_foot_on_the_capture_point_brings_the_pendulum_to_rest() {
        let p = humanoid();
        let (x0, v0) = (0.0f64, 0.45f64);
        let w = p.omega;

        // exact flow of ẍ = ω²(x − cop) over `dt`, in coordinates relative to the pressure point
        let step = |x: f64, v: f64, cop: f64, dt: f64| {
            let (c, sh) = ((w * dt).cosh(), (w * dt).sinh());
            let (e, ev) = (x - cop, v);
            (cop + e * c + ev * sh / w, e * w * sh + ev * c)
        };

        let cop = p.capture_point(x0, v0);
        let (mut x, mut v) = (x0, v0);
        for _ in 0..1000 {
            (x, v) = step(x, v, cop, 3e-3); // 3 s in exact 3 ms increments
        }
        // With the divergent coefficient exactly zero the motion is a single decaying exponential, so the
        // remaining velocity is not "small" but a specific number: v0 e^{-omega t}. Checking against that is
        // sharper than any tolerance, and it confirms the divergent mode is absent rather than merely quiet.
        let analytic_v = v0 * (-w * 3.0).exp();
        eprintln!("capture point at {cop:.4} m: after 3 s v = {v:.4e}, analytic v0 e^-wt = {analytic_v:.4e}");
        assert!((v - analytic_v).abs() / analytic_v < 1e-6, "the decay must be the pure exponential: {v:.4e} vs {analytic_v:.4e}");
        assert!((x - cop).abs() < 1e-4, "and settle at the capture point, off by {:.2e}", (x - cop).abs());
        // Deliberately not run further. By 6 s the analytic velocity is 1.1e-9 while round-off in the
        // divergent coefficient has been amplified by e^{19.8} = 4e8, so the simulation stops tracking the
        // exponential and a "still at rest" assertion would be testing floating point. The agreement above
        // is the real evidence.

        // A foot placed 5 cm short leaves a positive divergent coefficient, and the pendulum runs away —
        // the same integrator, so the contrast is about the placement and nothing else.
        let (mut x2, mut v2) = (x0, v0);
        let short = cop - 0.05;
        for _ in 0..333 {
            (x2, v2) = step(x2, v2, short, 3e-3); // 1 s
        }
        eprintln!("   a foot 5 cm short: after 1 s x = {x2:.4} m, v = {v2:.4} m/s - diverging");
        assert!(v2 > 0.5, "5 cm short must fail to stop it, velocity {v2:.3}");

        // The sharp statement: the capture point is exactly the placement whose divergent coefficient
        // vanishes, so a placement error `d` injects a divergent coefficient of `d/2` and nothing else.
        // Against the nominal trajectory that is an exact relation,
        //
        //     v_d(t) − v_0(t) = ω sinh(ωt) · d
        //
        // and it is *affine* in `d`, not proportional: the nominal run has its own decaying residual, which
        // is why the raw ratio v_d/d drifts with `d` and has to be differenced out.
        let t = 0.999_f64; // 333 steps of 3 ms
        let run = |cop: f64| {
            let (mut xa, mut va) = (x0, v0);
            for _ in 0..333 {
                (xa, va) = step(xa, va, cop, 3e-3);
            }
            va
        };
        let nominal = run(cop);
        let expected_slope = w * (w * t).sinh();
        for d in [0.01f64, 0.02, 0.04] {
            let slope = (run(cop - d) - nominal) / d;
            eprintln!("   placement off by {d:.2} m: divergent slope {slope:.3}, expected w sinh(wt) = {expected_slope:.3}");
            assert!((slope - expected_slope).abs() / expected_slope < 1e-6, "the divergent part must be exactly linear in the placement error: {slope} vs {expected_slope}");
        }
    }
}
