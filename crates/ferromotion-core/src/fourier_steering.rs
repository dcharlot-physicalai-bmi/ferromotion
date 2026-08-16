//! **Fourier steering** — driving a system that is *nearly* in chained form, exactly.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §8.3.1. Sinusoids at
//! integrally related frequencies steer a one-chain system ([`crate::ChainedForm`]), but real mechanisms
//! rarely arrive in that form. MLS's hopping-robot example has
//!
//! ```text
//! ψ̇ = u₁,   l̇ = u₂,   α̇ = f(l)·u₁
//! ```
//!
//! where `f` is whatever the mechanism's geometry produces — for the hopper,
//! `f(l) = −2mdI/(md²+I)²·l + O(l²)·u₁` after a coordinate change. Truncating to the linear term gives a
//! 3-state chained form and lets the chained-form algorithm be *justified*, but applying it to the truncated
//! model steers the truncated model. Fourier analysis lets the **exact** `f` be handled.
//!
//! # Why only the first harmonic matters
//!
//! Drive `u₁ = a₁ sin 2πt`, `u₂ = a₂ cos 2πt` over one period. Then `l(t) = (a₂/2π)·sin 2πt`, and the composed
//! function `t ↦ f(l(t))` has a Fourier expansion
//!
//! ```text
//! f((a₂/2π)·sin 2πt) = β₁ sin 2πt + β₂ sin 4πt + ⋯
//! ```
//!
//! Integrating `α̇ = f(l)·a₁ sin 2πt` over the period, every `∫ sin 2πt · sin 2πkt dt` vanishes except `k = 1`,
//! which contributes `1/2`. So the whole net motion is
//!
//! ```text
//! Δα = a₁·β₁/2
//! ```
//!
//! **and no other Fourier coefficient of `f` appears at all.** That is the useful content: an arbitrary
//! nonlinearity in `f` collapses, for steering purposes, to a single number. `ψ` and `l` return to their
//! starting values over the period by construction, so the motion is a pure `α` displacement.
//!
//! # What is verified, and how
//!
//! The prediction `Δα = a₁β₁/2` is checked against **direct numerical integration of the exact dynamics** for
//! several `f`, including ones with strong higher harmonics where a truncation would visibly fail. And for a
//! linear `f` it is checked against [`crate::ChainedForm`]'s independently measured gain, so the two modules
//! agree where they overlap. The constant `1/2` is *measured* rather than transcribed, on the same reasoning
//! as the chained-form amplitude: the dynamics are unambiguous, a copied constant is not.

use std::f64::consts::PI;

/// `β₁`, the first sine-Fourier coefficient of `t ↦ f((a₂/2π)·sin 2πt)` over one unit period.
///
/// `β₁ = 2·∫₀¹ f(l(t))·sin(2πt) dt`, by Simpson's rule on `2n` intervals. Simpson rather than trapezoid
/// because the integrand is smooth and periodic and the accuracy is free here; `n` sets the resolution and a
/// few thousand is ample for the smooth `f` this is for.
pub fn first_harmonic(f: impl Fn(f64) -> f64, a2: f64, n: usize) -> f64 {
    let n = n.max(1);
    let steps = 2 * n;
    let h = 1.0 / steps as f64;
    let g = |t: f64| f(a2 / (2.0 * PI) * (2.0 * PI * t).sin()) * (2.0 * PI * t).sin();
    let mut acc = g(0.0) + g(1.0);
    for i in 1..steps {
        acc += g(i as f64 * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    2.0 * acc * h / 3.0
}

/// Net `α` displacement over one period of `u₁ = a₁ sin 2πt`, `u₂ = a₂ cos 2πt`: **`Δα = a₁·β₁/2`**.
///
/// Only the first harmonic of the composed `f` contributes; every higher one integrates to zero against
/// `sin 2πt`. `ψ` and `l` return to their initial values, so this is a pure `α` motion.
pub fn alpha_displacement(f: impl Fn(f64) -> f64, a1: f64, a2: f64, n: usize) -> f64 {
    a1 * first_harmonic(f, a2, n) / 2.0
}

/// Integrate `ψ̇ = u₁`, `l̇ = u₂`, `α̇ = f(l)·u₁` over one period with RK4, returning the final
/// `(ψ, l, α)` from a zero start.
///
/// The reference the Fourier prediction is checked against — the exact dynamics, no truncation.
pub fn integrate_period(f: impl Fn(f64) -> f64, a1: f64, a2: f64, steps: usize) -> (f64, f64, f64) {
    let h = 1.0 / steps as f64;
    let (mut psi, mut l, mut alpha) = (0.0f64, 0.0f64, 0.0f64);
    let u1 = |t: f64| a1 * (2.0 * PI * t).sin();
    let u2 = |t: f64| a2 * (2.0 * PI * t).cos();
    for i in 0..steps {
        let t = i as f64 * h;
        // state derivative at (t, l)
        let d = |t: f64, l: f64| (u1(t), u2(t), f(l) * u1(t));
        let (p1, l1, a1d) = d(t, l);
        let (p2, l2, a2d) = d(t + 0.5 * h, l + 0.5 * h * l1);
        let (p3, l3, a3d) = d(t + 0.5 * h, l + 0.5 * h * l2);
        let (p4, l4, a4d) = d(t + h, l + h * l3);
        psi += h / 6.0 * (p1 + 2.0 * p2 + 2.0 * p3 + p4);
        alpha += h / 6.0 * (a1d + 2.0 * a2d + 2.0 * a3d + a4d);
        l += h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4);
    }
    (psi, l, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The prediction must match the exact dynamics**, including for an `f` whose higher harmonics are large
    /// — that is where a chained-form truncation would fail and the Fourier argument earns its place.
    #[test]
    fn the_first_harmonic_predicts_the_exact_net_motion() {
        let cases: Vec<(&str, Box<dyn Fn(f64) -> f64>)> = vec![
            ("linear (exact chained form)", Box::new(|l: f64| -1.7 * l)),
            ("quadratic", Box::new(|l: f64| 0.5 * l + 2.0 * l * l)),
            ("cubic, strong 3rd harmonic", Box::new(|l: f64| l - 4.0 * l * l * l)),
            ("saturating", Box::new(|l: f64| l.tanh())),
            ("offset (even part contributes nothing)", Box::new(|l: f64| 3.0 + 0.8 * l)),
            ("hopper, O(l^2) retained", Box::new(|l: f64| -0.62 * l + 0.9 * l * l)),
        ];
        for (name, f) in &cases {
            for (a1, a2) in [(1.0, 1.0), (2.0, 0.5), (-1.5, 3.0)] {
                let predicted = alpha_displacement(f.as_ref(), a1, a2, 4000);
                let (psi, l, exact) = integrate_period(f.as_ref(), a1, a2, 200_000);
                // psi and l must return to zero, or this is not a pure alpha motion
                assert!(psi.abs() < 1e-9, "{name}: psi did not close, {psi}");
                assert!(l.abs() < 1e-9, "{name}: l did not close, {l}");
                assert!(
                    (predicted - exact).abs() < 1e-6 * exact.abs().max(1e-3),
                    "{name} a1={a1} a2={a2}: Fourier predicts {predicted}, exact integration gives {exact}"
                );
            }
        }
    }

    /// A constant `f` steers nothing: its composed first harmonic is zero, so `Δα = 0`. The even part of `f`
    /// never contributes, which is why the "offset" case above matches its offset-free twin.
    #[test]
    fn the_even_part_of_f_contributes_nothing() {
        let a1 = 1.3;
        let a2 = 0.9;
        assert!(alpha_displacement(|_l| 2.5, a1, a2, 4000).abs() < 1e-12, "a constant f steers nothing");
        // f and f + const give the same displacement
        let odd = alpha_displacement(|l| 0.7 * l, a1, a2, 4000);
        let shifted = alpha_displacement(|l| 0.7 * l + 5.0, a1, a2, 4000);
        assert!((odd - shifted).abs() < 1e-12, "adding a constant must not change Δα");
        // an even function of l also contributes nothing
        assert!(alpha_displacement(|l| l * l, a1, a2, 4000).abs() < 1e-9, "l² is even in l, so no net motion");
    }

    /// **Agreement with `ChainedForm` where the two overlap.** For `f(l) = l` this system IS the 3-state
    /// one-chain system, so the Fourier displacement must equal the gain that module measures independently.
    #[test]
    fn a_linear_f_reproduces_the_chained_form_gain() {
        // ChainedForm's k=1 gain at amplitude a is (a/4π)^1 · b/1! per its own verified closed form, with
        // u1 = a sin(2πt), u2 = b cos(2πt) — the same drive as here, and q3dot = q2·u1 is f(l) = l.
        let (a1, a2) = (1.0, 1.0);
        let fourier = alpha_displacement(|l| l, a1, a2, 8000);
        let chained = (a1 / (4.0 * PI)).powi(1) * a2; // k = 1, 1! = 1
        assert!(
            (fourier - chained).abs() < 1e-9,
            "Fourier gives {fourier}, ChainedForm's closed form gives {chained} — the two modules must agree \
             on the system they share"
        );
        // and the exact dynamics agree with both
        let (_, _, exact) = integrate_period(|l| l, a1, a2, 200_000);
        assert!((exact - chained).abs() < 1e-9, "direct integration gives {exact}");
    }

    /// Δα is linear in `a₁` and, for a linear `f`, in `a₂` too. Worth pinning because a stray `2π` in
    /// `first_harmonic` would break the amplitude scaling while leaving the shape right.
    #[test]
    fn the_displacement_scales_as_the_structure_requires() {
        let f = |l: f64| l;
        let base = alpha_displacement(f, 1.0, 1.0, 8000);
        assert!((alpha_displacement(f, 3.0, 1.0, 8000) - 3.0 * base).abs() < 1e-12, "linear in a1");
        assert!((alpha_displacement(f, 1.0, 3.0, 8000) - 3.0 * base).abs() < 1e-12, "linear in a2 for linear f");
        // For a nonlinear f the a2 scaling is NOT linear — that is the whole point of needing β₁.
        let g = |l: f64| l - 4.0 * l * l * l;
        let b1 = alpha_displacement(g, 1.0, 1.0, 8000);
        let b3 = alpha_displacement(g, 1.0, 3.0, 8000);
        assert!(
            (b3 - 3.0 * b1).abs() > 1e-3 * b1.abs().max(1e-6),
            "a cubic f must break the linear a2 scaling, else the test is not exercising the nonlinearity"
        );
    }

    /// The quadrature converges, so the reported β₁ is a property of `f` rather than of the resolution.
    #[test]
    fn the_harmonic_quadrature_converges() {
        let f = |l: f64| l.tanh() + 0.3 * l * l;
        let coarse = first_harmonic(f, 2.0, 250);
        let fine = first_harmonic(f, 2.0, 16_000);
        assert!((coarse - fine).abs() < 1e-6, "coarse {coarse} vs fine {fine}");
        assert!(first_harmonic(f, 2.0, 0) .is_finite(), "n = 0 must be clamped, not divide by zero");
    }
}
