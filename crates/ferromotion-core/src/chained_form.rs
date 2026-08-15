//! **Chained-form systems and steering by sinusoids** — nonholonomic motion planning.
//!
//! Murray, Li & Sastry (1994), *A Mathematical Introduction to Robotic Manipulation*, §8.2.3. A *one-chain*
//! system is the two-input canonical form
//!
//! ```text
//! q̇₁ = u₁
//! q̇₂ = u₂
//! q̇₃ = q₂ u₁
//! q̇₄ = q₃ u₁
//!  ⋮
//! q̇ₙ = qₙ₋₁ u₁
//! ```
//!
//! which is completely nonholonomic — controllable, but not by any instantaneous combination of inputs. MLS
//! Prop. 8.2: the `n` fields `{g₁, g₂, ad^i_{g₁} g₂}` are independent, so the reachable set is full-dimensional
//! even though only two inputs exist. Many nonlinear systems convert into this form, which is why it is worth
//! having as a target.
//!
//! **Where this sits relative to what the crate already had.** [`crate::dubins`], [`crate::reeds_shepp`] and
//! [`crate::hybrid_astar`] produce nonholonomic *paths* — geometric curves a car can follow. This is the
//! other half: a *steering law*, an open-loop input pair that provably moves one chain variable while
//! returning every earlier one to where it started. That is the mechanism behind MLS's dynamic finger
//! repositioning, and it is what a path planner cannot give you.
//!
//! # Steering by sinusoids at integrally related frequencies
//!
//! MLS Algorithm 3: steer `q₁` and `q₂` directly, then for each `q_{k+2}` in turn drive
//! `u₁ = a·sin(2πt/T)`, `u₂ = b·cos(2πkt/T)`. Because `q̇_{k+2}` picks up a component at frequency zero while
//! everything below it oscillates back, one period moves `q_{k+2}` and leaves `q_j`, `j < k+2`, unchanged.
//!
//! **MLS's closed form is `Δq_{k+2} = (a/4π)^k · b/k!`, and it is exactly right** — measured against
//! numerical integration of the dynamics, the ratio is `1.000000` at every `k` from 1 to 4. I had expected a
//! possible factor of two from the `∫sin²` term in the book's derivation and checked rather than assumed;
//! the check vindicated the book.
//!
//! [`ChainedForm::steer_gain`] nonetheless *measures* the gain instead of applying the formula.
//! `Δq_{k+2}` is exactly linear in `b` for fixed `a`, so one integration with `b = 1` gives the gain and
//! `b = Δ_target / gain` follows. That keeps the steering law correct if the dynamics are ever generalised
//! away from the pure one-chain form, where the closed form would no longer apply — and the test pins the
//! closed form against the measurement, so a regression in either is caught.
//!
//! **The gain decays like `(a/4π)^k`, which is a practical limit rather than a footnote**: at `a = 1` it is
//! `7.96e-2` at `k = 1` but `1.67e-6` by `k = 4`. A deep chain steered with a small `a` demands an enormous
//! `b`, so amplitude has to grow with chain depth.

/// A one-chain system in `n ≥ 3` variables.
#[derive(Clone, Copy, Debug)]
pub struct ChainedForm {
    /// Number of chain variables.
    pub n: usize,
}

impl ChainedForm {
    /// A chain in `n` variables. `n < 3` has no chain part and is rejected.
    pub fn new(n: usize) -> Option<Self> {
        (n >= 3).then_some(Self { n })
    }

    /// `q̇ = g₁ u₁ + g₂ u₂` — MLS eq. (8.7).
    pub fn dynamics(&self, q: &[f64], u1: f64, u2: f64) -> Vec<f64> {
        let mut d = vec![0.0; self.n];
        d[0] = u1;
        d[1] = u2;
        for i in 2..self.n {
            d[i] = q[i - 1] * u1;
        }
        d
    }

    /// Integrate from `q0` under a time-varying input, with RK4 over `steps` uniform intervals.
    ///
    /// RK4 rather than Euler because the whole point of the steering law is a *cancellation* — every variable
    /// below the target returns to its start — and first-order integration error does not cancel, so Euler
    /// leaves a residual drift that looks exactly like the law being wrong.
    pub fn integrate(
        &self,
        q0: &[f64],
        u: impl Fn(f64) -> (f64, f64),
        t_final: f64,
        steps: usize,
    ) -> Vec<f64> {
        let h = t_final / steps as f64;
        let mut q = q0.to_vec();
        let add = |q: &[f64], d: &[f64], s: f64| -> Vec<f64> {
            q.iter().zip(d).map(|(a, b)| a + s * b).collect()
        };
        for i in 0..steps {
            let t = i as f64 * h;
            let (a1, b1) = u(t);
            let k1 = self.dynamics(&q, a1, b1);
            let (a2, b2) = u(t + 0.5 * h);
            let k2 = self.dynamics(&add(&q, &k1, 0.5 * h), a2, b2);
            let k3 = self.dynamics(&add(&q, &k2, 0.5 * h), a2, b2);
            let (a4, b4) = u(t + h);
            let k4 = self.dynamics(&add(&q, &k3, h), a4, b4);
            for j in 0..self.n {
                q[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
        }
        q
    }

    /// The sinusoid pair for chain level `k`: `u₁ = a·sin(2πt/T)`, `u₂ = b·cos(2πkt/T)`.
    pub fn sinusoids(k: usize, a: f64, b: f64, t_final: f64) -> impl Fn(f64) -> (f64, f64) {
        let w = 2.0 * std::f64::consts::PI / t_final;
        move |t: f64| (a * (w * t).sin(), b * (k as f64 * w * t).cos())
    }

    /// **Measured** gain `∂(Δq_{k+2}) / ∂b` at amplitude `a` over one period, from `q0`.
    ///
    /// `Δq_{k+2}` is linear in `b`, so this single integration determines the `b` that achieves any target
    /// displacement. See the module note on why this is measured rather than taken from the closed form.
    pub fn steer_gain(&self, q0: &[f64], k: usize, a: f64, t_final: f64, steps: usize) -> f64 {
        let base = self.integrate(q0, Self::sinusoids(k, a, 0.0, t_final), t_final, steps);
        let unit = self.integrate(q0, Self::sinusoids(k, a, 1.0, t_final), t_final, steps);
        unit[k + 1] - base[k + 1]
    }

    /// **MLS Algorithm 3.** Steer from `q0` to `target`, returning the achieved state and the input schedule
    /// as `(k, a, b, duration)` phases — phase `k = 0` is the direct `q₁`/`q₂` move.
    ///
    /// Each later phase moves one chain variable and returns every earlier one to where it started, so the
    /// phases compose without iteration. `a` sets how aggressively `u₁` swings; larger `a` needs smaller `b`
    /// for the same displacement, and the gain grows like `a^k`, so a small `a` on a long chain demands an
    /// enormous `b`.
    ///
    /// Returns `None` if a required gain vanishes, which happens when `a == 0` — there is then no `b` that
    /// moves the variable at all.
    #[allow(clippy::type_complexity)]
    pub fn steer(
        &self,
        q0: &[f64],
        target: &[f64],
        a: f64,
        phase_time: f64,
        steps: usize,
    ) -> Option<(Vec<f64>, Vec<(usize, f64, f64, f64)>)> {
        if q0.len() != self.n || target.len() != self.n {
            return None;
        }
        let mut q = q0.to_vec();
        let mut plan = Vec::new();

        // Step 1: q1 and q2 are directly actuated, so a constant input over one phase places them exactly.
        let (d1, d2) = (target[0] - q[0], target[1] - q[1]);
        let (c1, c2) = (d1 / phase_time, d2 / phase_time);
        q = self.integrate(&q, move |_| (c1, c2), phase_time, steps);
        plan.push((0, c1, c2, phase_time));

        // Step 2: each q_{k+2} in turn, k = 1.., using sinusoids at integrally related frequencies.
        for k in 1..=self.n - 2 {
            let need = target[k + 1] - q[k + 1];
            let gain = self.steer_gain(&q, k, a, phase_time, steps);
            if gain.abs() < 1e-14 {
                if need.abs() < 1e-12 {
                    continue; // already there and nothing to do
                }
                return None; // cannot move this variable at this amplitude
            }
            let b = need / gain;
            q = self.integrate(&q, Self::sinusoids(k, a, b, phase_time), phase_time, steps);
            plan.push((k, a, b, phase_time));
        }
        Some((q, plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_chain_is_controllable_in_practice_not_just_in_principle() {
        // MLS Prop. 8.2 says the one-chain system is completely nonholonomic. The operational version of
        // that claim is that Algorithm 3 actually reaches an arbitrary target, so steer to a few.
        for n in [3, 4, 5, 6] {
            let sys = ChainedForm::new(n).unwrap();
            let q0 = vec![0.0; n];
            let target: Vec<f64> = (0..n).map(|i| 0.3 - 0.11 * i as f64).collect();
            let (reached, plan) = sys.steer(&q0, &target, 1.0, 1.0, 4000).expect("steerable");
            for i in 0..n {
                assert!(
                    (reached[i] - target[i]).abs() < 1e-6,
                    "n={n}: q[{i}] reached {} want {}",
                    reached[i],
                    target[i]
                );
            }
            assert_eq!(plan.len(), n - 1, "one direct phase plus one per chain variable");
        }
    }

    #[test]
    fn one_period_moves_only_the_target_variable_and_returns_the_earlier_ones() {
        // THE property the steering law rests on, and the one that fails silently if the frequencies are not
        // integrally related. Drive level k and check every q_j for j < k+2 comes back to its start.
        let sys = ChainedForm::new(6).unwrap();
        let q0 = vec![0.0; 6];
        for k in 1..=3 {
            let out = sys.integrate(&q0, ChainedForm::sinusoids(k, 1.0, 1.0, 1.0), 1.0, 8000);
            for j in 0..k + 1 {
                assert!(
                    out[j].abs() < 1e-6,
                    "k={k}: q[{j}] should return to 0 after one period, got {}",
                    out[j]
                );
            }
            // The expected displacement falls like (a/4pi)^k / k!, so a fixed threshold fails at k=3 for a
            // correct implementation. Compare against the closed form instead.
            let factorial: f64 = (1..=k).map(|i| i as f64).product();
            let expect = (1.0 / (4.0 * std::f64::consts::PI)).powi(k as i32) / factorial;
            assert!(
                (out[k + 1] - expect).abs() < 1e-6 * expect.abs().max(1e-9),
                "k={k}: q[{}] moved {} but the closed form predicts {expect}",
                k + 1,
                out[k + 1]
            );
        }
    }

    #[test]
    fn the_displacement_is_linear_in_b_which_is_what_makes_one_probe_enough() {
        // steer_gain does a single integration and divides. That is only valid if Δq_{k+2} is exactly linear
        // in b — assert it rather than assume it.
        let sys = ChainedForm::new(5).unwrap();
        let q0 = vec![0.0; 5];
        for k in 1..=3 {
            let g = sys.steer_gain(&q0, k, 1.0, 1.0, 4000);
            for b in [0.5, 2.0, -3.0] {
                let out = sys.integrate(&q0, ChainedForm::sinusoids(k, 1.0, b, 1.0), 1.0, 4000);
                let predicted = g * b;
                assert!(
                    (out[k + 1] - predicted).abs() < 1e-8 * (1.0 + predicted.abs()),
                    "k={k} b={b}: got {} predicted {predicted}",
                    out[k + 1]
                );
            }
        }
    }

    #[test]
    fn the_measured_gain_matches_the_books_closed_form_exactly() {
        // MLS states Δq_{k+2} = (a/4π)^k · b/k!. Measured against numerical integration this is exact — the
        // ratio is 1.000000 at every k, not merely a consistent constant. I checked because the book's own
        // derivation carries an ∫sin² factor of one half that looked like it might have been dropped in
        // transcription; it had not been. Assert exactness, so a drift in either the dynamics or the formula
        // is caught rather than absorbed.
        let sys = ChainedForm::new(7).unwrap();
        let q0 = vec![0.0; 7];
        let (a, t) = (1.0, 1.0);
        let mut ratios = Vec::new();
        for k in 1..=4 {
            let measured = sys.steer_gain(&q0, k, a, t, 20000);
            let fourpi = 4.0 * std::f64::consts::PI;
            let factorial: f64 = (1..=k).map(|i| i as f64).product();
            let stated = (a / fourpi).powi(k as i32) / factorial;
            assert!(stated.abs() > 0.0);
            ratios.push(measured / stated);
        }
        for (i, r) in ratios.iter().enumerate() {
            assert!(
                (r - 1.0).abs() < 1e-5,
                "measured/stated at k={} is {r}, expected exactly 1 — MLS eq. for Δq_(k+2) is correct as \
                 printed and any deviation means the dynamics or the formula moved",
                i + 1
            );
        }
    }

    #[test]
    fn a_degenerate_amplitude_is_refused_rather_than_dividing_by_zero() {
        // a = 0 makes u1 identically zero, so no chain variable beyond q2 can move at all. Asking for one is
        // not satisfiable and must say so.
        let sys = ChainedForm::new(4).unwrap();
        let q0 = vec![0.0; 4];
        // Note the subtlety: even a "q1/q2 only" target is NOT reachable at a = 0, because the direct phase
        // itself perturbs the chain (q̇₃ = q₂·u₁ is non-zero while q₂ is being moved) and a = 0 leaves no way
        // to correct it. A first version of this test assumed otherwise and failed — correctly.
        let moves_q2 = vec![0.5, -0.2, 0.0, 0.0];
        assert!(sys.steer(&q0, &moves_q2, 0.0, 1.0, 500).is_none(), "the direct phase drags q3 along");
        // Staying put is reachable at a = 0: nothing needs correcting.
        assert!(sys.steer(&q0, &q0, 0.0, 1.0, 500).is_some(), "the trivial target needs no chain motion");
        // And with a real amplitude, the same q2 move becomes reachable.
        let (reached, _) = sys.steer(&q0, &moves_q2, 1.0, 1.0, 4000).expect("steerable with a=1");
        for i in 0..4 {
            assert!((reached[i] - moves_q2[i]).abs() < 1e-6, "q[{i}] = {} want {}", reached[i], moves_q2[i]);
        }
        assert!(ChainedForm::new(2).is_none(), "n < 3 has no chain part");
    }
}
