//! **P2 milestone M0: the distillation trap, demonstrated — and a stochasticity budget that follows from it.**
//!
//! Distillation is how a diffusion policy becomes real-time: replace a many-step stochastic sampler by a one-step
//! deterministic map fitted to it. P2 names the theoretical tension and calls it unaddressed — a deterministic
//! one-step policy is exactly the smooth-deterministic regime the exponential-compounding lower bound punishes,
//! so the distillation that makes a policy fast may silently remove it from the escape class.
//!
//! The linear-quadratic case makes the mechanism precise, and it is sharper than "determinism is risky". The
//! asymmetry is not about the *size* of a policy's error but about its **type**:
//!
//! * a **zero-mean** residual costs `tr[(BᵀPB + R)Σₑ]` — horizon-free, and bounded for *any* magnitude;
//! * a **state-dependent** residual `ΔK·x` changes the closed loop to `A_K + BΔK`, which is stable only below the
//!   `H∞` margin `γ* = 1/supω‖(e^{jω}I − A_K)⁻¹B‖`, and above it the cost grows as `e^{Θ(H)}`.
//!
//! And here is the trap, stated as a fact about what distillation *is*. A stochastic policy's residual can be
//! zero-mean, because it has a distribution to be centred. **A deterministic policy's residual is necessarily a
//! function of the state** — that is what deterministic means. So replacing a sampler by its conditional mean
//! converts a benign error type into one with a cliff, *at equal distributional error*. Nothing about the fit
//! quality changes; the failure mode does.
//!
//! That immediately gives the shape of P2's [C2.2] stochasticity budget: a distilled policy must keep enough of
//! its error zero-mean that the state-dependent remainder stays under `γ*`. This run computes that threshold and
//! then checks it against measured horizon growth.
//!
//! Run: `cargo run --release --example p2_distillation_trap -p ferromotion-control`

use ferromotion_control::{lqr_gain, measure_regret, ActionError, LqLoop};
use nalgebra::{DMatrix, DVector};

const SEED: u64 = 0x0D15_7111_A710_0BADu64;

/// Total average cost over a horizon under a given residual, for the horizon-growth test. Returns `None` if the
/// loop diverged, because an unstable loop has no average cost.
fn total_cost(l: &LqLoop, err: &ActionError, h: usize, noise: f64) -> Option<f64> {
    let (a, b, k, q, r) = (l.a.clone(), l.b.clone(), l.k.clone(), l.q.clone(), l.r.clone());
    let m = measure_regret(
        &|x, u, w| &a * x + &b * u + w,
        &|x| -(&k * x),
        &|x, u| (x.transpose() * &q * x)[0] + (u.transpose() * &r * u)[0],
        err,
        &DVector::from_row_slice(&[0.05, 0.0]),
        1,
        noise,
        h,
        0,
        SEED,
    );
    m.bounded.then_some(m.policy_cost * m.steps as f64)
}

fn main() {
    let dt = 0.1;
    let a = DMatrix::from_row_slice(2, 2, &[1.0, dt, 0.0, 1.0]);
    let b = DMatrix::from_row_slice(2, 1, &[0.5 * dt * dt, dt]);
    let q = DMatrix::identity(2, 2);
    let r = DMatrix::from_row_slice(1, 1, &[0.1]);
    let k = lqr_gain(&a, &b, &q, &r);
    let l = LqLoop { a, b, q, r, k };
    let gamma = l.stability_margin(4096).expect("a margin exists");
    let lam = l.contraction_rate().unwrap();

    println!("P2 / M0 - the distillation trap, and the stochasticity budget it implies");
    println!("(double integrator, LQR expert; lambda {lam:.5}, H-infinity margin gamma* {gamma:.4})\n");

    // ---- the asymmetry, at EQUAL distributional error ----
    //
    // Both residuals below have the same per-step second moment - the same eta a sampler bound would report - and
    // they behave completely differently. This is the whole trap in one table.
    println!("1. the same distributional error, two types, two fates");
    println!("   For each row, eta^2 = E||e||^2 is matched. Horizon growth is total cost at H = 4000 over H = 1000.");
    println!("   READ THE RATIO AGAINST 4.0, not against 1.0: a stable loop's total cost is proportional to the");
    println!("   horizon, so 4x is exactly what stability looks like. Anything above it is superlinear growth.");
    println!("      residual type                 ||dK||   eta^2 (matched)   H-growth   verdict");

    // stationary state covariance under the expert, so a state-dependent error of a given gain can be matched in
    // second moment to a zero-mean one
    let noise_sigma = 0.05;
    let sigma_w = DMatrix::from_diagonal(&DVector::from_row_slice(&[noise_sigma * noise_sigma, noise_sigma * noise_sigma]));
    let x_cov = ferromotion_core::solve_lyapunov_discrete(&l.closed_loop().transpose(), &sigma_w).expect("stationary covariance");
    let eta_sq_of_gain = |g: f64| {
        let dk = DMatrix::from_row_slice(1, 2, &[g, 0.0]);
        (&dk * &x_cov * dk.transpose())[0]
    };

    for &frac in &[0.5f64, 0.9, 1.0, 1.1, 1.5] {
        let g = frac * gamma;
        let eta_sq = eta_sq_of_gain(g);
        // the zero-mean residual matched to it
        let dk = DMatrix::from_row_slice(1, 2, &[g, 0.0]);
        let rows: [(&str, ActionError); 2] = [("zero-mean noise (a sampler)", ActionError::noise(eta_sq.sqrt())), ("state-dependent (a distillation)", ActionError::state_dependent(dk))];
        for (name, err) in rows {
            let growth = match (total_cost(&l, &err, 1000, noise_sigma), total_cost(&l, &err, 4000, noise_sigma)) {
                (Some(s), Some(long)) => Some(long / s.max(1e-300)),
                _ => None,
            };
            match growth {
                Some(gr) => println!("      {name:<30} {g:>6.3} {eta_sq:>17.6} {gr:>10.2}   {}", if gr < 5.0 { "bounded" } else { "GROWING" }),
                None => println!("      {name:<30} {g:>6.3} {eta_sq:>17.6} {:>10}   DIVERGED", "-"),
            }
        }
        println!();
    }
    println!("   The zero-mean rows are bounded at every magnitude - including magnitudes where the state-dependent");
    println!("   row of IDENTICAL second moment has already destabilised. The distributional error a sampler bound");
    println!("   reports does not distinguish them, and the closed loop does.");

    // ---- the trap, stated as what distillation does ----
    println!("\n2. why this is specifically about distillation");
    println!("   A stochastic policy's residual CAN be zero-mean: it has a distribution to be centred.");
    println!("   A deterministic policy's residual is a FUNCTION OF THE STATE - that is what deterministic means.");
    println!("   So replacing a sampler by its conditional mean converts the first error type into the second, at");
    println!("   unchanged distributional error. The fit quality need not degrade at all for the failure mode to");
    println!("   change, which is why the trap is invisible to the metric distillation is usually judged by.");

    // ---- the stochasticity budget ----
    //
    // Split a fixed total error budget between a state-dependent part and a zero-mean part. The state-dependent
    // part is what has a cliff, so the budget is the share that must remain stochastic.
    println!("\n3. the stochasticity budget: how much of a distilled policy's error must stay zero-mean");
    let total_eta_sq = eta_sq_of_gain(1.4 * gamma); // a total error budget that WOULD destabilise if all of it were state-dependent
    println!("   Fix a total budget eta^2 = {total_eta_sq:.6}, which destabilises if spent entirely on a gain error.");
    // largest gain the margin allows, and the eta^2 it accounts for
    let safe_eta_sq = eta_sq_of_gain(gamma);
    let min_stochastic_share = 1.0 - safe_eta_sq / total_eta_sq;
    println!("   The margin allows ||dK|| up to gamma* = {gamma:.4}, which accounts for eta^2 = {safe_eta_sq:.6}.");
    println!("   PREDICTED minimum stochastic share = 1 - {safe_eta_sq:.6}/{total_eta_sq:.6} = {:.1}%", 100.0 * min_stochastic_share);
    println!("   i.e. a one-step distillation of this teacher must retain at least that fraction of its error as");
    println!("   genuine conditional stochasticity, or it leaves the bounded regime at no change in fit quality.\n");

    println!("      stochastic share   ||dK|| for the rest   H-growth   verdict");
    for &share in &[0.0f64, 0.2, 0.4, 0.5, 0.6, 0.8] {
        // the deterministic part carries (1 - share) of the budget
        let det_eta_sq = (1.0 - share) * total_eta_sq;
        // invert eta_sq_of_gain, which is quadratic in g
        let g = (det_eta_sq / eta_sq_of_gain(1.0)).sqrt();
        let err = ActionError { sigma: (share * total_eta_sq).sqrt(), bias: None, gain: Some(DMatrix::from_row_slice(1, 2, &[g, 0.0])) };
        let growth = match (total_cost(&l, &err, 1000, noise_sigma), total_cost(&l, &err, 4000, noise_sigma)) {
            (Some(s), Some(long)) => Some(long / s.max(1e-300)),
            _ => None,
        };
        match growth {
            Some(gr) => println!("      {:>15.0}%   {g:>19.4} {gr:>10.2}   {}", 100.0 * share, if gr < 5.0 { "bounded" } else { "GROWING" }),
            None => println!("      {:>15.0}%   {g:>19.4} {:>10}   DIVERGED", 100.0 * share, "-"),
        }
    }
    println!("\n   The prediction is a DIVERGENCE threshold and lands on it: below {:.0}% the loop diverges outright,", 100.0 * min_stochastic_share);
    println!("   at {:.0}% it is marginal (||dK|| = gamma* exactly, so rho = 1), and full boundedness arrives a little", 100.0 * min_stochastic_share);
    println!("   above as the remaining gain error clears the margin with room. That is the right shape: gamma* marks");
    println!("   where stability is lost, not where cost becomes comfortable.");
    println!("\n   And the mechanism is worth naming precisely: adding stochasticity does not stabilise anything. It");
    println!("   DISPLACES error out of the channel that has a cliff into the channel that does not. A budget");
    println!("   statement, not a damping one.");

    // ---- the honest limit of this instance ----
    println!("\n4. what this does and does not show");
    println!("   Shown: a task where a deterministic residual incurs e^(Theta(H)) cost while a zero-mean residual of");
    println!("   identical second moment stays bounded - so the trap in [C2.2] is real, and the budget separating");
    println!("   the two regimes is computable from the plant and the expert alone.");
    println!("\n   NOT shown, and the gap is specific. In this linear-Gaussian instance the stochastic part does not");
    println!("   interact with the state-dependent part at all: stability is set by dK alone and the noise merely");
    println!("   adds bounded cost. So this demonstrates a budget by DISPLACEMENT - moving error between channels -");
    println!("   and not the mechanism Theorem 9.3 actually names, where heteroscedastic noise converts exponential");
    println!("   compounding to polynomial by breaking the adversarial alignment of a smooth imitator's errors");
    println!("   ACROSS steps. That mechanism needs an imitation setting with a state distribution that shifts, and");
    println!("   a fixed linear plant with a fixed expert does not have one.");
    println!("\n   So: [C2.2]'s trap is demonstrated, its budget has a computable form here, and the claim that");
    println!("   stochasticity per se buys the escape is NOT demonstrated - in this setting it does not, and the");
    println!("   benefit is entirely in which channel the error sits. Distinguishing 'stochasticity helps' from");
    println!("   'stochasticity relocates' matters for P2's sub-problem (a), because a heteroscedasticity");
    println!("   certificate that measures only the amount of noise would score these two policies identically.");
}
