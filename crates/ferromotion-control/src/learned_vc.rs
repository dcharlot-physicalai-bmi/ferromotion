//! **A learned virtual constraint** — Q1 milestone M1's first half: render an approximately invariant reduced
//! manifold by *training for invariance* rather than by picking a family in which it can be solved.
//!
//! M0 hand-picked a two-parameter family and solved the one scalar equation hybrid invariance imposes. That
//! works, and it is also the whole problem: with two parameters and one equation there is exactly one degree of
//! freedom left, so the gait's speed is whatever the algebra leaves and nothing else can be asked for. The
//! constraint M0 produced **scuffs the swing foot through the ground** mid-step, which the classical compass-gait
//! literature tolerates as an idealisation and a real robot does not.
//!
//! A richer parameterisation changes the character of the problem. Invariance is still one equation, so with `n`
//! weights there are `n − 1` directions along which it is satisfied exactly — and those are available for
//! whatever else the gait has to do. Here that is a target speed and the shallowest possible scuff, both trained
//! against, with invariance kept as a hard objective rather than a hope.
//!
//! # Why the objective is *shallowest* scuff and not zero
//!
//! Eliminating the scuff is **geometrically impossible for this robot**, and the argument is short enough to
//! settle before spending any optimisation on it. The swing-foot height is `l(cos θ₁ − cos θ₂)`, so clearance
//! requires `|θ₂| > |θ₁|` throughout the step. But `θ₂` must travel from `+α` to `−α` for the step to go
//! forward, so it passes through zero — and at that instant the height is `l(cos θ₁ − 1) ≤ 0`, negative unless
//! `θ₁` happens to be zero at the same moment. Point feet on straight legs cannot lift.
//!
//! So the best available is `θ₂ = 0` exactly when `θ₁ = 0`, which touches the ground without penetrating, and
//! the achievable objective is to minimise the deepest penetration. A scan over the two-weight family confirms
//! it: no member clears, at any invariance-satisfying point. Knees or telescoping legs are the real fix, and the
//! classical literature's habit of ignoring the scuff is a statement about the model rather than about tuning.
//!
//! ```text
//! h_w(θ₁) = −θ₁ + (α² − θ₁²)·Σ wᵢ (θ₁/α)ⁱ
//! ```
//!
//! The `(α² − θ₁²)` factor is the important structural choice: it makes `h_w(±α) = ∓α` hold for **every** weight
//! vector, so the step geometry — the swing leg arriving exactly where the next stance leg must be — is built in
//! and cannot be trained away. Only the interior shape is learned. `w₀` and `w₁` recover M0's two parameters, so
//! the hand-tuned family is a subspace of this one and the comparison is apples to apples.
//!
//! Training uses the analytic restricted map ([`CompassGait::restricted_map`](crate::CompassGait::restricted_map)),
//! which is a pair of quadratures rather than a simulation — so an objective evaluation costs microseconds and
//! the whole search is cheap. The full four-state model is then used to *verify*, never to train.

use crate::{CompassGait, SwingConstraint};

/// A virtual constraint whose interior shape is a learned polynomial.
#[derive(Clone, Debug)]
pub struct LearnedConstraint {
    /// Half the inter-leg angle at strike.
    pub alpha: f64,
    /// Shape weights, lowest order first. `w[0]` and `w[1]` correspond to M0's `c` and `e·α`.
    pub w: Vec<f64>,
}

impl LearnedConstraint {
    /// A constraint with `n` shape weights, all zero (the degenerate mirror — see
    /// [`VirtualConstraint`](crate::VirtualConstraint) for why that is not a usable starting point on its own).
    pub fn zeros(alpha: f64, n: usize) -> Self {
        LearnedConstraint { alpha, w: vec![0.0; n.max(1)] }
    }

    /// The shape polynomial and its first two derivatives with respect to `θ₁`.
    ///
    /// Differentiated term by term in `u = θ₁/α` and rescaled, rather than by any recursion that divides by
    /// `u` — the middle of a step is exactly `u = 0`, so a `1/u` anywhere here would be a singularity at the
    /// most-visited point of the trajectory.
    fn shape(&self, th1: f64) -> (f64, f64, f64) {
        let u = th1 / self.alpha;
        let (mut p, mut d, mut dd) = (0.0, 0.0, 0.0);
        for (i, wi) in self.w.iter().enumerate() {
            p += wi * u.powi(i as i32);
            if i >= 1 {
                d += wi * i as f64 * u.powi(i as i32 - 1);
            }
            if i >= 2 {
                dd += wi * (i * (i - 1)) as f64 * u.powi(i as i32 - 2);
            }
        }
        (p, d / self.alpha, dd / (self.alpha * self.alpha))
    }

    /// The shape at the two ends of the step, `(φ(−α), φ(+α))`.
    ///
    /// These two numbers decide whether the swing foot clears the ground, and the condition is not obvious.
    /// The foot lifts at the start only if `h'(−α) > −1` and descends at the strike only if `h'(α) > −1`; since
    /// `h'(±α) = −1 ∓ 2αφ(±α)`, that is **`φ(−α) > 0` and `φ(+α) < 0`**. M0's hand-tuned constraint has
    /// `φ(+α) = +6.19`, so its foot is *ascending* through the ground at the strike — that is the scuff, and it
    /// is a sign condition rather than a matter of tuning.
    pub fn shape_at_ends(&self) -> (f64, f64) {
        (self.shape(-self.alpha).0, self.shape(self.alpha).0)
    }

    /// Whether the sign conditions necessary for foot clearance hold.
    pub fn can_clear(&self) -> bool {
        let (lo, hi) = self.shape_at_ends();
        lo > 0.0 && hi < 0.0
    }
}

impl SwingConstraint for LearnedConstraint {
    fn desired(&self, th1: f64) -> (f64, f64, f64) {
        let (p, dp, ddp) = self.shape(th1);
        let f = self.alpha * self.alpha - th1 * th1;
        (-th1 + f * p, -1.0 - 2.0 * th1 * p + f * dp, -2.0 * p - 4.0 * th1 * dp + f * ddp)
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// What the training objective is asked to achieve, and how heavily each part counts.
#[derive(Clone, Copy, Debug)]
pub struct GaitGoal {
    /// Target squared stance rate at the fixed point, `ζ* = θ̇₁*²`.
    pub target_zeta: f64,
    /// Target for the worst swing-foot height over the step, in metres. **Zero is the best achievable value
    /// for this robot** and it is not reachable by the hand-tuned family — see the module docs for why positive
    /// clearance is geometrically impossible with point feet on straight legs.
    pub min_clearance: f64,
    /// Weight on the hybrid-invariance defect. Large, because invariance is what makes the reduction legal and
    /// everything else is negotiable.
    pub w_invariance: f64,
    pub w_speed: f64,
    pub w_clearance: f64,
    /// Weight on `‖w‖²`, to keep the shape from becoming needlessly wild between the sampled points.
    pub w_regularise: f64,
}

impl Default for GaitGoal {
    fn default() -> Self {
        GaitGoal { target_zeta: 4.7, min_clearance: 0.0, w_invariance: 1e8, w_speed: 1.0, w_clearance: 1e6, w_regularise: 1e-5 }
    }
}

/// The pieces of the objective, kept separate so a trained constraint can be reported honestly rather than as
/// a single number.
#[derive(Clone, Copy, Debug)]
pub struct GaitScore {
    /// `‖ẏ⁺‖` after the impact: the distance the impact leaves the state off `Z`.
    pub invariance_defect: f64,
    /// `δ²` of the restricted return map, or `None` if the reduction breaks down.
    pub delta_sq: Option<f64>,
    /// The fixed point `ζ*`, if a periodic gait exists.
    pub gait: Option<f64>,
    /// The worst (most negative, or smallest) swing-foot clearance over the step, in metres.
    pub worst_clearance: f64,
    pub total: f64,
}

/// The **hybrid-invariance defect** of a constraint: the post-impact velocity ratio against the one `Z` demands.
///
/// Exactly the scalar M0 solved for, and it stays a single number however many weights the constraint has —
/// which is the whole reason a richer parameterisation has room left over.
pub fn invariance_defect(r: &CompassGait, vc: &dyn SwingConstraint) -> f64 {
    let alpha = vc.alpha();
    let pre = vc.on_manifold(alpha, 1.0); // the impact is linear in velocity, so the scale is free
    let post = r.impact(&pre);
    (post.d2 / post.d1 - vc.desired(-alpha).1).abs()
}

/// The smallest swing-foot clearance over a step on `Z`, in metres. Negative means the foot passes through the
/// ground.
pub fn worst_clearance(r: &CompassGait, vc: &dyn SwingConstraint, samples: usize) -> f64 {
    let alpha = vc.alpha();
    let n = samples.max(8);
    // exclude the endpoints, where the foot is on the ground by construction
    (1..n)
        .map(|i| {
            let th1 = -alpha + 2.0 * alpha * i as f64 / n as f64;
            r.swing_foot_height(&vc.on_manifold(th1, 1.0))
        })
        .fold(f64::INFINITY, f64::min)
}

/// Score a constraint against a goal.
pub fn score(r: &CompassGait, vc: &dyn SwingConstraint, goal: &GaitGoal, weights: &[f64]) -> GaitScore {
    let defect = invariance_defect(r, vc);
    let clearance = worst_clearance(r, vc, 60);
    let map = r.restricted_map(vc, 600);
    let (delta_sq, gait) = match &map {
        Some(m) => (Some(m.delta_sq), m.gait()),
        None => (None, None),
    };

    let mut total = goal.w_invariance * defect * defect;
    total += goal.w_clearance * (goal.min_clearance - clearance).max(0.0).powi(2);
    total += goal.w_regularise * weights.iter().map(|x| x * x).sum::<f64>();
    // The penalty for "no stable gait" has to carry a *gradient*, or the optimiser sits on a plateau and looks
    // like a tuning problem. A flat constant here is what made the first network training runs fail outright:
    // every candidate scored the same, so there was no direction to move in. Steering delta^2 towards a
    // contracting value gives the search something to follow back into the feasible region.
    const TARGET_DELTA_SQ: f64 = 0.85;
    match (delta_sq, gait) {
        (Some(d), Some(z)) if d < 1.0 => total += goal.w_speed * (z - goal.target_zeta).powi(2),
        (Some(d), _) => total += 1e3 + 1e4 * (d - TARGET_DELTA_SQ).powi(2),
        (None, _) => total += 1e6,
    }
    GaitScore { invariance_defect: defect, delta_sq, gait, worst_clearance: clearance, total }
}

/// **Train the constraint** by gradient descent with central-difference gradients and a backtracking step.
///
/// Finite differences are the right tool here rather than a concession: an objective evaluation is two
/// quadratures and an impact, so it costs microseconds, and the weight count is small. Returns the trained
/// constraint and its score.
pub fn train(r: &CompassGait, alpha: f64, n_weights: usize, goal: &GaitGoal, init: &[f64], iters: usize) -> (LearnedConstraint, GaitScore) {
    let mut w: Vec<f64> = (0..n_weights).map(|i| init.get(i).copied().unwrap_or(0.0)).collect();
    let eval = |w: &[f64]| {
        let vc = LearnedConstraint { alpha, w: w.to_vec() };
        score(r, &vc, goal, w).total
    };
    let mut step = 0.05;
    let mut best = eval(&w);
    for _ in 0..iters {
        // central-difference gradient
        let eps = 1e-6;
        let grad: Vec<f64> = (0..n_weights)
            .map(|i| {
                let (mut wp, mut wm) = (w.clone(), w.clone());
                wp[i] += eps;
                wm[i] -= eps;
                (eval(&wp) - eval(&wm)) / (2.0 * eps)
            })
            .collect();
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < 1e-14 {
            break;
        }
        // backtracking line search
        let mut moved = false;
        let mut s = step;
        for _ in 0..40 {
            let cand: Vec<f64> = w.iter().zip(&grad).map(|(x, g)| x - s * g / gnorm).collect();
            let v = eval(&cand);
            if v < best {
                w = cand;
                best = v;
                step = s * 1.3;
                moved = true;
                break;
            }
            s *= 0.5;
        }
        if !moved {
            step *= 0.5;
            if step < 1e-14 {
                break;
            }
        }
    }
    let vc = LearnedConstraint { alpha, w: w.clone() };
    let sc = score(r, &vc, goal, &w);
    (vc, sc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualConstraint;

    /// The learned parameterisation must reproduce M0's family exactly on its first two weights, or the
    /// comparison between them means nothing. `h_w = −θ + (α²−θ²)(w₀ + w₁θ/α)` against
    /// `h_d = −θ + (α²−θ²)(c + eθ)`, so `w₀ = c` and `w₁ = eα`.
    #[test]
    fn the_learned_family_contains_the_hand_tuned_one() {
        let (alpha, c, e) = (0.22f64, 5.42474849f64, 3.5f64);
        let hand = VirtualConstraint { alpha, c, e };
        let learned = LearnedConstraint { alpha, w: vec![c, e * alpha] };
        for k in 0..=20 {
            let th1 = -alpha + 2.0 * alpha * k as f64 / 20.0;
            let (a0, a1, a2) = SwingConstraint::desired(&hand, th1);
            let (b0, b1, b2) = learned.desired(th1);
            assert!((a0 - b0).abs() < 1e-12, "h differs at {th1}: {a0} vs {b0}");
            assert!((a1 - b1).abs() < 1e-12, "h' differs at {th1}: {a1} vs {b1}");
            assert!((a2 - b2).abs() < 1e-12, "h'' differs at {th1}: {a2} vs {b2}");
        }
    }

    /// The derivatives are the derivatives, checked by finite-differencing `h` itself. A slip here would be
    /// invisible in the gait and fatal in the certificate, since `h'` enters the reduction and `h''` the
    /// restricted dynamics.
    #[test]
    fn the_analytic_derivatives_match_finite_differences() {
        let vc = LearnedConstraint { alpha: 0.25, w: vec![3.0, -1.2, 0.7, 2.1, -0.4] };
        let eps = 1e-6;
        for k in 1..20 {
            let th1 = -0.25 + 0.5 * k as f64 / 20.0;
            let h = |t: f64| vc.desired(t).0;
            let d_fd = (h(th1 + eps) - h(th1 - eps)) / (2.0 * eps);
            let dd_fd = (h(th1 + eps) - 2.0 * h(th1) + h(th1 - eps)) / (eps * eps);
            let (_, d, dd) = vc.desired(th1);
            assert!((d - d_fd).abs() < 1e-6, "h' wrong at {th1}: {d} vs {d_fd}");
            assert!((dd - dd_fd).abs() < 1e-3, "h'' wrong at {th1}: {dd} vs {dd_fd}");
        }
    }

    /// The step's endpoint geometry is **structural**: `h(±α) = ∓α` for every weight vector, so no amount of
    /// training can produce a constraint whose swing leg fails to arrive where the next stance leg must be.
    #[test]
    fn the_endpoint_geometry_survives_any_weights() {
        for w in [vec![0.0], vec![9.0, -40.0, 12.0], vec![-3.3, 0.1, 7.7, -18.0, 4.0, 2.2]] {
            let vc = LearnedConstraint { alpha: 0.2, w };
            assert!((vc.desired(0.2).0 + 0.2).abs() < 1e-14, "h(alpha) must be -alpha");
            assert!((vc.desired(-0.2).0 - 0.2).abs() < 1e-14, "h(-alpha) must be +alpha");
        }
    }

    /// The network's analytic derivatives are the derivatives. `h''` enters the restricted dynamics, so a slip
    /// here would land silently inside `δ²` and therefore inside the certificate.
    #[test]
    fn the_network_derivatives_match_finite_differences() {
        let mut vc = NeuralConstraint::new(0.24, 5);
        vc.v = vec![0.4, -0.9, 0.3, 1.1, -0.6];
        vc.c = 0.8;
        let eps = 1e-6;
        for k in 1..24 {
            let th1 = -0.24 + 0.48 * k as f64 / 24.0;
            let h = |t: f64| vc.desired(t).0;
            let (_, d, dd) = vc.desired(th1);
            let d_fd = (h(th1 + eps) - h(th1 - eps)) / (2.0 * eps);
            let dd_fd = (h(th1 + eps) - 2.0 * h(th1) + h(th1 - eps)) / (eps * eps);
            assert!((d - d_fd).abs() < 1e-6, "h' wrong at {th1}: {d} vs {d_fd}");
            assert!((dd - dd_fd).abs() < 1e-3, "h'' wrong at {th1}: {dd} vs {dd_fd}");
        }
        // and the endpoint geometry is structural for the network too
        assert!((vc.desired(0.24).0 + 0.24).abs() < 1e-14);
        assert!((vc.desired(-0.24).0 - 0.24).abs() < 1e-14);
    }

    /// **Positive clearance is impossible, and the reason is a sign condition rather than a tuning failure.**
    ///
    /// The necessary condition is `φ(−α) > 0` and `φ(+α) < 0`. M0's constraint has `φ(+α) = +6.19`, so its foot
    /// is ascending through the ground at the strike. But even satisfying both signs does not buy clearance,
    /// because `θ₂` must cross zero mid-step and the height is `l(cos θ₁ − 1) ≤ 0` there. This pins both facts.
    #[test]
    fn positive_foot_clearance_is_geometrically_impossible_here() {
        let r = CompassGait::default();
        let alpha = 0.22;
        let hand = VirtualConstraint { alpha, c: 5.42474849, e: 3.5 };
        let as_learned = LearnedConstraint { alpha, w: vec![5.42474849, 3.5 * alpha] };
        let (lo, hi) = as_learned.shape_at_ends();
        eprintln!("M0 constraint: phi(-a) = {lo:.3}, phi(+a) = {hi:.3}, can_clear = {}", as_learned.can_clear());
        assert!(hi > 0.0 && !as_learned.can_clear(), "M0 violates the sign condition at the strike");
        assert!(worst_clearance(&r, &hand, 400) < 0.0, "so it scuffs");

        // Even with both signs satisfied, the mid-step crossing forces a scuff: wherever theta2 = 0 with
        // theta1 != 0, the height is l(cos theta1 - 1) < 0.
        let signed_ok = LearnedConstraint { alpha, w: vec![2.0, -4.0] };
        assert!(signed_ok.can_clear(), "this one satisfies the necessary signs");
        let c = worst_clearance(&r, &signed_ok, 400);
        eprintln!("a constraint satisfying both signs still scuffs by {c:+.5} m - the signs are necessary, not sufficient");
        assert!(c < 0.0, "the mid-step zero crossing forces a scuff even so, got {c:+.5}");
    }

    /// **What the extra weights actually buy: a much shallower scuff at the same invariance and a real gait.**
    ///
    /// Since zero is unreachable, the honest objective is depth. The two-parameter family spends its only free
    /// direction on invariance and has nothing left; a six-weight constraint keeps invariance exact and uses the
    /// remaining five directions to flatten the penetration and hold a target speed.
    #[test]
    fn training_reduces_the_scuff_depth_the_hand_tuned_family_is_stuck_with() {
        let r = CompassGait::default();
        let alpha = 0.22;
        let hand = VirtualConstraint { alpha, c: 5.42474849, e: 3.5 };
        let hand_depth = worst_clearance(&r, &hand, 400);
        eprintln!("hand-tuned (M0): worst swing-foot height {hand_depth:+.5} m, invariance defect {:.2e}", invariance_defect(&r, &hand));

        let goal = GaitGoal::default();
        let (learned, sc) = train(&r, alpha, 6, &goal, &[0.0, 0.77, 0.0, 0.0, 0.0, 0.0], 6000);
        eprintln!("trained (6 weights): worst height {:+.6} m, invariance defect {:.2e}, delta^2 {:?}, zeta* {:?}", sc.worst_clearance, sc.invariance_defect, sc.delta_sq.map(|d| format!("{d:.6}")), sc.gait.map(|z| format!("{z:.4}")));
        eprintln!("   weights {:?}", learned.w.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());

        assert!(sc.invariance_defect < 1e-6, "invariance must stay near-exact, got {:.2e}", sc.invariance_defect);
        let d = sc.delta_sq.expect("a reduction must exist");
        assert!(d > 0.0 && d < 1.0, "the trained gait must still contract: delta^2 = {d}");
        assert!(sc.gait.is_some_and(|z| z > 0.5), "and carry a real gait");
        assert!(sc.worst_clearance > hand_depth, "the trained scuff must be shallower than the hand-tuned one: {:+.6} vs {hand_depth:+.6}", sc.worst_clearance);
        assert!(sc.worst_clearance > 0.5 * hand_depth, "and by a real margin, not a rounding: {:+.6} vs {hand_depth:+.6}", sc.worst_clearance);
    }
}

/// A **one-hidden-layer neural virtual constraint**:
///
/// ```text
/// h_θ(θ₁) = −θ₁ + (α² − θ₁²) · N_θ(θ₁/α),   N_θ(u) = Σⱼ vⱼ tanh(aⱼu + bⱼ) + c
/// ```
///
/// The same structural factor as [`LearnedConstraint`] keeps `h_θ(±α) = ∓α` exact for every parameter, so the
/// step geometry cannot be trained away and only the interior shape is learned.
///
/// A network rather than a polynomial matters for one reason beyond expressiveness: it is the realistic case
/// for **approximate** invariance. A polynomial family small enough to solve exactly is not what a learned
/// policy produces, and Q1 names precisely that risk — a learned policy may render no manifold even
/// approximately invariant, leaving no return map to certify. The answer is not to insist on exactness but to
/// let the residual enter a certificate as a disturbance, which is what the E-ISS route does.
///
/// The derivatives are analytic. `h''` enters the restricted dynamics directly, so a finite-difference
/// substitute would put quadrature noise inside the certificate.
#[derive(Clone, Debug)]
pub struct NeuralConstraint {
    pub alpha: f64,
    /// Input weights, one per hidden unit.
    pub a: Vec<f64>,
    /// Biases, one per hidden unit.
    pub b: Vec<f64>,
    /// Output weights, one per hidden unit.
    pub v: Vec<f64>,
    /// Output bias.
    pub c: f64,
}

impl NeuralConstraint {
    /// A network with `hidden` units, initialised to a spread of frequencies so the units are not degenerate
    /// with one another at the start. Deterministic, so a training run is reproducible.
    pub fn new(alpha: f64, hidden: usize) -> Self {
        let n = hidden.max(1);
        NeuralConstraint {
            alpha,
            a: (0..n).map(|j| 1.0 + 0.7 * j as f64).collect(),
            b: (0..n).map(|j| -1.0 + 2.0 * j as f64 / n as f64).collect(),
            v: vec![0.0; n],
            c: 0.0,
        }
    }

    /// Flatten the parameters for an optimiser: `[a, b, v, c]`.
    pub fn to_params(&self) -> Vec<f64> {
        let mut p = self.a.clone();
        p.extend(&self.b);
        p.extend(&self.v);
        p.push(self.c);
        p
    }

    /// Rebuild from a flat parameter vector of the same shape.
    pub fn from_params(alpha: f64, hidden: usize, p: &[f64]) -> Self {
        let n = hidden.max(1);
        NeuralConstraint { alpha, a: p[0..n].to_vec(), b: p[n..2 * n].to_vec(), v: p[2 * n..3 * n].to_vec(), c: p[3 * n] }
    }

    /// **Fit the output layer to a target shape by least squares.**
    ///
    /// With the input weights and biases fixed, `N(u) = Σⱼ vⱼ tanh(aⱼu + bⱼ) + c` is *linear* in `(v, c)`, so
    /// matching a given shape is a linear problem with an exact solution — no optimisation needed. Used to warm
    /// start from a shape already known to carry a gait, which matters because the objective's feasible region
    /// is not connected to an arbitrary initialisation: a network that renders no periodic gait gives the
    /// trainer nothing to descend.
    pub fn fit_shape(&mut self, target: &dyn Fn(f64) -> f64, samples: usize) {
        let n = self.a.len();
        let m = samples.max(n + 2);
        let mut ata = nalgebra::DMatrix::zeros(n + 1, n + 1);
        let mut atb = nalgebra::DVector::zeros(n + 1);
        for k in 0..m {
            let u = -1.0 + 2.0 * k as f64 / (m - 1) as f64;
            let mut row = nalgebra::DVector::zeros(n + 1);
            for j in 0..n {
                row[j] = (self.a[j] * u + self.b[j]).tanh();
            }
            row[n] = 1.0;
            let y = target(u);
            for i in 0..=n {
                atb[i] += row[i] * y;
                for l in 0..=n {
                    ata[(i, l)] += row[i] * row[l];
                }
            }
        }
        // a small ridge term, since tanh units at nearby frequencies are close to collinear
        for i in 0..=n {
            ata[(i, i)] += 1e-9;
        }
        if let Some(sol) = ata.lu().solve(&atb) {
            for j in 0..n {
                self.v[j] = sol[j];
            }
            self.c = sol[n];
        }
    }

    /// `(N, N', N'')` at `u`, analytically. Uses `d/dx sech²x = −2 tanh x · sech²x`.
    fn net(&self, u: f64) -> (f64, f64, f64) {
        let (mut y, mut d, mut dd) = (self.c, 0.0, 0.0);
        for j in 0..self.a.len() {
            let z = self.a[j] * u + self.b[j];
            let t = z.tanh();
            let sech2 = 1.0 - t * t;
            y += self.v[j] * t;
            d += self.v[j] * self.a[j] * sech2;
            dd += self.v[j] * self.a[j] * self.a[j] * (-2.0 * t * sech2);
        }
        (y, d, dd)
    }
}

impl SwingConstraint for NeuralConstraint {
    fn desired(&self, th1: f64) -> (f64, f64, f64) {
        let (n, dn, ddn) = self.net(th1 / self.alpha);
        let (dn, ddn) = (dn / self.alpha, ddn / (self.alpha * self.alpha));
        let f = self.alpha * self.alpha - th1 * th1;
        (-th1 + f * n, -1.0 - 2.0 * th1 * n + f * dn, -2.0 * n - 4.0 * th1 * dn + f * ddn)
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// Train a [`NeuralConstraint`] against a goal, by gradient descent with central differences and a
/// backtracking step. `invariance_weight` overrides the goal's, so a family of constraints with **deliberately
/// different invariance residuals** can be produced — which is what an E-ISS certificate has to be tested
/// against.
pub fn train_network(r: &CompassGait, alpha: f64, hidden: usize, goal: &GaitGoal, invariance_weight: f64, iters: usize) -> (NeuralConstraint, GaitScore) {
    let goal = GaitGoal { w_invariance: invariance_weight, ..*goal };
    // Warm start from a shape that already carries a gait. The M1 polynomial's shape is
    // `phi(u) = w0 + w1 u + ...`; fitting the output layer to it is a linear solve, and it puts the network
    // inside the feasible region rather than hoping the trainer finds its way in.
    let mut p = {
        let mut init = NeuralConstraint::new(alpha, hidden);
        let seed = [0.7524, -0.1387, 0.7535, -0.9080, 0.7539, -0.9078];
        init.fit_shape(&|u: f64| seed.iter().enumerate().map(|(i, w)| w * u.powi(i as i32)).sum(), 64);
        init.to_params()
    };
    let eval = |p: &[f64]| {
        let vc = NeuralConstraint::from_params(alpha, hidden, p);
        score(r, &vc, &goal, p).total
    };
    let mut step = 0.05;
    let mut best = eval(&p);
    for _ in 0..iters {
        let eps = 1e-6;
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut pp, mut pm) = (p.clone(), p.clone());
                pp[i] += eps;
                pm[i] -= eps;
                (eval(&pp) - eval(&pm)) / (2.0 * eps)
            })
            .collect();
        let gnorm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        if gnorm < 1e-14 {
            break;
        }
        let mut moved = false;
        let mut s = step;
        for _ in 0..40 {
            let cand: Vec<f64> = p.iter().zip(&grad).map(|(x, g)| x - s * g / gnorm).collect();
            let v = eval(&cand);
            if v < best {
                p = cand;
                best = v;
                step = s * 1.3;
                moved = true;
                break;
            }
            s *= 0.5;
        }
        if !moved {
            step *= 0.5;
            if step < 1e-14 {
                break;
            }
        }
    }
    let vc = NeuralConstraint::from_params(alpha, hidden, &p);
    let sc = score(r, &vc, &goal, &p);
    (vc, sc)
}
