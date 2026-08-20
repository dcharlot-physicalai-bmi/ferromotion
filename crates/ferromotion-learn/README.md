# ferromotion-learn

[![crates.io](https://img.shields.io/crates/v/ferromotion-learn.svg)](https://crates.io/crates/ferromotion-learn)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-learn)](https://docs.rs/ferromotion-learn)

**Differentiable, physics-informed learning** for physical AI. Where
[`ferromotion-core`](https://crates.io/crates/ferromotion-core) provides the physics — rigid-body dynamics,
variational integrators, contact, estimation — this crate provides the *learning*: models that blend physics
priors with data. It is the on-device, pure-Rust, WASM-clean counterpart to the PyTorch-based
physics-informed-ML stacks. **No BLAS, no Python, no GPU required.**

The organizing idea, from the physics-informed ML literature, is that a physics prior can enter a learned
model in one of three places:

| where | how | here |
|---|---|---|
| **guided** | in the data and features | engineered inputs, geometric projections |
| **informed** | in the loss | penalize violation of a differential equation — a PINN |
| **encoded** | in the architecture | Lagrangian and Hamiltonian nets, Neural ODEs, structure-preserving integrators |

Everything rests on one keystone: **automatic differentiation**. `autodiff` is reverse-mode, for gradients of
a scalar loss with respect to many parameters. `dual` is forward-mode, for the exact higher-order derivatives
with respect to a model's *inputs* that PINNs and Lagrangian nets need. Both are verified against finite
differences.

Also included: nonlinear least squares (`nls`), sparse identification of dynamics (`sindy`), a
model-structured network of local linear models (`msnn`), Neural ODEs, Deep Lagrangian Networks, Hamiltonian
networks, and calibration.

## Reinforcement learning

`env` and `ppo` are the training stage of a robot-learning pipeline: an environment boundary, a
diagonal-Gaussian policy with a learnable log-σ, generalized advantage estimation, and PPO's clipped
surrogate. Same `f64`, same tape, same WASM-clean constraint, so a policy trains **on your own device** with
no cloud in the loop.

Two details carry most of the correctness:

- **Terminated is not truncated.** A terminal state's value is exactly zero; a time-limit cutoff's successor
  has whatever value it has. The two flags enter the advantage estimate in *different places* — termination
  zeroes the bootstrap, either flag stops the recursion. Collapsing them into one `done` biases value
  estimates near the horizon downward, worst for the policies that survive longest.
- **Action scaling lives in the environment**, not the policy, so one set of weights means the same torque in
  simulation and on hardware.

The tests train against a scalar linear-quadratic problem, because it is the one control task whose optimal
policy is known in closed form. With a linear policy the learned weight *is* a feedback gain, comparable
directly against the Riccati solution — a bar that a rising reward curve does not clear.

What that oracle establishes, and what it does not: the achieved cost lands within a few percent of the
analytic optimum, but the **gain itself does not converge**. The LQR cost is quadratically flat near its
optimum (a gain 24% short costs 4.6%), and Adam normalizes its own step, so in a flat basin it random-walks
with no restoring force — measured by starting a policy exactly *at* the optimum and watching it wander off.
Annealing the learning rate roughly halves the error, which is why `final_lr_fraction` defaults on. It does
not remove it. The bounds the tests assert are set from that measurement rather than chosen in advance.

### Observation normalization

`train_normalized` estimates a per-channel mean and standard deviation from the observations as they arrive,
applies it before the policy sees anything, and `to_deployable_normalized` **ships it in the checkpoint** — so
`Policy`'s `obs_mean`/`obs_std` stop being dead fields.

This is not cosmetic. A joint angle of order 1 alongside a rate of order 30 saturates the first `tanh` layer on
the rate alone and the policy cannot see the error it is meant to null. The actuator bench in this workspace
would not learn until its observations were divided by hand inside the environment — which is the wrong place,
because the constants are then invisible to the checkpoint and changing a sensor's units silently reinterprets a
trained policy.

Three decisions worth knowing:

- **The transform is frozen for the duration of each batch**, updated from raw observations once it completes.
  The clipped surrogate compares a log-probability recomputed now against one recorded during collection; if the
  transform moved in between, the two are evaluated at different points and the importance ratio is meaningless.
- **Welford, not sum-of-squares.** Measured on a channel offset by `1e9` with a spread of `1e-3`: true variance
  `3.995e-6`, Welford `4.3e-6` relative error, naive `6272` — a relative error of `1.6e9`, fifteen orders worse.
  Welford is accurate, *not* exact, and cannot be: that offset-to-spread ratio leaves `f64` about six digits.
- **It is a passthrough below two samples**, because with one sample the only observation seen would map to
  exactly zero and the policy would be handed a constant.

### The value head was doing nothing

Measured before anything was built to fix it, on a `V(x) = −a x²` regression at three target magnitudes, MSE as
a fraction of the target variance:

| target scale | raw | normalized |
|---|---|---|
| 1 | 0.0059 | 0.0005 |
| 20 | **1.0000** | 0.0005 |
| 200 | **1.4577** | 0.0005 |

A ratio of `1.0` is exactly what predicting the mean scores. So at a target magnitude of 20 the head has learned
**nothing**, and at 200 it is *worse* than the mean: the `tanh` hidden layers saturate and the linear output
cannot reach the range. Both benches here have returns of order 50–100, so **their value baselines were inert and
GAE was differencing against a useless one the whole time.**

`normalize_value_targets` therefore defaults to **on**. It fits the head against normalized returns and
un-normalizes when `V(s)` is read, since GAE adds `V` to raw rewards and needs it in reward units.

**The update order is load-bearing and the value loss cannot see it.** The head predicts "normalized under the
statistics it was fit with", so those must be the statistics used to un-normalize on the next read. Updating them
after the fit instead of before sent the LQR oracle's achieved cost from −1.60 to **−176.5** and its recovered
gain error from 6% to 63% — while the value loss looked healthy throughout, because it is computed in whatever
units the fit used. `IterationReport::value_scale` exposes the pair, and there is a test that reads `V(x)` back
and checks it against an **empirically estimated** discounted return rather than a closed form.

### On a real 5-DoF arm, against the controller the stack already had

Everything above is a scalar problem. The bench `so101_reach_rl` points the same code at the **SO-101** loaded
from its own URDF: five joints, real inertias, torque control through `mass_matrix` and `inverse_dynamics`, and
a baseline of `solve_ik` + PD + `gravity_vector` that gets only the Cartesian target — never the joint
configuration the target was generated from, which the environment knows and withholds.

Three reward shapes, seeds 7 / 11 / 23, 150 × 3000 steps each, evaluated on the deterministic mean policy:

| controller | final error | reached 1 cm | mechanical | copper | electrical |
|---|---|---|---|---|---|
| IK + PD + gravity | **0.0008 m** | **100%** | 2.10 J | 3.34 J | **5.44 J** |
| PPO, linear reward | 0.0147 ± 0.0037 m | 36% | — | — | 16.8 J |
| PPO, quadratic reward | 0.0573 ± 0.0234 m | 0% | — | — | 21.5 J |
| PPO, peaked Gaussian | 0.0816 ± 0.0164 m | 3% | — | — | 20.8 J |

**The model-based controller wins on both axes: 18x the accuracy and 3.1x less energy.** That is not a verdict
on reinforcement learning, it is a statement about when it earns its keep — a reach with a known kinematic
model is the case where inverse kinematics is simply the right tool, and a bench that hid that would be
flattering the method it was built to exercise.

Two things the reward sweep establishes:

- **Gradient magnitude near the goal is what decides whether a policy settles.** `−‖e‖²` has gradient `2‖e‖`,
  so at 2 cm the signal driving the last centimetre is 25x weaker than the one that drove the approach from
  25 cm. The linear reward's constant gradient fixes it: 3.9x lower final error, and the per-seed ranges do not
  overlap — every linear seed beats every quadratic seed.
- **Sharpness where you want precision is worthless if the reward goes silent where you need guidance.** The
  peaked bonus `exp(−(‖e‖/σ)²)` with `σ` at the tolerance is the sharpest of the three exactly where the
  quadratic goes flat, and it came last. At the 25 cm start it is numerically zero, so it is *sparse* through
  the whole approach.

**One seed is not a measurement, and this bench carries the receipt.** The same quadratic configuration read 8%
reached in one run and 0% in the next, differing only by an LU-versus-Cholesky solve at `1e-15` amplified
through 150 training iterations. Across seeds the linear reward's success rate spans 17–50%. Every comparative
claim above is from the 3-seed sweep, and single-seed output says so on its face.

Energy is electrical, not mechanical. A servo holding a pose against gravity draws current at **zero speed**,
where `τ·q̇` reads exactly zero — so copper loss is 1.6x the mechanical work here, and a mechanical-work number
understates the draw by 2.5x. `k_t = V/ω₀` and `R = V/(τ_stall/k_t)` are derived from the two catalogue figures
the bench already declares, not fitted.

### Getting the policy onto hardware

`GaussianPolicy::to_deployable(&action_space)` converts a trained policy into a
[`ferromotion-policy`](https://crates.io/crates/ferromotion-policy) `Policy` — the on-device MLP runner, which
loads and saves JSON checkpoints and compiles to `wasm32`. `to_checkpoint_json` goes straight to the string.

Two things the conversion is careful about, because both are silent when wrong:

- **The action map ships with the weights.** `BoxSpace::from_unit` clamps to `[-1, 1]` and then applies
  `low + ½(u+1)(high − low)`, so the export carries `act_scale = (high−low)/2`, `act_bias = (high+low)/2`, and
  `clamp_unit = true`. Without that flag the deployed policy skips the clamp and commands out-of-range actions
  exactly where the net saturates, which is where a policy spends the start of every trajectory. There is a
  control test showing the flag changes the answer.
- **The learned log-σ is dropped.** It described exploration. Training samples; deployment should not, and
  carrying the noise to a real actuator costs stroke for nothing.

The end-to-end test trains on the LQR problem, exports, and rolls out the **deployed artefact** — a
weight-comparison test would pass with a wrong `act_scale`, since the weights would be identical.

Agreement with the training-time path is to floating-point rounding, not bit-exact: `mean` accumulates
`bias + Σwᵢaᵢ` in a scalar loop while the runner evaluates `W·x + b` through nalgebra. Measured worst deviation
over 80,000 samples across four architectures is **8.9e-16 absolute**, about 4 units in the last place.

```rust
use ferromotion_learn::Msnn;

// A blend of local linear models with partition-of-unity memberships. Interpretable by
// construction: each cell's slope is a readable local derivative.
let xs: Vec<f64> = (0..80).map(|i| -1.5 + 3.0 * i as f64 / 79.0).collect();
let ys: Vec<f64> = xs.iter().map(|x| (2.0 * x).sin()).collect();
let m = Msnn::fit(&xs, &ys, 10, -1.5, 1.5);

assert_eq!(m.degenerate_cells(), 0); // else a slope readout is a placeholder, not a derivative
let slope_near_zero = m.local_slope(5);
```

That `degenerate_cells` check is not decoration. A local slope needs two distinct weighted samples in its
cell to be identifiable at all, and a cell that has fewer reports `0.0` — indistinguishable from a genuinely
flat region. The rule this crate tries to hold to is that an unidentifiable parameter should say so rather
than return a readable-looking number.

Dual-licensed MIT OR Apache-2.0.
