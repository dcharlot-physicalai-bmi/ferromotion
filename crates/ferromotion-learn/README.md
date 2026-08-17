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
