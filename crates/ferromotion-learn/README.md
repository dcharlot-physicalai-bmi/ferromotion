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
