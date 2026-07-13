# ferromotion-fluid

[![crates.io](https://img.shields.io/crates/v/ferromotion-fluid.svg)](https://crates.io/crates/ferromotion-fluid)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-fluid)](https://docs.rs/ferromotion-fluid)

2D **incompressible Navier–Stokes** for fluid–robot interaction — the Aquarium track of
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion), in pure Rust (native + `wasm32`).

A **marker-and-cell (MAC) projection solver** on a staggered grid: velocities on cell faces, pressure
at cell centers — the layout that kills the checkerboard mode without artificial stabilization. Each
step is Chorin's fractional method — explicit advection + diffusion predictor, a pressure-Poisson
projection back onto the divergence-free manifold, then the velocity correction. The pressure Laplacian
is constant, so it is assembled once and factored with [`faer`](https://crates.io/crates/faer)'s sparse
Cholesky; every timestep is then a back-substitution.

Verified against the canonical **Ghia, Ghia & Shin (1982)** lid-driven-cavity benchmark at Re = 100:
the centerline `u`-velocity profile matches the published table to **max error ≈ 0.003** on a 64² grid,
alongside a hard internal check that the projected field is discretely divergence-free.

```rust
use ferromotion_fluid::MacFluid;

// Re = U·L/ν = 1·1/0.01 = 100
let mut fluid = MacFluid::new(64, 64, 0.01, 0.004, 1.0);
fluid.run_to_steady(8000, 2e-6);
assert!(fluid.max_divergence() < 1e-8);
let profile = fluid.centerline_u(); // (y, u) up the vertical centerline
```

The immersed-boundary coupling for a moving robot surface (full FSI, with gradients) builds on this
core next.

Dual-licensed MIT OR Apache-2.0.
