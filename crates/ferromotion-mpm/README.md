# ferromotion-mpm

[![crates.io](https://img.shields.io/crates/v/ferromotion-mpm.svg)](https://crates.io/crates/ferromotion-mpm)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-mpm)](https://docs.rs/ferromotion-mpm)

A **differentiable 2D Material Point Method** (MLS-MPM) for soft, elastic, and granular material —
the solid-mechanics companion to [`ferromotion-fluid`](https://crates.io/crates/ferromotion-fluid),
in pure Rust (native + `wasm32`), in the spirit of DiffTaichi / Genesis.

Particles carry mass, velocity, an APIC affine field `C`, and a deformation gradient `F`; a background
grid mediates forces. Each step is the MLS-MPM transfer — **P2G** scatters momentum and the internal
stress via quadratic B-splines, the grid velocity is updated (gravity, walls), **G2P** gathers it back
and evolves `F`. The neo-Hookean Kirchhoff stress `τ = μ(FFᵀ − I) + λ ln(J) I` is smooth in `F`, so the
pipeline is differentiable.

Since `μ, λ ∝ E`, the stress is **linear in Young's modulus** (`∂τ/∂E = τ/E`), giving an exact analytic
gradient of an outcome w.r.t. the material stiffness — verified against finite differences to machine
precision (`rel_err ~5e-12`). Verified physics: momentum conservation, free-fall, and a dropped elastic
block that deforms and stays bounded.

```rust
use ferromotion_mpm::{MpmSim, Particle};
use nalgebra::Vector2;

let mut sim = MpmSim { n: 24, dt: 5e-4, gravity: Vector2::new(0.0, -9.81),
    nu: 0.2, e: 80.0, mass: m, vol: v, walls: true, particles };
sim.step();
let (ke, dke_de) = sim.ke_and_dke_de(); // exact ∂KE/∂E
```

Dual-licensed MIT OR Apache-2.0.
