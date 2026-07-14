# ferromotion-tactile

[![crates.io](https://img.shields.io/crates/v/ferromotion-tactile.svg)](https://crates.io/crates/ferromotion-tactile)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-tactile)](https://docs.rs/ferromotion-tactile)

A **differentiable optical-tactile sensor** simulator (GelSight / DIGIT class), part of
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion), in pure Rust (native + `wasm32`),
in the spirit of DOT-Sim / Taccel.

An optical tactile sensor is an elastomer gel filmed from below: when an object presses in, the gel
deforms, and the camera reads that deformation as shading under colored lights (photometric stereo).
The forward model: a spherical **indenter** presses into the gel to a depth → a smooth surface-height
field `h(x,y)` → surface **normals** `n = (−hₓ, −h_y, 1)` → an RGB **photometric image**
`I_c = albedo·max(0, n·L_c)` under three colored lights.

Every stage is smooth (softplus contact), so the sensor is **differentiable**: `∂h/∂depth` is exact
(`= σ(·)`), verified against finite differences to machine precision (`rel ~1e-11`) — enabling
gradient-based tactile inference (estimate contact depth/pose from an image).

```rust
use ferromotion_tactile::{GelSim, Indenter, default_lights};

let gel = GelSim { n: 81, extent: 1.0, beta: 0.02 };
let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.35, depth: 0.15 };
let img = gel.tactile_image(&ind, &default_lights()); // RGB imprint
let (sum_h, d_sum_h_d_depth) = gel.total_deformation(&ind); // exact gradient
```

Dual-licensed MIT OR Apache-2.0.
