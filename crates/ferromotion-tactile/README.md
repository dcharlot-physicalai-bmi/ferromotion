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

## What the height field is, and is not

`h(x,y)` is **geometric**: the gel is taken to conform to the indenter, softplus-smoothed, and set to
exactly zero outside its footprint. It is not an elastic solution, and `beta` is a smoothing length,
not a material property. Measured against a 3D elastic solve of the same press
([`ferromotion-fem`](../ferromotion-fem), slab bonded to a rigid backing):

- just outside the contact edge the geometric field is **3.3x too small**;
- beyond the footprint it is exactly zero, where the elastic surface carries a fifth to a quarter of
  the total displacement;
- the elastic surface **bulges upward** in a ring around the contact, by **6.56% of the press depth at
  ν = 0.49**, growing 39x from ν = 0.20 as the material approaches incompressible. Silicone gel sits at
  the incompressible end.

No parameter choice reproduces the bulge: `h` is a softplus, so it can never be negative. Photometric
stereo reads normals, so a ring whose true slope has the opposite sign renders shading this model never
produces. Closing that needs an elastic surface response, which is not implemented here.
[`shear`](src/shear.rs) is the part of the crate that *is* elastic, via Cattaneo-Mindlin partial slip.

```rust
use ferromotion_tactile::{GelSim, Indenter, default_lights};

let gel = GelSim { n: 81, extent: 1.0, beta: 0.02 };
let ind = Indenter { cx: 0.0, cy: 0.0, radius: 0.35, depth: 0.15 };
let img = gel.tactile_image(&ind, &default_lights()); // RGB imprint
let (sum_h, d_sum_h_d_depth) = gel.total_deformation(&ind); // exact gradient
```

Dual-licensed MIT OR Apache-2.0.
