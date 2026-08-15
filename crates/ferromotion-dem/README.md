# ferromotion-dem

[![crates.io](https://img.shields.io/crates/v/ferromotion-dem.svg)](https://crates.io/crates/ferromotion-dem)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-dem)](https://docs.rs/ferromotion-dem)

**Discrete-element-method granular media** (Cundall & Strack, 1979) — sand, gravel, soil, pills: the
rigid-particle contact domain that complements the continuum deformables in
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion) (MPM, FEM, cloth, rod). Pure Rust
(native + `wasm32`), `nalgebra` only.

Each grain is a sphere. Contacts — grain-to-grain and grain-to-floor — resolve with a linear
spring–dashpot normal law and a Coulomb-limited tangential spring, so a contact stores elastic energy,
dissipates through the dashpot, and slips once the tangential force reaches `μ·f_n`.

Verified on the invariants granular contact must obey rather than on appearance:

- a frictionless head-on collision conserves momentum and, elastically, kinetic energy;
- a grain resting on the floor settles to the analytic spring penetration `mg/kₙ`;
- a poured pile comes to rest *above* the floor without exploding.

```rust
use ferromotion_dem::{DemSim, Grain};
use nalgebra::Vector3;

let grains = vec![Grain {
    x: Vector3::new(0.0, 0.0, 0.2),
    v: Vector3::zeros(),
    r: 0.02,
    m: 0.01,
}];
// kn = 1e4 N/m, damping 0.5, friction 0.3, dt = 1e-4
let mut sim = DemSim::new(grains, 1.0e4, 0.5, 0.3, 1e-4);
for _ in 0..2000 {
    sim.step();
}
let settled_z = sim.grains[0].x.z;
```

The spring–dashpot law is explicit, so the timestep has to resolve the contact stiffness: roughly
`dt ≪ 2·√(m/kₙ)`. Raising `kn` without lowering `dt` is the usual way to make a pile explode.

Two-way coupling to a deforming soft body lives in
[`ferromotion-coupled`](https://crates.io/crates/ferromotion-coupled). Dual-licensed MIT OR Apache-2.0.
