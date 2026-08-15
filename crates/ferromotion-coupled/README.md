# ferromotion-coupled

[![crates.io](https://img.shields.io/crates/v/ferromotion-coupled.svg)](https://crates.io/crates/ferromotion-coupled)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-coupled)](https://docs.rs/ferromotion-coupled)

**Two-way multiphysics coupling** between a volumetric soft body
([`ferromotion-fem`](https://crates.io/crates/ferromotion-fem)) and granular media
([`ferromotion-dem`](https://crates.io/crates/ferromotion-dem)) — two solvers each verified in isolation,
now interacting through a shared contact. Grains pile *on* a deforming body, a body sinks *into* grains,
and each pushes back. Pure Rust (native + `wasm32`).

The coupling is a symmetric penalty contact. A FEM surface vertex is treated as a small sphere, and every
FEM-vertex ↔ grain overlap applies an **equal-and-opposite** spring–dashpot force to both bodies. The
integrator owns gravity and the floor uniformly and advances both bodies together, so neither solver sees a
world the other does not.

**The oracle a correct two-way coupling must pass** is conservation: because every internal and contact
force is equal-and-opposite by construction, the coupled system conserves total linear momentum exactly —
not approximately, and not as a tuned result. That is what this crate is tested against.

```rust
use ferromotion_coupled::CoupledFemDem;
use ferromotion_dem::{DemSim, Grain};
use ferromotion_fem::FemSim;
use nalgebra::Vector3;

let fem = FemSim::box_grid(2, 2, 2, 0.05, 0.02, 3.0e3, 1.5e3, 2e-4);
let grains = vec![Grain { x: Vector3::new(0.05, 0.05, 0.4), v: Vector3::zeros(), r: 0.01, m: 0.002 }];
let dem = DemSim::new(grains, 1.0e4, 0.5, 0.3, 2e-4);

// vertex contact radius 0.01 m, coupling stiffness 1e4
let mut sim = CoupledFemDem::new(fem, dem, 0.01, 1.0e4);
for _ in 0..500 {
    sim.step();
}
```

Both sides are explicit, so the timestep must satisfy the *stiffer* of the two contact laws and the FEM
material — coupling does not relax either constraint, and the coupling stiffness `k_couple` adds a third.

`GraspFemSim` specializes the same machinery to a grasp: rigid fingers closing on a deformable object.
Dual-licensed MIT OR Apache-2.0.
