# ferromotion-models

[![crates.io](https://img.shields.io/crates/v/ferromotion-models.svg)](https://crates.io/crates/ferromotion-models)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-models)](https://docs.rs/ferromotion-models)

**Robot arms built from the tables their makers and their textbooks publish**, part of
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion), in pure Rust (native + `wasm32`).

Every model is a `Robot` assembled from a Denavit-Hartenberg table by `Robot::from_dh`, and every
table names its primary source, the DH convention that source uses (standard or modified), and any
unit conversion applied. DH parameters are public engineering data; URDF and mesh files carry licenses
and none are vendored here.

Each model is verified rather than transcribed: forward kinematics at a documented or hand-computed
pose, the central-difference Jacobian and Hessian check every `Robot` in the workspace is held to, and
a stated mutation showing the convention argument is load-bearing.

```rust
use ferromotion_models::Robot;
// constructors are added per model; each documents its source and its known-answer pose
```

Dual-licensed MIT OR Apache-2.0.
