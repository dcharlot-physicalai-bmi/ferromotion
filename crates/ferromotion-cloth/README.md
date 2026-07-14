# ferromotion-cloth

[![crates.io](https://img.shields.io/crates/v/ferromotion-cloth.svg)](https://crates.io/crates/ferromotion-cloth)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-cloth)](https://docs.rs/ferromotion-cloth)

A **differentiable FEM thin-shell cloth** solver — the third material domain alongside
[`ferromotion-fluid`](https://crates.io/crates/ferromotion-fluid) and
[`ferromotion-mpm`](https://crates.io/crates/ferromotion-mpm), in pure Rust (native + `wasm32`), in the
spirit of Diffclothai / DaXBench.

The cloth is a triangle mesh. Each triangle is a **constant-strain membrane element** with a
St. Venant–Kirchhoff material: from the deformation gradient `F = Ds·Dm⁻¹` come the Green strain
`E = ½(FᵀF − I)`, the 2nd-PK stress `S = 2μE + λ tr(E) I`, and the exact nodal forces
`H = −A₀·F·S·Dm⁻ᵀ`. Bending springs across quad diagonals give out-of-plane stiffness.

Everything is a smooth function of the vertex positions, so the solver is differentiable. Verified:
the membrane forces are the exact gradient of the elastic energy (`force = −∇E`, checked by finite
differences); a pinned cloth drapes to static equilibrium; and — since `S` is **linear in the Lamé
parameters** — the exact `∂KE/∂μ` matches finite differences to machine precision (`rel_err ~3e-14`).

```rust
use ferromotion_cloth::ClothSim;
let mut c = ClothSim::grid(20, 20, 0.05, 0.01, 500.0, 300.0, 20.0, 1e-3);
c.pinned[380] = true; c.pinned[399] = true; // pin two top corners
for _ in 0..2000 { c.step(); }               // drapes under gravity
let (ke, dke_dmu) = c.ke_and_dke_dmu();       // exact material gradient
```

Dual-licensed MIT OR Apache-2.0.
