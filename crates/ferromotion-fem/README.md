# ferromotion-fem

[![crates.io](https://img.shields.io/crates/v/ferromotion-fem.svg)](https://crates.io/crates/ferromotion-fem)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-fem)](https://docs.rs/ferromotion-fem)

**Differentiable volumetric tetrahedral FEM** for soft-body physical AI — the *solid* in
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion)'s deformable spectrum, alongside
the thin shell (`ferromotion-cloth`), the filament (`ferromotion-rod`), and the particle method
(`ferromotion-mpm`). Pure Rust (native + `wasm32`).

The body is a tetrahedral mesh of constant-strain elements. From the rest edge matrix `Dm` and the
current edge matrix `Ds` comes the deformation gradient `F = Ds·Dm⁻¹`, and from `F` a compressible
Neo-Hookean strain-energy density

```text
Ψ(F) = ½μ(I_C − 3) − μ ln J + ½λ(ln J)²
```

whose first Piola–Kirchhoff stress `P = ∂Ψ/∂F = μ(F − F⁻ᵀ) + λ ln(J)·F⁻ᵀ` gives the exact nodal forces
`−V₀·P·Dm⁻ᵀ`. This form is frame-indifferent — a rigid motion costs zero energy — and robust under large
deformation, unlike a linear or St.-Venant–Kirchhoff model.

Everything is a smooth function of the vertex positions, so forces are exactly `−∇energy` and the gradient
of an outcome with respect to material stiffness is available in closed form. Both are verified against
finite differences.

```rust
use ferromotion_fem::FemSim;

// a 2×2×2 tet-meshed block, 0.3 m cells, μ = 400, λ = 200
let mut sim = FemSim::box_grid(2, 2, 2, 0.3, 0.4, 4.0e2, 2.0e2, 3e-4);
for _ in 0..100 {
    sim.step();
}
let e = sim.energy();
```

**A note on the `ln J` barrier, since it bounds what this solver can be asked to do.** `ln J` diverges as
an element inverts, so `Ψ` is `+∞` at `J ≤ 0` while `forces` skips such elements — the barrier therefore has
no gradient, and an inverted tet neither recovers nor reports. Keep the timestep inside the range where
elements stay non-inverted. The literature's *stable* Neo-Hookean formulations exist precisely to remove
this barrier and remain finite and differentiable through inversion; adopting one is the natural next step
here, and until then the constraint is real rather than cosmetic.

A `gpu` feature adds a wgpu compute path for the force assembly. Dual-licensed MIT OR Apache-2.0.
