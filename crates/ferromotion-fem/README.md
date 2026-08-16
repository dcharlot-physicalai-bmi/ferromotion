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

## Material models

- **`plasticity`** — von Mises radial return in Hencky strain, with `det(F_p) = 1` held exactly, so plastic
  flow is volume-preserving by construction rather than to within a tolerance.
- **`viscoelastic`** — generalized Maxwell (Prony series): creep, stress relaxation, and a frequency-dependent
  complex modulus. This is the inelasticity that is fully recoverable and still not elastic, and the one whose
  symptoms read as sensor drift. A gripper pad with a 4 MPa instantaneous and 1 MPa equilibrium modulus keeps
  only **a quarter** of its grip force once it has finished relaxing; the position never changed and the
  commanded force never changed, so nothing in a position loop can see it coming.

  Its update is **exact for strain linear across the step, at any timestep** — asserted across four orders of
  magnitude of `dt` — because the branch ODE has a closed-form solution for constant strain rate. A
  viscoelastic material therefore adds no integration error of its own to a ramp.

  Two independent cross-checks rather than algebra restated: a time-domain sinusoid reproduces the closed-form
  storage and loss moduli in **both amplitude and phase**, and the creep integrator agrees with the relaxation
  modulus through the Laplace identity `s² Ê(s) Ĵ(s) = 1`. Negative branch stiffness is rejected at
  construction, because it gives `E'' < 0` over some band and a material that *generates* energy per cycle —
  which is exactly what an unconstrained least-squares fit to noisy data produces.

## A note on inversion, since it bounds what this solver can be asked to do

`ln J` diverges as an element inverts, so `Ψ` is unbounded at `J → 0⁺`. Below `J = 0` the energy switches to a
quadratic recovery well `½k(J − 0.1)²`, whose gradient routes through `∂J/∂F = cof(F)` and is therefore defined
at `det F = 0`, the configuration a flattening tet passes through. That replaced a bare `continue`, which had
left the advertised `+∞` barrier with **no gradient at all**: an inverted tet contributed no force in either
direction and could never recover, while `energy()` read `+inf` forever and the simulation carried on.

**This turns a zero gradient into a correctly-signed one. It does not make the model inversion-safe.** The
cofactor entries are cross-products of `F`'s columns, so the restoring force *vanishes* as the element
flattens — it dies exactly where it is most needed. Measured on an inverted unit tet with gravity off, `J`
rises monotonically from `−0.2` to `−0.0346` over 200k steps and does not cross zero. Escaping a flattened
element needs an energy whose gradient does not pass through `∂J/∂F`, which is what the literature's *stable*
Neo-Hookean formulations exist for. Keep the timestep inside the range where elements stay non-inverted.

A `gpu` feature adds a wgpu compute path for the force assembly. Dual-licensed MIT OR Apache-2.0.
