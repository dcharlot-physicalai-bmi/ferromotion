# ferromotion-circuit

[![crates.io](https://img.shields.io/crates/v/ferromotion-circuit.svg)](https://crates.io/crates/ferromotion-circuit)
[![docs.rs](https://img.shields.io/docsrs/ferromotion-circuit)](https://docs.rs/ferromotion-circuit)

**Electrical-network dynamics as a conservation-law physics domain** — the electrical member of
[Ferromotion](https://github.com/dcharlot-physicalai-bmi/ferromotion)'s substrate, built in the same shape
as the mechanical ones: a conservation law, an implicit integrator, and analytic oracles. Pure Rust
(native + `wasm32`), `nalgebra` only.

A circuit is a dynamical system whose conservation law is Kirchhoff's — current conserved at every node,
voltage around every loop. Energy lives in the reactive elements (`½C·v²`, `½L·i²`) and dissipates as `i²R`,
which is exactly the storage-and-dissipation bookkeeping the mechanical domains keep.

The engine is **Modified Nodal Analysis**: stamp each element into a conductance matrix `G` and a reactive
matrix `C`, giving the differential-algebraic system

```text
G·x + C·ẋ = b(t)
```

where the unknowns `x` are node voltages plus a branch current for every inductor and voltage source.
Transient response uses the **trapezoidal rule**, SPICE's default. It is A-stable, and because the bilinear
transform maps an undamped `LC` pole exactly onto the unit circle, it neither grows nor damps a lossless
oscillator — **an `LC` tank conserves its energy to rounding**, which is the analytic oracle here.

Nonlinear devices — a Shockley diode and a MOSFET-as-switch — turn each timestep into a Newton solve and
make digital logic expressible: see `Mna::dc_nonlinear_stepped` and `Mna::transient_nonlinear`.

```rust
use ferromotion_circuit::Circuit;

let mut c = Circuit::new(2);      // node 0 is ground
// build an RC or LC network by stamping elements, then solve the transient
```

**Grown circuits.** The `morpho` module carries a morphogenesis idea end to end: one recursive rule grows an
adder of any width as a graph, the graph is lowered to transistors, and the *analog node voltages* decide
whether the grown design actually computes. The physics is what adjudicates the design, rather than a
symbolic check standing in for it.

Dual-licensed MIT OR Apache-2.0.
