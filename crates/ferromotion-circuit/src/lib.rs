//! **Electrical-network dynamics** — the electrical domain of the ferromotion physics substrate, in
//! the same shape as the mechanical ones: a conservation-law system integrated implicitly and checked
//! against analytic oracles. A circuit is a dynamical system whose conservation law is Kirchhoff's:
//! current is conserved at every node (KCL) and voltage around every loop (KVL). Energy lives in the
//! reactive elements — `½C·v²` in a capacitor, `½L·i²` in an inductor — and dissipates as `i²R` in the
//! resistors, exactly the storage/dissipation bookkeeping the mechanical domains keep.
//!
//! The engine is **Modified Nodal Analysis** (MNA): stamp each element into a conductance matrix `G`
//! and a reactive matrix `C`, giving the differential-algebraic system `G·x + C·ẋ = b(t)`, where the
//! unknowns `x` are the node voltages plus a branch current for every inductor and voltage source.
//! Transient response is the **trapezoidal rule** (SPICE's default): A-stable, and — because the
//! bilinear transform maps an undamped `LC` pole exactly onto the unit circle — it neither grows nor
//! damps a lossless oscillator, so an `LC` tank conserves its energy to rounding. Pure `nalgebra`,
//! WASM-clean, differentiable-ready.
//!
//! Nonlinear devices (a Shockley diode and a MOSFET-as-switch) turn each timestep into a Newton solve
//! and make digital logic expressible: see [`Mna::dc_nonlinear_stepped`] and [`Mna::transient_nonlinear`].
//!
//! This is the physics under the "growing circuits" idea: a morphogenesis pass grows a netlist as a
//! graph, and this domain simulates the electrical behaviour of what grew. [`morpho`] carries that
//! through end to end — one recursive rule grows an adder of any width, it is lowered to transistors,
//! and the analog node voltages decide whether the grown design actually computes.

pub mod morpho;

use nalgebra::{DMatrix, DVector};

/// A source waveform, evaluated at a time `t` (seconds).
#[derive(Clone, Copy, Debug)]
pub enum Waveform {
    /// Constant value.
    Dc(f64),
    /// `amp·sin(2π·freq·t + phase)`.
    Sine { amp: f64, freq: f64, phase: f64 },
    /// `0` before `t0`, then `v` (a step).
    Step { v: f64, t0: f64 },
}

impl Waveform {
    /// The waveform value at time `t`.
    pub fn at(&self, t: f64) -> f64 {
        match *self {
            Waveform::Dc(v) => v,
            Waveform::Sine { amp, freq, phase } => amp * (std::f64::consts::TAU * freq * t + phase).sin(),
            Waveform::Step { v, t0 } => {
                if t >= t0 {
                    v
                } else {
                    0.0
                }
            }
        }
    }
}

/// A two-terminal (or four-terminal, for controlled sources) circuit element. Node `0` is ground.
#[derive(Clone, Copy, Debug)]
pub enum Element {
    /// Resistor of resistance `r` ohms between nodes `a`,`b`.
    Resistor { a: usize, b: usize, r: f64 },
    /// Capacitor of capacitance `c` farads between nodes `a`,`b`.
    Capacitor { a: usize, b: usize, c: f64 },
    /// Inductor of inductance `l` henries between nodes `a`,`b`.
    Inductor { a: usize, b: usize, l: f64 },
    /// Independent voltage source: `v(a) − v(b) = wave(t)` (`a` is +).
    Vsource { a: usize, b: usize, wave: Waveform },
    /// Independent current source: forces current `wave(t)` from node `a` to node `b`.
    Isource { a: usize, b: usize, wave: Waveform },
    /// Voltage-controlled voltage source: `v(p) − v(n) = gain·(v(cp) − v(cn))` (an ideal op-amp gain,
    /// a small-signal transistor, a buffer).
    Vcvs { p: usize, n: usize, cp: usize, cn: usize, gain: f64 },
    /// Voltage-controlled current source: current `gm·(v(cp) − v(cn))` flows from `a` to `b`
    /// (a transconductance — the active element of a transistor small-signal model).
    Vccs { a: usize, b: usize, cp: usize, cn: usize, gm: f64 },
    /// **Nonlinear** Shockley diode from `a` (anode) to `b` (cathode): current `Is·(exp(V/(n·Vt)) − 1)`
    /// with `V = v(a) − v(b)`. Rectifies; the canonical nonlinear device (needs the Newton transient).
    Diode { a: usize, b: usize, is: f64, n_vt: f64 },
    /// **Nonlinear** voltage-controlled switch (a MOSFET-as-switch) conducting between `a` and `b` with a
    /// conductance that swings from `1/roff` to `1/ron` as the control voltage `v(cp) − v(cn)` crosses
    /// `vth` over width `vslope`. The digital primitive: wire it as a pull-down and it inverts.
    Switch { a: usize, b: usize, cp: usize, cn: usize, ron: f64, roff: f64, vth: f64, vslope: f64 },
}

/// A circuit: a node count (node `0` is ground) and a list of elements.
#[derive(Clone, Debug, Default)]
pub struct Circuit {
    n_nodes: usize,
    elements: Vec<Element>,
}

impl Circuit {
    /// A circuit with `n_nodes` nodes, node `0` being ground.
    pub fn new(n_nodes: usize) -> Self {
        Circuit { n_nodes, elements: Vec::new() }
    }
    /// Add a resistor `r` ohms between `a` and `b`.
    pub fn resistor(&mut self, a: usize, b: usize, r: f64) -> &mut Self {
        self.elements.push(Element::Resistor { a, b, r });
        self
    }
    /// Add a capacitor `c` farads between `a` and `b`.
    pub fn capacitor(&mut self, a: usize, b: usize, c: f64) -> &mut Self {
        self.elements.push(Element::Capacitor { a, b, c });
        self
    }
    /// Add an inductor `l` henries between `a` and `b`.
    pub fn inductor(&mut self, a: usize, b: usize, l: f64) -> &mut Self {
        self.elements.push(Element::Inductor { a, b, l });
        self
    }
    /// Add an independent voltage source (`a` is +) with the given waveform.
    pub fn vsource(&mut self, a: usize, b: usize, wave: Waveform) -> &mut Self {
        self.elements.push(Element::Vsource { a, b, wave });
        self
    }
    /// Add an independent current source pushing current `wave(t)` from `a` to `b`.
    pub fn isource(&mut self, a: usize, b: usize, wave: Waveform) -> &mut Self {
        self.elements.push(Element::Isource { a, b, wave });
        self
    }
    /// Add a voltage-controlled voltage source `v(p,n) = gain·v(cp,cn)`.
    pub fn vcvs(&mut self, p: usize, n: usize, cp: usize, cn: usize, gain: f64) -> &mut Self {
        self.elements.push(Element::Vcvs { p, n, cp, cn, gain });
        self
    }
    /// Add a voltage-controlled current source `i(a→b) = gm·v(cp,cn)`.
    pub fn vccs(&mut self, a: usize, b: usize, cp: usize, cn: usize, gm: f64) -> &mut Self {
        self.elements.push(Element::Vccs { a, b, cp, cn, gm });
        self
    }
    /// Add a Shockley diode (anode `a` → cathode `b`), saturation current `is`, thermal scale `n·Vt`
    /// (≈ 0.026 V at room temperature for `n = 1`). Nonlinear — solve with the Newton transient.
    pub fn diode(&mut self, a: usize, b: usize, is: f64, n_vt: f64) -> &mut Self {
        self.elements.push(Element::Diode { a, b, is, n_vt });
        self
    }
    /// Add a voltage-controlled switch between `a` and `b`, controlled by `v(cp) − v(cn)`: `ron` when on
    /// (control ≫ `vth`), `roff` when off, transitioning over `vslope`. An n-type MOSFET switch.
    #[allow(clippy::too_many_arguments)]
    pub fn switch(&mut self, a: usize, b: usize, cp: usize, cn: usize, ron: f64, roff: f64, vth: f64, vslope: f64) -> &mut Self {
        self.elements.push(Element::Switch { a, b, cp, cn, ron, roff, vth, vslope });
        self
    }

    /// Assemble the MNA system: `G·x + C·ẋ = b(t)`.
    pub fn build(&self) -> Mna {
        let nv = self.n_nodes.saturating_sub(1); // node voltages (ground excluded)
        // count auxiliary current unknowns (inductors + voltage sources + VCVS)
        let mut n_aux = 0;
        for e in &self.elements {
            if matches!(e, Element::Inductor { .. } | Element::Vsource { .. } | Element::Vcvs { .. }) {
                n_aux += 1;
            }
        }
        let n = nv + n_aux;
        let mut g = DMatrix::zeros(n, n);
        let mut cm = DMatrix::zeros(n, n);
        // node → matrix index (ground `0` → None)
        let idx = |node: usize| -> Option<usize> { if node == 0 { None } else { Some(node - 1) } };
        // stamp a value into a matrix cell, skipping any ground row/col
        let stamp = |m: &mut DMatrix<f64>, r: Option<usize>, c: Option<usize>, v: f64| {
            if let (Some(r), Some(c)) = (r, c) {
                m[(r, c)] += v;
            }
        };

        let mut caps: Vec<(usize, usize, f64)> = Vec::new();
        let mut inds: Vec<Ind> = Vec::new();
        let mut vsrcs: Vec<Vsrc> = Vec::new();
        let mut isrcs: Vec<Isrc> = Vec::new();
        let mut nonlin: Vec<NonLin> = Vec::new();
        let mut next_aux = nv;

        for e in &self.elements {
            match *e {
                Element::Resistor { a, b, r } => {
                    let (ia, ib) = (idx(a), idx(b));
                    let gg = 1.0 / r;
                    stamp(&mut g, ia, ia, gg);
                    stamp(&mut g, ib, ib, gg);
                    stamp(&mut g, ia, ib, -gg);
                    stamp(&mut g, ib, ia, -gg);
                }
                Element::Capacitor { a, b, c } => {
                    let (ia, ib) = (idx(a), idx(b));
                    stamp(&mut cm, ia, ia, c);
                    stamp(&mut cm, ib, ib, c);
                    stamp(&mut cm, ia, ib, -c);
                    stamp(&mut cm, ib, ia, -c);
                    caps.push((a, b, c));
                }
                Element::Inductor { a, b, l } => {
                    let (ia, ib) = (idx(a), idx(b));
                    let k = next_aux;
                    next_aux += 1;
                    let ik = Some(k);
                    // KCL: current i_L (a→b) leaves a, enters b
                    stamp(&mut g, ia, ik, 1.0);
                    stamp(&mut g, ib, ik, -1.0);
                    // branch: v_a − v_b − L·i_L' = 0
                    stamp(&mut g, ik, ia, 1.0);
                    stamp(&mut g, ik, ib, -1.0);
                    cm[(k, k)] = -l;
                    inds.push(Ind { aux: k, l });
                }
                Element::Vsource { a, b, wave } => {
                    let (ia, ib) = (idx(a), idx(b));
                    let k = next_aux;
                    next_aux += 1;
                    let ik = Some(k);
                    stamp(&mut g, ia, ik, 1.0);
                    stamp(&mut g, ib, ik, -1.0);
                    stamp(&mut g, ik, ia, 1.0);
                    stamp(&mut g, ik, ib, -1.0);
                    vsrcs.push(Vsrc { aux: k, wave });
                }
                Element::Isource { a, b, wave } => {
                    isrcs.push(Isrc { a: idx(a), b: idx(b), wave });
                }
                Element::Vcvs { p, n: nn, cp, cn, gain } => {
                    let (ip, inn, icp, icn) = (idx(p), idx(nn), idx(cp), idx(cn));
                    let k = next_aux;
                    next_aux += 1;
                    let ik = Some(k);
                    stamp(&mut g, ip, ik, 1.0);
                    stamp(&mut g, inn, ik, -1.0);
                    // v_p − v_n − gain·(v_cp − v_cn) = 0
                    stamp(&mut g, ik, ip, 1.0);
                    stamp(&mut g, ik, inn, -1.0);
                    stamp(&mut g, ik, icp, -gain);
                    stamp(&mut g, ik, icn, gain);
                }
                Element::Vccs { a, b, cp, cn, gm } => {
                    let (ia, ib, icp, icn) = (idx(a), idx(b), idx(cp), idx(cn));
                    // current gm·(v_cp − v_cn) leaves a, enters b
                    stamp(&mut g, ia, icp, gm);
                    stamp(&mut g, ia, icn, -gm);
                    stamp(&mut g, ib, icp, -gm);
                    stamp(&mut g, ib, icn, gm);
                }
                // nonlinear devices contribute nothing to the constant G/C; they are stamped per
                // Newton iteration from the current voltages
                Element::Diode { a, b, is, n_vt } => {
                    nonlin.push(NonLin::Diode { a: idx(a), b: idx(b), is, n_vt });
                }
                Element::Switch { a, b, cp, cn, ron, roff, vth, vslope } => {
                    nonlin.push(NonLin::Switch { a: idx(a), b: idx(b), cp: idx(cp), cn: idx(cn), g_on: 1.0 / ron, g_off: 1.0 / roff, vth, vslope });
                }
            }
        }

        Mna { n, nv, g, cm, caps, inds, vsrcs, isrcs, nonlin }
    }
}

#[derive(Clone, Copy, Debug)]
struct Ind {
    aux: usize,
    l: f64,
}
#[derive(Clone, Copy, Debug)]
struct Vsrc {
    aux: usize,
    wave: Waveform,
}
#[derive(Clone, Copy, Debug)]
struct Isrc {
    a: Option<usize>,
    b: Option<usize>,
    wave: Waveform,
}
/// A nonlinear device, with its terminals resolved to matrix indices (`None` = ground).
#[derive(Clone, Copy, Debug)]
enum NonLin {
    Diode { a: Option<usize>, b: Option<usize>, is: f64, n_vt: f64 },
    Switch { a: Option<usize>, b: Option<usize>, cp: Option<usize>, cn: Option<usize>, g_on: f64, g_off: f64, vth: f64, vslope: f64 },
}

/// The assembled Modified Nodal Analysis system `G·x + C·ẋ = b(t)`, ready to solve. The unknown
/// vector `x` is `[node voltages (node 1..), then one branch current per inductor / voltage source /
/// VCVS in the order they were added]`.
pub struct Mna {
    n: usize,
    nv: usize,
    g: DMatrix<f64>,
    cm: DMatrix<f64>,
    caps: Vec<(usize, usize, f64)>,
    inds: Vec<Ind>,
    vsrcs: Vec<Vsrc>,
    isrcs: Vec<Isrc>,
    nonlin: Vec<NonLin>,
}

impl Mna {
    /// Number of unknowns.
    pub fn dim(&self) -> usize {
        self.n
    }
    /// Matrix index of a node's voltage in `x` (ground → `None`).
    pub fn node_index(&self, node: usize) -> Option<usize> {
        if node == 0 {
            None
        } else {
            Some(node - 1)
        }
    }
    /// Node voltage from a state vector.
    pub fn voltage(&self, x: &DVector<f64>, node: usize) -> f64 {
        self.node_index(node).map(|i| x[i]).unwrap_or(0.0)
    }
    /// The `x`-index of the `k`-th inductor's current (0-based, add order).
    pub fn inductor_current_index(&self, k: usize) -> usize {
        self.inds[k].aux
    }

    /// The source right-hand side `b(t)`.
    fn rhs(&self, t: f64) -> DVector<f64> {
        let mut b = DVector::zeros(self.n);
        for s in &self.isrcs {
            let i = s.wave.at(t);
            if let Some(a) = s.a {
                b[a] -= i;
            }
            if let Some(bb) = s.b {
                b[bb] += i;
            }
        }
        for s in &self.vsrcs {
            b[s.aux] = s.wave.at(t);
        }
        b
    }

    /// DC operating point: solve `G·x = b(0)` (capacitors open, inductors shorted). `None` if `G` is
    /// singular (a node with no DC reference).
    pub fn dc(&self) -> Option<DVector<f64>> {
        self.g.clone().lu().solve(&self.rhs(0.0))
    }

    /// Trapezoidal transient from a zero initial state.
    pub fn transient(&self, dt: f64, steps: usize) -> Solution {
        self.transient_from(&DVector::zeros(self.n), dt, steps)
    }

    /// Trapezoidal transient from an explicit initial state `x0` (set initial capacitor voltages via
    /// node entries and inductor currents via [`inductor_current_index`](Self::inductor_current_index)).
    /// Integrates `G·x + C·ẋ = b` with `(C/dt + G/2)·xₙ₊₁ = (C/dt − G/2)·xₙ + ½(bₙ + bₙ₊₁)`.
    pub fn transient_from(&self, x0: &DVector<f64>, dt: f64, steps: usize) -> Solution {
        let a = &self.cm / dt + &self.g * 0.5;
        let m = &self.cm / dt - &self.g * 0.5;
        let lu = a.lu();
        let mut x = x0.clone();
        let mut xs = Vec::with_capacity(steps + 1);
        let mut ts = Vec::with_capacity(steps + 1);
        xs.push(x.clone());
        ts.push(0.0);
        for k in 0..steps {
            let t0 = k as f64 * dt;
            let t1 = t0 + dt;
            let rhs = &m * &x + (self.rhs(t0) + self.rhs(t1)) * 0.5;
            x = lu.solve(&rhs).expect("MNA system singular during transient");
            xs.push(x.clone());
            ts.push(t1);
        }
        Solution { t: ts, x: xs }
    }

    /// Whether the circuit has any nonlinear devices (diodes / switches).
    pub fn is_nonlinear(&self) -> bool {
        !self.nonlin.is_empty()
    }

    /// Add each nonlinear device's current into the KCL residual `f` and its small-signal conductance
    /// into the Jacobian `j`, evaluated at the present voltages `x`. This is the per-Newton-iteration
    /// linearization (companion model) of the diodes and switches.
    fn stamp_nonlin(&self, x: &DVector<f64>, f: &mut DVector<f64>, j: &mut DMatrix<f64>) {
        let volt = |n: Option<usize>| n.map(|i| x[i]).unwrap_or(0.0);
        // add current `cur` (leaving `a`, entering `b`) and conductance `g` (∂cur/∂V) as a resistor-like stamp
        let mut add = |f: &mut DVector<f64>, j: &mut DMatrix<f64>, a: Option<usize>, b: Option<usize>, cur: f64, g: f64| {
            if let Some(a) = a {
                f[a] += cur;
                j[(a, a)] += g;
            }
            if let Some(b) = b {
                f[b] -= cur;
                j[(b, b)] += g;
            }
            if let (Some(a), Some(b)) = (a, b) {
                j[(a, b)] -= g;
                j[(b, a)] -= g;
            }
        };
        for d in &self.nonlin {
            match *d {
                NonLin::Diode { a, b, is, n_vt } => {
                    let v = volt(a) - volt(b);
                    // junction limiting: cap the exponent to avoid overflow (series R keeps V modest in practice)
                    let arg = (v / n_vt).min(40.0);
                    let e = arg.exp();
                    let cur = is * (e - 1.0);
                    let g = (is / n_vt) * e; // dI/dV
                    add(f, j, a, b, cur, g);
                }
                NonLin::Switch { a, b, cp, cn, g_on, g_off, vth, vslope } => {
                    let vc = volt(cp) - volt(cn);
                    let vab = volt(a) - volt(b);
                    let u = (vc - vth) / vslope;
                    let s = 1.0 / (1.0 + (-u).exp()); // logistic gate: 0 (off) → 1 (on)
                    let g = g_off + (g_on - g_off) * s; // conductance a↔b
                    let cur = g * vab;
                    // ∂cur/∂v(a) = g, ∂cur/∂v(b) = −g (resistor-like)
                    add(f, j, a, b, cur, g);
                    // ∂cur/∂control = (g_on−g_off)·σ'(u)/vslope · vab, σ' = s(1−s)
                    let dg_dvc = (g_on - g_off) * s * (1.0 - s) / vslope;
                    let dcur_dvc = dg_dvc * vab;
                    if let Some(a) = a {
                        if let Some(cp) = cp {
                            j[(a, cp)] += dcur_dvc;
                        }
                        if let Some(cn) = cn {
                            j[(a, cn)] -= dcur_dvc;
                        }
                    }
                    if let Some(b) = b {
                        if let Some(cp) = cp {
                            j[(b, cp)] -= dcur_dvc;
                        }
                        if let Some(cn) = cn {
                            j[(b, cn)] += dcur_dvc;
                        }
                    }
                }
            }
        }
    }

    /// One Newton solve of `base·x + i_nl(x) = rhs`, returning the converged `x` (or the last iterate).
    fn newton_solve(&self, base: &DMatrix<f64>, rhs: &DVector<f64>, x0: &DVector<f64>, tol: f64, max_iter: usize) -> DVector<f64> {
        let mut x = x0.clone();
        for _ in 0..max_iter {
            let mut f = base * &x - rhs;
            let mut jac = base.clone();
            self.stamp_nonlin(&x, &mut f, &mut jac);
            if f.norm() < tol {
                break;
            }
            match jac.lu().solve(&(-&f)) {
                Some(dx) => {
                    // damped step for robustness on the stiff diode exponential
                    let step = if dx.amax() > 1.0 { 1.0 / dx.amax() } else { 1.0 };
                    x += step * dx;
                }
                None => break,
            }
        }
        x
    }

    /// Nonlinear DC operating point: Newton on `G·x + i_nl(x) = b(0)` (capacitors open, inductors shorted).
    pub fn dc_nonlinear(&self) -> DVector<f64> {
        self.newton_solve(&self.g, &self.rhs(0.0), &DVector::zeros(self.n), 1e-9, 200)
    }

    /// Nonlinear DC operating point by **source stepping**: bring the sources up from zero in `steps`
    /// increments, warm-starting Newton from the previous solution each time. Plain Newton from a cold
    /// start diverges on large switching circuits (many gates can sit at their threshold, where the
    /// device derivative is largest and the iteration has no reason to pick a side); walking the
    /// supply up hands each solve an initial guess that is already close. This is the standard remedy
    /// and is what makes a few hundred coupled logic gates solvable.
    pub fn dc_nonlinear_stepped(&self, steps: usize) -> DVector<f64> {
        let b = self.rhs(0.0);
        let mut x = DVector::zeros(self.n);
        for k in 1..=steps.max(1) {
            let alpha = k as f64 / steps.max(1) as f64;
            x = self.newton_solve(&self.g, &(&b * alpha), &x, 1e-9, 200);
        }
        x
    }

    /// Nonlinear DC solve started from an explicit guess. With a good guess (the operating point of a
    /// nearby input combination) this converges in a handful of iterations, which is what makes
    /// re-solving a large switching circuit interactive. Check [`dc_residual`](Self::dc_residual) on the
    /// result and fall back to [`dc_nonlinear_stepped`](Self::dc_nonlinear_stepped) if the guess was bad.
    pub fn dc_nonlinear_from(&self, x0: &DVector<f64>) -> DVector<f64> {
        self.newton_solve(&self.g, &self.rhs(0.0), x0, 1e-9, 100)
    }

    /// Norm of the DC residual `‖G·x + i_nl(x) − b(0)‖` at a state — how well a solve actually converged.
    pub fn dc_residual(&self, x: &DVector<f64>) -> f64 {
        let mut f = &self.g * x - self.rhs(0.0);
        let mut j = self.g.clone();
        self.stamp_nonlin(x, &mut f, &mut j);
        f.norm()
    }

    /// Backward-Euler transient for circuits with nonlinear devices: at each step Newton-solves
    /// `(C/dt + G)·xₙ₊₁ + i_nl(xₙ₊₁) = bₙ₊₁ + (C/dt)·xₙ`. Reduces to the exact linear BE step when there
    /// are no nonlinear devices. Backward Euler (not trapezoidal) for the unconditional stability that
    /// switching/rectifying circuits need.
    pub fn transient_nonlinear_from(&self, x0: &DVector<f64>, dt: f64, steps: usize) -> Solution {
        let base = &self.cm / dt + &self.g;
        let mut x = x0.clone();
        let mut xs = Vec::with_capacity(steps + 1);
        let mut ts = Vec::with_capacity(steps + 1);
        xs.push(x.clone());
        ts.push(0.0);
        for k in 0..steps {
            let t1 = (k + 1) as f64 * dt;
            let rhs = self.rhs(t1) + &self.cm / dt * &x;
            x = self.newton_solve(&base, &rhs, &x, 1e-9, 100);
            xs.push(x.clone());
            ts.push(t1);
        }
        Solution { t: ts, x: xs }
    }

    /// Nonlinear transient from a zero initial state.
    pub fn transient_nonlinear(&self, dt: f64, steps: usize) -> Solution {
        self.transient_nonlinear_from(&DVector::zeros(self.n), dt, steps)
    }

    /// Energy stored in the reactive elements at state `x`: `Σ ½C·v_C² + Σ ½L·i_L²` (joules).
    pub fn stored_energy(&self, x: &DVector<f64>) -> f64 {
        let mut e = 0.0;
        for &(a, b, c) in &self.caps {
            let v = self.voltage(x, a) - self.voltage(x, b);
            e += 0.5 * c * v * v;
        }
        for ind in &self.inds {
            let i = x[ind.aux];
            e += 0.5 * ind.l * i * i;
        }
        e
    }
}

/// A transient solution: matched times and state vectors.
pub struct Solution {
    pub t: Vec<f64>,
    pub x: Vec<DVector<f64>>,
}

impl Solution {
    /// The number of stored steps (including the initial state).
    pub fn len(&self) -> usize {
        self.t.len()
    }
    /// Whether the solution is empty.
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

#[cfg(test)]
mod verification {
    use super::*;

    // The nonlinear DC operating point of a source→R→diode→gnd loop is self-consistent: the current
    // through R (Ohm) equals the diode's Shockley current at the same junction voltage, to Newton tol.
    #[test]
    fn diode_series_r_dc_is_self_consistent() {
        let (vs, r, is, n_vt) = (5.0, 1_000.0, 1e-12, 0.026);
        let mut ckt = Circuit::new(3); // 0=gnd, 1=source, 2=anode
        ckt.vsource(1, 0, Waveform::Dc(vs)).resistor(1, 2, r).diode(2, 0, is, n_vt);
        let mna = ckt.build();
        let x = mna.dc_nonlinear();
        let vd = mna.voltage(&x, 2); // junction voltage (cathode = gnd)
        let i_r = (vs - vd) / r; // current through the resistor
        let i_d = is * ((vd / n_vt).exp() - 1.0); // Shockley current
        eprintln!("diode DC: Vd {:.4} V, I_R {:.4e} A, I_D {:.4e} A, mismatch {:.2e}", vd, i_r, i_d, (i_r - i_d).abs());
        assert!(vd > 0.4 && vd < 0.9, "junction voltage not in the diode-drop range: {vd}");
        assert!((i_r - i_d).abs() / i_r.abs().max(1e-12) < 1e-6, "KCL not satisfied at the diode node");
    }

    // A half-wave rectifier: a sine source through a diode into an R load passes the positive half and
    // blocks the negative half — the output never goes meaningfully below zero, and its positive peak is
    // the source peak minus a diode drop.
    #[test]
    fn half_wave_rectifier_clips_the_negative_half() {
        let (amp, freq, r) = (5.0, 1_000.0, 1_000.0);
        let mut ckt = Circuit::new(3); // 0=gnd, 1=source, 2=output
        ckt.vsource(1, 0, Waveform::Sine { amp, freq, phase: 0.0 })
            .diode(1, 2, 1e-12, 0.026)
            .resistor(2, 0, r);
        let mna = ckt.build();
        let dt = 1.0 / freq / 400.0;
        let sol = mna.transient_nonlinear(dt, 800); // two periods
        let (mut vmin, mut vmax) = (f64::INFINITY, f64::NEG_INFINITY);
        for x in sol.x.iter().skip(400) {
            let vo = mna.voltage(x, 2);
            vmin = vmin.min(vo);
            vmax = vmax.max(vo);
        }
        eprintln!("half-wave rectifier: output range [{vmin:.3}, {vmax:.3}] V (source ±{amp})");
        assert!(vmin > -0.05, "diode failed to block the negative half: vmin {vmin}");
        assert!(vmax > amp - 1.0 && vmax < amp, "positive peak not ~source − diode drop: {vmax}");
    }

    // With no nonlinear devices, the Newton transient must reproduce the linear (backward-Euler) result:
    // an RC step still charges toward V₀(1 − e^{−t/RC}).
    #[test]
    fn nonlinear_transient_reduces_to_linear_rc() {
        let (r, c, v0) = (1_000.0, 1e-6, 5.0);
        let mut ckt = Circuit::new(3);
        ckt.vsource(1, 0, Waveform::Dc(v0)).resistor(1, 2, r).capacitor(2, 0, c);
        let mna = ckt.build();
        assert!(!mna.is_nonlinear());
        let tau = r * c;
        let dt = tau / 2000.0; // small dt: backward Euler → analytic
        let sol = mna.transient_nonlinear(dt, 8000);
        let mut worst = 0.0f64;
        for (k, x) in sol.x.iter().enumerate() {
            let analytic = v0 * (1.0 - (-sol.t[k] / tau).exp());
            worst = worst.max((mna.voltage(x, 2) - analytic).abs());
        }
        eprintln!("nonlinear-solver on a linear RC: worst |Δ| {worst:.3e} V");
        assert!(worst < 1e-2, "Newton transient diverged from the linear RC: {worst} V");
    }

    // A resistor-load NMOS inverter: pull-up R from Vdd to the output, an n-switch (gate = input) from
    // output to ground. Input HIGH turns the switch on and pulls the output LOW; input LOW leaves it off
    // and the output rises to Vdd. The logic primitive the MorphoHDL bench is built from.
    #[test]
    fn nmos_inverter_inverts() {
        let vdd = 5.0;
        let invert = |vin: f64| {
            let mut ckt = Circuit::new(4); // 0=gnd, 1=Vdd, 2=out, 3=input
            ckt.vsource(1, 0, Waveform::Dc(vdd))
                .vsource(3, 0, Waveform::Dc(vin))
                .resistor(1, 2, 10_000.0) // pull-up
                .switch(2, 0, 3, 0, 200.0, 1e9, 2.5, 0.2); // out→gnd, gate=input, Ron 200Ω, Roff 1GΩ, Vth mid-rail 2.5V
            let mna = ckt.build();
            mna.voltage(&mna.dc_nonlinear(), 2)
        };
        let out_lo = invert(vdd); // input HIGH → output LOW
        let out_hi = invert(0.0); // input LOW  → output HIGH
        eprintln!("NMOS inverter: in HIGH → out {out_lo:.3} V, in LOW → out {out_hi:.3} V (Vdd {vdd})");
        assert!(out_lo < 0.2, "input HIGH should pull output LOW: {out_lo}");
        assert!(out_hi > 0.95 * vdd, "input LOW should leave output HIGH: {out_hi}");
    }

    // An RC low-pass driven by a voltage step charges toward the source exactly as V0(1 − e^{−t/RC}).
    #[test]
    fn rc_step_matches_analytic() {
        let (r, c, v0) = (1_000.0, 1e-6, 5.0); // τ = RC = 1 ms
        let mut ckt = Circuit::new(3); // 0=gnd, 1=source, 2=cap node
        ckt.vsource(1, 0, Waveform::Dc(v0)) // step already on at t=0
            .resistor(1, 2, r)
            .capacitor(2, 0, c);
        let mna = ckt.build();
        let tau = r * c;
        let dt = tau / 500.0;
        let sol = mna.transient(dt, 2000); // 4τ
        let mut worst = 0.0f64;
        for (k, x) in sol.x.iter().enumerate() {
            let t = sol.t[k];
            let analytic = v0 * (1.0 - (-t / tau).exp());
            worst = worst.max((mna.voltage(x, 2) - analytic).abs());
        }
        eprintln!("RC step: worst |v_C − analytic| = {worst:.3e} V");
        assert!(worst < 5e-3, "RC transient diverged from analytic: {worst} V");
    }

    // A lossless LC tank conserves its energy: the trapezoidal rule puts the undamped pole exactly on
    // the unit circle, so ½C·v² + ½L·i² is constant to rounding, and it oscillates at ω₀ = 1/√(LC).
    #[test]
    fn lc_tank_conserves_energy_and_frequency() {
        let (l, c, v0) = (1e-3, 1e-6, 2.0); // ω₀ = 1/√(LC) = 3.16e4 rad/s, f₀ ≈ 5.03 kHz
        let mut ckt = Circuit::new(2); // 0=gnd, 1=tank node
        ckt.capacitor(1, 0, c).inductor(1, 0, l);
        let mna = ckt.build();
        // initial state: capacitor charged to v0, inductor current 0
        let mut x0 = DVector::zeros(mna.dim());
        x0[mna.node_index(1).unwrap()] = v0;
        let w0 = 1.0 / (l * c).sqrt();
        let period = std::f64::consts::TAU / w0;
        let dt = period / 400.0;
        let steps = 400 * 25; // 25 periods
        let sol = mna.transient_from(&x0, dt, steps);
        let e0 = mna.stored_energy(&sol.x[0]);
        let (mut emin, mut emax) = (e0, e0);
        for x in &sol.x {
            let e = mna.stored_energy(x);
            emin = emin.min(e);
            emax = emax.max(e);
        }
        let drift = (emax - emin) / e0;
        // frequency: count zero-crossings of the tank voltage over the run
        let mut crossings = 0;
        for k in 1..sol.x.len() {
            let (a, b) = (mna.voltage(&sol.x[k - 1], 1), mna.voltage(&sol.x[k], 1));
            if a.signum() != b.signum() {
                crossings += 1;
            }
        }
        let f_meas = crossings as f64 / 2.0 / (steps as f64 * dt);
        let f0 = w0 / std::f64::consts::TAU;
        eprintln!("LC tank: energy drift {drift:.2e} over 25 periods; f_meas {f_meas:.1} Hz vs f₀ {f0:.1} Hz");
        assert!(drift < 1e-3, "LC tank did not conserve energy (drift {drift})");
        assert!((f_meas - f0).abs() / f0 < 0.02, "LC frequency off: {f_meas} vs {f0}");
    }

    // A series RLC step response is a damped sinusoid; the trapezoidal solve reproduces its damped
    // natural frequency ω_d = ω₀√(1 − ζ²) and decays (ζ < 1, underdamped).
    #[test]
    fn series_rlc_is_underdamped_and_decays() {
        let (r, l, c) = (20.0, 1e-3, 1e-6); // ω₀=3.16e4, ζ = (R/2)·√(C/L) = 0.316 (underdamped)
        // series loop: src(1)+ → R → node2 → L → node3 → C → gnd
        let mut ckt = Circuit::new(4);
        ckt.vsource(1, 0, Waveform::Step { v: 1.0, t0: 0.0 })
            .resistor(1, 2, r)
            .inductor(2, 3, l)
            .capacitor(3, 0, c);
        let mna = ckt.build();
        let w0 = 1.0 / (l * c).sqrt();
        let zeta = 0.5 * r * (c / l).sqrt();
        let wd = w0 * (1.0 - zeta * zeta).sqrt();
        let dt = (std::f64::consts::TAU / wd) / 400.0;
        let sol = mna.transient(dt, 400 * 8);
        // capacitor voltage overshoots 1.0 (underdamped) then rings down toward the final 1.0 V
        let vc: Vec<f64> = sol.x.iter().map(|x| mna.voltage(x, 3)).collect();
        let peak = vc.iter().cloned().fold(0.0f64, f64::max);
        let final_v = *vc.last().unwrap();
        // measure the ringing period from the first two successive maxima's spacing via zero-crossings
        // of (vc − final): underdamped means it crosses the settling value multiple times.
        let mut crossings = 0;
        for k in 1..vc.len() {
            if (vc[k - 1] - 1.0).signum() != (vc[k] - 1.0).signum() {
                crossings += 1;
            }
        }
        eprintln!("series RLC: peak {peak:.3} V (overshoot > 1), final {final_v:.3} V, ζ {zeta:.3}, {crossings} crossings of 1V");
        assert!(peak > 1.05, "underdamped step should overshoot 1 V, got peak {peak}");
        assert!((final_v - 1.0).abs() < 0.05, "should settle to the 1 V source, got {final_v}");
        assert!(crossings >= 3, "underdamped ringing should cross the settling value repeatedly, got {crossings}");
    }

    // KCL holds at every node at every step: the sum of element currents leaving each node is zero to
    // machine precision — the discrete conservation law the solver enforces.
    #[test]
    fn kcl_holds_at_every_node() {
        // a small mixed network: source → R → node2 (C to gnd) → R → node3 (L to gnd)
        let (r1, r2, c, l) = (500.0, 800.0, 2e-6, 5e-3);
        let mut ckt = Circuit::new(4);
        ckt.vsource(1, 0, Waveform::Sine { amp: 3.0, freq: 200.0, phase: 0.0 })
            .resistor(1, 2, r1)
            .capacitor(2, 0, c)
            .resistor(2, 3, r2)
            .inductor(3, 0, l);
        let mna = ckt.build();
        let dt = 1.0 / 200.0 / 200.0;
        let sol = mna.transient(dt, 600);
        // check KCL at nodes 2 and 3 mid-run using the trapezoidal derivative estimate for reactive currents
        let k = 400;
        let (xm, x0, xp) = (&sol.x[k - 1], &sol.x[k], &sol.x[k + 1]);
        let v = |x: &DVector<f64>, n: usize| mna.voltage(x, n);
        // node 2: i(R1: 1→2) into 2 = (v1 − v2)/r1 ; i(R2: 2→3) leaving ; i(C: 2→0) = C dv2/dt
        let dv2 = (v(xp, 2) - v(xm, 2)) / (2.0 * dt);
        let kcl2 = (v(x0, 1) - v(x0, 2)) / r1 - (v(x0, 2) - v(x0, 3)) / r2 - c * dv2;
        // node 3: i(R2 into 3) = i(L leaving 3) ; inductor current is an explicit unknown
        let il = sol.x[k][mna.inductor_current_index(0)];
        let kcl3 = (v(x0, 2) - v(x0, 3)) / r2 - il;
        eprintln!("KCL residuals: node2 {kcl2:.2e} A, node3 {kcl3:.2e} A");
        assert!(kcl2.abs() < 1e-6, "KCL violated at node 2: {kcl2} A");
        assert!(kcl3.abs() < 1e-9, "KCL violated at node 3: {kcl3} A");
    }

    // DC operating point of a resistive voltage divider equals the analytic ratio.
    #[test]
    fn voltage_divider_dc() {
        let (r1, r2, vin) = (2_000.0, 3_000.0, 10.0);
        let mut ckt = Circuit::new(3);
        ckt.vsource(1, 0, Waveform::Dc(vin)).resistor(1, 2, r1).resistor(2, 0, r2);
        let mna = ckt.build();
        let x = mna.dc().expect("divider has a DC solution");
        let v2 = mna.voltage(&x, 2);
        let analytic = vin * r2 / (r1 + r2);
        eprintln!("divider DC: v2 {v2:.4} V vs analytic {analytic:.4} V");
        assert!((v2 - analytic).abs() < 1e-9, "divider DC wrong: {v2} vs {analytic}");
    }

    // A voltage-controlled voltage source is a linear amplifier: the DC output is the gain times the
    // input — the active building block (op-amp / small-signal transistor) that makes logic possible.
    #[test]
    fn vcvs_amplifies() {
        let (vin, gain) = (0.1, 20.0);
        let mut ckt = Circuit::new(3);
        // node1 = input (driven), node2 = amplified output = gain·v(1,0)
        ckt.vsource(1, 0, Waveform::Dc(vin))
            .vcvs(2, 0, 1, 0, gain)
            .resistor(2, 0, 1_000.0); // load so node2 has a reference
        let mna = ckt.build();
        let x = mna.dc().expect("amplifier DC solution");
        let vout = mna.voltage(&x, 2);
        eprintln!("VCVS amp: vout {vout:.4} V vs expected {:.4} V", gain * vin);
        assert!((vout - gain * vin).abs() < 1e-9, "VCVS gain wrong: {vout} vs {}", gain * vin);
    }
}
