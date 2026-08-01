//! **Grown circuits** — build a netlist by recursive rewriting instead of by declaring it, then lower
//! it to transistors so the electrical engine can decide whether it works.
//!
//! This is the morphogenetic style of hardware description: a *base case* plus a *recursive rule*
//! generate a circuit of any size, and no bus width is ever declared. [`Netlist::adder`] is the worked
//! example: an adder of width `n` is two adders of half the width with the carry chained, bottoming out
//! in a one-bit full adder. The same two rules yield a 1-bit or a 16-bit machine.
//!
//! Everything is expressed in one universal gate, NAND, so the grown object is a pure graph.
//! [`Netlist::lower`] turns that graph into a real circuit: each NAND becomes a resistor pull-up with
//! two switches in series to ground, which is a resistor-load NMOS gate. Solving it with
//! [`Mna::dc_nonlinear_stepped`](crate::Mna::dc_nonlinear_stepped) gives analog node voltages, so the
//! grown design is correct only if those voltages land on the right side of the switching threshold.
//! The physics is the referee, not a logic table.

use crate::{Circuit, Waveform};

/// Device and rail parameters for lowering a grown netlist to transistors.
#[derive(Clone, Copy, Debug)]
pub struct Tech {
    pub vdd: f64,
    /// Pull-up resistor from the rail to each gate output.
    pub r_pullup: f64,
    /// Switch resistance when conducting.
    pub r_on: f64,
    /// Switch resistance when off.
    pub r_off: f64,
    /// Gate switching threshold.
    pub vth: f64,
    /// Transition width of the gate. Too soft and a marginal input leaks, degrading the next stage.
    pub vslope: f64,
}

impl Default for Tech {
    fn default() -> Self {
        Tech { vdd: 5.0, r_pullup: 10_000.0, r_on: 200.0, r_off: 1e9, vth: 2.5, vslope: 0.06 }
    }
}

impl Tech {
    /// The logic-low a conducting gate produces: two series switches against the pull-up, as a divider.
    /// The analytic value the solved circuit is checked against.
    pub fn predicted_low(&self) -> f64 {
        self.vdd * (2.0 * self.r_on) / (2.0 * self.r_on + self.r_pullup)
    }
}

/// A grown netlist in one universal gate. Node `0` is ground and node `1` is the supply rail; every
/// other node is a primary input or the output of a NAND.
#[derive(Clone, Debug, Default)]
pub struct Netlist {
    n_nodes: usize,
    nands: Vec<(usize, usize, usize)>,
}

impl Netlist {
    pub fn new() -> Self {
        Netlist { n_nodes: 2, nands: Vec::new() } // 0 = ground, 1 = Vdd
    }
    /// Total node count (ground and rail included).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }
    /// The grown gates, as `(a, b, out)` with `out = NAND(a, b)`.
    pub fn gates(&self) -> &[(usize, usize, usize)] {
        &self.nands
    }
    /// Allocate a fresh node (used for primary inputs).
    pub fn node(&mut self) -> usize {
        self.n_nodes += 1;
        self.n_nodes - 1
    }
    /// The one primitive. Everything else is built from it.
    pub fn nand(&mut self, a: usize, b: usize) -> usize {
        let out = self.node();
        self.nands.push((a, b, out));
        out
    }
    pub fn not(&mut self, a: usize) -> usize {
        self.nand(a, a)
    }
    pub fn and(&mut self, a: usize, b: usize) -> usize {
        let n = self.nand(a, b);
        self.not(n)
    }
    pub fn or(&mut self, a: usize, b: usize) -> usize {
        let (na, nb) = (self.not(a), self.not(b));
        self.nand(na, nb)
    }
    pub fn xor(&mut self, a: usize, b: usize) -> usize {
        let c = self.nand(a, b);
        let (x, y) = (self.nand(a, c), self.nand(b, c));
        self.nand(x, y)
    }

    /// **The base case**: a one-bit full adder, returning `(sum, carry_out)`.
    pub fn full_adder(&mut self, a: usize, b: usize, cin: usize) -> (usize, usize) {
        let s1 = self.xor(a, b);
        let sum = self.xor(s1, cin);
        let c1 = self.and(a, b);
        let c2 = self.and(s1, cin);
        let cout = self.or(c1, c2);
        (sum, cout)
    }

    /// **The recursive rule**: an adder of width `n` is two adders of half the width with the carry
    /// chained. Nothing here names a width, so one definition grows every size.
    pub fn adder(&mut self, a: &[usize], b: &[usize], cin: usize) -> (Vec<usize>, usize) {
        assert_eq!(a.len(), b.len(), "operands must be the same width");
        if a.len() == 1 {
            let (s, c) = self.full_adder(a[0], b[0], cin);
            return (vec![s], c);
        }
        let h = a.len() / 2;
        let (mut lo, c) = self.adder(&a[..h], &b[..h], cin);
        let (hi, cout) = self.adder(&a[h..], &b[h..], c);
        lo.extend(hi);
        (lo, cout)
    }

    /// Lower to a transistor-level circuit: every NAND becomes a pull-up plus two series switches to
    /// ground. `inputs` drives each primary input node to a logic level.
    pub fn lower(&self, inputs: &[(usize, bool)], t: Tech) -> Circuit {
        let mut ckt = Circuit::new(self.n_nodes + self.nands.len()); // one midpoint node per gate
        ckt.vsource(1, 0, Waveform::Dc(t.vdd));
        for &(node, hi) in inputs {
            ckt.vsource(node, 0, Waveform::Dc(if hi { t.vdd } else { 0.0 }));
        }
        for (i, &(a, b, out)) in self.nands.iter().enumerate() {
            let mid = self.n_nodes + i;
            ckt.resistor(1, out, t.r_pullup);
            ckt.switch(out, mid, a, 0, t.r_on, t.r_off, t.vth, t.vslope); // upper device, gate = a
            ckt.switch(mid, 0, b, 0, t.r_on, t.r_off, t.vth, t.vslope); // lower device, gate = b
        }
        ckt
    }
}

/// A grown adder: the netlist plus the nodes that are its interface.
#[derive(Clone, Debug)]
pub struct Grown {
    pub netlist: Netlist,
    pub a: Vec<usize>,
    pub b: Vec<usize>,
    pub sum: Vec<usize>,
    pub cout: usize,
}

/// What one electrical evaluation of a grown adder produced.
#[derive(Clone, Debug)]
pub struct Solved {
    /// The number read off the output nodes by comparing each against the switching threshold.
    pub value: u32,
    /// Output voltage of every grown gate, in growth order.
    pub gate_v: Vec<f64>,
    /// Voltage of each sum bit, then the carry out.
    pub out_v: Vec<f64>,
    pub worst_high: f64,
    pub worst_low: f64,
    pub residual: f64,
    /// The raw MNA solution, so a nearby input combination can warm-start from this operating point.
    pub state: nalgebra::DVector<f64>,
}

/// Grow an `n`-bit adder. The recursion is the whole description; nothing states a width.
pub fn grow_adder(n: usize) -> Grown {
    let mut netlist = Netlist::new();
    let a: Vec<usize> = (0..n).map(|_| netlist.node()).collect();
    let b: Vec<usize> = (0..n).map(|_| netlist.node()).collect();
    let (sum, cout) = netlist.adder(&a, &b, 0); // carry-in tied to ground
    Grown { netlist, a, b, sum, cout }
}

impl Grown {
    /// Drive `x + y` into the grown circuit and solve it electrically. `steps` is the source-stepping
    /// count; more is more robust and slower, and [`recommended_steps`] picks a workable value.
    pub fn evaluate(&self, x: u32, y: u32, t: Tech, steps: usize) -> Solved {
        self.evaluate_warm(x, y, t, steps, None)
    }

    /// As [`evaluate`](Self::evaluate), but first try Newton from `warm`, the operating point of a
    /// nearby input combination. A good guess converges in a few iterations instead of walking the
    /// supply up from zero; if it does not converge, this falls back to source stepping, so the answer
    /// is the same either way and only the time taken changes.
    pub fn evaluate_warm(&self, x: u32, y: u32, t: Tech, steps: usize, warm: Option<&nalgebra::DVector<f64>>) -> Solved {
        let n = self.a.len();
        let mut inputs = Vec::with_capacity(2 * n);
        for i in 0..n {
            inputs.push((self.a[i], (x >> i) & 1 == 1));
            inputs.push((self.b[i], (y >> i) & 1 == 1));
        }
        let mna = self.netlist.lower(&inputs, t).build();
        let mut sol = match warm {
            Some(w) if w.len() == mna.dim() => mna.dc_nonlinear_from(w),
            _ => mna.dc_nonlinear_stepped(steps),
        };
        if mna.dc_residual(&sol) > 1e-9 {
            sol = mna.dc_nonlinear_stepped(steps); // the guess was not good enough
        }

        let gate_v: Vec<f64> = self.netlist.gates().iter().map(|&(_, _, out)| mna.voltage(&sol, out)).collect();
        let (mut value, mut worst_high, mut worst_low) = (0u32, f64::INFINITY, f64::NEG_INFINITY);
        let mut out_v = Vec::with_capacity(n + 1);
        for (i, &node) in self.sum.iter().chain(std::iter::once(&self.cout)).enumerate() {
            let v = mna.voltage(&sol, node);
            out_v.push(v);
            if v > t.vth {
                value |= 1 << i;
                worst_high = worst_high.min(v);
            } else {
                worst_low = worst_low.max(v);
            }
        }
        let residual = mna.dc_residual(&sol);
        Solved { value, gate_v, out_v, worst_high, worst_low, residual, state: sol }
    }
}

/// A source-stepping count that reliably solves a circuit of this many gates. Fewer than about forty
/// increments does not converge even on a single gate: the sharp switching characteristic is what makes
/// the system stiff, not the size, so this only ever scales up.
pub fn recommended_steps(gates: usize) -> usize {
    (gates / 2).clamp(40, 120)
}

/// Grow an `n`-bit adder, evaluate `x + y` on it electrically, and return
/// `(value, worst_high, worst_low, residual)`.
pub fn grow_and_add(n: usize, x: u32, y: u32, t: Tech) -> (u32, f64, f64, f64) {
    let g = grow_adder(n);
    let s = g.evaluate(x, y, t, 40);
    (s.value, s.worst_high, s.worst_low, s.residual)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One rule set grows every width, and the gate count scales exactly with it.
    #[test]
    fn growth_is_width_agnostic() {
        for &n in &[1usize, 2, 4, 8, 16] {
            let mut nl = Netlist::new();
            let a: Vec<usize> = (0..n).map(|_| nl.node()).collect();
            let b: Vec<usize> = (0..n).map(|_| nl.node()).collect();
            let (sum, _cout) = nl.adder(&a, &b, 0);
            assert_eq!(sum.len(), n, "grew the wrong number of sum bits");
            assert_eq!(nl.gates().len(), 15 * n, "a full adder is 15 NANDs, so width n must grow 15n");
        }
    }

    /// The grown adder is arithmetically correct when solved as an analog circuit: every input pair for
    /// 1 and 2 bits, checked through the transistors rather than through a logic table.
    #[test]
    fn grown_adder_computes_through_the_transistors() {
        let t = Tech::default();
        for &n in &[1usize, 2] {
            let m = 1u32 << n;
            for x in 0..m {
                for y in 0..m {
                    let (got, hi, lo, res) = grow_and_add(n, x, y, t);
                    assert_eq!(got, x + y, "{n}-bit grown adder: {x} + {y} gave {got}");
                    assert!(res < 1e-9, "solve did not converge: residual {res}");
                    assert!(!hi.is_finite() || hi > 0.9 * t.vdd, "logic high degraded to {hi} V");
                    assert!(!lo.is_finite() || lo < 0.25 * t.vdd, "logic low degraded to {lo} V");
                }
            }
        }
    }

    /// The logic-low the engine produces is the analytic resistor divider of a conducting gate, so the
    /// solve is reproducing the circuit rather than snapping to logic levels.
    #[test]
    fn logic_low_matches_the_analytic_divider() {
        let t = Tech::default();
        let (_v, _hi, lo, _r) = grow_and_add(2, 1, 1, t);
        let predicted = t.predicted_low();
        assert!((lo - predicted).abs() < 1e-3, "logic low {lo} V is not the divider value {predicted} V");
    }

    /// Source stepping is what makes a large grown circuit solvable; a cold start does not converge.
    #[test]
    fn source_stepping_is_what_makes_it_converge() {
        let t = Tech::default();
        let n = 4;
        let mut nl = Netlist::new();
        let a: Vec<usize> = (0..n).map(|_| nl.node()).collect();
        let b: Vec<usize> = (0..n).map(|_| nl.node()).collect();
        let _ = nl.adder(&a, &b, 0);
        let inputs: Vec<(usize, bool)> = a.iter().chain(b.iter()).map(|&node| (node, true)).collect();
        let mna = nl.lower(&inputs, t).build();
        let cold = mna.dc_residual(&mna.dc_nonlinear());
        let stepped = mna.dc_residual(&mna.dc_nonlinear_stepped(40));
        eprintln!("4-bit grown adder residual: cold start {cold:.2e}, source stepped {stepped:.2e}");
        assert!(stepped < 1e-9, "source stepping failed to converge: {stepped}");
        assert!(stepped < cold, "source stepping should beat a cold start ({stepped} vs {cold})");
    }
}
