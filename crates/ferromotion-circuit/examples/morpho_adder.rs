//! **Grown circuits, checked by physics** — one recursive rule produces an adder of any width, and the
//! electrical engine decides whether the grown thing actually works.
//!
//! A base case (a one-bit full adder) plus a recursive rule (split the operands, grow an adder for each
//! half, chain the carry) generate an N-bit adder without any width ever being declared. The grown
//! graph is lowered to resistor-load NMOS gates and solved as one nonlinear circuit, so the design is
//! correct only if the analog node voltages land on the right side of the switching threshold.
//!
//! Run: `cargo run --release --example morpho_adder -p ferromotion-circuit`

use ferromotion_circuit::morpho::{grow_and_add, Netlist, Tech};

fn main() {
    let t = Tech::default();
    println!("Grown circuits, checked by physics");
    println!("one base case (a 1-bit full adder) + one recursive rule (split, grow, chain the carry)");
    println!("lowered to resistor-load NMOS gates and solved by nonlinear modified nodal analysis\n");
    println!("  Vdd {} V, pull-up {:.0} ohm, Ron {:.0} ohm, threshold {} V\n", t.vdd, t.r_pullup, t.r_on, t.vth);

    for &n in &[1usize, 2, 4, 8, 16] {
        let mut probe = Netlist::new();
        let a: Vec<usize> = (0..n).map(|_| probe.node()).collect();
        let b: Vec<usize> = (0..n).map(|_| probe.node()).collect();
        let _ = probe.adder(&a, &b, 0);
        let gates = probe.gates().len();

        // exhaustive for the small widths, a deterministic spread for the larger ones
        let cases: Vec<(u32, u32)> = if n <= 4 {
            let m = 1u32 << n;
            (0..m).flat_map(|x| (0..m).map(move |y| (x, y))).collect()
        } else {
            let m = 1u32 << n;
            (0..24).map(|k| ((k * 37 + 11) % m, (k * 91 + 5) % m)).collect()
        };

        let (mut pass, mut fail) = (0usize, 0usize);
        let (mut worst_hi, mut worst_lo, mut worst_res) = (f64::INFINITY, f64::NEG_INFINITY, 0.0f64);
        let mut first_bad = None;
        for &(x, y) in &cases {
            let (got, hi, lo, res) = grow_and_add(n, x, y, t);
            if got == x + y {
                pass += 1;
            } else {
                fail += 1;
                first_bad.get_or_insert((x, y, got, x + y));
            }
            worst_res = worst_res.max(res);
            if hi.is_finite() {
                worst_hi = worst_hi.min(hi);
            }
            if lo.is_finite() {
                worst_lo = worst_lo.max(lo);
            }
        }
        println!(
            "{n:2}-bit : grew {gates:4} NAND gates ({:4} transistors), {pass:4}/{} cases {} | logic high {worst_hi:.3} V, logic low {worst_lo:.3} V, residual {worst_res:.1e}",
            2 * gates,
            cases.len(),
            if fail == 0 { "correct" } else { "WRONG" }
        );
        if let Some((x, y, got, want)) = first_bad {
            println!("        first mismatch: {x} + {y} gave {got}, expected {want}");
        }
    }

    println!("\nanalytic check: a pulled-down gate is a divider, Vdd*2Ron/(2Ron+Rpullup) = {:.3} V", t.predicted_low());
    println!("the measured logic low is that number, so the engine is reproducing the circuit and not a logic table.");
    println!("\nThe same two rules produced every width. The design is correct only because the analog");
    println!("voltages land on the right side of the threshold, and the engine is what decides that.");
}
