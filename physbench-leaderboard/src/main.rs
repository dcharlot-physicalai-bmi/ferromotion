//! The scorer. Discovers every submission (via build.rs), scores each on the conservation invariants, ranks
//! them, and writes the persistent standings: `standings.json` (canonical, for the website and any consumer)
//! and `STANDINGS.md` (human-readable). CI runs this on every push; the committed standings are the board.
mod bench;
use bench::{score, Report};
include!(concat!(env!("OUT_DIR"), "/registry.rs")); // generated: fn all() -> Vec<(&Meta, Box<dyn Model>)>

fn main() {
    let mut reps: Vec<Report> = all().iter().map(|(meta, m)| score(meta, m.as_ref())).collect();
    // rank: passing first, then smallest energy drift
    reps.sort_by(|a, b| b.pass.cmp(&a.pass)
        .then(a.energy_drift.partial_cmp(&b.energy_drift).unwrap_or(std::cmp::Ordering::Equal)));
    for (i, r) in reps.iter_mut().enumerate() { r.rank = i + 1; }

    // standings.json — canonical machine-readable artifact
    let json = serde_json::to_string_pretty(&reps).unwrap();
    std::fs::write("standings.json", format!("{json}\n")).expect("write standings.json");

    // STANDINGS.md — human-readable board
    let pct = |x: f64| if x.is_finite() { format!("{:.2}%", x * 100.0) } else { "diverged".into() };
    let mut md = String::from("# Physics-Fidelity Benchmark — Standings\n\nSystem: frictionless pendulum (energy constant, flow reversible). Verdict = energy drift < 5% AND time-reversibility error < 5%. Scored by CI; do not edit by hand.\n\n| # | model | author | energy drift | reversibility | 1-step RMSE | verdict |\n|---|-------|--------|--------------|---------------|-------------|---------|\n");
    for r in &reps {
        md.push_str(&format!("| {} | {} ({}) | {} | {} | {} | {:.2e} | {} |\n",
            r.rank, r.name, r.kind, r.author, pct(r.energy_drift), pct(r.reversibility), r.one_step_rmse,
            if r.pass { "PASS" } else { "FAIL" }));
    }
    md.push_str("\nThe tell: a model can be accurate step to step and still violate the invariant over a rollout. Structure, not per-step accuracy, is what earns a PASS. Submit yours — see [README](./README.md).\n");
    std::fs::write("STANDINGS.md", &md).expect("write STANDINGS.md");

    // console echo
    println!("Scored {} submissions:\n", reps.len());
    for r in &reps {
        println!("  {:>2}. {:<28} {:<10}  energy {:>9}  rev {:>9}  {}",
            r.rank, format!("{} ({})", r.name, r.kind), r.author, pct(r.energy_drift), pct(r.reversibility),
            if r.pass { "PASS" } else { "FAIL" });
    }
    println!("\nWrote standings.json and STANDINGS.md.");
}
