//! **The constraint solve against MuJoCo, on MuJoCo's own rows** — every row type, not only contacts.
//!
//! The rows come from the oracle (`scripts/mujoco_efc_oracle.py`): `efc_J`, `efc_aref`, `efc_D`,
//! `efc_frictionloss` and the mass matrix, exactly as MuJoCo built them. That isolates the SOLVER from the
//! assembly that produces those rows, which is the same separation that made the contact work tractable —
//! when this disagrees it is the solve, and nothing else.
//!
//! ⛔ A row's law comes from its POSITION: `mj_constraintUpdate_impl` tests `i < ne` and `i < ne + nf`, so
//! the row order — equality, friction, then limits and contacts — is part of the problem statement.
//!
//! ```text
//! cargo run --release --example menagerie_efc -- <efc oracle txt> [substring]
//! ```
//!
//! The objective is strictly convex, so its minimiser is unique and the comparison is the ANSWER, never the
//! path: a disagreement is reported as a failure only if OUR cost is also worse than MuJoCo's.

use ferromotion_core::{mujoco_constraint_update_blocks, solve_constraints_newton_blocks, Cone, ConeContact, EfcBlock};
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    state: usize,
    nv: usize,
    nefc: usize,
    ne: usize,
    nf: usize,
    elliptic: bool,
    types: Vec<String>,
    aref: Vec<f64>,
    d: Vec<f64>,
    floss: Vec<f64>,
    qacc_smooth: Vec<f64>,
    qacc: Vec<f64>,
    j: Vec<Vec<f64>>,
    m: Vec<Vec<f64>>,
    /// `(efc_address, condim, mu, friction)` for each contact that actually has rows
    contacts: Vec<(usize, usize, f64, [f64; 5])>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: menagerie_efc <efc oracle txt> [substring]");
        std::process::exit(2);
    }
    let filter = args.get(2).cloned();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s:?}"));
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "case" => cases.push(Case {
                rel: t[1].to_string(),
                state: t[2].parse().unwrap(),
                nv: t[3].parse().unwrap(),
                nefc: t[4].parse().unwrap(),
                ne: t[5].parse().unwrap(),
                nf: t[6].parse().unwrap(),
                elliptic: t[7] == "1",
                ..Default::default()
            }),
            "type" => cases.last_mut().unwrap().types = t[1..].iter().map(|s| s.to_string()).collect(),
            "contact" => {
                let fr = [f(t[4]), f(t[5]), f(t[6]), f(t[7]), f(t[8])];
                cases.last_mut().unwrap().contacts.push((t[1].parse().unwrap(), t[2].parse().unwrap(), f(t[3]), fr));
            }
            "J" => cases.last_mut().unwrap().j.push(t[1..].iter().map(|x| f(x)).collect()),
            "M" => cases.last_mut().unwrap().m.push(t[1..].iter().map(|x| f(x)).collect()),
            "aref" | "D" | "R" | "floss" | "qacc_smooth" | "qacc" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let c = cases.last_mut().unwrap();
                match t[0] {
                    "aref" => c.aref = v,
                    "D" => c.d = v,
                    "floss" => c.floss = v,
                    "qacc_smooth" => c.qacc_smooth = v,
                    "qacc" => c.qacc = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let (mut seen, mut ok, mut better, mut worse) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = (0.0f64, String::new());
    let mut by_kind: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        // the blocks, in MuJoCo's order: ne equality rows, nf friction rows, then limits and contacts
        let mut blocks: Vec<EfcBlock> = Vec::new();
        if c.ne > 0 {
            blocks.push(EfcBlock::Equality { rows: c.ne });
        }
        for i in c.ne..c.ne + c.nf {
            blocks.push(EfcBlock::Friction { loss: c.floss[i] });
        }
        let by_addr: BTreeMap<usize, (usize, f64, [f64; 5])> = c.contacts.iter().map(|(a, d, mu, fr)| (*a, (*d, *mu, *fr))).collect();
        let mut i = c.ne + c.nf;
        let mut malformed = None;
        while i < c.nefc {
            match c.types[i].as_str() {
                "limit_joint" | "limit_tendon" => {
                    blocks.push(EfcBlock::Limit);
                    i += 1;
                }
                ty if ty.starts_with("contact") => {
                    let Some(&(condim, mu, friction)) = by_addr.get(&i) else {
                        malformed = Some(format!("no contact at efc row {i}"));
                        break;
                    };
                    let cone = if ty == "contact_elliptic" { Cone::Elliptic } else { Cone::Pyramidal };
                    let cc = ConeContact { cone, condim, mu, friction };
                    i += EfcBlock::Contact(cc).rows();
                    blocks.push(EfcBlock::Contact(cc));
                }
                other => {
                    malformed = Some(format!("row {i} is '{other}'"));
                    break;
                }
            }
        }
        if let Some(why) = malformed {
            notes.push(format!("{} [{}]: {why}", c.rel, c.state));
            continue;
        }
        if i != c.nefc {
            notes.push(format!("{} [{}]: blocks cover {i} rows, MuJoCo has {}", c.rel, c.state, c.nefc));
            continue;
        }
        seen += 1;
        let kind = if c.ne > 0 {
            "with equality rows"
        } else if c.nf > 0 {
            "with dof friction"
        } else if c.types.iter().any(|t| t.starts_with("limit")) {
            "with limits"
        } else if c.elliptic {
            "contacts only, elliptic"
        } else {
            "contacts only, pyramidal"
        };
        by_kind.entry(kind).or_default().1 += 1;

        let m = DMatrix::from_fn(c.nv, c.nv, |r, k| c.m[r][k]);
        let jac = DMatrix::from_fn(c.nefc, c.nv, |r, k| c.j[r][k]);
        let a_smooth = DVector::from_row_slice(&c.qacc_smooth);
        let want = DVector::from_row_slice(&c.qacc);
        match solve_constraints_newton_blocks(&m, &a_smooth, &jac, &c.aref, &c.d, &blocks, 1e-14, 300) {
            Ok(sol) => {
                let scale = want.amax().max(1.0);
                let err = (&sol.qacc - &want).amax() / scale;
                if err > worst.0 {
                    worst = (err, format!("{} [{}]", c.rel, c.state));
                }
                if err < 1e-6 {
                    ok += 1;
                    by_kind.entry(kind).or_default().0 += 1;
                } else {
                    // the objective is strictly convex, so the minimiser is unique: if we are FURTHER down
                    // the same cost, MuJoCo's solver stopped first and the disagreement is not ours
                    let cost_at = |a: &DVector<f64>| {
                        let jar: Vec<f64> = (&jac * a - DVector::from_row_slice(&c.aref)).iter().copied().collect();
                        let u = mujoco_constraint_update_blocks(&blocks, &c.d, &jar);
                        let da = a - &a_smooth;
                        0.5 * (da.transpose() * &m * &da)[(0, 0)] + u.cost
                    };
                    let (ours, theirs) = (cost_at(&sol.qacc), cost_at(&want));
                    if ours <= theirs {
                        better += 1;
                        by_kind.entry(kind).or_default().0 += 1;
                    } else {
                        worse += 1;
                        if notes.len() < 10 {
                            notes.push(format!("{} [{}]: qacc off {err:.2e} at a HIGHER cost ({ours} vs {theirs}), {} rows ne {} nf {}", c.rel, c.state, c.nefc, c.ne, c.nf));
                        }
                    }
                }
            }
            Err(e) => {
                worse += 1;
                if notes.len() < 10 {
                    notes.push(format!("{} [{}]: solve refused: {e}", c.rel, c.state));
                }
            }
        }
    }
    println!("constraint sets solved from MuJoCo's own rows: {seen}");
    println!("  reach MuJoCo's qacc within 1e-6 relative: {ok}");
    println!("  differ, but at a cost no worse than MuJoCo's (its solver stopped first): {better}");
    println!("  differ AND cost more — ours is wrong: {worse}");
    println!("  worst relative qacc {:.2e} on {}", worst.0, worst.1);
    for (k, (good, n)) in &by_kind {
        println!("    {good:>4} of {n:<4} {k}");
    }
    for n in notes.iter().take(10) {
        println!("  {n}");
    }
}
