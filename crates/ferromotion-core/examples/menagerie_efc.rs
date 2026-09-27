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
//! The comparison is the ANSWER, never the path: a disagreement with the `qacc` a model ships is reported
//! as a failure unless ours is the answer MuJoCo reaches when told to converge and, where no noslip pass
//! runs, our cost is no worse than MuJoCo's.
//!
//! ⛔⛔ and that verdict is only as good as OUR cost. The rows here are MuJoCo's to the last bit, but the
//! map from a row's residual to a cost is this port's own, and a row's LINEAR TAIL — the part that bites
//! once the row is driven past its bound — is exactly where two implementations could differ. So the
//! verdict is settled OUTSIDE this file:
//!
//! ```text
//! DUMP_QACC=/tmp/ours.txt cargo run --release --example menagerie_efc -- <efc oracle txt>
//! python scripts/mujoco_cost_at.py <menagerie root> <efc oracle txt> /tmp/ours.txt
//! ```
//!
//! which rebuilds each state and scores both answers with `mj_constraintUpdate`.
//!
//! ⭐ The verdict no longer rests on a cost. The oracle also records `qacc_conv`: the same state with
//! MuJoCo's main solve told to converge (`iterations` 1000, `tolerance` 0), noslip still running after it
//! where the model asks for one. The optimum of the rows is unique, so that is the answer a converged solver
//! owes, and it is MuJoCo's own. Measured against MuJoCo 3.13.0 on Menagerie: 503 of 504 states within 1e-9
//! of it, all 504 within 1e-6. Against `qacc` as each model ships, 476 of 504 agree within 1e-6, and every
//! one of the 28 that differ is MuJoCo stopping short under its own options:
//!
//! - 21 on files that cap `iterations`: 19 on the MJX variants that cap it at 1, one on
//!   `rainbow_robotics_rby1` at 30 and one on `tetheria_aero_hand_open` at 5;
//! - 4 on `i2rt_yam` and `toddlerbot_2xc`, where MuJoCo's DEFAULT `tolerance` of 1e-8 ends the main solve
//!   with `qacc` still 2e-5 to 6e-5 from the optimum;
//! - 3 on `hello_robot_stretch` and `hello_robot_stretch_3`, the same early stop, with noslip then run on
//!   the unconverged forces.
//!
//! ⛔⛔ CORRECTED 2026-09-27: this header used to say MuJoCo's solver "stalls short of the optimum of its own
//! rows" on `robot_soccer_kit`, `hello_robot_stretch`, `hello_robot_stretch_3`, `flybody`, `umi_gripper` and
//! `i2rt_yam`, "which all leave the default 100 in place". Five of the six set `noslip_iterations`, so the
//! `qacc` MuJoCo reports is the answer AFTER noslip, which is meant to leave the optimum of these rows; the
//! cost comparison could not see that, because it judged both answers by the main problem alone. With
//! `mujoco_noslip` applied, 36 of the 39 states on noslip models agree to 1e-6. The sixth, `i2rt_yam`, and the three that
//! remain are not a stall either: MuJoCo reaches our answer when told to converge.
//! (The 2026-09-24 correction still stands: `rainbow_robotics_rby1` and `tetheria_aero_hand_open` cap their
//! solver, so they never showed anything about an uncapped one.)

use ferromotion_core::{mujoco_constraint_update_blocks, mujoco_noslip, solve_constraints_newton_blocks, Cone, ConeContact, EfcBlock};
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
    /// the same state with MuJoCo's main solve told to converge
    qacc_conv: Vec<f64>,
    j: Vec<Vec<f64>>,
    m: Vec<Vec<f64>>,
    /// `(efc_address, condim, mu, friction)` for each contact that actually has rows
    contacts: Vec<(usize, usize, f64, [f64; 5])>,
    /// `(noslip_iterations, noslip_tolerance, meaninertia, iterations, tolerance)`
    opt: Option<(usize, f64, f64, usize, f64)>,
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
            "opt" => cases.last_mut().unwrap().opt = Some((t[1].parse().unwrap(), f(t[2]), f(t[3]), t[4].parse().unwrap(), f(t[5]))),
            "J" => cases.last_mut().unwrap().j.push(t[1..].iter().map(|x| f(x)).collect()),
            "M" => cases.last_mut().unwrap().m.push(t[1..].iter().map(|x| f(x)).collect()),
            "aref" | "D" | "R" | "floss" | "qacc_smooth" | "qacc" | "qacc_conv" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let c = cases.last_mut().unwrap();
                match t[0] {
                    "aref" => c.aref = v,
                    "D" => c.d = v,
                    "floss" => c.floss = v,
                    "qacc_smooth" => c.qacc_smooth = v,
                    "qacc" => c.qacc = v,
                    "qacc_conv" => c.qacc_conv = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let (mut seen, mut ok, mut better, mut worse) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = (0.0f64, String::new());
    // against MuJoCo CONVERGED: (compared, within 1e-9, within 1e-6, worst)
    let mut conv = (0usize, 0usize, 0usize, (0.0f64, String::new()));
    let mut worst_gap = (0.0f64, String::new());
    let dump_qacc = std::env::var("DUMP_QACC").ok();
    if let Some(p) = &dump_qacc {
        let _ = std::fs::remove_file(p);
    }
    let mut by_kind: BTreeMap<&'static str, (usize, usize)> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    // every state where the answers differ at a cost no worse than MuJoCo's, and what MuJoCo was told
    let mut short: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut noslip_states = 0usize;
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
        let solved = solve_constraints_newton_blocks(&m, &a_smooth, &jac, &c.aref, &c.d, &blocks, 1e-14, 300).and_then(|mut sol| {
            // ⭐ MuJoCo's noslip pass, where the model asks for one: the `qacc` MuJoCo reports is then
            // AFTER that pass, not the optimum of the rows, so the solve is compared after it too
            if let Some((iters, tol, meaninertia, ..)) = c.opt.filter(|o| o.0 > 0) {
                let mut force = sol.force.clone();
                sol.qacc = mujoco_noslip(&m, &a_smooth, &jac, &c.aref, &c.d, &blocks, &mut force, iters, tol, meaninertia)?;
                sol.force = force;
            }
            Ok(sol)
        });
        let noslip = c.opt.is_some_and(|o| o.0 > 0);
        noslip_states += usize::from(noslip);
        match solved {
            Ok(sol) => {
                // ⭐ our answer, written out so MuJoCo's OWN cost function can be evaluated at it. "Our
                // cost is no worse" is only as good as our cost: the rows are MuJoCo's, but the map from
                // a row's residual to a cost is this port's, and a friction row's linear tails are
                // exactly where the two could differ. `scripts/mujoco_cost_at.py` takes this file and
                // asks `mj_constraintUpdate` instead.
                if let Some(path) = &dump_qacc {
                    use std::io::Write;
                    let mut fh = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                    writeln!(fh, "ours\t{}\t{}\t{}", c.rel, c.state, sol.qacc.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join("\t")).unwrap();
                }
                let scale = want.amax().max(1.0);
                let err = (&sol.qacc - &want).amax() / scale;
                // ⭐ the claim: against MuJoCo told to converge, the optimum is unique, so the two must agree
                let err_conv = (!c.qacc_conv.is_empty()).then(|| (&sol.qacc - DVector::from_row_slice(&c.qacc_conv)).amax() / scale);
                if let Some(e) = err_conv {
                    conv.0 += 1;
                    conv.1 += usize::from(e < 1e-9);
                    conv.2 += usize::from(e < 1e-6);
                    // a NaN is the worst error, never a skipped one
                    if e.is_nan() || e > conv.3 .0 {
                        conv.3 = (e, format!("{} [{}]", c.rel, c.state));
                    }
                }
                if err > worst.0 {
                    worst = (err, format!("{} [{}]", c.rel, c.state));
                }
                if err < 1e-6 {
                    ok += 1;
                    by_kind.entry(kind).or_default().0 += 1;
                } else {
                    // The objective is STRICTLY convex, so its minimiser is unique: the Gauss term
                    // ½(a−a₀)ᵀM(a−a₀) has Hessian M, which is positive definite, and adding the convex
                    // row costs cannot make it less than strictly convex. Two different answers cannot
                    // both be optimal, so when ours costs strictly less, MuJoCo stopped short of the one
                    // optimum. The cost is the test because it is the one quantity both answers are
                    // judged by; the deficit is reported so a gap at rounding level reads as rounding.
                    //
                    // ⛔ CORRECTED 2026-09-24: this comment used to claim the objective was "convex but
                    // NOT strictly so" and that "the argmin is a SET". That was wrong. A row driven past
                    // its bound does contribute a linear piece, but the Gauss term is strictly convex on
                    // its own, and strict convexity survives the sum. It was written to explain why
                    // MuJoCo's answer could differ from ours at an equal cost, and the explanation was
                    // never needed: MuJoCo's cost is strictly HIGHER in every one of the 40 cases.
                    let cost_at = |a: &DVector<f64>| {
                        let jar: Vec<f64> = (&jac * a - DVector::from_row_slice(&c.aref)).iter().copied().collect();
                        let u = mujoco_constraint_update_blocks(&blocks, &c.d, &jar);
                        let da = a - &a_smooth;
                        0.5 * (da.transpose() * &m * &da)[(0, 0)] + u.cost
                    };
                    let (ours, theirs) = (cost_at(&sol.qacc), cost_at(&want));
                    // two witnesses that MuJoCo, not this port, is the one short of the optimum: ours
                    // is the answer MuJoCo itself reaches when told to converge, and (without noslip,
                    // whose answer is MEANT to leave the optimum of these rows) ours costs no more
                    if err_conv.is_some_and(|e| e < 1e-6) && (noslip || ours <= theirs) {
                        better += 1;
                        let gap = (theirs - ours) / theirs.abs().max(1.0);
                        let told = match c.opt {
                            Some((_, _, _, it, _)) if it < 100 => format!("its file caps iterations at {it}"),
                            Some((_, _, _, _, tol)) if tol > 1e-8 => format!("its file loosens tolerance to {tol:e}"),
                            Some((ns, _, _, it, tol)) => format!("MuJoCo's defaults (iterations {it}, tolerance {tol:e}) stop the main solve early{}", if ns > 0 { ", and noslip runs on those forces" } else { "" }),
                            None => "options not recorded".into(),
                        };
                        short.entry(told).or_default().push(format!("{} [{}] qacc off {err:.1e}{}", c.rel, c.state, if noslip { String::new() } else { format!(", cost deficit {gap:.1e}") }));
                        if !noslip && gap > worst_gap.0 {
                            worst_gap = (gap, format!("{} [{}]: qacc off {err:.2e}", c.rel, c.state));
                        }
                        by_kind.entry(kind).or_default().0 += 1;
                    } else {
                        worse += 1;
                        if notes.len() < 10 {
                            let conv = err_conv.map_or("not recorded".into(), |e| format!("{e:.2e}"));
                            notes.push(format!("{} [{}]: qacc off {err:.2e}{}, off MuJoCo-converged {conv}, cost ours {ours} vs {theirs}, {} rows ne {} nf {}", c.rel, c.state, if noslip { " after noslip" } else { "" }, c.nefc, c.ne, c.nf));
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
    println!("  against MuJoCo told to CONVERGE: {} of {} within 1e-9 relative, {} within 1e-6; worst {:.2e} on {}", conv.1, conv.0, conv.2, conv.3 .0, conv.3 .1);
    println!("  against MuJoCo as the model ships it: {ok} within 1e-6 relative");
    println!("  differ, because MuJoCo stops short (ours is MuJoCo's own converged answer): {better}; worst cost DEFICIT (relative) {:.2e} on {} — the minimiser is unique (the Gauss term is strictly convex), so a deficit above rounding means MuJoCo's solver stopped short of it", worst_gap.0, worst_gap.1);
    println!("  differ otherwise — ours is wrong: {worse}");
    println!("  worst relative qacc {:.2e} on {}", worst.0, worst.1);
    for (k, (good, n)) in &by_kind {
        println!("    {good:>4} of {n:<4} {k}");
    }
    println!("  {noslip_states} states are on models that ask for noslip, and are compared after it");
    println!("  where MuJoCo stopped short, by what it was told:");
    for (told, v) in &short {
        println!("    {:>4}  {told}", v.len());
        for x in v.iter().take(if told.starts_with("MuJoCo's defaults") { 20 } else { 3 }) {
            println!("            {x}");
        }
    }
    for n in notes.iter().take(10) {
        println!("  {n}");
    }
}
