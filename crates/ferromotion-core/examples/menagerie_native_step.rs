//! **MuJoCo's own trajectory, step by step, to the bit** — [`ferromotion_core::MjcfTree::mujoco_native_step`]
//! from the same start as MuJoCo's `mj_step`, compared after EVERY step: `qpos`, `qvel`, `act` and
//! `qacc_warmstart` (the solver's starting point, which MuJoCo carries from step to step).
//!
//! ```text
//! python scripts/mujoco_native_step_oracle.py <menagerie root> <trajectory txt> [steps]
//! cargo run --release --example menagerie_native_step -- <menagerie root> <trajectory txt> [substring]
//! ```
//!
//! A trajectory that agrees to the bit cannot part later for any reason but a bug: this is the comparison
//! that `examples/menagerie_step.rs` could not make, because a state one ulp off lands elsewhere wherever
//! MuJoCo's step is not continuous.

use ferromotion_core::{tree_from_mjcf, MjNativeState};
use std::collections::BTreeMap;

fn hex(s: &str) -> f64 {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let s = s.strip_prefix("0x").expect("a hex float");
    let (mant, exp) = s.split_once('p').expect("an exponent");
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits = frac.to_string();
    while digits.len() < 13 {
        digits.push('0');
    }
    let fbits = u64::from_str_radix(&digits, 16).unwrap();
    let exp: i64 = exp.parse().unwrap();
    let bits = match (int, fbits) {
        ("0", 0) => 0,
        ("0", f) => f,
        _ => (((exp + 1023) as u64) << 52) | fbits,
    };
    let v = f64::from_bits(bits);
    if neg { -v } else { v }
}

#[derive(Default)]
struct Case {
    rel: String,
    start: MjNativeState,
    ctrl: Vec<f64>,
    steps: Vec<MjNativeState>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_native_step <menagerie root> <trajectory txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let mut cases: Vec<Case> = Vec::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        let vals = || t.get(1).map_or(Vec::new(), |s| s.split_whitespace().map(hex).collect::<Vec<f64>>());
        let c = cases.last_mut();
        match t[0] {
            "model" => cases.push(Case { rel: t[1].to_string(), ..Default::default() }),
            "s" => c.unwrap().steps.push(MjNativeState::default()),
            key => {
                let c = c.unwrap();
                let st = if c.steps.is_empty() { &mut c.start } else { c.steps.last_mut().unwrap() };
                match key {
                    "qpos" => st.qpos = vals(),
                    "qvel" => st.qvel = vals(),
                    "act" => st.act = vals(),
                    "warm" => st.qacc_warmstart = vals(),
                    "ctrl" => c.ctrl = vals(),
                    _ => {}
                }
            }
        }
    }
    let (mut exact, mut compared, mut refused_load) = (0usize, 0usize, 0usize);
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    let mut first_miss: Vec<String> = Vec::new();
    for c in &cases {
        if filter.as_ref().is_some_and(|f| !c.rel.contains(f.as_str())) {
            continue;
        }
        let path = root.join(&c.rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            refused_load += 1;
            continue;
        };
        let mut st = c.start.clone();
        let mut ctrl = c.ctrl.clone();
        st.qacc_warmstart = vec![0.0; st.qvel.len()];
        let mut ok = true;
        let mut counted = false;
        for (k, want) in c.steps.iter().enumerate() {
            let next = match t.mujoco_native_step(&st, &ctrl) {
                Ok(n) => n,
                Err(why) => {
                    *refused.entry(why).or_default() += 1;
                    ok = false;
                    break;
                }
            };
            if !counted {
                compared += 1;
                counted = true;
            }
            let same = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            if !(same(&next.qpos, &want.qpos) && same(&next.qvel, &want.qvel) && same(&next.act, &want.act) && same(&next.qacc_warmstart, &want.qacc_warmstart)) {
                let off = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
                if first_miss.len() < 25 {
                    first_miss.push(format!(
                        "{}: step {}: qpos off {:.2e}, qvel {:.2e}, qacc_warmstart {:.2e}",
                        c.rel,
                        k + 1,
                        off(&next.qpos, &want.qpos),
                        off(&next.qvel, &want.qvel),
                        off(&next.qacc_warmstart, &want.qacc_warmstart)
                    ));
                }
                ok = false;
                break;
            }
            // `mj_resetData` zeroed the control: the rest of MuJoCo's run has none
            if next.reset {
                ctrl.iter_mut().for_each(|u| *u = 0.0);
            }
            // carry MuJoCo's own state forward, so a miss is charged to the step that made it
            st = want.clone();
        }
        if ok && counted {
            exact += 1;
        }
    }
    let nsteps = cases.first().map_or(0, |c| c.steps.len());
    println!("mj_step against MuJoCo 3.13.0, BIT FOR BIT at every one of {nsteps} steps: {exact} of {compared} models stepped ({refused_load} the loader refuses)");
    for (why, n) in &refused {
        println!("  not stepped: {n} models: {why}");
    }
    for m in &first_miss {
        println!("  {m}");
    }
}
