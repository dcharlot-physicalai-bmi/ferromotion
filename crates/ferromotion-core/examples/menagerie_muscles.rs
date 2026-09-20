//! **MuJoCo's muscle curves, held against MuJoCo's own** — `mju_muscleGain`, `mju_muscleBias` and
//! `mju_muscleDynamics` on a grid chosen to land in every branch of all four piecewise curves, plus the
//! compiled parameters of every muscle in Menagerie.
//!
//! ```text
//! python scripts/mujoco_muscle_oracle.py <menagerie root> /tmp/muscle_oracle.txt
//! cargo run --release --example menagerie_muscles -- <menagerie root> /tmp/muscle_oracle.txt
//! ```
//!
//! ⛔ A muscle is the one actuator whose law is not `gain·input + bias` with constant coefficients, so it
//! is the one where a port can be exactly right in the middle of the operating range and wrong at both
//! ends — where a leg actually spends its stance. The grid runs the normalised length from below `lmin` to
//! past `lmax` and the velocity past both ends of the force-velocity curve, on purpose.

use ferromotion_core::mujoco_actuator::{muscle_bias, muscle_dynamics, muscle_gain, ActBias, ActDyn, ActGain};
use ferromotion_core::tree_from_mjcf;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_muscles <menagerie root> <muscle oracle txt>");
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s:?}"));
    let (mut curves, mut curves_ok, mut worst_g, mut worst_b) = (0usize, 0usize, 0.0f64, 0.0f64);
    let (mut dyns, mut dyns_ok, mut worst_d, mut worst_d_where) = (0usize, 0usize, 0.0f64, String::new());
    let (mut compiled, mut with_muscle) = (0usize, std::collections::BTreeSet::new());
    #[allow(clippy::type_complexity)]
    let mut by_model: BTreeMap<String, Vec<(String, [i32; 3], Vec<f64>)>> = BTreeMap::new();
    let mut worst_where = String::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "curve" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let (len, vel) = (v[0], v[1]);
                let lr = [v[2], v[3]];
                let acc0 = v[4];
                let mut prm = [0.0; 9];
                prm.copy_from_slice(&v[5..14]);
                let (g, b) = (v[14], v[15]);
                curves += 1;
                let (mg, mb) = (muscle_gain(len, vel, &lr, acc0, &prm), muscle_bias(len, &lr, acc0, &prm));
                let (eg, eb) = ((mg - g).abs() / g.abs().max(1.0), (mb - b).abs() / b.abs().max(1.0));
                if eg.max(eb) < 1e-12 {
                    curves_ok += 1;
                } else if worst_where.is_empty() || eg.max(eb) > worst_g.max(worst_b) {
                    worst_where = format!("len {len:.6} vel {vel:.6} lr {lr:?} acc0 {acc0:.3e} prm {prm:?}: gain {mg:.12e} vs {g:.12e}, bias {mb:.12e} vs {b:.12e}");
                }
                worst_g = worst_g.max(eg);
                worst_b = worst_b.max(eb);
            }
            "dyn" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let prm = [v[2], v[3], v[4]];
                dyns += 1;
                let mine = muscle_dynamics(v[0], v[1], &prm);
                let e = (mine - v[5]).abs() / v[5].abs().max(1.0);
                if e < 1e-12 {
                    dyns_ok += 1;
                } else if e > worst_d {
                    worst_d = e;
                    worst_d_where = format!("ctrl {} act {} prm {prm:?}: {mine:.12e} vs {:.12e}", v[0], v[1], v[5]);
                }
                worst_d = worst_d.max(e);
            }
            "compiled" => {
                compiled += 1;
                with_muscle.insert(t[1].to_string());
                let v: Vec<f64> = t[6..].iter().map(|x| f(x)).collect();
                by_model.entry(t[1].to_string()).or_default().push((t[2].to_string(), [t[3].parse::<i32>().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap()], v));
            }
            _ => {}
        }
    }
    println!("muscle curves sampled from MuJoCo: {curves}");
    println!("  gain AND bias within 1e-12 relative: {curves_ok}; worst gain {worst_g:.2e}, worst bias {worst_b:.2e}");
    if curves_ok != curves {
        println!("  {worst_where}");
    }
    println!("activation dynamics sampled: {dyns}");
    println!("  within 1e-12 relative: {dyns_ok}; worst {worst_d:.2e} {worst_d_where}");
    println!("compiled muscles in the corpus: {compiled} across {} models", with_muscle.len());

    // ⭐ the parameters MuJoCo's COMPILER produced, against this loader's: the curves above are only worth
    // as much as the numbers handed to them, and `lengthrange` and `acc0` are both resolved by the
    // compiler rather than stated in the file.
    let root = std::path::Path::new(&args[1]);
    let (mut acts, mut acts_ok, mut worst_p, mut worst_p_where) = (0usize, 0usize, 0.0f64, String::new());
    let mut refused: BTreeMap<String, usize> = BTreeMap::new();
    let (mut acc0_ok, mut acc0_tried) = (0usize, 0usize);
    let (mut acc0_bad, mut acc0_bad_trn) = (0usize, 0usize);
    let mut state0: BTreeMap<String, BTreeMap<String, (f64, f64)>> = BTreeMap::new();
    for (rel, want) in &by_model {
        let model = root.join(rel);
        let dir = model.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&model) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            *refused.entry(format!("{rel}: the loader refuses the model")).or_default() += 1;
            continue;
        };
        let mine: BTreeMap<&str, &ferromotion_core::Actuator> = t.actuators.iter().map(|a| (a.name.as_str(), a)).collect();
        let nv = t.tree.joints.len();
        let st0 = t.actuator_state(&t.reference_q, &vec![0.0; nv]);
        let e = state0.entry(rel.clone()).or_default();
        for (a, s) in t.actuators.iter().zip(&st0) {
            let n: f64 = s.moment.iter().map(|(_, m)| m * m).sum::<f64>().sqrt();
            e.insert(a.name.clone(), (s.length, n));
        }
        for (name, kinds, v) in want {
            acts += 1;
            let Some(a) = mine.get(name.as_str()) else {
                *refused.entry(format!("{rel}: not carried")).or_default() += 1;
                continue;
            };
            // mjGAIN_MUSCLE = 2, mjBIAS_MUSCLE = 2, mjDYN_MUSCLE = 4
            let kind_ok = (kinds[0] == 2) == (a.gain == ActGain::Muscle)
                && (kinds[1] == 2) == (a.bias == ActBias::Muscle)
                && (kinds[2] == 4) == (a.dynamics == ActDyn::Muscle);
            let mut e: f64 = 0.0;
            for (k, x) in v[..9].iter().enumerate() {
                e = e.max((a.gainprm[k] - x).abs() / x.abs().max(1.0));
            }
            for (k, x) in v[9..18].iter().enumerate() {
                e = e.max((a.biasprm[k] - x).abs() / x.abs().max(1.0));
            }
            for (k, x) in v[18..21].iter().enumerate() {
                e = e.max((a.dynprm[k] - x).abs() / x.abs().max(1.0));
            }
            for (k, x) in v[21..23].iter().enumerate() {
                e = e.max((a.lengthrange[k] - x).abs() / x.abs().max(1.0));
            }
            let e_acc0 = (a.acc0 - v[23]).abs() / v[23].abs().max(1.0);
            // the transmission at qpos0, which is what `acc0` is taken through
            let st = &state0[rel][name];
            let e_trn = ((st.0 - v[24]).abs() / v[24].abs().max(1.0)).max((st.1 - v[25]).abs() / v[25].abs().max(1.0));
            if e_acc0 >= 1e-9 {
                acc0_bad += 1;
                if e_trn >= 1e-9 {
                    acc0_bad_trn += 1;
                }
            }
            if kind_ok && e < 1e-12 && e_acc0 < 1e-9 {
                acts_ok += 1;
            }
            if kind_ok && e < 1e-12 {
                acc0_tried += 1;
                if e_acc0 < 1e-9 {
                    acc0_ok += 1;
                }
            }
            if !(kind_ok && e < 1e-12 && e_acc0 < 1e-9) && std::env::var("DUMP_ACC0").is_ok() {
                println!("    acc0 {rel} / {name}: {:.9} vs {:.9}", a.acc0, v[23]);
            }
            if !kind_ok || e.max(e_acc0) > worst_p {
                worst_p = e.max(e_acc0);
                worst_p_where = format!("{rel} / {name}: kinds ok {kind_ok}, prm off {e:.2e}, acc0 {:.9} vs {:.9}", a.acc0, v[23]);
            }
        }
    }
    println!("compiled muscle parameters compared: {acts}");
    println!("  gain/bias/dyn types, all 9+9+3 parameters, lengthrange AND acc0 all matching: {acts_ok}; worst {worst_p:.2e} on {worst_p_where}");
    println!("  of those with the types and parameters right, acc0 within 1e-9: {acc0_ok} of {acc0_tried}");
    println!("  ⛔ of the {acc0_bad} whose acc0 differs, {acc0_bad_trn} also have a TRANSMISSION that differs at qpos0 — `acc0` is `‖M⁻¹·moment‖`, so those are the tendon path, not the muscle");
    for (k, n) in &refused {
        println!("    {n:>5}  {k}");
    }
}
