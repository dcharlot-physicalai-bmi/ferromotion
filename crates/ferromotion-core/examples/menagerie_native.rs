//! **MuJoCo's own arrays, to the bit** — [`ferromotion_core::MjcfTree::mujoco_native_model`] and
//! [`ferromotion_core::MjcfTree::mujoco_native_forward`] against `mjModel` and `mjData` after `mj_forward`,
//! array by array, on every Menagerie model at `qpos0` and at random states.
//!
//! ```text
//! python scripts/mujoco_native_oracle.py <menagerie root> <native oracle txt> [states per model]
//! cargo run --release --example menagerie_native -- <menagerie root> <native oracle txt> [substring]
//! ```
//!
//! Each array is counted once per model (`mjModel`) or once per model-state (`mjData`), and passes only
//! if EVERY element has MuJoCo's bits. The first differences are listed, so the stage that breaks first is
//! the one to read.

use ferromotion_core::tree_from_mjcf;
use std::collections::BTreeMap;

/// An integer, or Python's `float.hex`, to the bit.
fn num(s: &str) -> f64 {
    if !s.contains("0x") {
        return s.parse::<i64>().expect("an integer") as f64;
    }
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
struct State {
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    ctrl: Vec<f64>,
    act: Vec<f64>,
    arrays: Vec<(String, Vec<f64>)>,
}

#[derive(Default)]
struct Model {
    rel: String,
    arrays: Vec<(String, Vec<f64>)>,
    states: Vec<State>,
}

/// (compared, bit-identical, worst |difference|, where)
type Tally = (usize, usize, f64, String);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_native <menagerie root> <native oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let mut models: Vec<Model> = Vec::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        let vals = |i: usize| t.get(i).map_or(Vec::new(), |s| s.split_whitespace().map(num).collect());
        match t[0] {
            "model" => models.push(Model { rel: t[1].to_string(), ..Default::default() }),
            "m" => models.last_mut().unwrap().arrays.push((t[1].to_string(), vals(2))),
            "qpos" => models.last_mut().unwrap().states.push(State { qpos: vals(1), ..Default::default() }),
            "qvel" => models.last_mut().unwrap().states.last_mut().unwrap().qvel = vals(1),
            "ctrl" => models.last_mut().unwrap().states.last_mut().unwrap().ctrl = vals(1),
            "act" => models.last_mut().unwrap().states.last_mut().unwrap().act = vals(1),
            "d" => models.last_mut().unwrap().states.last_mut().unwrap().arrays.push((t[1].to_string(), vals(2))),
            _ => {}
        }
    }
    let mut tally: BTreeMap<String, Tally> = BTreeMap::new();
    let mut notes: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let (mut nmodels, mut refused) = (0usize, 0usize);
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();
    let mut compare = |kind: &str, name: &str, rel: &str, ours: &[f64], want: &[f64]| {
        let key = format!("{kind} {name}");
        let e = tally.entry(key.clone()).or_insert((0, 0, 0.0, String::new()));
        e.0 += 1;
        let same = ours.len() == want.len() && ours.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits());
        if same {
            e.1 += 1;
            return;
        }
        let worst = if ours.len() == want.len() { ours.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) } else { f64::INFINITY };
        if worst >= e.2 {
            e.2 = worst;
            e.3 = rel.to_string();
        }
        let n = notes.entry(key).or_default();
        if n.len() < 3 {
            let k = ours.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits());
            n.push(format!("{rel}: {} vs {} values; first at {k:?}: ours {:?} MuJoCo {:?}", ours.len(), want.len(), k.and_then(|k| ours.get(k)), k.and_then(|k| want.get(k))));
        }
    };
    for m in &models {
        if filter.as_ref().is_some_and(|f| !m.rel.contains(f.as_str())) {
            continue;
        }
        let path = root.join(&m.rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            refused += 1;
            continue;
        };
        // a model with anything the native pipeline does not compute yet is counted by what, not compared
        let missing = t.mujoco_native_unsupported();
        if !missing.is_empty() {
            for why in missing {
                *unsupported.entry(why).or_default() += 1;
            }
            continue;
        }
        nmodels += 1;
        let ours: BTreeMap<&str, Vec<f64>> = t.mujoco_native_model().into_iter().collect();
        for (name, want) in &m.arrays {
            if let Some(o) = ours.get(name.as_str()) {
                compare("mjModel", name, &m.rel, o, want);
            }
        }
        for (s, st) in m.states.iter().enumerate() {
            let ours: BTreeMap<&str, Vec<f64>> = t.mujoco_native_forward(&st.qpos, &st.qvel, &st.ctrl, &st.act).into_iter().collect();
            for (name, want) in &st.arrays {
                if let Some(o) = ours.get(name.as_str()) {
                    compare("mjData ", name, &format!("{} state {s}", m.rel), o, want);
                }
            }
        }
    }
    println!("MuJoCo 3.13.0's own arrays, BIT FOR BIT: {nmodels} models compared ({refused} refused)");
    for (why, n) in &unsupported {
        println!("  not compared: {n} models with {why}");
    }
    for (k, (n, same, worst, at)) in &tally {
        println!("  {k:<26} {same:>5} of {n:<5}{}", if same < n { format!(" worst {worst:.2e} on {at}") } else { String::new() });
    }
    for (k, ns) in &notes {
        for n in ns {
            println!("  {k}: {n}");
        }
    }
}
