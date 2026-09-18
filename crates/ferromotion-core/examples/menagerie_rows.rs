//! **The constraint rows this port ASSEMBLES, against the ones MuJoCo assembles** — `mj_makeConstraint` for
//! dof friction and joint limits.
//!
//! `examples/menagerie_efc` proved the solver by handing it MuJoCo's own rows. This is the other half: build
//! the rows ourselves from a loaded model and compare `efc_J`, `efc_aref` and `efc_D` against MuJoCo's. With
//! the solver already pinned, a disagreement here can only be the assembly.
//!
//! ```text
//! cargo run --release --example menagerie_rows -- <menagerie root> <efc oracle txt> [substring]
//! ```

use ferromotion_core::{tree_from_mjcf, EfcBlock, MjcfJointKind};
use std::collections::BTreeMap;

#[derive(Default)]
struct Case {
    rel: String,
    nv: usize,
    ne: usize,
    nf: usize,
    types: Vec<String>,
    joints: Vec<(String, String, usize)>,
    qpos: Vec<f64>,
    qvel: Vec<f64>,
    aref: Vec<f64>,
    d: Vec<f64>,
    floss: Vec<f64>,
    j: Vec<Vec<f64>>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_rows <menagerie root> <efc oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let text = std::fs::read_to_string(&args[2]).unwrap();
    let f = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s:?}"));
    let mut cases: Vec<Case> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "case" => cases.push(Case {
                rel: t[1].to_string(),
                nv: t[3].parse().unwrap(),
                ne: t[5].parse().unwrap(),
                nf: t[6].parse().unwrap(),
                ..Default::default()
            }),
            "type" => cases.last_mut().unwrap().types = t[1..].iter().map(|s| s.to_string()).collect(),
            "joint" => cases.last_mut().unwrap().joints.push((t[1].to_string(), t[2].to_string(), t[3].parse().unwrap())),
            "J" => cases.last_mut().unwrap().j.push(t[1..].iter().map(|x| f(x)).collect()),
            "qpos" | "qvel" | "aref" | "D" | "floss" => {
                let v: Vec<f64> = t[1..].iter().map(|x| f(x)).collect();
                let c = cases.last_mut().unwrap();
                match t[0] {
                    "qpos" => c.qpos = v,
                    "qvel" => c.qvel = v,
                    "aref" => c.aref = v,
                    "D" => c.d = v,
                    "floss" => c.floss = v,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    let (mut seen, mut ok) = (0usize, 0usize);
    let (mut worst_j, mut worst_aref, mut worst_d) = (0.0f64, 0.0f64, 0.0f64);
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut notes: Vec<String> = Vec::new();
    let mut cache: BTreeMap<String, Option<ferromotion_core::MjcfTree>> = BTreeMap::new();
    for c in &cases {
        if filter.as_ref().is_some_and(|fl| !c.rel.contains(fl.as_str())) {
            continue;
        }
        // tendon rows are a transmission this port does not carry, so a case with any is out of scope
        if c.types.iter().any(|t| t.ends_with("_tendon")) {
            *counts.entry("has tendon rows").or_default() += 1;
            continue;
        }
        let t = cache.entry(c.rel.clone()).or_insert_with(|| {
            let model = root.join(&c.rel);
            let dir = model.parent().unwrap().to_path_buf();
            let xml = std::fs::read_to_string(&model).ok()?;
            tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()).ok()
        });
        let Some(t) = t.as_ref() else {
            *counts.entry("the loader refuses the model").or_default() += 1;
            continue;
        };
        let lined_up = t.joints.len() == c.joints.len()
            && t.joints.iter().zip(&c.joints).all(|(a, b)| matches!((a.kind, b.1.as_str()), (MjcfJointKind::Hinge, "hinge") | (MjcfJointKind::Slide, "slide")));
        if !lined_up {
            *counts.entry("a free or ball joint: the dof columns do not correspond").or_default() += 1;
            continue;
        }
        let qposadr: Vec<usize> = c.joints.iter().map(|j| j.2).collect();
        let Ok(q) = t.q_from_qpos(&c.qpos, &qposadr) else {
            *counts.entry("qpos does not map onto our coordinates").or_default() += 1;
            continue;
        };
        let iw = t.dof_invweight0();
        if std::env::var("DUMP_IW").is_ok() {
            println!("IW {}: {:?}", c.rel, iw);
        }
        if !t.equalities_unsupported.is_empty() {
            *counts.entry("an equality this port does not carry (connect, weld, tendon)").or_default() += 1;
            continue;
        }
        let mine = t.joint_constraint_rows(&q, &c.qvel, &iw);
        // MuJoCo's order: equality, friction, limits, contacts — and this port builds the first three
        let nl = c.types.iter().filter(|x| x.starts_with("limit")).count();
        let theirs: Vec<usize> = (0..c.ne + c.nf + nl).collect();
        seen += 1;
        if theirs.len() != mine.blocks.len() {
            *counts.entry("we and MuJoCo disagree on how many joint rows there are").or_default() += 1;
            if notes.len() < 8 {
                let want: Vec<&str> = theirs.iter().map(|&i| c.types[i].as_str()).collect();
                notes.push(format!("{}: {} rows, MuJoCo has {} {:?}", c.rel, mine.blocks.len(), theirs.len(), want));
            }
            continue;
        }
        let (mut dj, mut da, mut dd) = (0.0f64, 0.0f64, 0.0f64);
        for (k, &row) in theirs.iter().enumerate() {
            for col in 0..c.nv {
                dj = dj.max((mine.jac[(k, col)] - c.j[row][col]).abs());
            }
            da = da.max((mine.aref[k] - c.aref[row]).abs() / c.aref[row].abs().max(1.0));
            dd = dd.max((mine.d[k] - c.d[row]).abs() / c.d[row].abs().max(1.0));
            // and the block must agree with MuJoCo's own type for that row
            let want = c.types[row].as_str();
            let got = match mine.blocks[k] {
                EfcBlock::Equality { .. } => "equality",
                EfcBlock::Friction { .. } => "friction_dof",
                EfcBlock::Limit => "limit_joint",
                EfcBlock::Contact(_) => "contact",
            };
            if want != got && notes.len() < 8 {
                notes.push(format!("{}: row {k} is {got} here and {want} in MuJoCo", c.rel));
            }
            if let EfcBlock::Friction { loss } = mine.blocks[k]
                && (loss - c.floss[row]).abs() > 1e-12
                && notes.len() < 8
            {
                notes.push(format!("{}: frictionloss {loss} vs {}", c.rel, c.floss[row]));
            }
        }
        worst_j = worst_j.max(dj);
        worst_aref = worst_aref.max(da);
        worst_d = worst_d.max(dd);
        if dj < 1e-12 && da < 1e-9 && dd < 1e-9 {
            ok += 1;
        } else if notes.len() < 8 {
            notes.push(format!("{}: J off {dj:.2e}, aref off {da:.2e}, D off {dd:.2e} over {} rows", c.rel, theirs.len()));
            for (k, &row) in theirs.iter().enumerate().take(3) {
                notes.push(format!("      row {k} ({}): aref {} vs {}, D {} vs {}", c.types[row], mine.aref[k], c.aref[row], mine.d[k], c.d[row]));
            }
        }
    }
    println!("states whose joint rows were assembled and compared: {seen}");
    println!("  every row matching MuJoCo's: {ok}");
    println!("  worst J {worst_j:.2e}, worst aref (relative) {worst_aref:.2e}, worst D (relative) {worst_d:.2e}");
    if !counts.is_empty() {
        println!("  not compared:");
        let mut v: Vec<_> = counts.iter().collect();
        v.sort_by_key(|(_, n)| std::cmp::Reverse(**n));
        for (k, n) in v {
            println!("    {n:>5}  {k}");
        }
    }
    for n in &notes {
        println!("  {n}");
    }
}
