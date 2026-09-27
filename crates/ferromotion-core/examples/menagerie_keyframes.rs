//! **`MjcfTree::keyframes` against MuJoCo's compiled `key_*`, on every Menagerie model that declares one.**
//!
//! ```text
//! python scripts/mujoco_keyframe_oracle.py <menagerie root> <keyframe oracle txt>
//! cargo run --release --example menagerie_keyframes -- <menagerie root> <keyframe oracle txt>
//! ```

use ferromotion_core::tree_from_mjcf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_keyframes <menagerie root> <keyframe oracle txt>");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    // (model, [(name, time, [qpos, qvel, act, ctrl])])
    type Key = (String, f64, [Vec<f64>; 4]);
    let mut models: Vec<(String, Vec<Key>)> = Vec::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        let v = || t[1..].iter().map(|x| x.parse::<f64>().unwrap()).collect::<Vec<f64>>();
        match t[0] {
            "model" => models.push((t[1].to_string(), Vec::new())),
            "key" => models.last_mut().unwrap().1.push((t[1].to_string(), t[2].parse().unwrap(), Default::default())),
            "qpos" | "qvel" | "act" | "ctrl" => {
                let i = ["qpos", "qvel", "act", "ctrl"].iter().position(|x| *x == t[0]).unwrap();
                models.last_mut().unwrap().1.last_mut().unwrap().2[i] = v();
            }
            _ => {}
        }
    }
    let (mut keys, mut ok, mut refused) = (0usize, 0usize, 0usize);
    let mut notes = Vec::new();
    for (rel, want) in &models {
        let path = root.join(rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(e) => {
                refused += 1;
                notes.push(format!("{rel}: loader refused: {e}"));
                continue;
            }
        };
        if t.keyframes.len() != want.len() {
            notes.push(format!("{rel}: {} keyframes here, {} in MuJoCo", t.keyframes.len(), want.len()));
            continue;
        }
        for (k, (name, time, w)) in t.keyframes.iter().zip(want) {
            keys += 1;
            let got = [&k.qpos, &k.qvel, &k.act, &k.ctrl];
            let same = k.name == *name && k.time == *time && got.iter().zip(w).all(|(a, b)| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-12 * y.abs().max(1.0)));
            if same {
                ok += 1;
            } else if notes.len() < 20 {
                let fields = ["qpos", "qvel", "act", "ctrl"];
                let mut why = Vec::new();
                if k.name != *name || k.time != *time {
                    why.push(format!("name/time '{}' {} vs '{name}' {time}", k.name, k.time));
                }
                for (f, (a, b)) in fields.iter().zip(got.iter().zip(w)) {
                    if a.len() != b.len() {
                        why.push(format!("{f} has {} values here, {} in MuJoCo", a.len(), b.len()));
                    } else if let Some(i) = a.iter().zip(b.iter()).position(|(x, y)| (x - y).abs() > 1e-12 * y.abs().max(1.0)) {
                        why.push(format!("{f}[{i}] {} vs MuJoCo {}", a[i], b[i]));
                    }
                }
                notes.push(format!("{rel} key '{name}': {}", why.join("; ")));
            }
        }
    }
    println!("keyframes against MuJoCo 3.13.0: {ok} of {keys} identical (name, time, qpos, qvel, act, ctrl); {refused} models the loader refuses");
    for n in &notes {
        println!("  {n}");
    }
}
