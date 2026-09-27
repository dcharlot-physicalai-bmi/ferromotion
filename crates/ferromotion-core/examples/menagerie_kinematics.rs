//! **MuJoCo's forward kinematics, bit for bit** — [`ferromotion_core::MjcfTree::geom_frames_mujoco`] against
//! MuJoCo's own `geom_xpos`/`geom_xmat` after `mj_kinematics`, at `qpos0` and at random states that move every
//! joint (unnormalised quaternions included, which MuJoCo normalises).
//!
//! ```text
//! python scripts/mujoco_kinematics_oracle.py <menagerie root> <kinematics oracle txt> [states per model]
//! cargo run --release --example menagerie_kinematics -- <menagerie root> <kinematics oracle txt>
//! ```

use ferromotion_core::tree_from_mjcf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_kinematics <menagerie root> <kinematics oracle txt>");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let text = std::fs::read_to_string(&args[2]).unwrap();
    // model -> [(qpos, [xpos ++ xmat per geom])]
    let mut models: Vec<(String, Vec<(Vec<f64>, Vec<[f64; 12]>)>)> = Vec::new();
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "model" => models.push((t[1].to_string(), Vec::new())),
            "qpos" => models.last_mut().unwrap().1.push((t[1..].iter().filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect(), Vec::new())),
            "geom" => {
                let v: Vec<f64> = t[2..].iter().map(|x| x.parse().unwrap()).collect();
                models.last_mut().unwrap().1.last_mut().unwrap().1.push(v.try_into().unwrap());
            }
            _ => {}
        }
    }
    let (mut tried, mut exact, mut refused, mut worst) = (0usize, 0usize, 0usize, 0.0f64);
    let mut notes = Vec::new();
    for (rel, states) in &models {
        let path = root.join(rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let Ok(t) = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) else {
            refused += 1;
            continue;
        };
        for (s, (qpos, want)) in states.iter().enumerate() {
            let got = t.geom_frames_mujoco(qpos);
            if got.len() != want.len() {
                notes.push(format!("{rel}: {} geoms here, {} in MuJoCo", got.len(), want.len()));
                break;
            }
            for (g, ((p, m), w)) in got.iter().zip(want).enumerate() {
                tried += 1;
                let ours = [p[0], p[1], p[2], m[(0, 0)], m[(0, 1)], m[(0, 2)], m[(1, 0)], m[(1, 1)], m[(1, 2)], m[(2, 0)], m[(2, 1)], m[(2, 2)]];
                if ours.iter().zip(w).all(|(a, b)| a.to_bits() == b.to_bits()) {
                    exact += 1;
                } else {
                    let e = ours.iter().zip(w).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
                    worst = worst.max(e);
                    if notes.len() < 12 {
                        notes.push(format!("{rel} state {s} geom {g}: off {e:.2e}"));
                    }
                }
            }
        }
    }
    println!("geom_frames_mujoco against MuJoCo 3.13.0's mj_kinematics, BIT FOR BIT: {exact} of {tried} geom-states (worst {worst:.2e}); {refused} models the loader refuses");
    for n in &notes {
        println!("  {n}");
    }
}
