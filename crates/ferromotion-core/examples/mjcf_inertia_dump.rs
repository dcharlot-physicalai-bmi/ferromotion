//! **What does the loader think each body of an MJCF model weighs?** — one line per body: mass, centre of mass
//! and inertia tensor in the link frame, and whether it was stated or inferred from geoms. The companion to
//! `menagerie_parity` for reading a single disagreement.
//!
//! ```text
//! cargo run --release -p ferromotion-core --example mjcf_inertia_dump -- <model.xml> [body-name-filter]
//! ```

use ferromotion_core::tree_from_mjcf;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: mjcf_inertia_dump <model.xml> [body-name-filter]");
        std::process::exit(2);
    }
    let model = Path::new(&args[1]);
    let filter = args.get(2).cloned().unwrap_or_default();
    let dir = model.parent().unwrap().to_path_buf();
    let xml = std::fs::read_to_string(model).expect("read model");
    let resolve = |p: &str| std::fs::read(dir.join(p)).ok();
    let t = match tree_from_mjcf(&xml, &resolve) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("REFUSED: {e}");
            std::process::exit(1);
        }
    };
    println!("{} tree joints, {} MuJoCo joints, {} bodies on joints, {} world-fixed; {} state no <inertial>, {} inferred from geoms", t.tree.dof(), t.joints.len(), t.body_frames.len(), t.world_fixed.len(), t.no_inertial.len(), t.inferred_from_geoms.len());
    for (name, idx) in &t.tree.link_names {
        if !filter.is_empty() && !name.contains(&filter) {
            continue;
        }
        let li = &t.tree.inertia[*idx];
        let how = if t.inferred_from_geoms.contains(name) { "inferred" } else if t.no_inertial.contains(name) { "massless" } else { "stated" };
        let i = &li.inertia;
        println!("{name:<32} {how:<8} m={:.9e} com=[{:.6e} {:.6e} {:.6e}] I=[{:.6e} {:.6e} {:.6e} | {:.6e} {:.6e} {:.6e}]", li.mass, li.com.x, li.com.y, li.com.z, i[(0, 0)], i[(1, 1)], i[(2, 2)], i[(0, 1)], i[(0, 2)], i[(1, 2)]);
    }
}
