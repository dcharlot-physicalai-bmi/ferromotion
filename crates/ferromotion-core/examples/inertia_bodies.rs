//! Print compiled body mass and principal inertia for one MJCF. Not a pass.
use ferromotion_core::tree_from_mjcf;
use std::fs;

fn main() {
    match ferromotion_core::install_policy_from_env() {
        Ok(policy) => eprintln!("{}", ferromotion_core::policy_receipt(policy)),
        Err(error) => {
            eprintln!("policy refused: {error:?}");
            std::process::exit(2);
        }
    }
    let path = std::env::args().nth(1).expect("xml");
    let xml = fs::read_to_string(&path).unwrap();
    let dir = std::path::Path::new(&path).parent().unwrap().to_path_buf();
    let (tree, samples) = ferromotion_core::capture_ops(|| {
        tree_from_mjcf(&xml, &|p: &str| fs::read(dir.join(p)).ok())
    });
    let tree = tree.unwrap_or_else(|e| {
        eprintln!("load {e}");
        std::process::exit(1);
    });
    if std::env::args().any(|a| a == "--mesh") {
        for sample in samples.iter().filter(|sample| sample.op == "mesh_inertia") {
            println!(
                "mesh\t{}\t{}",
                sample.inputs.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(","),
                sample.outputs.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(",")
            );
        }
        return;
    }
    let arrays = tree.mujoco_native_model();
    let mass = arrays.iter().find(|(n, _)| *n == "body_mass").unwrap().1.clone();
    let inertia = arrays.iter().find(|(n, _)| *n == "body_inertia").unwrap().1.clone();
    for i in 0..mass.len() {
        let a = inertia[3 * i];
        let b = inertia[3 * i + 1];
        let c = inertia[3 * i + 2];
        println!("{i}\t{}\t{}\t{}\t{}", mass[i].to_bits(), a.to_bits(), b.to_bits(), c.to_bits());
    }
}
