//! Dump instrumented kinematics ops at `qpos0` for every Menagerie model this loader accepts.
//! One JSON object per line. This is not a pass/fail report.

use ferromotion_core::tree_from_mjcf;
use std::fs;
use std::path::Path;

fn main() {
    match ferromotion_core::install_policy_from_env() {
        Ok(policy) => eprintln!("{}", ferromotion_core::policy_receipt(policy)),
        Err(error) => {
            eprintln!("policy refused: {error:?}");
            std::process::exit(2);
        }
    }
    let root = std::env::args().nth(1).expect("menagerie root");
    let mut models = Vec::new();
    for ent in fs::read_dir(&root).unwrap().flatten() {
        if !ent.path().is_dir() {
            continue;
        }
        for xml in fs::read_dir(ent.path()).unwrap().flatten() {
            if xml.path().extension().and_then(|s| s.to_str()) == Some("xml") {
                models.push(xml.path());
            }
        }
    }
    models.sort();
    let root = Path::new(&root);
    for path in models {
        let rel = path.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/");
        let Ok(xml) = fs::read_to_string(&path) else { continue };
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(tree) = tree_from_mjcf(&xml, &|p: &str| fs::read(dir.join(p)).ok()) else {
            println!("{{\"model\":{rel:?},\"loaded\":false}}");
            continue;
        };
        let qpos = tree.qpos0();
        let samples = tree.trace_mujoco_kinematics_ops(&qpos);
        print!("{{\"model\":{rel:?},\"loaded\":true,\"ops\":[");
        for (i, sample) in samples.iter().enumerate() {
            if i > 0 {
                print!(",");
            }
            print!("{{\"op\":\"{}\",\"in\":[", sample.op);
            for (j, bits) in sample.inputs.iter().enumerate() {
                if j > 0 {
                    print!(",");
                }
                print!("{bits}");
            }
            print!("],\"out\":[");
            for (j, bits) in sample.outputs.iter().enumerate() {
                if j > 0 {
                    print!(",");
                }
                print!("{bits}");
            }
            print!("]}}");
        }
        println!("]}}");
    }
}
