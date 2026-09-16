//! **One geom pair, run through the collision pipeline with everything printed** — the instrument for a
//! single disagreement found by `menagerie_contacts`, which reports thousands of pairs and can only name one.
//!
//! ```text
//! cargo run --release --example ccd_one_pair -- <menagerie root> <model.xml> <geom1> <geom2> \
//!     <pos1 x y z> <mat1 9> <pos2 x y z> <mat2 9>
//! ```
//!
//! The poses are MuJoCo's own `geom_xpos`/`geom_xmat` for the sample in question, so the only thing under
//! test is the pair routine.

use ferromotion_core::{ccd, collide_pair_with, tree_from_mjcf, CcdConfig, CcdObj, CollideOptions, CollisionGeom, GeomPose, GeomType};
use nalgebra::{Matrix3, Vector3};

fn geom_type(s: &str) -> GeomType {
    match s {
        "plane" => GeomType::Plane,
        "hfield" => GeomType::HField,
        "sphere" => GeomType::Sphere,
        "capsule" => GeomType::Capsule,
        "ellipsoid" => GeomType::Ellipsoid,
        "cylinder" => GeomType::Cylinder,
        "box" => GeomType::Box,
        _ => GeomType::Mesh,
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 7 {
        eprintln!("usage: ccd_one_pair <menagerie root> <model.xml> <type1> <size1 x y z> <mesh1|-> <type2> <size2 x y z> <mesh2|-> <pos1 3> <mat1 9> <pos2 3> <mat2 9>");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&a[1]);
    let model = root.join(&a[2]);
    let dir = model.parent().unwrap().to_path_buf();
    let xml = std::fs::read_to_string(&model).expect("model");
    let t = tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()).expect("loads");

    let arg: Vec<String> = a[3..].to_vec();
    let f = |i: usize| arg[i].parse::<f64>().unwrap_or_else(|_| panic!("bad number {:?}", arg[i]));
    let (ty1, m1, ty2, m2) = (arg[0].clone(), arg[4].clone(), arg[5].clone(), arg[9].clone());
    let s1 = [f(1), f(2), f(3)];
    let s2 = [f(6), f(7), f(8)];
    let p1 = Vector3::new(f(10), f(11), f(12));
    let r1: Vec<f64> = (13..22).map(f).collect();
    let p2 = Vector3::new(f(22), f(23), f(24));
    let r2: Vec<f64> = (25..34).map(f).collect();

    let hull = |n: &str| if n == "-" { None } else { Some(t.mesh_hulls.get(n).expect("mesh hull")) };
    let g1 = CollisionGeom { kind: geom_type(&ty1), pose: GeomPose { pos: p1, mat: Matrix3::from_row_slice(&r1) }, size: s1, hull: hull(&m1) };
    let g2 = CollisionGeom { kind: geom_type(&ty2), pose: GeomPose { pos: p2, mat: Matrix3::from_row_slice(&r2) }, size: s2, hull: hull(&m2) };
    println!("centre 1 -> 2: {:?}", (g2.pose.pos - g1.pose.pos).normalize());
    // MULTICCD=0 takes the single-contact path, which isolates GJK/EPA from the multi-contact clipping
    let opts = CollideOptions { multiccd: std::env::var("MULTICCD").map(|v| v != "0").unwrap_or(true), ..Default::default() };
    println!("multiccd: {}", opts.multiccd);
    // the raw CCD, before the pair dispatcher and the multi-contact stage
    {
        let mut o1 = CcdObj::new(g1.kind, g1.pose, g1.size, g1.hull, 0.0);
        let mut o2 = CcdObj::new(g2.kind, g2.pose, g2.size, g2.hull, 0.0);
        let cfg = CcdConfig { max_iterations: 35, tolerance: 1e-6, max_contacts: 1, dist_cutoff: 0.0 };
        let st = ccd(&cfg, &mut o1, &mut o2);
        println!("raw ccd: epa_status={} separated={} dist={:?}", st.epa_status, st.separated, st.dist);
        for (a, b) in st.x1.iter().zip(&st.x2) {
            println!("   x1={:?} x2={:?}  x1-x2={:?}", a.as_slice(), b.as_slice(), (a - b).normalize().as_slice());
        }
    }
    match collide_pair_with(&opts, 0.0, &g1, &g2) {
        Ok(cs) => {
            println!("{} contacts:", cs.len());
            for c in &cs {
                println!("  dist={} pos={:?} normal={:?}", c.dist, c.pos.as_slice(), c.normal.as_slice());
                println!("    normal · (centre2 − centre1) = {}", c.normal.dot(&(g2.pose.pos - g1.pose.pos).normalize()));
            }
        }
        Err(e) => println!("refused: {e}"),
    }
}
