//! **Run [`ferromotion_core::qhull::mesh_graph`] over a vertex dump** (`menagerie_qhull --dump`'s format) and
//! print each graph the way a reference build of qhull prints it, so the two can be diffed line by line.
//!
//! ```text
//! cargo run --release --example qhull_graph_dump -- <verts txt>
//! ```
//!
//! A mesh the port refuses (a qhull path it does not carry) prints `refused <reason>`.

use ferromotion_core::qhull::mesh_graph;

fn main() {
    let path = std::env::args().nth(1).expect("usage: qhull_graph_dump <verts txt>");
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let t: Vec<&str> = line.split('\t').collect();
        if t.len() != 5 || t[0] != "mesh" {
            continue;
        }
        let cap: i64 = t[3].parse().unwrap();
        let n: usize = t[4].parse().unwrap();
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| {
                let b: Vec<f64> = lines.next().unwrap().split_whitespace().map(|x| f64::from_bits(u64::from_str_radix(x, 16).unwrap())).collect();
                [b[0], b[1], b[2]]
            })
            .collect();
        match mesh_graph(&pts, (cap >= 0).then_some(cap as usize)) {
            Ok(g) => println!("graph\t{}\t{}\t{}", t[1], t[2], g.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")),
            Err(e) => println!("refused\t{}\t{}\t{e}", t[1], t[2]),
        }
    }
}
