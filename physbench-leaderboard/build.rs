//! Auto-discover every submission in `submissions/*.rs` and generate a registry. A contributor only has to
//! drop a file in `submissions/` that defines `pub struct M; impl crate::bench::Model for M {…}` and
//! `pub const META: crate::bench::Meta = …;` — no manual registration, no edit to any shared file.
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=submissions");
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("submissions");
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("submissions/ directory must exist")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map_or(false, |x| x == "rs"))
        .collect();
    files.sort();

    let mut mods = String::new();
    let mut regs = String::new();
    for p in &files {
        let stem = p.file_stem().unwrap().to_str().unwrap();
        let abs = fs::canonicalize(p).unwrap();
        println!("cargo:rerun-if-changed={}", p.display());
        mods.push_str(&format!("#[path = {:?}]\nmod sub_{};\n", abs.to_str().unwrap(), stem));
        regs.push_str(&format!(
            "        (&sub_{s}::META, Box::new(sub_{s}::M) as Box<dyn crate::bench::Model>),\n", s = stem));
    }
    let code = format!(
        "{mods}\npub fn all() -> Vec<(&'static crate::bench::Meta, Box<dyn crate::bench::Model>)> {{\n    vec![\n{regs}    ]\n}}\n");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("registry.rs");
    fs::write(out, code).unwrap();
}
