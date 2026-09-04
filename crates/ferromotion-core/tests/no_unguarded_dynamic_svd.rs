//! **No library code may call a dynamically sized SVD without a finiteness guard.**
//!
//! nalgebra's `DMatrix` SVD does not terminate on a matrix holding a `NaN` (measured in
//! `nalgebra_nan_behaviour.rs`), and a hang reports nothing. `ferromotion_core::finite_svd` and
//! `finite_singular_values` check first and return `None`; this gate keeps the raw calls from coming
//! back. A fixed-size SVD (`Matrix3`, `Matrix4`) returns on a `NaN` and is exempt, but the exemption
//! must be deliberate: put `FIXED-SIZE SVD` in a comment on the call's line or the line above it.

use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

#[test]
fn every_dynamic_svd_call_in_library_code_is_guarded_or_explicitly_fixed_size() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates/").to_path_buf();
    let mut files = Vec::new();
    for c in std::fs::read_dir(&root).expect("crates/ readable").flatten() {
        let src = c.path().join("src");
        if src.is_dir() {
            rust_files(&src, &mut files);
        }
    }
    files.sort();
    assert!(files.len() > 100, "the walk should find the whole workspace, found {}", files.len());

    let mut offenders = Vec::new();
    for f in &files {
        if f.ends_with("numerics.rs") {
            continue; // the one guarded home
        }
        let text = std::fs::read_to_string(f).expect("readable");
        let lines: Vec<&str> = text.lines().collect();
        let test_start = lines.iter().position(|l| l.contains("#[cfg(test)]")).unwrap_or(lines.len());
        for (i, l) in lines.iter().enumerate().take(test_start) {
            if !(l.contains(".singular_values()") || l.contains(".svd(")) {
                continue;
            }
            let prev = if i > 0 { lines[i - 1] } else { "" };
            if l.contains("FIXED-SIZE SVD") || prev.contains("FIXED-SIZE SVD") {
                continue;
            }
            let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
            offenders.push(format!("{rel}:{}  {}", i + 1, l.trim()));
        }
    }

    if !offenders.is_empty() {
        eprintln!("\n=== {} unguarded SVD call(s) in library code ===", offenders.len());
        for o in &offenders {
            eprintln!("  {o}");
        }
        eprintln!();
    }
    assert!(offenders.is_empty(), "{} unguarded dynamic-SVD call(s); use finite_svd / finite_singular_values, or mark a fixed-size call with FIXED-SIZE SVD", offenders.len());
}
