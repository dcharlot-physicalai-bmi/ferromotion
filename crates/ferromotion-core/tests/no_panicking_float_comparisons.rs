//! **No library code may compare floats in a way that panics on a `NaN`.**
//!
//! `a.partial_cmp(&b).unwrap()` panics the moment one operand is `NaN`, and
//! `.unwrap_or(Ordering::Equal)` silently calls a `NaN` equal to everything. Both were widespread here
//! and both were reachable from public entry points with ordinary caller data: a LiDAR dropout, an
//! invalid depth pixel, a diverged simulation state.
//!
//! `f64::total_cmp` is the fix for ORDERING, but it is not a `NaN` guard on its own: it orders
//! `-NaN < -inf < … < +inf < +NaN`, so under a maximum a `NaN` is selected as the best element and
//! under a descending sort it lands first. Sites that pick a best element, or that treat a position in
//! a sorted list as "best", must SKIP non-finite values, not merely order them. This gate does not try
//! to judge which case applies; it only ensures no site can panic.
//!
//! Escape hatch, for a comparison provably never given a `NaN`: put `PARTIAL_CMP OK` in a comment on
//! the line or the line above, with the reason.

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
fn no_library_file_unwraps_a_float_comparison() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("crates/").to_path_buf();
    let mut files = Vec::new();
    for c in std::fs::read_dir(&root).expect("crates/ readable").flatten() {
        let src = c.path().join("src");
        if src.is_dir() {
            rust_files(&src, &mut files);
        }
    }
    files.sort();
    assert!(files.len() > 100, "the walk should cover the workspace, found {}", files.len());

    let mut offenders = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).expect("readable");
        let lines: Vec<&str> = text.lines().collect();
        let test_start = lines.iter().position(|l| l.contains("#[cfg(test)]")).unwrap_or(lines.len());
        for (i, l) in lines.iter().enumerate().take(test_start) {
            if l.trim_start().starts_with("//") {
                continue;
            }
            if !l.contains("partial_cmp") {
                continue;
            }
            if !(l.contains(".unwrap()") || l.contains(".unwrap_or(")) {
                continue;
            }
            let prev = if i > 0 { lines[i - 1] } else { "" };
            if l.contains("PARTIAL_CMP OK") || prev.contains("PARTIAL_CMP OK") {
                continue;
            }
            let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
            offenders.push(format!("{rel}:{}  {}", i + 1, l.trim()));
        }
    }

    if !offenders.is_empty() {
        eprintln!("\n=== {} panicking float comparison(s) in library code ===", offenders.len());
        for o in &offenders {
            eprintln!("  {o}");
        }
        eprintln!();
    }
    assert!(offenders.is_empty(), "{} panicking float comparison(s); use f64::total_cmp for ordering, SKIP non-finite values where a best element is chosen, or mark the line PARTIAL_CMP OK with a reason", offenders.len());
}
