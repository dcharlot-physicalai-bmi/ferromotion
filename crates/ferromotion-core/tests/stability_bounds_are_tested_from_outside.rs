//! **Every stability bound must have a test that runs the system PAST it.**
//!
//! This workspace shipped eight `max_stable_dt`/`stability_limit`/`stable_timestep` methods. An audit
//! ran each one's system on both sides of its bound and found **two wrong — and both were among the
//! ones nothing tested from outside. Every bound that had such a test was correct.**
//!
//! `LuGre::max_stable_dt` returned the bristle time constant where the limit is twice it: conservative,
//! so nothing ever diverged and no user would report it. `Rotor::max_stable_dt` returned the UNDAMPED
//! `2/ω_n` for a step explicit in the damper, so nine tenths of the documented bound diverged: that one
//! fails in the field and never in a test. The properties their tests did check — monotonicity, scaling,
//! degenerate cases — are all preserved by both errors. Only crossing the bound separates them.
//!
//! So every such method must NAME the test that crosses it, in a comment reading
//! `CROSSED BY: <test fn name>`, and this gate checks that the named test exists in the same file. A
//! first version of this gate tried to RECOGNISE a crossing test by its syntax and flagged three
//! correct bounds — including the exemplary one, whose test uses literal step sizes rather than a
//! multiplier. A gate that fires on good code teaches people to silence it, so the claim is explicit
//! and human-made, and the gate only verifies it points somewhere real.
//!
//! If the bound is deliberately NOT the stability edge — a conservative practical bound, or an upper
//! envelope verified another way — say so with `BOUND NOT THE EDGE:` and the reason.

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
fn every_stability_bound_has_a_test_that_crosses_it() {
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

    let mut unguarded = Vec::new();
    let mut cited: Vec<String> = Vec::new();
    let mut exempt = Vec::new();
    for f in &files {
        let text = std::fs::read_to_string(f).expect("readable");
        for name in ["max_stable_dt", "stability_limit", "stable_timestep"] {
            let decl = format!("pub fn {name}");
            let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
            // EVERY declaration site, not just the first: two files declare the same bound twice, and
            // checking only the first would have passed one of them on the other's citation.
            for (at, _) in text.match_indices(&decl) {
            // the marker must sit in the doc block immediately above the declaration
            let head = &text[..at];
            let block: String = head.lines().rev().take_while(|l| {
                let t = l.trim();
                t.starts_with("///") || t.starts_with("//") || t.starts_with("#[") || t.is_empty()
            }).collect::<Vec<_>>().join("\n");
            if block.contains("BOUND NOT THE EDGE:") {
                exempt.push(format!("{rel}::{name}"));
                continue;
            }
            match block.split("CROSSED BY:").nth(1).and_then(|r| r.split_whitespace().next()) {
                Some(test_fn) => {
                    let decl_of_test = format!("fn {}(", test_fn.trim_end_matches(&['`', ',', '.'][..]));
                    if text.contains(&decl_of_test) {
                        cited.push(format!("{rel}::{name} -> {test_fn}"));
                    } else {
                        unguarded.push(format!("{rel}::{name} names `{test_fn}`, which does not exist in this file"));
                    }
                }
                None => unguarded.push(format!("{rel}::{name} (no CROSSED BY:)")),
            }
            }
        }
    }

    eprintln!("\n  stability bounds: {} cite a crossing test, {} exempt, {} unguarded", cited.len(), exempt.len(), unguarded.len());
    for c in &cited {
        eprintln!("    ok:     {c}");
    }
    for e in &exempt {
        eprintln!("    exempt: {e}");
    }
    for u in &unguarded {
        eprintln!("    UNGUARDED: {u}");
    }
    eprintln!();
    assert!(
        unguarded.is_empty(),
        "{} stability bound(s) with no test that crosses them. Two of the eight this workspace shipped \
         were wrong, and both were in exactly this state. Add a test that runs the system just past the \
         bound and requires it to diverge, then cite it as `CROSSED BY: <test fn>`; or mark the method \
         `BOUND NOT THE EDGE:` with the reason.",
        unguarded.len()
    );
}
