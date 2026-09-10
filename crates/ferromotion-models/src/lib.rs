//! ferromotion-models — **robot arms built from the tables their makers and their textbooks publish.**
//!
//! Every constructor here is a [`Robot`] assembled by [`Robot::from_dh`] from a Denavit–Hartenberg table,
//! and every table states where it came from: the manufacturer document, the paper, or the textbook
//! edition, with the convention that source uses ([`DhConvention::Standard`] or
//! [`DhConvention::Modified`]) and any unit conversion applied. DH parameters are public engineering
//! data. Files — URDFs, meshes — carry licenses, and none are vendored here.
//!
//! Each model is verified rather than transcribed: forward kinematics at a documented pose or a
//! hand-computed one, the same central-difference Jacobian and Hessian check every `Robot` in the
//! workspace is held to, and a stated mutation showing the convention argument is load-bearing — a table
//! read under the wrong convention builds a plausible-looking wrong arm, and that is the failure mode
//! these tests exist to catch.
//!
//! Where a source gives no joint limit, effort or velocity, the model says so by leaving it `None`
//! rather than inventing one. Pure `nalgebra` → WASM-clean.

pub use ferromotion_core::{DhConvention, DhRow, Robot};

pub mod ur;
pub mod classic;
pub mod franka;
pub mod kuka;
pub mod kinova;
pub mod abb;
pub mod rethink;
pub mod others;

/// **What independently pins each arm's geometry — and, honestly, where nothing does.**
///
/// Every arm in this crate is verified against its own table: a known answer, a central-difference
/// Jacobian and Hessian, and a convention-swap mutation. ⛔ **All of those pass on a WRONG arm.** The
/// Hessian of a mistyped link length still matches its own finite differences, and a known answer
/// "computed by hand from the table" was computed from whatever the table says — so if a length were
/// mistyped on the day the table was written, the hand computation would have produced that wrong
/// arm's pose and every test would have been green ever since. A value derived from the artefact
/// cannot audit the artefact.
///
/// The defence is a figure published by someone other than this table: a reach on a data sheet, a
/// height printed on a drawing, a worked example in a textbook. This module records which arms have
/// one, cites it, and — the part that matters — records which do not.
#[cfg(test)]
mod geometry_oracles {
    /// An independent figure that pins an arm's geometry, or the honest absence of one.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Oracle {
        /// A number published by someone other than this table, and the test that asserts it.
        Published(&'static str),
        /// ⚠ **No independent figure was located by this review.** Not "there is none" — see the
        /// workspace rule on absence claims. The table is verified against itself and against the
        /// convention-swap mutation, which catches a great deal, but a transcription error made on day
        /// one would not be caught by anything here. Sourcing a reach or an envelope figure for these
        /// is real work and the number must never be guessed at.
        TableOnly,
    }
    use Oracle::{Published, TableOnly};

    /// Every arm this crate ships. The gate below fails if a constructor is added without an entry,
    /// which is the point: the decision gets made once, in the open, rather than defaulting to silence.
    const ARMS: &[(&str, Oracle)] = &[
        // --- abb.rs ---
        ("irb140", Published("ABB drawing: axis-1 to axis-5 = 70 + 380 mm and base-to-axis-2 352 + 360 mm arm, asserted as the wrist centre (0.450, 0, 0.712) m")),
        ("irb120", TableOnly),
        ("irb1600_1_45", TableOnly),
        ("yumi_single_arm", TableOnly),
        // --- classic.rs ---
        ("puma560", TableOnly),
        ("puma560_modified", TableOnly),
        ("stanford", Published("Paul, 'Robot Manipulators' Table 2.1 p.9 — a source-stated worked example, not a pose recomputed from this table")),
        ("two_link_planar", TableOnly),
        ("three_link_planar", TableOnly),
        // --- franka.rs ---
        ("panda", TableOnly),
        ("fr3", TableOnly),
        // --- kinova.rs ---
        ("gen3_7dof", TableOnly),
        ("gen3_6dof", Published("Gen3 User Guide Figure 89: the 410 mm link length, which settles a transcription hazard the printed Table 95's column headers create")),
        ("gen3_lite", TableOnly),
        // --- kuka.rs ---
        ("kuka_lbr_iiwa_7_r800", Published("Spec Fig. 4-1 flange height 1266 mm, printed on the drawing, and the Section 4.2.1 reach of 800 mm")),
        ("kuka_lbr_iiwa_14_r820", Published("Spec Fig. 4-4 flange height 1306 mm and the Section 4.3.1 reach of 820 mm")),
        ("kuka_kr_5_arc", Published("R1412 printed on the drawing's top view, asserted as the reach a1 + a2 + hypot(d4, a3)")),
        // --- others.rs ---
        ("xarm5", TableOnly),
        ("xarm6", TableOnly),
        ("xarm7", TableOnly),
        ("lite6", TableOnly),
        ("fanuc_lr_mate_200id", Published("LR Mate 200iD data sheet reach 717 mm, asserted to a millimetre")),
        ("denso_vs6556", TableOnly),
        // --- rethink.rs ---
        ("baxter", Published("Williams' stated 0.80764 m, which the table must reproduce and which the doc notes differs from the 0.88664 a naive reading gives")),
        ("sawyer", Published("the source paper's stated q = 0 pose (eq. 101, zero configuration), not a pose recomputed from this table")),
        // --- ur.rs: the UR article publishes the DH table itself and no pose; no reach or envelope
        //     figure is recorded in this repo for any of the eight, so none can be asserted here
        //     without sourcing a number from outside it. ---
        ("ur3", TableOnly),
        ("ur5", TableOnly),
        ("ur10", TableOnly),
        ("ur3e", TableOnly),
        ("ur5e", TableOnly),
        ("ur10e", TableOnly),
        ("ur16e", TableOnly),
        ("ur20", TableOnly),
    ];

    /// The sources, read at compile time so the enumeration cannot drift from what is shipped.
    const SOURCES: &[(&str, &str)] = &[
        ("abb.rs", include_str!("abb.rs")),
        ("classic.rs", include_str!("classic.rs")),
        ("franka.rs", include_str!("franka.rs")),
        ("kinova.rs", include_str!("kinova.rs")),
        ("kuka.rs", include_str!("kuka.rs")),
        ("others.rs", include_str!("others.rs")),
        ("rethink.rs", include_str!("rethink.rs")),
        ("ur.rs", include_str!("ur.rs")),
    ];

    /// Every `pub fn NAME() -> Robot` the crate actually ships.
    fn shipped_arms() -> Vec<String> {
        let mut out = Vec::new();
        for (_, src) in SOURCES {
            for line in src.lines() {
                let t = line.trim_start();
                if let Some(rest) = t.strip_prefix("pub fn ")
                    && let Some((name, tail)) = rest.split_once('(')
                    && tail.starts_with(") -> Robot")
                {
                    out.push(name.to_string());
                }
            }
        }
        out.sort();
        out
    }

    /// ⛔ **The gate.** Adding an arm without deciding what independently pins its geometry fails here.
    /// Silence was the previous default and it is what let 1 of 33 arms carry an external check for as
    /// long as it did.
    #[test]
    fn every_shipped_arm_is_recorded_with_its_geometry_oracle() {
        let shipped = shipped_arms();
        assert!(shipped.len() >= 30, "the source scan found only {} arms, so it is not finding them", shipped.len());

        let mut listed: Vec<&str> = ARMS.iter().map(|(n, _)| *n).collect();
        listed.sort_unstable();
        let missing: Vec<&String> = shipped.iter().filter(|n| !listed.contains(&n.as_str())).collect();
        let stale: Vec<&&str> = listed.iter().filter(|n| !shipped.iter().any(|s| s == *n)).collect();
        assert!(missing.is_empty(), "arms shipped with no oracle entry: {missing:?} — decide and record, TableOnly is a valid answer");
        assert!(stale.is_empty(), "oracle entries for arms that no longer exist: {stale:?}");
        assert_eq!(shipped.len(), ARMS.len(), "one entry per arm");
    }

    /// The coverage itself, printed. Not a pass/fail on the ratio — that would freeze a number nobody
    /// chose — but a floor, so the external checks that exist cannot be quietly deleted.
    #[test]
    fn the_external_geometry_coverage_is_reported_and_does_not_regress() {
        let published: Vec<&str> = ARMS.iter().filter(|(_, o)| matches!(o, Published(_))).map(|(n, _)| *n).collect();
        eprintln!(
            "  geometry oracles: {} of {} arms carry an INDEPENDENT published figure; {} are verified against their own table only",
            published.len(),
            ARMS.len(),
            ARMS.len() - published.len()
        );
        for (n, o) in ARMS.iter() {
            if let Published(cite) = o {
                eprintln!("    ⭐ {n}: {cite}");
            }
        }
        assert!(
            published.len() >= 9,
            "external geometry checks must not regress below the 9 recorded on 2026-09-10, got {}",
            published.len()
        );
    }
}
