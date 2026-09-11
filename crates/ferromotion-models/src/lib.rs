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

/// **The working envelope, computed once for the whole crate.**
///
/// Max over configuration of the horizontal distance from the base rotation axis to a chosen point:
/// random restarts followed by coordinate refinement. Verified to converge — 200 restarts and 2000
/// agree to six decimals on every arm tried — and it deliberately IGNORES joint limits, so every
/// figure is an upper bound over the unrestricted configuration space. Both facts matter when a number
/// disagrees with a datasheet: without the first the gap reads as a bad search, without the second as
/// the limits.
///
/// ⛔ **[`wrist`] and [`flange`] are one fixed transform apart and manufacturers disagree about which
/// they publish.** `Robot::from_dh` folds the final DH row into `ee_offset`, so `frame_pose(q, dof)`
/// stops short of `fk(q)`. Measured: ABB and DENSO quote reach to the WRIST, Kinova and Franka to the
/// FLANGE. Always measure both and say which the number is — a test naming the wrong one still passes
/// and misleads the next reader, which is exactly what happened here once already.
///
/// One implementation, reached by every module's tests, because four copies of a search like this is
/// four chances for one of them to drift into a weaker refinement schedule.
#[cfg(test)]
pub(crate) mod envelope {
    use ferromotion_core::Robot;
    use std::f64::consts::PI;

    fn search_n(r: &Robot, restarts: usize, point: impl Fn(&Robot, &[f64]) -> f64) -> (f64, Vec<f64>) {
        let n = r.dof();
        let mut s = 0xC0FF_EE00_1234_5678u64;
        let mut rnd = || {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 * PI - PI
        };
        let mut global = 0.0f64;
        let mut global_q: Vec<f64> = vec![0.0; n];
        for _ in 0..restarts {
            let mut best: Vec<f64> = (0..n).map(|_| rnd()).collect();
            let mut bv = point(r, &best);
            loop {
                let mut improved = false;
                for j in 0..n {
                    // ⭐ down to 1e-7: the coarse schedule locates the maximum, the fine tail pins the
                    // ARGUMENT. `franka::the_arm_reaches_maximum_extension_at_frankas_published_elbow_angle`
                    // needs the joint angle to nanoradians, not just the radius.
                    for d in [0.4, -0.4, 0.1, -0.1, 0.02, -0.02, 4e-3, -4e-3, 1e-3, -1e-3, 2e-4, -2e-4, 5e-5, -5e-5, 1e-5, -1e-5, 1e-6, -1e-6, 1e-7, -1e-7] {
                        let mut q = best.clone();
                        q[j] += d;
                        let v = point(r, &q);
                        if v > bv {
                            bv = v;
                            best = q;
                            improved = true;
                        }
                    }
                }
                if !improved {
                    break;
                }
            }
            if bv > global {
                global = bv;
                global_q = best;
            }
        }
        (global, global_q)
    }

    /// Restart count. ⭐ 60 is not a guess: 60, 200 and 2000 restarts agree to 1e-9 on every arm in
    /// this crate, so the extra work buys nothing but seconds in the debug profile CI uses. The
    /// convergence check itself lives in `the_envelope_search_has_converged`.
    const RESTARTS: usize = 60;

    fn search(r: &Robot, point: impl Fn(&Robot, &[f64]) -> f64) -> f64 {
        search_n(r, RESTARTS, point).0
    }

    /// The flange envelope AND the configuration that achieves it. The argument is the interesting half
    /// when a manufacturer publishes the POSE of maximum extension rather than its magnitude — Franka
    /// does, to fifteen figures, and that is a far stronger oracle than a rounded reach number.
    pub(crate) fn flange_argmax(r: &Robot) -> (f64, Vec<f64>) {
        search_n(r, RESTARTS, |r, q| {
            let p = r.fk(q).translation.vector;
            p.x.hypot(p.y)
        })
    }

    /// To the last JOINT frame — what `frame_pose(q, dof)` returns. ABB and DENSO publish reach here.
    pub(crate) fn wrist(r: &Robot) -> f64 {
        let n = r.dof();
        search(r, move |r, q| {
            let p = r.frame_pose(q, n).translation.vector;
            p.x.hypot(p.y)
        })
    }

    /// To the tool flange — what `fk(q)` returns, one folded transform beyond [`wrist`]. Kinova and
    /// Franka publish reach here.
    pub(crate) fn flange(r: &Robot) -> f64 {
        search(r, |r, q| {
            let p = r.fk(q).translation.vector;
            p.x.hypot(p.y)
        })
    }

    /// ⛔ **The premise every reach assertion rests on.** If the search returned a local maximum the
    /// figures would be too small and a correct table would look wrong — which is precisely how the
    /// Panda's apparent "smaller than published" anomaly could have been misread. Cheap restart counts
    /// must agree with expensive ones, or `RESTARTS` is set too low.
    #[ignore = "envelope search: seconds per arm in debug; release --ignored lane"]
    #[test]
    fn the_envelope_search_has_converged() {
        let arms = [
            ("panda", crate::franka::panda()),
            ("gen3_7dof", crate::kinova::gen3_7dof()),
            ("irb120", crate::abb::irb120()),
            ("denso", crate::others::denso_vs6556()),
        ];
        let mut worst = 0.0f64;
        for (name, r) in &arms {
            let cheap = search_n(r, RESTARTS, |r, q| {
                let p = r.fk(q).translation.vector;
                p.x.hypot(p.y)
            })
            .0;
            let dear = search_n(r, 240, |r, q| {
                let p = r.fk(q).translation.vector;
                p.x.hypot(p.y)
            })
            .0;
            assert!(
                (cheap - dear).abs() < 1e-9,
                "{name}: {RESTARTS} restarts gave {cheap:.9} and 240 gave {dear:.9} — RESTARTS is too low"
            );
            worst = worst.max((cheap - dear).abs());
        }
        eprintln!("  envelope search: {RESTARTS} vs 240 restarts agree to {worst:.1e} over {} arms", arms.len());
    }
}

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
        ///
        /// ⭐ Each entry below carries its COMPUTED envelope, so the remaining work is only to source
        /// the manufacturer's figure and compare — the measurement half is done. Two of them are
        /// marked SYNTHETIC: textbook constructs whose reach is `L1 + L2` by construction, where a
        /// reach check would be circular and `TableOnly` is the permanently correct answer.
        TableOnly,
        /// ⚠ **A published figure exists but is NOMINAL — usable only as a gross-error bound.**
        ///
        /// Sourcing one and finding it imprecise is a different state from not finding one, and
        /// collapsing the two would lose the measurement. Universal Robots publishes a reach for every
        /// arm (R500, R850, R1300 on the dimension drawings), and those figures are ROUNDED envelope
        /// numbers, not quantities this table can reproduce: the computed swept radius exceeds the
        /// published reach by 10.3 to 76.5 mm across the eight arms — 0.7% to 11.5% relatively, whose
        /// ordering differs from the millimetre one — and the excess is not a consistent function of
        /// any wrist parameter. Asserting them to a millimetre, the way
        /// the FANUC and KUKA figures are asserted, would have shipped a test whose tolerance was
        /// invented to make it pass.
        ///
        /// What they DO support is a bound on gross error — a transposed digit or a factor-of-two
        /// slip in a dominant link length — which `the_ur_envelopes_bound_their_published_nominal_reach`
        /// asserts at 15%, a threshold set from the measured 11.5% worst case with margin, and stated as such.
        NominalOnly(&'static str),
    }
    use Oracle::{NominalOnly, Published, TableOnly};

    /// Every arm this crate ships. The gate below fails if a constructor is added without an entry,
    /// which is the point: the decision gets made once, in the open, rather than defaulting to silence.
    const ARMS: &[(&str, Oracle)] = &[
        // --- abb.rs ---
        ("irb140", Published("ABB drawing: axis-1 to axis-5 = 70 + 380 mm and base-to-axis-2 352 + 360 mm arm, asserted as the wrist centre (0.450, 0, 0.712) m")),
        ("irb120", Published("ABB Product specification 3HAC035960: 580 mm reach; the computed envelope is 0.5800061 m, 6.1 um off")),
        ("irb1600_1_45", Published("ABB's model designation IRB 1600-X/1.45 — the suffix IS the reach in metres; the computed envelope is 1.4499999 m, 0.1 um off")),
        // ⚠ The envelopes below are to the WRIST (`frame_pose(q, dof)`), not the flange — `from_dh`
        //   folds the last DH row into `ee_offset`. Where an arm carries a tool the flange is further
        //   out, and which one a manufacturer's "reach" means has to be checked per maker: ABB quotes
        //   the wrist, and the Panda's figure lines up with the flange.
        ("yumi_single_arm", TableOnly), // wrist 0.591656 / flange 0.609656 m
        // --- classic.rs ---
        ("puma560", TableOnly), // wrist 0.873129 / flange 0.929379 m
        ("puma560_modified", TableOnly), // wrist 0.873129 m, identical to the standard-DH table as it must be
        ("stanford", Published("Paul, 'Robot Manipulators' Table 2.1 p.9 — a source-stated worked example, not a pose recomputed from this table")),
        ("two_link_planar", TableOnly), // SYNTHETIC: a textbook construct with chosen link lengths. Reach is L1+L2 BY CONSTRUCTION, so a reach check would be circular. Permanently TableOnly, and correctly so.
        ("three_link_planar", TableOnly), // SYNTHETIC, as above
        // --- franka.rs ---
        ("panda", Published("Franka FCI spec: maximum extension occurs at q4 = -0.467002423653011 rad; the table's argmax is -0.467002428685570, 5.0e-9 rad off. ⛔ The ANGLE is the oracle, not a reach: the FCI page states no reach in mm at all (checked 2026-09-10), so the widely-quoted 855 mm is not from the document this crate cites. Flange envelope 0.857893 m, recorded but unasserted.")),
        ("fr3", Published("same FCI elbow-flip angle as the Panda, which the two share along with their chain")),
        // --- kinova.rs ---
        ("gen3_7dof", Published("Kinova spec TS-014: maximum reach 902 mm; the FLANGE envelope is 0.902912 m, 0.9 mm over")),
        ("gen3_6dof", Published("Gen3 User Guide Figure 89: the 410 mm link length, which settles a transcription hazard the printed Table 95's column headers create")),
        ("gen3_lite", NominalOnly("Kinova publishes 760 mm — a ROUND number where the same maker quotes the 7 DoF as 902 mm; the flange envelope is 0.763551 m, 3.55 mm over (0.47%), bounded at 1.5% rather than asserted to a millimetre")),
        // --- kuka.rs ---
        ("kuka_lbr_iiwa_7_r800", Published("Spec Fig. 4-1 flange height 1266 mm, printed on the drawing, and the Section 4.2.1 reach of 800 mm")),
        ("kuka_lbr_iiwa_14_r820", Published("Spec Fig. 4-4 flange height 1306 mm and the Section 4.3.1 reach of 820 mm")),
        ("kuka_kr_5_arc", Published("R1412 printed on the drawing's top view, asserted as the reach a1 + a2 + hypot(d4, a3)")),
        // --- others.rs ---
        ("xarm5", TableOnly), // wrist 0.716645 / flange 0.763873 m. ⛔ VERIFIED ABSENCE, not an unchecked one: UFACTORY's own technical specification (docs.xarm.ufactory.cc, read 2026-09-10) publishes a CARTESIAN RANGE of ±700 mm and no reach or working-radius figure. The commonly quoted "700 mm reach" is that operating box, not an envelope — and the computed wrist envelope EXCEEDS it, as a software-limited box should be exceeded by the physical arm.
        ("xarm6", TableOnly), // wrist 0.716645 / flange 0.763873 m, identical to the xArm 5; same verified absence
        ("xarm7", TableOnly), // wrist 0.724825 / flange 0.772053 m; same verified absence
        ("lite6", TableOnly), // wrist 0.443661 / flange 0.505161 m
        ("fanuc_lr_mate_200id", Published("LR Mate 200iD data sheet reach 717 mm, asserted to a millimetre")),
        ("denso_vs6556", Published("DENSO WAVE VS-6556 specification: maximum arm reach 653 mm; the WRIST envelope is 0.653423 m, 0.42 mm over")),
        // --- rethink.rs ---
        ("baxter", Published("Williams' stated 0.80764 m, which the table must reproduce and which the doc notes differs from the 0.88664 a naive reading gives")),
        ("sawyer", Published("the source paper's stated q = 0 pose (eq. 101, zero configuration), not a pose recomputed from this table")),
        // --- ur.rs: the UR article publishes the DH table and no pose. The datasheets DO publish a
        //     reach per arm, and it was sourced and measured against the tables (2026-09-10) — it is
        //     nominal, not exact. See `NominalOnly`. ---
        ("ur3", NominalOnly("UR datasheet reach 500 mm (drawing R500); computed envelope 0.5538 m, +53.8 mm")),
        ("ur5", NominalOnly("UR datasheet reach 850 mm (drawing R850); computed envelope 0.9184 m, +68.4 mm")),
        ("ur10", NominalOnly("UR datasheet reach 1300 mm (drawing R1300); computed envelope 1.3103 m, +10.3 mm")),
        ("ur3e", NominalOnly("UR datasheet reach 500 mm; computed envelope 0.5577 m, +57.7 mm — the largest RELATIVE excess, 11.5%, which is what the 15% bound is set from")),
        ("ur5e", NominalOnly("UR datasheet reach 850 mm; computed envelope 0.9265 m, +76.5 mm — the largest ABSOLUTE excess")),
        ("ur10e", NominalOnly("UR datasheet reach 1300 mm; computed envelope 1.3157 m, +15.7 mm")),
        ("ur16e", NominalOnly("UR datasheet reach 900 mm; computed envelope 0.9740 m, +73.9 mm")),
        ("ur20", NominalOnly("UR datasheet reach 1750 mm; computed envelope 1.7615 m, +11.5 mm")),
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
        let nominal = ARMS.iter().filter(|(_, o)| matches!(o, NominalOnly(_))).count();
        eprintln!(
            "  geometry oracles: {} of {} arms carry an EXACT independent figure, {} a NOMINAL one (gross-error bound only), {} their own table only",
            published.len(),
            ARMS.len(),
            nominal,
            ARMS.len() - published.len() - nominal
        );
        for (n, o) in ARMS.iter() {
            if let Published(cite) = o {
                eprintln!("    ⭐ {n}: {cite}");
            }
        }
        assert!(
            published.len() >= 15,
            "external geometry checks must not regress below the 15 recorded on 2026-09-10, got {}",
            published.len()
        );
    }
}
