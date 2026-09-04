//! **Which public entry points fault on a non-finite input, measured rather than asserted.**
//!
//! A claim that "the remaining unwrapped float comparisons are on internally derived values, not
//! caller data" was made twice in this repo's history and was wrong both times. This probe settles it
//! by calling each candidate entry point with a non-finite value and recording what happens, instead
//! of reading the code and guessing. Every probe runs under `catch_unwind`, so one fault does not hide
//! the rest, and the panic hook is silenced so the report is readable.
//!
//! `Roadmap` is deliberately absent: it has a private field, so a caller cannot build one holding a
//! non-finite node from outside this crate, and `PrmStar::build` validates its bounds. That site is
//! unreachable rather than unguarded, which is what this probe was written to distinguish.

use ferromotion_core::*;
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use std::panic::{catch_unwind, AssertUnwindSafe};

const NAN: f64 = f64::NAN;

/// Run one probe on its own thread with a watchdog, print the verdict immediately, and return it.
///
/// The watchdog exists because the first version of this probe HUNG: a non-finite input does not only
/// panic, it can also make an iterative routine never converge. A single collected report at the end
/// could not say which entry point was responsible, so each one now answers for itself.
fn probe(name: &'static str, f: impl FnOnce() + Send + 'static) -> (&'static str, &'static str) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(catch_unwind(AssertUnwindSafe(f)).is_ok());
    });
    let verdict = match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(true) => "returned",
        Ok(false) => "PANICKED",
        Err(_) => "DID NOT TERMINATE in 5s",
    };
    eprintln!("  {name:<45} {verdict}");
    (name, verdict)
}

#[test]
fn report_which_public_entry_points_fault_on_a_non_finite_input() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    eprintln!("\n=== non-finite input probe ===");
    let mut rows = Vec::new();
    rows.push(probe("Bvh::build (a NaN AABB)", || {
        let boxes = vec![
            Aabb { min: Vector3::new(0.0, 0.0, 0.0), max: Vector3::new(1.0, 1.0, 1.0) },
            Aabb { min: Vector3::new(NAN, 0.0, 0.0), max: Vector3::new(1.0, 1.0, 1.0) },
            Aabb { min: Vector3::new(2.0, 2.0, 2.0), max: Vector3::new(3.0, 3.0, 3.0) },
        ];
        let _ = Bvh::build(&boxes);
    }));
    rows.push(probe("SdfScene::distance (a NaN query point)", || {
        let s = SdfScene { prims: vec![Sdf::Sphere { center: Vector3::zeros(), radius: 1.0 }, Sdf::Sphere { center: Vector3::new(3.0, 0.0, 0.0), radius: 1.0 }] };
        let _ = s.distance(&Vector3::new(NAN, 0.0, 0.0));
    }));
    rows.push(probe("Sdf::Box::distance (a NaN query point)", || {
        let _ = Sdf::Box { center: Vector3::zeros(), half: Vector3::new(1.0, 1.0, 1.0) }.distance(&Vector3::new(NAN, 0.0, 0.0));
    }));
    rows.push(probe("w1_empirical_1d (a NaN sample)", || {
        let _ = w1_empirical_1d(&[0.0, 1.0, NAN], &[0.0, 1.0, 2.0]);
    }));
    rows.push(probe("cramer_distance (a NaN sample)", || {
        let _ = cramer_distance(&[0.0, 1.0, NAN], &[0.0, 1.0, 2.0]);
    }));
    rows.push(probe("time_delay (a NaN sample)", || {
        let _ = time_delay(&[0.0, 1.0, NAN, 0.5], &[0.0, 1.0, 0.0, 0.5]);
    }));
    rows.push(probe("screw_log_so3 (a NaN rotation)", || {
        let mut m = Matrix3::identity();
        m[(0, 0)] = NAN;
        let _ = screw_log_so3(&m);
    }));
    rows.push(probe("real_roots (a NaN coefficient)", || {
        let _ = real_roots(&[1.0, 0.0, NAN], 1e-9);
    }));
    rows.push(probe("singular_values (a NaN Jacobian entry)", || {
        let mut j = DMatrix::<f64>::identity(3, 3);
        j[(1, 1)] = NAN;
        let _ = singular_values(&j);
    }));
    rows.push(probe("modal_analysis (a NaN stiffness entry)", || {
        let m = DMatrix::<f64>::identity(3, 3);
        let mut k = DMatrix::<f64>::identity(3, 3);
        k[(2, 2)] = NAN;
        let _ = modal_analysis(&m, &k, 2);
    }));
    rows.push(probe("pca (a NaN point)", || {
        let _ = pca(&[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(NAN, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)]);
    }));
    rows.push(probe("w2_gaussian (a NaN covariance entry)", || {
        let m1 = DVector::from_vec(vec![0.0, 0.0]);
        let m2 = DVector::from_vec(vec![1.0, 0.0]);
        let s1 = DMatrix::<f64>::identity(2, 2);
        let mut s2 = DMatrix::<f64>::identity(2, 2);
        s2[(0, 0)] = NAN;
        let _ = w2_gaussian(&m1, &s1, &m2, &s2);
    }));

    std::panic::set_hook(prev);
    let bad: Vec<&str> = rows.iter().filter(|(_, v)| *v != "returned").map(|(n, _)| *n).collect();
    eprintln!("\n{} entry points probed, {} did not simply return\n", rows.len(), bad.len());
    assert!(bad.is_empty(), "a public entry point must report a non-finite input, not fault or spin on it: {bad:?}");
}

/// The probe above proves only that nothing faults or spins. This asserts what each entry point
/// actually returns, so every guard is covered by a test rather than merely present.
///
/// The policy, applied consistently: `None` where the signature already carries a failure; drop what is
/// independently invalid in a point set; propagate `NaN` where the object is one indivisible
/// mathematical entity and the return type cannot say "no answer".
#[test]
fn each_entry_point_honours_its_documented_non_finite_contract() {
    // --- signature already carries the failure: None ---
    assert!(w1_empirical_1d(&[0.0, 1.0, 2.0], &[0.0, 1.0, 3.0]).is_some(), "control");
    assert!(w1_empirical_1d(&[0.0, 1.0, NAN], &[0.0, 1.0, 2.0]).is_none(), "w1 refuses a non-finite sample");
    assert!(cramer_distance(&[0.0, 1.0, 2.0], &[0.0, 1.0, 3.0]).is_some(), "control");
    assert!(cramer_distance(&[0.0, 1.0, NAN], &[0.0, 1.0, 2.0]).is_none(), "cramer refuses one");
    assert_eq!(time_delay(&[0.0, 1.0, 0.0, 0.0], &[0.0, 0.0, 1.0, 0.0]), Some(1), "control");
    assert!(time_delay(&[0.0, 1.0, NAN, 0.5], &[0.0, 1.0, 0.0, 0.5]).is_none(), "time_delay refuses one");
    assert!(time_delay(&[], &[]).is_none(), "and empty input");

    // --- indivisible object: NaN out ---
    let mut m = Matrix3::identity();
    m[(0, 1)] = NAN;
    assert!(screw_log_so3(&m).iter().all(|v| !v.is_finite()), "log_so3 propagates rather than faulting");
    assert!(screw_log_so3(&Matrix3::identity()).iter().all(|v| v.is_finite()), "control stays finite");
    // This one assertion runs under a watchdog. If the guard regresses, nalgebra's SVD does not
    // return, and a test that HANGS is worse in CI than one that fails: the mutation run that removed
    // the guard had to be killed at 400 s rather than reporting anything.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut j = DMatrix::<f64>::identity(3, 4);
        j[(1, 1)] = NAN;
        let _ = tx.send(singular_values(&j));
    });
    let sv = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("singular_values must RETURN on a non-finite matrix, not spin forever");
    assert_eq!(sv.len(), 3, "one value per singular direction even when unusable");
    assert!(sv.iter().all(|v| !v.is_finite()), "singular_values propagates rather than spinning forever");
    assert!(singular_values(&DMatrix::<f64>::identity(3, 4)).iter().all(|v| v.is_finite()), "control");
    let mut k = DMatrix::<f64>::identity(3, 3);
    k[(2, 2)] = NAN;
    let mm = modal_analysis(&DMatrix::<f64>::identity(3, 3), &k, 2);
    assert_eq!((mm.freq.len(), mm.basis.nrows(), mm.project.ncols()), (2, 3, 3), "shape is preserved");
    assert!(mm.freq.iter().all(|v| !v.is_finite()), "modal_analysis propagates rather than faulting");

    // --- point set: drop what is independently invalid ---
    let clean = [Vector3::new(0.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0)];
    let mut dirty = clean.to_vec();
    dirty.insert(2, Vector3::new(NAN, 5.0, 5.0));
    dirty.push(Vector3::new(0.0, f64::INFINITY, 0.0));
    let (a, b) = (pca(&clean), pca(&dirty));
    assert!((a.centroid - b.centroid).norm() < 1e-12, "the invalid points must not move the centroid");
    assert!((a.variances - b.variances).norm() < 1e-12, "nor the variances");
    assert!(pca(&[Vector3::new(NAN, NAN, NAN)]).centroid.iter().all(|v| !v.is_finite()), "nothing real leaves no axis");
}
