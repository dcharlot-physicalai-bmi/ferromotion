//! **Why `ferromotion_core::numerics` exists: which nalgebra routines survive a `NaN`.**
//!
//! This records an UPSTREAM behaviour our guards depend on. The dynamically sized SVD does not
//! terminate on a matrix holding a `NaN`, while the fixed-size one and every other decomposition
//! return. If a future nalgebra makes the dynamic SVD terminate, this test fails and tells us the
//! guards in `numerics.rs` could be relaxed. Each call runs behind a 3 s watchdog, so the suite
//! reports instead of hanging.
use nalgebra::{DMatrix, Matrix3};
fn watch<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> &'static str {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || { let _ = tx.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_ok()); });
    match rx.recv_timeout(std::time::Duration::from_secs(3)) { Ok(true) => "returned", Ok(false) => "panicked", Err(_) => "HUNG" }
}
#[test]
fn which_nalgebra_routines_survive_a_nan() {
    let h = std::panic::take_hook(); std::panic::set_hook(Box::new(|_| {}));
    let mk = |n: usize| { let mut m = DMatrix::<f64>::identity(n, n); m[(0, 0)] = f64::NAN; m };
    let rows = [
        ("DMatrix 3x3 .singular_values()", watch(move || { let _ = mk(3).singular_values(); })),
        ("DMatrix 6x6 .singular_values()", watch(move || { let _ = mk(6).singular_values(); })),
        ("DMatrix 3x3 .svd(true,true)", watch(move || { let _ = mk(3).svd(true, true); })),
        ("Matrix3 .svd(true,true)", watch(move || { let mut m = Matrix3::<f64>::identity(); m[(0,0)] = f64::NAN; let _ = m.svd(true, true); })),
        ("DMatrix 3x3 .symmetric_eigen()", watch(move || { let _ = mk(3).symmetric_eigen(); })),
        ("DMatrix 3x3 .cholesky()", watch(move || { let _ = mk(3).cholesky(); })),
        ("DMatrix 3x3 .try_inverse()", watch(move || { let _ = mk(3).try_inverse(); })),
        ("DMatrix 3x3 .lu().solve()", watch(move || { let m = mk(3); let b = nalgebra::DVector::from_element(3, 1.0); let _ = m.lu().solve(&b); })),
        ("DMatrix 3x3 .qr()", watch(move || { let _ = mk(3).qr(); })),
    ];
    std::panic::set_hook(h);
    eprintln!("\n=== nalgebra on a matrix holding one NaN ===");
    for (n, v) in &rows { eprintln!("  {n:<34} {v}"); }
    eprintln!();

    let verdict = |name: &str| rows.iter().find(|(n, _)| *n == name).map(|(_, v)| *v).expect("probe present");
    // The hazard the guards exist for. If any of these starts returning, numerics.rs can be relaxed.
    for name in ["DMatrix 3x3 .singular_values()", "DMatrix 6x6 .singular_values()", "DMatrix 3x3 .svd(true,true)"] {
        assert_eq!(verdict(name), "HUNG", "{name} no longer hangs on a NaN: the guards in ferromotion_core::numerics may now be unnecessary, and this record needs updating");
    }
    // The exemptions the static gate allows. If any of these starts hanging, the exemptions are unsafe.
    for name in ["Matrix3 .svd(true,true)", "DMatrix 3x3 .symmetric_eigen()", "DMatrix 3x3 .cholesky()", "DMatrix 3x3 .try_inverse()", "DMatrix 3x3 .lu().solve()", "DMatrix 3x3 .qr()"] {
        assert_ne!(verdict(name), "HUNG", "{name} now hangs on a NaN: any FIXED-SIZE SVD exemption or unguarded use of it is no longer safe");
    }
}
