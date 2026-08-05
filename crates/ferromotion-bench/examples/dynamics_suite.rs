//! **The dynamics hot paths, measured** — with the scaling exponents that the algorithm claims imply.
//!
//! A timing table is a weak claim: it says how fast this machine ran this code today. The stronger claim, and the one
//! that survives a hardware change, is the **scaling exponent**. ABA is advertised as `O(n)` in joint count and CRBA
//! as `O(n^2)`, so the suite runs each at several robot sizes and fits the exponent, which either confirms the
//! implementation realises its algorithm or shows it does not.

use ferromotion_bench::{format_time, Bench, Suite};
use ferromotion_cloth::vbd::{hanging_chain, VbdSolver};
use ferromotion_core::{crba, forward_dynamics_aba, from_urdf_full, saltation_matrix, solve_contacts_pgs, LinkInertia, PgsContact, Robot};
use nalgebra::{DMatrix, DVector, Vector3};

/// An `n`-revolute serial arm, so the same code is measured at several sizes.
fn arm(n: usize) -> (Robot, Vec<LinkInertia>) {
    let mut urdf = String::from("<robot name=\"a\"><link name=\"l0\"/>");
    for i in 1..=n {
        urdf += &format!(
            "<link name=\"l{i}\"><inertial><mass value=\"1.5\"/><origin xyz=\"0 0 0.15\"/>\
             <inertia ixx=\"0.02\" iyy=\"0.02\" izz=\"0.006\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/></inertial></link>"
        );
    }
    for i in 1..=n {
        // alternate axes so the chain is not degenerate
        let axis = ["0 0 1", "0 1 0", "1 0 0"][(i - 1) % 3];
        urdf += &format!(
            "<joint name=\"j{i}\" type=\"revolute\"><parent link=\"l{}\"/><child link=\"l{i}\"/>\
             <origin xyz=\"0 0 0.25\"/><axis xyz=\"{axis}\"/>\
             <limit lower=\"-3\" upper=\"3\" effort=\"50\" velocity=\"5\"/></joint>",
            i - 1
        );
    }
    urdf += "</robot>";
    from_urdf_full(&urdf, "l0", &format!("l{n}")).expect("generated urdf parses")
}

/// Local exponent between the two largest sizes — the asymptotic one, and the only one not contaminated by the fixed
/// per-call cost that dominates at small `n`. The first version of this suite fitted the whole range and reported CRBA
/// at 1.427 against a claimed 2.0; the local exponents were rising (1.45 then 1.59), which is the signature of
/// overhead at the small end rather than a wrong algorithm.
fn tail_exponent(points: &[(usize, f64)]) -> f64 {
    let n = points.len();
    if n < 2 {
        return f64::NAN;
    }
    let ((k0, t0), (k1, t1)) = (points[n - 2], points[n - 1]);
    (t1 / t0).ln() / (k1 as f64 / k0 as f64).ln()
}

/// Least-squares slope of `log(time)` against `log(n)` over the whole range.
fn exponent(points: &[(usize, f64)]) -> f64 {
    let n = points.len() as f64;
    let (sx, sy): (f64, f64) = points.iter().map(|(k, t)| ((*k as f64).ln(), t.ln())).fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    let (mx, my) = (sx / n, sy / n);
    let num: f64 = points.iter().map(|(k, t)| ((*k as f64).ln() - mx) * (t.ln() - my)).sum();
    let den: f64 = points.iter().map(|(k, _)| ((*k as f64).ln() - mx).powi(2)).sum();
    num / den
}

fn main() {
    let bench = Bench::default();
    const SIZES: [usize; 6] = [6, 12, 24, 48, 96, 192];

    // --- articulated-body algorithm: claimed O(n)
    let mut suite = Suite::default();
    let mut aba_points = Vec::new();
    let mut crba_points = Vec::new();
    for n in SIZES {
        let (robot, inertia) = arm(n);
        let q: Vec<f64> = (0..n).map(|i| 0.1 * (i as f64).sin()).collect();
        let qd: Vec<f64> = (0..n).map(|i| 0.05 * (i as f64).cos()).collect();
        let tau = vec![0.3; n];
        let g = Vector3::new(0.0, 0.0, -9.81);

        let m = bench.run_value(|| forward_dynamics_aba(&robot, &inertia, &q, &qd, &tau, g));
        suite.push(format!("aba forward dynamics ({n} dof)"), m);
        aba_points.push((n, m.median));

        let m = bench.run_value(|| crba(&robot, &inertia, &q));
        suite.push(format!("crba mass matrix ({n} dof)"), m);
        crba_points.push((n, m.median));
    }

    // --- contact solve, scaling in contact count
    let mut pgs_points = Vec::new();
    for nc in [4usize, 8, 16, 32] {
        let dof = 12;
        let mass = DMatrix::identity(dof, dof) * 2.0;
        let v_free = DVector::from_fn(dof, |i, _| 0.1 * (i as f64) - 0.5);
        let contacts: Vec<PgsContact> = (0..nc)
            .map(|i| PgsContact { j: DMatrix::from_fn(3, dof, |r, c| if (r + c + i) % 4 == 0 { 1.0 } else { 0.0 }), phi: -1e-5, mu: 0.7 })
            .collect();
        let m = bench.run_value(|| solve_contacts_pgs(&mass, &v_free, &contacts, 1e-3, 40, None));
        suite.push(format!("contact pgs solve ({nc} contacts, 12 dof)"), m);
        pgs_points.push((nc, m.median));
    }

    // --- the deformable solver, and the saltation matrix
    for links in [16usize, 64, 256] {
        let solver = VbdSolver::new(1.0 / 60.0, 1).unwrap();
        let (verts, springs) = hanging_chain(links, 0.05, 0.02, 1e5);
        let m = bench.run(|| {
            let mut v = verts.clone();
            solver.step(&mut v, &springs);
        });
        suite.push(format!("vbd step, 1 sweep ({links} links)"), m);
    }

    let reset = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, -0.6]);
    let normal = DVector::from_row_slice(&[1.0, 0.0]);
    let fm = DVector::from_row_slice(&[-3.0, -9.81]);
    let fp = DVector::from_row_slice(&[1.8, -9.81]);
    suite.push("saltation matrix (2x2)", bench.run_value(|| saltation_matrix(&reset, &normal, &fm, &fp)));

    suite.print("ferromotion dynamics suite");

    // --- the claim that outlives the hardware
    println!("\nmeasured scaling exponents");
    println!("  {:<32} {:>8}  {:>10}  {:>10}  verdict", "algorithm", "claimed", "full range", "asymptotic");
    for (name, claimed, points) in [
        ("aba forward dynamics, in dof", 1.0, &aba_points),
        ("crba mass matrix, in dof", 2.0, &crba_points),
        // Gauss-Seidel at a fixed iteration count and a fixed dof is LINEAR in contacts: each sweep touches each
        // contact once and does O(dof) work. The quadratic cost would be forming a dense Delassus operator, which
        // this solver does not do. An earlier version of this suite claimed 2.0 here and that was simply wrong.
        ("contact pgs, in contact count", 1.0, &pgs_points),
    ] {
        let (full, tail) = (exponent(points), tail_exponent(points));
        let ok = (tail - claimed).abs() < 0.3;
        println!("  {name:<32} {claimed:>8.1}  {full:>10.3}  {tail:>10.3}  {}", if ok { "consistent" } else { "CHECK" });
        for (k, t) in points {
            println!("      {k:>5} -> {}", format_time(*t));
        }
    }
    println!("\n  The two exponent columns differ because a fixed per-call cost dominates the small end. The");
    println!("  asymptotic column is the one that tests the algorithm; the full-range column is what a careless");
    println!("  sweep would report, and for CRBA it reads 30% low.");
    println!("\n  An exponent is the part of a benchmark that survives a hardware change. A table of medians says what");
    println!("  this machine did today; an exponent says whether the implementation realises the algorithm it claims.");
}
