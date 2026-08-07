//! **Is the soft-finger torsional generator's premise true?**
//!
//! `primitive_wrenches_spatial` documents its torsional generator as "a pure moment about the contact normal, bounded
//! by the normal force which is unit by construction here". This measures whether the normal force IS unit by
//! construction, by reading it off the generators the function actually emits.
//!
//! The cone-edge forces are built as `(n + mu*t).normalize()`, so their TOTAL magnitude is one. Their NORMAL component
//! is whatever that normalisation leaves. If the two differ, the torsional generator is scaled against a normal force
//! no cone edge delivers, and the mix of conventions inflates the torsional capacity of a soft finger.
//!
//!   cargo run -p ferromotion-core --example soft_finger_normal_force --release

use ferromotion_core::{force_closure_q1_spatial, grasp_matrix, primitive_wrenches_spatial, wrench_rank, GraspContact3};
use nalgebra::Vector3;

/// Three contacts around a unit-radius object, NOT coplanar, so the grasp can actually reach rank 6 and Q1 > 0.
/// A flat grasp returns exactly zero at every mu and would hide any change.
fn tetra_grasp(mu: f64, mu_torsion: f64) -> Vec<GraspContact3> {
    let dirs = [
        Vector3::new(1.0, 0.0, -0.35),
        Vector3::new(-0.5, 0.866, -0.35),
        Vector3::new(-0.5, -0.866, -0.35),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    dirs
        .iter()
        .map(|d| {
            let p = d.normalize();
            GraspContact3 { pos: p, normal: -p, mu, mu_torsion }
        })
        .collect()
}

fn main() {
    println!("== 1. What is the normal component of each cone-edge force? ==");
    println!("   mu     |f|        f.n        1/sqrt(1+mu^2)   sqrt(1+mu^2)");
    for mu in [0.0f64, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] {
        let c = GraspContact3 { pos: Vector3::new(1.0, 0.0, 0.0), normal: -Vector3::x(), mu, mu_torsion: 0.0 };
        let ws = primitive_wrenches_spatial(std::slice::from_ref(&c), 24);
        let n = c.normal.normalize();
        // every cone edge shares the same normal component by symmetry; take the extremes to prove it
        let mut fmag_lo = f64::INFINITY;
        let mut fmag_hi: f64 = 0.0;
        let mut fn_lo = f64::INFINITY;
        let mut fn_hi = f64::NEG_INFINITY;
        for w in &ws {
            let f = Vector3::new(w[0], w[1], w[2]);
            if f.norm() == 0.0 {
                continue; // a torsional generator has zero force; none here since mu_torsion = 0
            }
            fmag_lo = fmag_lo.min(f.norm());
            fmag_hi = fmag_hi.max(f.norm());
            fn_lo = fn_lo.min(f.dot(&n));
            fn_hi = fn_hi.max(f.dot(&n));
        }
        let predicted = 1.0 / (1.0 + mu * mu).sqrt();
        println!(
            "   {mu:4.2}   {fmag_lo:.6}   {fn_lo:.6}   {predicted:.6}         {:.6}   (|f| spread {:.1e}, f.n spread {:.1e})",
            (1.0 + mu * mu).sqrt(),
            fmag_hi - fmag_lo,
            fn_hi - fn_lo
        );
    }

    println!();
    println!("== 2. The torsional generator's magnitude vs the normal force actually available ==");
    println!("   The generator is  mu_torsion * n, i.e. scaled as though f.n = 1.");
    println!("   mu     f.n at a cone edge   generator/(mu_torsion * f.n)   overstatement");
    for mu in [0.25f64, 0.5, 0.75, 1.0, 1.5, 2.0] {
        let fn_edge = 1.0 / (1.0 + mu * mu).sqrt();
        let ratio = 1.0 / fn_edge;
        println!("   {mu:4.2}   {fn_edge:.6}             {ratio:.6}                      {:.2}x", ratio);
    }

    println!();
    println!("== 3. Does it change a Q1 that is not identically zero? ==");
    println!("   Non-coplanar 4-contact grasp, 24 facets, 20000 directions.");
    println!("   mu    mu_tors   rank   Q1 (as shipped)   Q1 (torsion scaled by f.n)   ratio");
    for mu in [0.3f64, 0.5, 1.0, 1.5] {
        for mt in [0.0, 0.1, 0.3] {
            let cs = tetra_grasp(mu, mt);
            let shipped = force_closure_q1_spatial(&cs, 24, 20_000);
            let rank = wrench_rank(&cs, 24);
            // the consistent version: emit torsion scaled by the normal force a unit-total-force cone edge delivers
            let scaled = q1_with_scaled_torsion(&cs, 24, 20_000, 1.0 / (1.0 + mu * mu).sqrt());
            let ratio = if scaled > 0.0 { shipped / scaled } else { f64::NAN };
            println!("   {mu:3.1}   {mt:5.2}     {rank}      {shipped:.6}          {scaled:.6}               {ratio:.4}");
        }
    }

    println!();
    println!("== 4. Is the mixed convention visible in the grasp matrix columns? ==");
    let cs = tetra_grasp(1.5, 0.3);
    let g = grasp_matrix(&cs, 24);
    let mut force_cols = 0usize;
    let mut torsion_cols = 0usize;
    for j in 0..g.ncols() {
        let f = Vector3::new(g[(0, j)], g[(1, j)], g[(2, j)]);
        if f.norm() > 1e-12 {
            force_cols += 1;
        } else {
            torsion_cols += 1;
        }
    }
    println!("   mu = 1.5, mu_torsion = 0.3: {force_cols} force columns, {torsion_cols} torsional columns");
    println!("   a force column carries normal force {:.6}", 1.0 / (1.0 + 1.5 * 1.5f64).sqrt());
    println!("   a torsional column is scaled as though it carried normal force 1.000000");
}

/// Recompute Q1 with the torsional generators scaled by `fn_edge`, the normal force a unit-total-force cone edge
/// actually delivers. Rebuilt here rather than in the crate, so the shipped behaviour is measured, not replaced.
fn q1_with_scaled_torsion(contacts: &[GraspContact3], facets: usize, dirs: usize, fn_edge: f64) -> f64 {
    let scaled: Vec<GraspContact3> =
        contacts.iter().map(|c| GraspContact3 { mu_torsion: c.mu_torsion * fn_edge, ..*c }).collect();
    force_closure_q1_spatial(&scaled, facets, dirs)
}
