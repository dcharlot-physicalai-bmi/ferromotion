//! **R7.1 / M1 — is a sampled funnel a sound funnel?**
//!
//! The composition theorem takes the inflow and outflow sets as given. For a learned skill nobody hands you those
//! sets; you *measure* them, by running the skill from sampled starts and taking the box that holds the results.
//! This example asks what that measurement costs, on a 4-DOF reach skill with torque saturation.
//!
//! The claim under test: uniform sampling of the inflow box gives a sound outflow certificate.

use ferromotion_control::{compose_chain, Region, SkillFunnel, Xorshift};

const DIM: usize = 4;
const HORIZON: usize = 12;
const DT: f64 = 0.05;
const GAIN: f64 = 6.0;
const SAT: f64 = 1.2; // torque limit - this is what makes the funnel non-uniform

/// A saturated proportional reach: drive every coordinate toward `target` for a fixed horizon.
fn run_skill(start: &[f64], target: &[f64]) -> Vec<f64> {
    let mut x = start.to_vec();
    for _ in 0..HORIZON {
        for i in 0..DIM {
            x[i] += (GAIN * (target[i] - x[i])).clamp(-SAT, SAT) * DT;
        }
    }
    x
}

/// The tightest box holding every image of `samples`.
fn bounding_box(images: &[Vec<f64>]) -> Region {
    let mut lo = vec![f64::INFINITY; DIM];
    let mut hi = vec![f64::NEG_INFINITY; DIM];
    for img in images {
        for i in 0..DIM {
            lo[i] = lo[i].min(img[i]);
            hi[i] = hi[i].max(img[i]);
        }
    }
    Region::new(lo, hi).expect("non-empty")
}

/// Uniform interior samples of a box, which is how a basin normally gets estimated.
fn uniform_samples(region: &Region, n: usize, rng: &mut Xorshift) -> Vec<Vec<f64>> {
    (0..n).map(|_| (0..DIM).map(|i| region.lo[i] + rng.uniform() * (region.hi[i] - region.lo[i])).collect()).collect()
}

/// Every corner of a box: 2^DIM points, and the only ones saturation can hurt most.
fn corners(region: &Region) -> Vec<Vec<f64>> {
    (0..1usize << DIM).map(|m| (0..DIM).map(|i| if m >> i & 1 == 1 { region.hi[i] } else { region.lo[i] }).collect()).collect()
}

fn width(r: &Region) -> f64 {
    (0..DIM).map(|i| r.hi[i] - r.lo[i]).fold(0.0, f64::max)
}

fn main() {
    let mut rng = Xorshift::new(0x5EED_1234);
    let inflow = Region::new(vec![-1.0; DIM], vec![1.0; DIM]).unwrap();
    let target = vec![0.0; DIM];

    println!("R7.1 / M1 - measuring a skill's outflow set");
    println!("  4-DOF saturated reach, horizon {HORIZON}, gain {GAIN}, torque limit {SAT}");
    println!("  inflow = [-1, 1]^4, target = origin\n");

    // --- route A: uniform interior sampling, at several budgets
    println!("  route A: uniform samples of the inflow box");
    println!("    {:>8}  {:>12}  {:>12}", "samples", "outflow width", "max |x|inf");
    let mut sampled: Vec<(usize, Region)> = Vec::new();
    for n in [50usize, 200, 1_000, 10_000, 100_000] {
        let pts = uniform_samples(&inflow, n, &mut rng);
        let imgs: Vec<Vec<f64>> = pts.iter().map(|p| run_skill(p, &target)).collect();
        let out = bounding_box(&imgs);
        let far = imgs.iter().map(|v| v.iter().fold(0.0f64, |a, b| a.max(b.abs()))).fold(0.0, f64::max);
        println!("    {n:>8}  {:>12.6}  {far:>12.6}", width(&out));
        sampled.push((n, out));
    }

    // --- route B: the corners, where saturation binds hardest
    let corner_pts = corners(&inflow);
    let corner_imgs: Vec<Vec<f64>> = corner_pts.iter().map(|p| run_skill(p, &target)).collect();
    let corner_out = bounding_box(&corner_imgs);
    println!("\n  route B: the {} corners of the inflow box", corner_pts.len());
    println!("    outflow width {:.6}, max |x|inf {:.6}", width(&corner_out), corner_imgs.iter().map(|v| v.iter().fold(0.0f64, |a, b| a.max(b.abs()))).fold(0.0, f64::max));

    // --- is the sampled certificate sound?
    println!("\n  does each sampled outflow contain the corner images?");
    for (n, out) in &sampled {
        let escapes = corner_imgs.iter().filter(|img| !out.contains(img)).count();
        let deficit = corner_out.overhang(out).iter().fold(0.0f64, |a, b| a.max(*b));
        println!("    n = {n:>6}: {escapes:>2} of {} corner images outside the certificate, worst deficit {deficit:.6}", corner_imgs.len());
    }

    // --- held-out soundness: fresh points the certificate never saw
    println!("\n  held-out soundness: 200000 FRESH inflow points vs each certificate");
    println!("    {:>14}  {:>10}  {:>12}  {:>12}", "certificate", "escapes", "escape rate", "worst escape");
    for (label, out) in [("sampled n=50", &sampled[0].1), ("sampled n=200", &sampled[1].1), ("sampled n=1000", &sampled[2].1), ("sampled n=100000", &sampled[4].1), ("corners", &corner_out)] {
        let mut esc = 0usize;
        let mut worst = 0.0f64;
        for _ in 0..200_000 {
            let p: Vec<f64> = (0..DIM).map(|i| inflow.lo[i] + rng.uniform() * (inflow.hi[i] - inflow.lo[i])).collect();
            let img = run_skill(&p, &target);
            if !out.contains(&img) {
                esc += 1;
                worst = worst.max((0..DIM).map(|i| (out.lo[i] - img[i]).max(img[i] - out.hi[i]).max(0.0)).fold(0.0, f64::max));
            }
        }
        println!("    {label:>14}  {esc:>10}  {:>11.4}%  {worst:>12.2e}", 100.0 * esc as f64 / 200_000.0);
    }
    println!("    (the corner bound is the only one nothing escapes - and 200000 held-out points confirm");
    println!("     the worst case for this monotone saturated map really is a corner, not merely assumed to be)");

    // --- the consequence for a chain. Size the downstream skill to exactly what the upstream certificate
    // promises, which is the natural engineering choice: its inflow is the measured outflow, coordinate for
    // coordinate. Then the composition is structurally perfect by construction, and every handoff the real
    // system gets wrong is a handoff the certificate said could not happen.
    println!("\n  chain: a downstream skill sized to exactly what each certificate promises");
    println!("    {:>16}  {:>10}  {:>14}  {:>16}", "upstream cert", "composes", "reported P(task)", "real breach rate");
    for (label, out) in [("sampled n=50", &sampled[0].1), ("sampled n=200", &sampled[1].1), ("sampled n=1000", &sampled[2].1), ("sampled n=100000", &sampled[4].1), ("corners", &corner_out)] {
        let chain = vec![
            SkillFunnel { name: "reach", inflow: inflow.clone(), outflow: out.clone(), reliability: 0.97, detection: 0.85 },
            SkillFunnel { name: "insert", inflow: out.clone(), outflow: Region::new(vec![-0.05; DIM], vec![0.05; DIM]).unwrap(), reliability: 0.97, detection: 0.85 },
        ];
        let verdict = compose_chain(&chain);
        let mut breaches = 0usize;
        for _ in 0..200_000 {
            let p: Vec<f64> = (0..DIM).map(|i| inflow.lo[i] + rng.uniform() * (inflow.hi[i] - inflow.lo[i])).collect();
            if !out.contains(&run_skill(&p, &target)) {
                breaches += 1;
            }
        }
        let rate = 100.0 * breaches as f64 / 200_000.0;
        match verdict {
            Ok(c) => println!("    {label:>16}  {:>10}  {:>14.4}  {rate:>15.4}%", "yes", c.with_detection),
            Err(e) => println!("    {label:>16}  {:>10}  {:>14}  {rate:>15.4}%", "no", format!("break at {}", e.at)),
        }
    }
    println!("\n    Every row composes, and every row reports the same 0.9908. The reliability arithmetic never sees");
    println!("    the sampling error, because the sets are its inputs, not its outputs. A 50-sample basin estimate");
    println!("    and a sound one produce identical certificates that differ by 15 percentage points in what the");
    println!("    hardware does. The binding uncertainty is the confidence interval on the SET, not on the rate,");
    println!("    and the composition theorem has no slot for it.");

    // --- the missing slot, as a law. A sampled bounding box in d dimensions has 2d faces, and a fresh point
    // exceeds the max of n iid samples along one direction with probability 1/(n+1). So the escape rate of a
    // sampled set-certificate is about 2d/(n+1) - independent of the dynamics entirely.
    println!("\n  the missing slot: how a sampled set-certificate escapes, measured over an ENSEMBLE of boxes");
    println!("    (one realisation is high-variance because a bounding box is a maximum; 400 boxes each)");
    println!("    {:>4}  {:>8}  {:>14}  {:>14}  {:>7}", "dim", "n", "measured rate", "2d/(n+1)", "ratio");
    for dim in [2usize, 4, 8] {
        for n in [50usize, 200, 1_000, 5_000] {
            let mut total = 0.0f64;
            for _ in 0..400 {
                // sample a box from n images, then estimate its escape rate on fresh points
                let mut lo = vec![f64::INFINITY; dim];
                let mut hi = vec![f64::NEG_INFINITY; dim];
                for _ in 0..n {
                    for k in 0..dim {
                        // the image coordinate; the law is dynamics-independent so any fixed map works
                        let v = (0..HORIZON).fold(2.0 * rng.uniform() - 1.0, |x, _| x + (GAIN * (0.0 - x)).clamp(-SAT, SAT) * DT);
                        lo[k] = lo[k].min(v);
                        hi[k] = hi[k].max(v);
                    }
                }
                let mut esc = 0usize;
                const FRESH: usize = 400;
                for _ in 0..FRESH {
                    let mut out = false;
                    for k in 0..dim {
                        let v = (0..HORIZON).fold(2.0 * rng.uniform() - 1.0, |x, _| x + (GAIN * (0.0 - x)).clamp(-SAT, SAT) * DT);
                        if v < lo[k] || v > hi[k] {
                            out = true;
                        }
                    }
                    if out {
                        esc += 1;
                    }
                }
                total += esc as f64 / FRESH as f64;
            }
            let measured = total / 400.0;
            let predicted = 2.0 * dim as f64 / (n as f64 + 1.0);
            println!("    {dim:>4}  {n:>8}  {:>13.4}%  {:>13.4}%  {:>7.3}", 100.0 * measured, 100.0 * predicted, measured / predicted);
        }
    }
    println!("\n    The rate is 2d/(n+1) and has nothing to do with the dynamics: it is a property of taking a");
    println!("    bounding box from finitely many samples. That gives the slot the composition theorem lacks -");
    println!("    to certify a handoff at rate eps you need n = 2d/eps samples of the skill's image.");
    println!("    d = 4 at 0.1%: 8000 rollouts. A 30-DOF humanoid (d = 60) at 0.1%: 120000 rollouts per skill.");
    println!("    Linear in dimension, which is the good news; the 2^d corners of the exact route are the bad.");
}
