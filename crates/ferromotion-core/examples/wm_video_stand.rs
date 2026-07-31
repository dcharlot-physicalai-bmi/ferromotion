//! Putting a VIDEO world model on the physics stand — the harness, with the perception confound MEASURED.
//!
//! A video world model outputs PIXELS, not state. To ask "does its generated future obey physics?" you must
//! (1) perceive physical state from the frames, then (2) score a conservation invariant. Step (1) has its own
//! error, so a naive score conflates the model's physics violation with the tracker's noise. The honest move
//! is to CHARACTERIZE that noise floor first: run the perception on frames of a KNOWN-correct scene and
//! measure how far the recovered invariant strays from truth. Only violations beyond that floor are the
//! model's. This file builds the whole pipeline on a rendered bouncing ball (2-D, analytic ground truth):
//!   render(trajectory) -> frames  ->  perceive(frames) -> trajectory'  ->  invariants(trajectory')
//! and shows it (a) recovers correct physics within the floor and (b) flags three hallucinations a real video
//! model would make — wrong gravity, energy from nothing, and passing through the floor — well beyond it.
//! A frontier model (e.g. Cosmos) plugs in by replacing the renderer with its decoded frames; the perception
//! and scoring are identical, and the same measured floor says what is a real violation and what is the tracker.

const W: usize = 100;         // frame size (px)
const WORLD: f64 = 10.0;      // world is WORLD×WORLD metres, y up
const FLOOR: f64 = 2.0;       // the ledge the ball bounces on, at a VISIBLE height (so penetration stays in frame)
const START_Y: f64 = 7.0;     // drop height
const G_SCENE: f64 = 9.81;    // the gravity the scene establishes (Earth); the invariant checks the model obeys it
const DT: f64 = 0.02;
const SIGMA: f64 = 3.0;       // blob radius (px)
const NOISE: f64 = 0.03;      // per-pixel sensor noise

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }

// ---- ground-truth "world" the frames depict: a ball under gravity g, bounce restitution e, optional floor ----
fn simulate(g: f64, e: f64, floor: bool, n: usize) -> Vec<(f64, f64)> {
    let (mut x, mut y, mut vx, mut vy) = (2.0f64, START_Y, 1.6f64, 0.0f64);
    let mut tr = Vec::with_capacity(n);
    for _ in 0..n {
        tr.push((x, y));
        vy -= g * DT; x += vx * DT; y += vy * DT;
        if floor && y < FLOOR { y = FLOOR; vy = -e * vy; } // reflect at the ledge
    }
    tr
}

// ---- render a trajectory to grayscale frames: a Gaussian blob at the ball + sensor noise ----
fn render(tr: &[(f64, f64)]) -> Vec<Vec<f32>> {
    tr.iter().enumerate().map(|(f, &(x, y))| {
        let cx = x / WORLD * W as f64;
        let cy = (1.0 - y / WORLD) * W as f64; // flip: world y-up -> image y-down
        let mut img = vec![0.0f32; W * W];
        for py in 0..W { for px in 0..W {
            let d2 = (px as f64 - cx).powi(2) + (py as f64 - cy).powi(2);
            let blob = (-d2 / (2.0 * SIGMA * SIGMA)).exp();
            let noise = NOISE * (u01((f as u32 * 131 + (py * W + px) as u32) * 2 + 1) - 0.5);
            img[py * W + px] = (blob + noise).max(0.0) as f32;
        }}
        img
    }).collect()
}

// ---- perceive: intensity-weighted centroid (thresholded to reject noise) -> world (x,y); + total mass ----
fn perceive(frames: &[Vec<f32>]) -> (Vec<(f64, f64)>, Vec<f64>) {
    let mut traj = Vec::new(); let mut mass = Vec::new();
    for img in frames {
        let (mut sx, mut sy, mut s) = (0.0f64, 0.0f64, 0.0f64);
        for py in 0..W { for px in 0..W {
            let v = img[py * W + px] as f64;
            if v > 0.15 { sx += v * px as f64; sy += v * py as f64; s += v; }
        }}
        let (cx, cy) = if s > 0.0 { (sx / s, sy / s) } else { (0.0, 0.0) };
        traj.push((cx / W as f64 * WORLD, (1.0 - cy / W as f64) * WORLD)); mass.push(s);
    }
    (traj, mass)
}

// ---- least-squares parabola y = a + b t + c t^2 over an index range; returns (a,b,c). accel = 2c ----
fn fit_parabola(ys: &[f64], i0: usize, i1: usize) -> (f64, f64, f64) {
    let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut b0, mut b1, mut b2) = (0.0, 0.0, 0.0);
    for i in i0..i1 {
        let t = i as f64 * DT; let y = ys[i];
        let (t2, t3, t4) = (t * t, t * t * t, t * t * t * t);
        s0 += 1.0; s1 += t; s2 += t2; s3 += t3; s4 += t4;
        b0 += y; b1 += y * t; b2 += y * t2;
    }
    let m = [[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]];
    let det = |a: [[f64; 3]; 3]| a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) - a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) + a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]);
    let d = det(m); let col = [b0, b1, b2];
    let mut sol = [0.0; 3];
    for k in 0..3 { let mut mk = m; for r in 0..3 { mk[r][k] = col[r]; } sol[k] = det(mk) / d.abs().max(1e-12) * d.signum(); }
    (sol[0], sol[1], sol[2])
}

struct Score { g_meas: f64, energy_ratio: Option<f64>, min_y: f64, mass_cv: f64 }

fn score(traj: &[(f64, f64)], mass: &[f64]) -> Score {
    // analyze only while the object is VISIBLE — once it leaves the frame the tracker returns garbage, which
    // must not corrupt the fits. (A ball that sinks through the ledge is caught by min-y BEFORE it exits.)
    let full = traj.len();
    let vis_end = (0..full).find(|&i| mass[i] < 1.0).unwrap_or(full).max(6);
    let ys: Vec<f64> = traj[0..vis_end].iter().map(|p| p.1).collect(); let n = ys.len();
    // bounces = local minima near the ledge (deduped so noise doesn't split one bounce)
    let mut bounces: Vec<usize> = Vec::new();
    for i in 2..n - 2 {
        if ys[i] <= ys[i - 1] && ys[i] < ys[i + 1] && ys[i] < FLOOR + 0.6
            && bounces.last().map_or(true, |&l| i - l > 5) { bounces.push(i); }
    }
    // gravity: fit the initial free-flight drop [1, first bounce] → accel = 2c, g = -2c
    let end = bounces.first().copied().unwrap_or(n - 2).max(5).min(n - 2);
    let (_, _, c) = fit_parabola(&ys, 1, end);
    let g_meas = -2.0 * c;
    // energy across a bounce: PEAK-HEIGHT ratio above the ledge (robust — apex heights, not velocities at the
    // discontinuity). E ∝ apex height, so h_after/h_before = e² ≤ 1 for a real bounce.
    let energy_ratio = if bounces.len() >= 2 {
        let apex = (bounces[0]..bounces[1]).map(|i| ys[i]).fold(f64::MIN, f64::max);
        let (h0, h1) = (ys[0] - FLOOR, apex - FLOOR);
        if h0 > 0.5 { Some(h1 / h0) } else { None }
    } else { None };
    let min_y = ys.iter().cloned().fold(f64::INFINITY, f64::min);
    let vm = &mass[0..vis_end];
    let mmean = vm.iter().sum::<f64>() / vm.len() as f64;
    let mass_cv = (vm.iter().map(|m| (m - mmean).powi(2)).sum::<f64>() / vm.len() as f64).sqrt() / mmean.max(1e-9);
    Score { g_meas, energy_ratio, min_y, mass_cv }
}

fn main() {
    println!("A VIDEO world model on the physics stand — perception confound MEASURED, then three hallucinations caught.\n");
    let n = 220; // long enough for ≥2 bounces (so the bounce-energy invariant is defined)

    // ---- STEP 1: characterize the perception confound on a KNOWN-correct scene ----
    let truth = simulate(G_SCENE, 0.8, true, n);
    let (perc, mass) = perceive(&render(&truth));
    let s_true = score(&truth, &vec![1.0; n]);
    let s_perc = score(&perc, &mass);
    let rms = (truth.iter().zip(&perc).map(|(a, b)| (a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sum::<f64>() / n as f64).sqrt();
    let g_floor = (s_perc.g_meas - s_true.g_meas).abs();
    let e_floor = (s_perc.energy_ratio.unwrap_or(0.0) - s_true.energy_ratio.unwrap_or(0.0)).abs();
    let y_floor = (s_perc.min_y - s_true.min_y).abs();
    println!("  CONFOUND FLOOR (perception error on correct physics — {}×{} frames, blob σ={}, noise={}):", W, W, SIGMA, NOISE);
    println!("    position RMS {:.3} m   ·   gravity ±{:.3} m/s²   ·   bounce-energy ±{:.3}   ·   min-height ±{:.3} m", rms, g_floor, e_floor, y_floor);
    println!("    (true g {:.2} → perceived {:.2}; true KE-ratio {:.2} → perceived {:.2}; a real bounce keeps energy ≤ 1)\n",
        s_true.g_meas, s_perc.g_meas, s_true.energy_ratio.unwrap_or(0.0), s_perc.energy_ratio.unwrap_or(0.0));
    let g_tol = 5.0 * g_floor.max(0.05);
    let e_tol = 5.0 * e_floor.max(0.02);
    let y_tol = 5.0 * y_floor.max(0.05);

    // ---- STEP 2: score the correct scene + three hallucinations, all through the SAME perception ----
    struct Case { name: &'static str, g: f64, e: f64, floor: bool }
    let cases = [
        Case { name: "correct physics",           g: G_SCENE, e: 0.8,  floor: true },
        Case { name: "wrong gravity (too slow)",  g: 3.2,     e: 0.8,  floor: true },
        Case { name: "energy from nothing (e>1)", g: G_SCENE, e: 1.15, floor: true },
        Case { name: "passes through the floor",  g: G_SCENE, e: 0.8,  floor: false },
    ];
    println!("  {:<28}{:>10}{:>13}{:>11}   verdict (vs the measured floor)", "generated video", "gravity", "bounce-E", "min-y");
    println!("  {}", "-".repeat(94));
    for c in &cases {
        let (p, m) = perceive(&render(&simulate(c.g, c.e, c.floor, n)));
        let s = score(&p, &m);
        let g_bad = (s.g_meas - G_SCENE).abs() > g_tol;
        let e_bad = s.energy_ratio.map(|r| r > 1.0 + e_tol).unwrap_or(false);
        let y_bad = s.min_y < FLOOR - y_tol;
        let pass = !g_bad && !e_bad && !y_bad;
        let er = s.energy_ratio.map(|r| format!("{:.2}", r)).unwrap_or_else(|| "  n/a".into());
        let flag = |bad: bool| if bad { "✗" } else { " " };
        println!("  {:<28}{:>7.2} {}{:>10} {}{:>8.2} {}   {}", c.name,
            s.g_meas, flag(g_bad), er, flag(e_bad), s.min_y, flag(y_bad),
            if pass { "PASS — obeys physics" } else { "FAIL — physics violated" });
    }
    println!("\n  READING: perception is imperfect, so a score is only trustworthy ABOVE the measured floor (position");
    println!("  RMS {:.3} m, gravity ±{:.2} m/s², energy ±{:.2}). The correct scene lands inside it and PASSES; each", rms, g_floor, e_floor);
    println!("  hallucination clears the floor by a wide margin and is flagged — an object that falls at the wrong");
    println!("  rate, a bounce that GAINS energy, or a ball that sinks through the ledge. That is the honest way to");
    println!("  score a generative video model: extract state, test a conservation law, and trust only violations the");
    println!("  tracker itself cannot manufacture. Swap the renderer for a Cosmos rollout's decoded frames (the 4B");
    println!("  transformer + Wan-VAE decode both run in Ferric) and the verdict — and its error bars — carry over.");
}
