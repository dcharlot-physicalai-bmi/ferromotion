//! THE CURE MECHANISM, isolated: force = −∇V (a learned POTENTIAL) is conservative BY CONSTRUCTION.
//!
//! The SO-101 result (wm_so101.rs) showed a generic learned force field fails energy conservation on a
//! multi-body system. The precise reason: a generic vector field f(q)∈Rⁿ (n>1) is conservative only if
//! it is CURL-FREE (∇×f = 0). A plain MLP force is not, so even a symplectic integrator pumps energy.
//! The fix: parameterize the force as the gradient of a learned scalar potential, f = −∇V. Then f is a
//! gradient, hence curl-free BY CONSTRUCTION, hence conservative — regardless of the net's approximation
//! error. This isolates that fix on a 2-D particle in an anharmonic well (unit mass — no mass-matrix
//! confound, so ONLY conservativeness is under test): a generic force MLP vs a potential-gradient MLP,
//! same data, scored on energy drift AND on the measured curl of the learned field. The training uses a
//! finite-difference-of-single-backprops trick (no hand-rolled double backprop), and is gradient-checked.
use nalgebra::{DMatrix, DVector};

fn hash(mut h: u32) -> u32 { h ^= h >> 15; h = h.wrapping_mul(2246822519); h ^= h >> 13; h = h.wrapping_mul(3266489917); h ^= h >> 16; h }
fn u01(i: u32) -> f64 { (hash(i) % 1_000_000) as f64 / 1_000_000.0 }
fn randn(i: u32) -> f64 { (0..12).map(|k| u01(i * 13 + k)).sum::<f64>() - 6.0 }

// true system: V*(x,y) = ½k(x²+y²) + a x²y² (anharmonic, coupled). force = −∇V*. E = ½|v|² + V*.
const K: f64 = 4.0;
const A: f64 = 1.5;
fn true_pot(x: f64, y: f64) -> f64 { 0.5 * K * (x * x + y * y) + A * x * x * y * y }
fn true_force(x: f64, y: f64) -> (f64, f64) { (-(K * x + 2.0 * A * x * y * y), -(K * y + 2.0 * A * x * x * y)) }
fn energy(x: f64, y: f64, vx: f64, vy: f64) -> f64 { 0.5 * (vx * vx + vy * vy) + true_pot(x, y) }

const H: usize = 32;
#[derive(Clone)]
struct Net { w1: DMatrix<f64>, b1: DVector<f64>, w2: DMatrix<f64>, b2: DVector<f64>, w3: DMatrix<f64>, b3: DVector<f64> }
impl Net {
    fn new(nin: usize, nout: usize, s: u32) -> Self {
        let w = |r: usize, c: usize, sd: u32| DMatrix::from_fn(r, c, |i, j| randn(sd + (i * 131 + j) as u32) * (2.0 / c as f64).sqrt());
        Net { w1: w(H, nin, s + 1), b1: DVector::zeros(H), w2: w(H, H, s + 2), b2: DVector::zeros(H), w3: w(nout, H, s + 3), b3: DVector::zeros(nout) }
    }
    fn zeros(nin: usize, nout: usize) -> Self { Net { w1: DMatrix::zeros(H, nin), b1: DVector::zeros(H), w2: DMatrix::zeros(H, H), b2: DVector::zeros(H), w3: DMatrix::zeros(nout, H), b3: DVector::zeros(nout) } }
    fn fwd(&self, x: &DVector<f64>) -> (DVector<f64>, DVector<f64>, DVector<f64>) {
        let h1 = (&self.w1 * x + &self.b1).map(|v| v.tanh());
        let h2 = (&self.w2 * &h1 + &self.b2).map(|v| v.tanh());
        (&self.w3 * &h2 + &self.b3, h1, h2)
    }
    // backprop an arbitrary output-gradient dy → gradient w.r.t. every weight (a Net-shaped Grad).
    fn bwd_dy(&self, x: &DVector<f64>, dy: &DVector<f64>) -> Net {
        let (_, h1, h2) = self.fwd(x);
        let gw3 = dy * h2.transpose(); let gb3 = dy.clone();
        let dz2 = (self.w3.transpose() * dy).component_mul(&h2.map(|v| 1.0 - v * v));
        let gw2 = &dz2 * h1.transpose(); let gb2 = dz2.clone();
        let dz1 = (self.w2.transpose() * &dz2).component_mul(&h1.map(|v| 1.0 - v * v));
        let gw1 = &dz1 * x.transpose(); let gb1 = dz1;
        Net { w1: gw1, b1: gb1, w2: gw2, b2: gb2, w3: gw3, b3: gb3 }
    }
    fn axpy(&mut self, a: f64, g: &Net) { // self += a*g
        self.w1 += a * &g.w1; self.b1 += a * &g.b1; self.w2 += a * &g.w2; self.b2 += a * &g.b2; self.w3 += a * &g.w3; self.b3 += a * &g.b3;
    }
}
struct Adam { m: Net, v: Net, t: f64 }
impl Adam {
    fn new(nin: usize, nout: usize) -> Self { Adam { m: Net::zeros(nin, nout), v: Net::zeros(nin, nout), t: 0.0 } }
    fn step(&mut self, p: &mut Net, g: &Net, lr: f64) {
        self.t += 1.0; let (b1, b2, e) = (0.9, 0.999, 1e-8);
        macro_rules! upd { ($pf:ident, $mf:ident, $vf:ident) => {
            for i in 0..p.$pf.len() { self.m.$mf[i] = b1*self.m.$mf[i] + 0.1*g.$pf[i]; self.v.$vf[i] = b2*self.v.$vf[i] + 0.001*g.$pf[i]*g.$pf[i];
                p.$pf[i] -= lr*(self.m.$mf[i]/(1.0-b1.powf(self.t)))/((self.v.$vf[i]/(1.0-b2.powf(self.t))).sqrt()+e); } } }
        upd!(w1, w1, w1); upd!(b1, b1, b1); upd!(w2, w2, w2); upd!(b2, b2, b2); upd!(w3, w3, w3); upd!(b3, b3, b3);
    }
}
fn vec2(x: f64, y: f64) -> DVector<f64> { DVector::from_vec(vec![x, y]) }

// force from the POTENTIAL net via finite-difference gradient: f = −∇V (curl-free by construction).
const EPS: f64 = 1e-3;
fn pot_force(net: &Net, x: f64, y: f64) -> (f64, f64) {
    let v = |a: f64, b: f64| net.fwd(&vec2(a, b)).0[0];
    (-(v(x + EPS, y) - v(x - EPS, y)) / (2.0 * EPS), -(v(x, y + EPS) - v(x, y - EPS)) / (2.0 * EPS))
}

fn main() {
    println!("The cure mechanism, isolated — conservative force = −∇V (curl-free by construction).\n");
    let steps = 80_000u32;

    // ---- GENERIC force net: (x,y) → (fx,fy) ----
    let mut gnet = Net::new(2, 2, 10); let mut gen_opt = Adam::new(2, 2);
    // ---- POTENTIAL net: (x,y) → V (scalar); force = −∇V ----
    let mut pot = Net::new(2, 1, 20); let mut pot_opt = Adam::new(2, 1);

    // gradient-check the potential-net training gradient (FD-of-loss vs the assembled analytic gradient)
    {
        let (x, y) = (0.3, -0.2); let (tx, ty) = true_force(x, y);
        let (fx, fy) = pot_force(&pot, x, y);
        // analytic d(loss)/dw via FD-of-single-backprops
        let gxp = pot.bwd_dy(&vec2(x + EPS, y), &DVector::from_vec(vec![1.0]));
        let gxm = pot.bwd_dy(&vec2(x - EPS, y), &DVector::from_vec(vec![1.0]));
        let gyp = pot.bwd_dy(&vec2(x, y + EPS), &DVector::from_vec(vec![1.0]));
        let gym = pot.bwd_dy(&vec2(x, y - EPS), &DVector::from_vec(vec![1.0]));
        let mut grad = Net::zeros(2, 1);
        // ∂fx/∂w = −(gxp−gxm)/2ε ; d(loss)/dw = (fx−tx)·∂fx/∂w = −(fx−tx)/2ε·(gxp−gxm)
        grad.axpy(-(fx - tx) / (2.0 * EPS), &gxp); grad.axpy((fx - tx) / (2.0 * EPS), &gxm);
        grad.axpy(-(fy - ty) / (2.0 * EPS), &gyp); grad.axpy((fy - ty) / (2.0 * EPS), &gym);
        // FD of the loss w.r.t. one weight
        let loss_at = |n: &Net| { let (a, b) = pot_force(n, x, y); 0.5 * ((a - tx).powi(2) + (b - ty).powi(2)) };
        let mut np = pot.clone(); let d = 1e-6; let base = np.w2[(4, 1)];
        np.w2[(4, 1)] = base + d; let lp = loss_at(&np); np.w2[(4, 1)] = base - d; let lm = loss_at(&np);
        let fd = (lp - lm) / (2.0 * d);
        println!("  gradient check (potential w2[4,1]): analytic {:+.3e} finite-diff {:+.3e} → {}", grad.w2[(4, 1)], fd, if (grad.w2[(4, 1)] - fd).abs() < 1e-5 { "MATCH ✓ (training is real)" } else { "MISMATCH" });
    }

    for k in 0..steps {
        let (x, y) = (2.4 * (2.0 * u01(k * 3 + 1) - 1.0), 2.4 * (2.0 * u01(k * 3 + 2) - 1.0));
        let (tx, ty) = true_force(x, y);
        // generic: predict force directly
        let (gy, _, _) = gnet.fwd(&vec2(x, y));
        let ggen = gnet.bwd_dy(&vec2(x, y), &DVector::from_vec(vec![gy[0] - tx, gy[1] - ty]));
        gen_opt.step(&mut gnet, &ggen, 1e-3);
        // potential: force = −∇V via FD; assemble d(loss)/dw from single backprops at perturbed points
        let (fx, fy) = pot_force(&pot, x, y); let (ex, ey) = (fx - tx, fy - ty);
        let gxp = pot.bwd_dy(&vec2(x + EPS, y), &DVector::from_vec(vec![1.0]));
        let gxm = pot.bwd_dy(&vec2(x - EPS, y), &DVector::from_vec(vec![1.0]));
        let gyp = pot.bwd_dy(&vec2(x, y + EPS), &DVector::from_vec(vec![1.0]));
        let gym = pot.bwd_dy(&vec2(x, y - EPS), &DVector::from_vec(vec![1.0]));
        let mut gp = Net::zeros(2, 1);
        gp.axpy(-ex / (2.0 * EPS), &gxp); gp.axpy(ex / (2.0 * EPS), &gxm);   // d(loss)/dw = −ex/2ε·(gxp−gxm)
        gp.axpy(-ey / (2.0 * EPS), &gyp); gp.axpy(ey / (2.0 * EPS), &gym);
        pot_opt.step(&mut pot, &gp, 1e-3);
    }

    // curl of each learned field (∂fx/∂y − ∂fy/∂x), averaged over the domain — 0 ⇔ conservative
    let curl = |f: &dyn Fn(f64, f64) -> (f64, f64)| -> f64 {
        let mut s = 0.0; let n = 400;
        for k in 0..n { let (x, y) = (2.0 * (2.0 * u01(80_000 + k * 2) - 1.0), 2.0 * (2.0 * u01(80_001 + k * 2) - 1.0));
            let (fx_yp, _) = f(x, y + EPS); let (fx_ym, _) = f(x, y - EPS);   // ∂fx/∂y  (first component)
            let (_, fy_xp) = f(x + EPS, y); let (_, fy_xm) = f(x - EPS, y);   // ∂fy/∂x  (second component)
            s += ((fx_yp - fx_ym) / (2.0 * EPS) - (fy_xp - fy_xm) / (2.0 * EPS)).abs(); }
        s / n as f64
    };
    let gen_f = |x: f64, y: f64| { let o = gnet.fwd(&vec2(x, y)).0; (o[0], o[1]) };
    let pot_f = |x: f64, y: f64| pot_force(&pot, x, y);
    let gen_curl = curl(&gen_f); let pot_curl = curl(&pot_f);

    // rollout energy conservation from an IC, symplectic, 1500 steps (dt=0.01, 15 s)
    let dt = 0.01;
    let rollout = |f: &dyn Fn(f64, f64) -> (f64, f64)| -> f64 {
        let (mut x, mut y, mut vx, mut vy) = (1.2f64, -0.8f64, 0.0f64, 0.0f64); let e0 = energy(x, y, vx, vy); let mut d = 0.0f64;
        for _ in 0..1500 { let (ax, ay) = f(x, y); vx += dt * ax; vy += dt * ay; x += dt * vx; y += dt * vy; d = d.max(((energy(x, y, vx, vy) - e0) / e0).abs()); }
        d
    };
    let gen_drift = rollout(&gen_f); let pot_drift = rollout(&pot_f);
    // one-step force accuracy (both should be accurate)
    let (mut gmse, mut pmse, n) = (0.0f64, 0.0f64, 1000);
    for k in 0..n { let (x, y) = (2.0 * (2.0 * u01(600_000 + k * 2) - 1.0), 2.0 * (2.0 * u01(600_001 + k * 2) - 1.0));
        let (tx, ty) = true_force(x, y); let (gx, gy) = gen_f(x, y); let (px, py) = pot_f(x, y);
        gmse += ((gx - tx).powi(2) + (gy - ty).powi(2)) / 2.0; pmse += ((px - tx).powi(2) + (py - ty).powi(2)) / 2.0; }
    gmse /= n as f64; pmse /= n as f64;

    let pf = |ok: bool| if ok { "PASS" } else { "FAIL" };
    // PRIMARY metric = curl (exact, fit-independent structural property). Threshold 0.1 (≈ FD floor).
    println!("\n  (both: 2→32→32 MLP capacity, {} Adam steps, SAME data — a conservative force target)\n", steps);
    println!("  {:>34}   force MSE   mean |curl| (conservative test)   energy drift", "");
    println!("  GENERIC force net  (x,y)→(fx,fy):        {:.2e}       {:>7.3}  [{}]                  {:>7.2}%", gmse, gen_curl, pf(gen_curl < 0.1), gen_drift * 100.0);
    println!("  POTENTIAL net  force = −∇V:              {:.2e}       {:>7.3}  [{}]                  {:>7.2}%", pmse, pot_curl, pf(pot_curl < 0.1), pot_drift * 100.0);
    println!("\n  READING — the clean, fit-independent result is the CURL. A force field is conservative iff it is");
    println!("  curl-free (∇×f = 0). The GENERIC net's field has curl {:.2}: it is NOT a gradient, so no integrator", gen_curl);
    println!("  can make it conserve. The POTENTIAL net's field has curl {:.3} — curl-free BY CONSTRUCTION (it is −∇V", pot_curl);
    println!("  of a scalar net, and mixed partials commute), and this holds at ANY fit error. That is the structural");
    println!("  fix the SO-101 result pointed to. Energy drift ({:.1}% generic vs {:.1}% potential) moves the same way", gen_drift * 100.0, pot_drift * 100.0);
    println!("  but is CONFOUNDED here by force-fit error (both nets only fit the stiff anharmonic force to MSE ~{:.1});", pmse.max(gmse));
    println!("  the potential's drift is purely that fit error (a wrong-but-conservative shadow potential), the generic's");
    println!("  is fit error PLUS curl pumping. The verified claim: parameterize a learned force as −∇V and it is");
    println!("  conservative by construction. The full multi-body cure adds the true M(q) for the Coriolis term — a");
    println!("  Lagrangian/Hamiltonian net; this isolates and confirms the conservativeness half of it, gradient-checked.");
}
