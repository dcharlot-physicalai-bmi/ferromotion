//! **Real-Time Chunking (RTC)** — asynchronous execution of action-chunking flow policies (Physical
//! Intelligence, NeurIPS 2025). An action-chunking policy emits a whole chunk of `H` actions at once,
//! but inference takes time: by the moment the next chunk is ready, the first few actions of the current
//! chunk have already run. Re-planning from scratch makes the new chunk jump away from what is executing
//! — a visible discontinuity. RTC removes it by **inpainting**: the new chunk is generated with its
//! overlap region *frozen* to the actions already committed, and a short *soft-guided* region easing off
//! the freeze, so the executed action stream is continuous and the tail still reacts to fresh
//! observations. It is a pure inference-time wrapper over the crate's flow runner ([`crate::sample_field`])
//! — no retraining, no new weights — and it reduces bit-identically to naive chunking when the latency,
//! hence the frozen region, is zero. Pure Rust → WASM-clean.

use crate::Integrator;

/// Per-action guidance weights for a chunk of `n_actions` actions of `action_dim` scalars each: `1.0`
/// (hard-frozen) for the first `frozen` actions, a linear ramp `1 → 0` over the next `soft` actions,
/// then `0.0` (free). Broadcast to every scalar of each action.
pub fn rtc_mask(n_actions: usize, action_dim: usize, frozen: usize, soft: usize) -> Vec<f64> {
    let mut w = vec![0.0; n_actions * action_dim];
    for k in 0..n_actions {
        let wk = if k < frozen {
            1.0
        } else if soft > 0 && k < frozen + soft {
            1.0 - (k - frozen + 1) as f64 / (soft + 1) as f64
        } else {
            0.0
        };
        for d in 0..action_dim {
            w[k * action_dim + d] = wk;
        }
    }
    w
}

/// Guided (inpainting) flow sampling: integrate `da/dt = v(a,t)` from `a0`, but at each step blend each
/// coordinate toward its straight-line target path `(1−t)·a0 + t·target` by its weight `w ∈ [0,1]`.
/// `w = 1` pins the coordinate exactly to `target` (its value is known); `w = 0` leaves it to the field.
pub fn guided_sample(
    v: &dyn Fn(&[f64], f64) -> Vec<f64>,
    a0: &[f64],
    target: &[f64],
    weights: &[f64],
    steps: usize,
    method: Integrator,
) -> Vec<f64> {
    let h = 1.0 / steps as f64;
    let n = a0.len();
    let mut a = a0.to_vec();
    for k in 0..steps {
        let t = k as f64 * h;
        let tn = (k + 1) as f64 * h;
        // one field step (Euler/Heun) on the current state
        let k1 = v(&a, t);
        match method {
            Integrator::Euler => {
                for i in 0..n {
                    a[i] += h * k1[i];
                }
            }
            Integrator::Heun => {
                let a_pred: Vec<f64> = (0..n).map(|i| a[i] + h * k1[i]).collect();
                let k2 = v(&a_pred, t + h);
                for i in 0..n {
                    a[i] += 0.5 * h * (k1[i] + k2[i]);
                }
            }
        }
        // inpaint: pull guided coordinates onto their known interpolation path toward the target
        for i in 0..n {
            let w = weights[i];
            if w > 0.0 {
                let interp = (1.0 - tn) * a0[i] + tn * target[i];
                a[i] = (1.0 - w) * a[i] + w * interp;
            }
        }
    }
    a
}

/// Sample the next action chunk with RTC: freeze the `frozen`-action overlap to `prev_chunk` (the actions
/// already committed to execute during inference latency), soft-guide the next `soft`, free the rest.
/// `prev_chunk` is the target for the guided region; only its frozen/soft entries are used.
pub fn sample_rtc(
    v: &dyn Fn(&[f64], f64) -> Vec<f64>,
    a0: &[f64],
    prev_chunk: &[f64],
    n_actions: usize,
    action_dim: usize,
    frozen: usize,
    soft: usize,
    steps: usize,
    method: Integrator,
) -> Vec<f64> {
    let weights = rtc_mask(n_actions, action_dim, frozen, soft);
    guided_sample(v, a0, prev_chunk, &weights, steps, method)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample_field;

    // A constant-velocity flow field: da/dt = g ⇒ the chunk integrates to a0 + g. `g` (per scalar)
    // stands in for an obs-conditioned policy; different obs ⇒ different g ⇒ different chunk.
    fn const_field(g: Vec<f64>) -> impl Fn(&[f64], f64) -> Vec<f64> {
        move |_a: &[f64], _t: f64| g.clone()
    }

    #[test]
    fn zero_freeze_reduces_bit_identically_to_naive_chunking() {
        // With no frozen/soft region (zero latency), RTC must equal the plain flow sample exactly.
        let g = vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.2];
        let v = const_field(g.clone());
        let a0 = vec![0.1, 0.0, -0.1, 0.2, 0.05, -0.05];
        let prev = vec![9.0; 6]; // irrelevant when nothing is frozen
        let plain = sample_field(&v, &a0, 8, Integrator::Heun);
        let rtc = sample_rtc(&v, &a0, &prev, 3, 2, 0, 0, 8, Integrator::Heun);
        assert!(plain.iter().zip(&rtc).all(|(p, r)| (p - r).abs() < 1e-15), "RTC(0,0) must equal naive: {plain:?} vs {rtc:?}");
    }

    #[test]
    fn the_frozen_prefix_is_preserved_exactly() {
        // The frozen actions come out exactly equal to the committed previous-chunk actions — the
        // property that makes the chunk boundary jump-free.
        let v = const_field(vec![0.5; 6]);
        let a0 = vec![0.0; 6];
        let prev = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 actions × 2 dims
        let out = sample_rtc(&v, &a0, &prev, 3, 2, 1, 1, 10, Integrator::Euler);
        // action 0 hard-frozen ⇒ equals prev exactly
        assert!((out[0] - 1.0).abs() < 1e-12 && (out[1] - 2.0).abs() < 1e-12, "frozen action 0 not preserved: {out:?}");
    }

    #[test]
    fn full_freeze_returns_the_target_and_a_free_tail_is_the_plain_sample() {
        let g = vec![0.4, 0.4, 0.4, 0.4];
        let v = const_field(g.clone());
        let a0 = vec![0.0; 4];
        let prev = vec![7.0, 8.0, 9.0, 10.0];
        // all frozen ⇒ exactly the target
        let all = sample_rtc(&v, &a0, &prev, 2, 2, 2, 0, 6, Integrator::Heun);
        assert!(all.iter().zip(&prev).all(|(x, p)| (x - p).abs() < 1e-12), "full freeze must return target");
        // free tail (frozen=0) ⇒ plain sample a0+g
        let free = sample_rtc(&v, &a0, &prev, 2, 2, 0, 0, 6, Integrator::Heun);
        let plain = sample_field(&v, &a0, 6, Integrator::Heun);
        assert!(free.iter().zip(&plain).all(|(x, p)| (x - p).abs() < 1e-14));
    }

    #[test]
    fn the_soft_mask_ramps_monotonically_from_one_to_zero() {
        let w = rtc_mask(6, 1, 2, 3); // 2 frozen, 3-action soft ramp, 1 free
        assert_eq!(&w[..2], &[1.0, 1.0], "first two hard-frozen");
        assert!(w[2] < 1.0 && w[2] > w[3] && w[3] > w[4] && w[4] > 0.0, "monotone ramp: {w:?}");
        assert_eq!(w[5], 0.0, "last action free");
    }

    #[test]
    fn rtc_makes_the_chunk_boundary_continuous_where_naive_jumps() {
        // Two successive plans under CHANGED observations. After executing `d` actions of chunk A, a new
        // chunk is generated. Naive re-planning ignores A ⇒ the executed stream jumps at the switch.
        // RTC freezes the `d`-action overlap to A ⇒ the stream is continuous there.
        let (n_actions, da, d) = (6usize, 1usize, 2usize);
        let a0 = vec![0.0; n_actions * da];
        let obs_a = const_field(vec![0.20; n_actions * da]); // chunk A ≈ 0.20 everywhere
        let obs_b = const_field(vec![0.55; n_actions * da]); // new obs wants ≈ 0.55 — a big change
        let chunk_a = sample_field(&obs_a, &a0, 8, Integrator::Euler);

        // executed stream: A[0..d], then the new chunk from index d onward.
        let naive = sample_field(&obs_b, &a0, 8, Integrator::Euler);
        let rtc = sample_rtc(&obs_b, &a0, &chunk_a, n_actions, da, d, 2, 8, Integrator::Euler);

        let jump = |newc: &[f64]| (newc[d * da] - chunk_a[(d - 1) * da]).abs(); // A[d-1] → newchunk[d]
        let naive_jump = jump(&naive);
        let rtc_jump = jump(&rtc);
        // RTC's frozen overlap matches A exactly …
        assert!((rtc[(d - 1) * da] - chunk_a[(d - 1) * da]).abs() < 1e-12, "overlap must equal chunk A");
        // … and the transition into the free tail is far smoother than a naive re-plan's jump.
        assert!(rtc_jump < 0.5 * naive_jump, "RTC should cut the boundary jump: {rtc_jump} vs naive {naive_jump}");
    }
}
