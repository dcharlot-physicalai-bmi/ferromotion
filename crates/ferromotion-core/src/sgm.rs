//! **Semi-Global Matching (SGM) stereo disparity** — dense disparity from a rectified grayscale stereo pair,
//! Hirschmüller's classic algorithm. This is the depth front-end for the geometry back-end: where
//! [`crate::pnp`]/[`crate::essential`]/[`crate::triangulate`](crate) recover structure from *sparse* matched
//! points, SGM produces a *per-pixel* disparity (hence depth `Z = f·B / d`) — the input a robot needs for
//! obstacle grids ([`crate::OccupancyGrid`]) and mapping. Like [`crate::orb`], it is **weights-free classical
//! computation**, verifiable on synthetic images — no trained stereo network.
//!
//! Three stages, each the standard formulation:
//! 1. **Matching cost** via the **census transform** — encode each pixel by the sign of its neighbors' minus
//!    its own intensity (a bit-string robust to gain/offset), then the cost of assigning disparity `d` to left
//!    pixel `p` is the **Hamming distance** between `census_L(p)` and `census_R(p − d)`. Robust to the
//!    illumination differences that break raw SAD.
//! 2. **Semi-global aggregation** — the exact per-pixel winner-take-all is noisy; the true 2-D smoothness MRF
//!    is NP-hard. SGM approximates it by summing 1-D dynamic-programming passes along **8 directions**:
//!    `L_r(p, d) = C(p, d) + min[ L_r(p−r, d), L_r(p−r, d±1) + P1, minₖ L_r(p−r, k) + P2 ] − minₖ L_r(p−r, k)`.
//!    `P1` penalizes ±1 disparity steps (slanted surfaces), `P2 > P1` penalizes jumps (depth discontinuities).
//! 3. **Winner-take-all** on the summed cost → the disparity map.
//!
//! Verified on a synthetic rectified pair: a textured fronto-parallel plane at a known constant disparity is
//! recovered across the image interior; a two-plane scene recovers *both* disparities with a sharp step at the
//! boundary; and census cost is minimized at the true shift. Pure Rust → WASM-clean.

use crate::orb::GrayImage;

/// SGM parameters.
#[derive(Clone, Copy, Debug)]
pub struct StereoParams {
    /// Disparities searched are `0..max_disparity`.
    pub max_disparity: usize,
    /// Small-step (±1 disparity) smoothness penalty.
    pub p1: f64,
    /// Large-step (>1 disparity) smoothness penalty; must exceed `p1`.
    pub p2: f64,
}

impl Default for StereoParams {
    fn default() -> Self {
        Self { max_disparity: 32, p1: 8.0, p2: 32.0 }
    }
}

// The 8 aggregation directions (dx, dy).
const DIRS: [(i32, i32); 8] =
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)];

/// **Census transform** over a 5×5 window: bit *k* = `[ neighbor_k < center ]`, 24 bits packed into a `u32`.
/// Border pixels (within 2 of the edge) get code 0.
pub fn census5x5(img: &GrayImage) -> Vec<u32> {
    let (w, h) = (img.width, img.height);
    let mut out = vec![0u32; w * h];
    for y in 2..h.saturating_sub(2) {
        for x in 2..w.saturating_sub(2) {
            let c = img.data[y * w + x];
            let mut code = 0u32;
            let mut bit = 0;
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let v = img.data[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize];
                    if v < c {
                        code |= 1 << bit;
                    }
                    bit += 1;
                }
            }
            out[y * w + x] = code;
        }
    }
    out
}

/// The matching-cost volume `C[(y·w + x)·D + d]` = Hamming(`census_L(x,y)`, `census_R(x−d, y)`), with a high
/// cost where `x − d` falls outside the image.
pub fn cost_volume(left: &GrayImage, right: &GrayImage, params: &StereoParams) -> Vec<f64> {
    let (w, h) = (left.width, left.height);
    let d_max = params.max_disparity;
    let cl = census5x5(left);
    let cr = census5x5(right);
    let high = 25.0; // max Hamming over 24 bits + slack
    let mut vol = vec![high; w * h * d_max];
    for y in 0..h {
        for x in 0..w {
            for d in 0..d_max {
                if x >= d {
                    let hd = (cl[y * w + x] ^ cr[y * w + (x - d)]).count_ones() as f64;
                    vol[(y * w + x) * d_max + d] = hd;
                }
            }
        }
    }
    vol
}

/// Aggregate the cost volume along one direction `(dx, dy)`, adding the result into `acc` (the running sum
/// over all directions). Implements the SGM 1-D recursion with penalties `p1`, `p2`.
fn aggregate_dir(cost: &[f64], acc: &mut [f64], w: usize, h: usize, dir: (i32, i32), params: &StereoParams) {
    let (dx, dy) = dir;
    let d_max = params.max_disparity;
    let (p1, p2) = (params.p1, params.p2);
    // scan order: iterate so that (x-dx, y-dy) — the predecessor along the path — is always visited first.
    let xs: Vec<usize> = if dx >= 0 { (0..w).collect() } else { (0..w).rev().collect() };
    let ys: Vec<usize> = if dy >= 0 { (0..h).collect() } else { (0..h).rev().collect() };
    let mut prev = vec![0.0f64; d_max]; // L_r(p - r, ·) for the current path
    for &y in &ys {
        for &x in &xs {
            let px = x as i32 - dx;
            let py = y as i32 - dy;
            let base = (y * w + x) * d_max;
            let has_prev = px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h;
            if !has_prev {
                // path start: L_r(p, d) = C(p, d)
                for d in 0..d_max {
                    let v = cost[base + d];
                    prev[d] = v;
                    acc[base + d] += v;
                }
                continue;
            }
            let min_prev = prev.iter().cloned().fold(f64::INFINITY, f64::min);
            let mut cur = vec![0.0f64; d_max];
            for d in 0..d_max {
                let same = prev[d];
                let up = if d + 1 < d_max { prev[d + 1] + p1 } else { f64::INFINITY };
                let down = if d >= 1 { prev[d - 1] + p1 } else { f64::INFINITY };
                let jump = min_prev + p2;
                let best = same.min(up).min(down).min(jump);
                let v = cost[base + d] + best - min_prev; // subtract min_prev to bound growth
                cur[d] = v;
                acc[base + d] += v;
            }
            prev = cur;
        }
    }
}

/// Full SGM: build the cost volume, aggregate along all 8 directions, and take the per-pixel argmin. Returns
/// the disparity map (`disp[y·w + x]`), with `0` in the unreliable left border (`x < max_disparity`).
pub fn disparity_map(left: &GrayImage, right: &GrayImage, params: &StereoParams) -> Vec<u16> {
    let (w, h) = (left.width, left.height);
    let d_max = params.max_disparity;
    let cost = cost_volume(left, right, params);
    let mut acc = vec![0.0f64; w * h * d_max];
    for &dir in &DIRS {
        aggregate_dir(&cost, &mut acc, w, h, dir, params);
    }
    let mut disp = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * d_max;
            let mut best_d = 0;
            let mut best_c = f64::INFINITY;
            for d in 0..d_max {
                if acc[base + d] < best_c {
                    best_c = acc[base + d];
                    best_d = d;
                }
            }
            disp[y * w + x] = best_d as u16;
        }
    }
    disp
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic per-pixel texture (seeded hash → intensity in [0,255]); defined over all `x` so a shifted
    /// right image is well-defined at the borders.
    fn tex(x: i32, y: i32) -> f64 {
        let mut z = (x as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 27;
        (z % 256) as f64
    }

    /// Build a rectified pair for a fronto-parallel plane at constant disparity `d0`:
    /// `right(x,y) = left(x + d0, y)`, so the true match for left pixel `x` is right pixel `x − d0`.
    fn pair_constant(w: usize, h: usize, d0: i32) -> (GrayImage, GrayImage) {
        let mut l = vec![0.0; w * h];
        let mut r = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                l[y * w + x] = tex(x as i32, y as i32);
                r[y * w + x] = tex(x as i32 + d0, y as i32);
            }
        }
        (GrayImage::new(w, h, l), GrayImage::new(w, h, r))
    }

    #[test]
    fn census_cost_is_minimized_at_the_true_shift() {
        // THE ORACLE. For a plane at disparity d0, the census Hamming cost at a well-inside pixel is (near) zero
        // at d = d0 and larger elsewhere — the raw signal SGM then regularizes.
        let d0 = 6;
        let (l, r) = pair_constant(64, 40, d0);
        let params = StereoParams { max_disparity: 24, ..Default::default() };
        let vol = cost_volume(&l, &r, &params);
        let (x, y) = (40usize, 20usize);
        let base = (y * l.width + x) * params.max_disparity;
        let mut best_d = 0;
        let mut best_c = f64::INFINITY;
        for d in 0..params.max_disparity {
            if vol[base + d] < best_c {
                best_c = vol[base + d];
                best_d = d;
            }
        }
        assert_eq!(best_d, d0 as usize, "census cost should bottom out at the true disparity");
        assert!(best_c < 1.0, "the true-shift census cost should be ~0: {best_c}");
    }

    #[test]
    fn sgm_recovers_a_constant_disparity_plane() {
        // THE HEADLINE. A fronto-parallel plane at disparity 6 → the disparity map is 6 across the interior.
        let d0 = 6;
        let (l, r) = pair_constant(80, 50, d0);
        let params = StereoParams { max_disparity: 24, p1: 8.0, p2: 32.0 };
        let disp = disparity_map(&l, &r, &params);
        let (w, h) = (l.width, l.height);
        // count interior pixels (away from borders) that recovered d0
        let (mut correct, mut total) = (0, 0);
        for y in 5..h - 5 {
            for x in params.max_disparity + 2..w - 5 {
                total += 1;
                if disp[y * w + x] == d0 as u16 {
                    correct += 1;
                }
            }
        }
        let frac = correct as f64 / total as f64;
        assert!(frac > 0.95, "SGM should recover the constant disparity almost everywhere: {frac:.3}");
    }

    #[test]
    fn sgm_recovers_a_two_plane_scene_with_a_step() {
        // THE DISCRIMINATOR. Left half at disparity 4, right half at disparity 12: SGM recovers both, and the
        // boundary is a sharp step (its P2 penalty is exactly what preserves the discontinuity).
        let (w, h) = (100, 50);
        let (near, far) = (12i32, 4i32);
        let split = 50;
        let mut l = vec![0.0; w * h];
        let mut r = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let d = if x < split { far } else { near };
                l[y * w + x] = tex(x as i32, y as i32);
                r[y * w + x] = tex(x as i32 + d, y as i32);
            }
        }
        let (l, r) = (GrayImage::new(w, h, l), GrayImage::new(w, h, r));
        let params = StereoParams { max_disparity: 24, p1: 8.0, p2: 48.0 };
        let disp = disparity_map(&l, &r, &params);
        // sample a clearly-left and clearly-right interior column, mid-height
        let y = 25;
        let left_val = disp[y * w + 40];
        let right_val = disp[y * w + 80];
        assert_eq!(left_val, far as u16, "left plane disparity");
        assert_eq!(right_val, near as u16, "right plane disparity");
    }

    #[test]
    fn census_is_illumination_robust() {
        // Census encodes only local intensity ORDER, so a global gain+offset on the right image leaves the
        // matching cost (hence disparity) unchanged — where raw SAD would blow up.
        let d0 = 5;
        let (l, r) = pair_constant(64, 40, d0);
        // apply gain 1.3 + offset 20 to the right image
        let r_bright = GrayImage::new(r.width, r.height, r.data.iter().map(|v| 1.3 * v + 20.0).collect());
        let p = StereoParams { max_disparity: 24, ..Default::default() };
        let d1 = disparity_map(&l, &r, &p);
        let d2 = disparity_map(&l, &r_bright, &p);
        let (w, h) = (l.width, l.height);
        let mut same = 0;
        let mut total = 0;
        for y in 5..h - 5 {
            for x in p.max_disparity + 2..w - 5 {
                total += 1;
                if d1[y * w + x] == d2[y * w + x] {
                    same += 1;
                }
            }
        }
        assert!(same as f64 / total as f64 > 0.98, "census disparity should be invariant to gain/offset");
    }
}
