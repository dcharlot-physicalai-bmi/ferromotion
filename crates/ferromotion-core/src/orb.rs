//! **ORB features — FAST corner detection + oriented BRIEF descriptor** — the classical, weights-free
//! keypoint front-end for visual odometry / SLAM. This is the piece the crate's geometry back-end
//! ([`crate::pnp`], [`crate::essential`], [`crate::homography`], [`crate::ransac`], [`crate::bundle`]) has
//! always assumed someone hands it: pixel correspondences. ORB produces them from raw grayscale, with **no
//! trained network** — unlike the learned front-ends (SuperPoint/NetVLAD) that need model weights and have no
//! analytic oracle, ORB is pure computation and verifiable on synthetic images.
//!
//! Three stages, each the standard algorithm:
//! 1. **FAST-9** corner detection — a pixel is a corner if ≥9 *contiguous* pixels on the radius-3 Bresenham
//!    circle are all brighter than `center + t` or all darker than `center − t`. Corners are where intensity
//!    varies in *two* directions (blobs/edges are rejected), so they localize well.
//! 2. **Orientation** by the intensity centroid — the angle from the patch center to its brightness centroid
//!    (`atan2(m01, m10)`), which makes the descriptor rotation-aware.
//! 3. **BRIEF** descriptor — a 256-bit string, bit *i* = `[ I(p_i) < I(q_i) ]` for a fixed deterministic set
//!    of intra-patch sample pairs, *steered* by the keypoint orientation so the same physical patch gives the
//!    same bits under rotation. Matching is **Hamming distance** with Lowe's ratio test.
//!
//! Verified on synthetic images: FAST fires exactly on the corners of a bright square and stays silent on a
//! flat field and a straight edge; a patch's BRIEF descriptor matches itself at Hamming 0; and a
//! translated-image match recovers the correct pixel correspondences. Pure Rust, integer/float only →
//! WASM-clean.

/// A grayscale image: `data[y*width + x]` intensity in `[0, 255]`.
#[derive(Clone, Debug)]
pub struct GrayImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f64>,
}

impl GrayImage {
    /// Build from a row-major slice.
    pub fn new(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height, "data length must be width*height");
        Self { width, height, data }
    }
    #[inline]
    fn at(&self, x: usize, y: usize) -> f64 {
        self.data[y * self.width + x]
    }
}

/// A detected keypoint: pixel location, orientation (radians), and corner strength.
#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    pub x: usize,
    pub y: usize,
    pub angle: f64,
    pub score: f64,
}

/// One BRIEF sample pair: two intra-patch `(dx, dy)` offsets whose intensities are compared for one bit.
pub type SamplePair = ((i32, i32), (i32, i32));

/// A 256-bit BRIEF descriptor.
#[derive(Clone, Copy)]
pub struct Descriptor(pub [u64; 4]);

impl Descriptor {
    /// Hamming distance (number of differing bits) between two descriptors.
    pub fn hamming(&self, other: &Descriptor) -> u32 {
        (0..4).map(|i| (self.0[i] ^ other.0[i]).count_ones()).sum()
    }
}

// The 16 offsets of the radius-3 Bresenham circle, clockwise from the top.
const CIRCLE: [(i32, i32); 16] = [
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
];

/// **FAST-9 corner detection.** Returns keypoints where ≥9 contiguous circle pixels are uniformly brighter
/// than `center + threshold` or darker than `center − threshold`. `threshold` in intensity units (e.g. 20).
pub fn fast_corners(img: &GrayImage, threshold: f64) -> Vec<Keypoint> {
    let mut kps = Vec::new();
    let (w, h) = (img.width, img.height);
    if w < 7 || h < 7 {
        return kps;
    }
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let c = img.at(x, y);
            // classify each circle pixel: +1 brighter, -1 darker, 0 similar
            let mut cls = [0i8; 16];
            for (k, &(dx, dy)) in CIRCLE.iter().enumerate() {
                let p = img.at((x as i32 + dx) as usize, (y as i32 + dy) as usize);
                cls[k] = if p > c + threshold {
                    1
                } else if p < c - threshold {
                    -1
                } else {
                    0
                };
            }
            // longest contiguous run (circular) of the same nonzero sign, need ≥9
            if let Some(score) = contiguous_arc(&cls, img, x, y, c) {
                kps.push(Keypoint { x, y, angle: 0.0, score });
            }
        }
    }
    kps
}

/// If there's a contiguous circular arc of ≥9 same-sign pixels, return the corner score (sum of abs
/// intensity differences on the arc), else `None`.
fn contiguous_arc(cls: &[i8; 16], img: &GrayImage, x: usize, y: usize, c: f64) -> Option<f64> {
    for &sign in &[1i8, -1i8] {
        let mut best = 0usize;
        let mut run = 0usize;
        // go around twice to catch wrap-around arcs
        for i in 0..32 {
            if cls[i % 16] == sign {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        }
        // a full circle counts once, not twice
        let best = best.min(16);
        if best >= 9 {
            let score: f64 = CIRCLE
                .iter()
                .map(|&(dx, dy)| (img.at((x as i32 + dx) as usize, (y as i32 + dy) as usize) - c).abs())
                .sum();
            return Some(score);
        }
    }
    None
}

/// Non-maximum suppression: keep a keypoint only if its score is ≥ every other keypoint within `radius`
/// pixels. Reduces clustered detections to one per corner.
pub fn nms(kps: &[Keypoint], radius: i32) -> Vec<Keypoint> {
    let mut out = Vec::new();
    for (i, k) in kps.iter().enumerate() {
        let mut is_max = true;
        for (j, o) in kps.iter().enumerate() {
            if i == j {
                continue;
            }
            let dx = k.x as i32 - o.x as i32;
            let dy = k.y as i32 - o.y as i32;
            if dx * dx + dy * dy <= radius * radius && o.score > k.score {
                is_max = false;
                break;
            }
        }
        if is_max {
            out.push(*k);
        }
    }
    out
}

/// Compute the **intensity-centroid orientation** of the patch of half-size `half` around a keypoint, and
/// write it into a copy. `atan2(m01, m10)` where `m10 = Σ x·I`, `m01 = Σ y·I` over the patch.
pub fn orient(img: &GrayImage, kp: &Keypoint, half: usize) -> Keypoint {
    let (mut m10, mut m01) = (0.0, 0.0);
    let h = half as i32;
    for dy in -h..=h {
        for dx in -h..=h {
            let px = kp.x as i32 + dx;
            let py = kp.y as i32 + dy;
            if px >= 0 && py >= 0 && (px as usize) < img.width && (py as usize) < img.height {
                let v = img.at(px as usize, py as usize);
                m10 += dx as f64 * v;
                m01 += dy as f64 * v;
            }
        }
    }
    let mut out = *kp;
    out.angle = m01.atan2(m10);
    out
}

/// A deterministic BRIEF sampling pattern: 256 pairs of `(dx, dy)` offsets within a patch of the given
/// half-size, generated by a seeded LCG so the pattern is fixed and reproducible (no `rand`).
pub fn brief_pattern(half: i32) -> Vec<SamplePair> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        // SplitMix64
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    };
    let span = (2 * half + 1) as u64;
    let mut off = || ((next() % span) as i32 - half, (next() % span) as i32 - half);
    (0..256).map(|_| (off(), off())).collect()
}

/// Compute the **steered BRIEF descriptor** at a keypoint: rotate each sample pair by the keypoint angle,
/// then set bit *i* iff `I(p_i) < I(q_i)`. Out-of-bounds samples read as 0.
pub fn brief_descriptor(img: &GrayImage, kp: &Keypoint, pattern: &[SamplePair]) -> Descriptor {
    let (s, c) = kp.angle.sin_cos();
    let sample = |dx: i32, dy: i32| -> f64 {
        // steer: rotate the offset by the keypoint orientation
        let rx = (c * dx as f64 - s * dy as f64).round() as i32;
        let ry = (s * dx as f64 + c * dy as f64).round() as i32;
        let px = kp.x as i32 + rx;
        let py = kp.y as i32 + ry;
        if px >= 0 && py >= 0 && (px as usize) < img.width && (py as usize) < img.height {
            img.at(px as usize, py as usize)
        } else {
            0.0
        }
    };
    let mut bits = [0u64; 4];
    for (i, &((ax, ay), (bx, by))) in pattern.iter().enumerate() {
        if sample(ax, ay) < sample(bx, by) {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    Descriptor(bits)
}

/// A brute-force match: for each descriptor in `a`, its nearest and second-nearest in `b` (by Hamming),
/// accepted only if it passes **Lowe's ratio test** (`best < ratio · second`). Returns `(i, j, distance)`.
pub fn match_descriptors(a: &[Descriptor], b: &[Descriptor], ratio: f64) -> Vec<(usize, usize, u32)> {
    let mut matches = Vec::new();
    for (i, da) in a.iter().enumerate() {
        let (mut best, mut second, mut best_j) = (u32::MAX, u32::MAX, 0usize);
        for (j, db) in b.iter().enumerate() {
            let d = da.hamming(db);
            if d < best {
                second = best;
                best = d;
                best_j = j;
            } else if d < second {
                second = d;
            }
        }
        if (best as f64) < ratio * second as f64 {
            matches.push((i, best_j, best));
        }
    }
    matches
}

/// Convenience: the full ORB pipeline on one image — detect, NMS, orient, describe. Returns oriented
/// keypoints paired with their descriptors, at most `max_kps` strongest.
pub fn detect_and_describe(img: &GrayImage, threshold: f64, max_kps: usize) -> Vec<(Keypoint, Descriptor)> {
    let pattern = brief_pattern(15);
    let mut kps = nms(&fast_corners(img, threshold), 3);
    kps.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    kps.truncate(max_kps);
    kps.iter()
        .map(|k| {
            let ok = orient(img, k, 15);
            let d = brief_descriptor(img, &ok, &pattern);
            (ok, d)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Render a `w×h` image with a filled bright square on a dark field.
    fn square_image(w: usize, h: usize, x0: usize, y0: usize, side: usize) -> GrayImage {
        let mut data = vec![30.0; w * h];
        for y in y0..(y0 + side).min(h) {
            for x in x0..(x0 + side).min(w) {
                data[y * w + x] = 220.0;
            }
        }
        GrayImage::new(w, h, data)
    }

    #[test]
    fn fast_stays_silent_on_a_flat_field() {
        // THE ORACLE (negative). No intensity variation → no corners, ever.
        let flat = GrayImage::new(30, 30, vec![128.0; 900]);
        assert!(fast_corners(&flat, 20.0).is_empty(), "a flat image has no corners");
    }

    #[test]
    fn fast_stays_silent_on_a_straight_edge() {
        // THE DISCRIMINATOR. FAST rejects straight edges (variation in ONE direction) — only true corners fire.
        let mut data = vec![30.0; 30 * 30];
        for y in 0..30 {
            for x in 15..30 {
                data[y * 30 + x] = 220.0; // vertical edge down the middle, no corners
            }
        }
        let img = GrayImage::new(30, 30, data);
        assert!(fast_corners(&img, 20.0).is_empty(), "a straight edge is not a corner");
    }

    #[test]
    fn fast_fires_on_the_corners_of_a_square() {
        // THE HEADLINE. A bright square has exactly four corners; FAST (after NMS) localizes near each.
        let img = square_image(40, 40, 12, 12, 16); // corners at (12,12),(27,12),(12,27),(27,27)
        let kps = nms(&fast_corners(&img, 20.0), 4);
        assert!(!kps.is_empty(), "the square's corners should be detected");
        let expected = [(12, 12), (27, 12), (12, 27), (27, 27)];
        for &(ex, ey) in &expected {
            let found = kps.iter().any(|k| {
                let dx = k.x as i32 - ex;
                let dy = k.y as i32 - ey;
                dx * dx + dy * dy <= 9 // within 3px
            });
            assert!(found, "no keypoint near corner ({ex},{ey}); got {:?}", kps.iter().map(|k| (k.x, k.y)).collect::<Vec<_>>());
        }
    }

    #[test]
    fn a_descriptor_matches_itself_at_hamming_zero() {
        // THE INVARIANT. The same keypoint in the same image gives the identical descriptor.
        let img = square_image(50, 50, 15, 15, 20);
        let pat = brief_pattern(15);
        let kp = orient(&img, &Keypoint { x: 15, y: 15, angle: 0.0, score: 1.0 }, 15);
        let d1 = brief_descriptor(&img, &kp, &pat);
        let d2 = brief_descriptor(&img, &kp, &pat);
        assert_eq!(d1.hamming(&d2), 0, "a descriptor must equal itself");
    }

    #[test]
    fn brief_distinguishes_different_patches() {
        // Two structurally different patches (a corner vs. a flat region) must have far-apart descriptors.
        let img = square_image(50, 50, 15, 15, 20);
        let pat = brief_pattern(15);
        let corner = orient(&img, &Keypoint { x: 15, y: 15, angle: 0.0, score: 1.0 }, 15);
        let flat = orient(&img, &Keypoint { x: 40, y: 40, angle: 0.0, score: 1.0 }, 15);
        let dc = brief_descriptor(&img, &corner, &pat);
        let df = brief_descriptor(&img, &flat, &pat);
        assert!(dc.hamming(&df) > 20, "distinct patches should differ in many bits: {}", dc.hamming(&df));
    }

    /// An asymmetric scene: several rectangles of different sizes/positions, so each corner has a
    /// *distinctive* local neighborhood (unlike a lone symmetric square, whose four identical corners are
    /// genuinely ambiguous and correctly rejected by the ratio test). Optional global shift `(sx, sy)`.
    fn scene(w: usize, h: usize, sx: i32, sy: i32) -> GrayImage {
        let mut data = vec![30.0; w * h];
        let rects = [(10, 10, 8, 20), (28, 15, 14, 9), (18, 34, 22, 6), (44, 30, 6, 15)];
        for &(x0, y0, ww, hh) in &rects {
            for y in 0..hh {
                for x in 0..ww {
                    let px = x0 + x + sx;
                    let py = y0 + y + sy;
                    if px >= 0 && py >= 0 && (px as usize) < w && (py as usize) < h {
                        data[py as usize * w + px as usize] = 220.0;
                    }
                }
            }
        }
        GrayImage::new(w, h, data)
    }

    #[test]
    fn matching_recovers_correspondences_under_translation() {
        // THE APPLICATION. Detect on an image and on its 5px-shifted copy; matched keypoints should be offset
        // by ~(5,0), i.e. the descriptor is translation-invariant and matching finds the right pairs.
        let a = scene(64, 64, 0, 0);
        let b = scene(64, 64, 5, 0); // whole scene shifted +5 in x
        let fa = detect_and_describe(&a, 20.0, 20);
        let fb = detect_and_describe(&b, 20.0, 20);
        assert!(!fa.is_empty() && !fb.is_empty(), "both images should yield keypoints");
        let da: Vec<_> = fa.iter().map(|(_, d)| *d).collect();
        let db: Vec<_> = fb.iter().map(|(_, d)| *d).collect();
        let matches = match_descriptors(&da, &db, 0.8);
        assert!(!matches.is_empty(), "translation should produce matches");
        // most matches should show the +5,0 pixel offset
        let good = matches
            .iter()
            .filter(|&&(i, j, _)| {
                let dx = fb[j].0.x as i32 - fa[i].0.x as i32;
                let dy = fb[j].0.y as i32 - fa[i].0.y as i32;
                (dx - 5).abs() <= 2 && dy.abs() <= 2
            })
            .count();
        assert!(good * 2 >= matches.len(), "majority of matches should recover the +5px shift: {good}/{}", matches.len());
    }

    #[test]
    fn orientation_tracks_the_brightness_centroid() {
        // A keypoint whose patch is brighter to the +x side should get an angle near 0; brighter to +y → ~π/2.
        let mut data = vec![30.0; 40 * 40];
        for y in 15..25 {
            for x in 20..30 {
                data[y * 40 + x] = 220.0; // bright block to the +x side of (20,20)
            }
        }
        let img = GrayImage::new(40, 40, data);
        let k = orient(&img, &Keypoint { x: 20, y: 20, angle: 0.0, score: 1.0 }, 12);
        assert!(k.angle.abs() < PI / 4.0, "centroid is to the +x, angle≈0: {}", k.angle);
    }
}
