//! **MuJoCo's spatial tendons in MuJoCo's own arithmetic** — the spatial half of `mj_tendon`
//! (`engine_core_smooth.c`) and `mju_wrap` with its circle and inside wraps (`engine_util_misc.c`), as
//! MuJoCo 3.13.0 runs them on arm64: the path's length through its sites and around its wrap geoms, and
//! its Jacobian, segment by segment, from `mj_jacDifPair` (sparse, common dofs kept) projected on the
//! segment's direction and merged into the tendon's static sparsity (`mju_combineSparseInc`).
//!
//! Contraction follows clang's `-ffp-contract=on`: a product added to anything is fused, the LEFT product of
//! a sum of two, one expression at a time.

use crate::mujoco_kinematics::{mj_local2global, KinGeom, KinState};
use crate::mujoco_smooth::SmoothModel;

const MJ_MINVAL: f64 = 1e-15;

/// One object of a spatial tendon's path, as `wrap_type`/`wrap_objid`/`wrap_prm` hold it.
#[derive(Clone, Debug)]
pub(crate) enum TendonObj {
    Site(KinGeom),
    /// a sphere or cylinder the path wraps around, with `geom_size[0]` and its optional side site
    Wrap { geom: KinGeom, radius: f64, cylinder: bool, side: Option<KinGeom> },
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
}

/// `mju_norm3`
fn norm3(v: &[f64; 3]) -> f64 {
    v[2].mul_add(v[2], v[0].mul_add(v[0], v[1] * v[1])).sqrt()
}

/// `mju_normalize3`: shorter than `mjMINVAL` becomes the x axis.
fn normalize3(v: &mut [f64; 3]) -> f64 {
    let norm = norm3(v);
    if norm < MJ_MINVAL {
        *v = [1.0, 0.0, 0.0];
    } else {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    norm
}

/// `mju_dot` of two 2-vectors: `0 + (a0·b0 + a1·b1)`, the tail of the four-sum loop
fn dot2(a: &[f64], b: &[f64]) -> f64 {
    0.0 + a[0].mul_add(b[0], a[1] * b[1])
}

/// `mju_normalize` in two dimensions
fn normalize2(v: &mut [f64; 2]) {
    let norm = dot2(v, v).sqrt();
    if norm < MJ_MINVAL {
        *v = [1.0, 0.0];
    } else {
        let inv = 1.0 / norm;
        v[0] *= inv;
        v[1] *= inv;
    }
}

/// `mji_cross`
fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[1].mul_add(b[2], -(a[2] * b[1])), a[2].mul_add(b[0], -(a[0] * b[2])), a[0].mul_add(b[1], -(a[1] * b[0]))]
}

/// `mju_dist3`
fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    norm3(&[a[0] - b[0], a[1] - b[1], a[2] - b[2]])
}

/// `is_intersect`: do the 2-D segments p1→p2 and p3→p4 cross?
fn is_intersect(p1: &[f64], p2: &[f64], p3: &[f64], p4: &[f64]) -> bool {
    let det = (p4[1] - p3[1]).mul_add(p2[0] - p1[0], -((p4[0] - p3[0]) * (p2[1] - p1[1])));
    if det.abs() < MJ_MINVAL {
        return false;
    }
    let a = (p4[0] - p3[0]).mul_add(p1[1] - p3[1], -((p4[1] - p3[1]) * (p1[0] - p3[0]))) / det;
    let b = (p2[0] - p1[0]).mul_add(p1[1] - p3[1], -((p2[1] - p1[1]) * (p1[0] - p3[0]))) / det;
    (0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b)
}

/// `length_circle`
fn length_circle(p0: &[f64; 2], p1: &[f64; 2], ind: usize, radius: f64) -> f64 {
    let (mut p0n, mut p1n) = (*p0, *p1);
    normalize2(&mut p0n);
    normalize2(&mut p1n);
    let mut angle = dot2(&p0n, &p1n).acos();
    let cross = p0[1].mul_add(p1[0], -(p0[0] * p1[1]));
    if (cross > 0.0 && ind != 0) || (cross < 0.0 && ind == 0) {
        angle = 2.0 * std::f64::consts::PI - angle;
    }
    radius * angle
}

/// `wrap_circle`: the two tangent points and the arc's length, or `None` (MuJoCo's −1).
fn wrap_circle(end: &[f64; 4], side: Option<&[f64; 2]>, radius: f64) -> Option<([f64; 4], f64)> {
    let sqlen0 = end[0].mul_add(end[0], end[1] * end[1]);
    let sqlen1 = end[2].mul_add(end[2], end[3] * end[3]);
    let sqrad = radius * radius;
    if sqlen0 < sqrad || sqlen1 < sqrad || radius < MJ_MINVAL {
        return None;
    }
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0].mul_add(dif[0], dif[1] * dif[1]);
    if dd < MJ_MINVAL {
        return None;
    }
    // clamped to the segment (`clamp` is MuJoCo's if-chain, NaN through as NaN)
    let a = (-dif[0].mul_add(end[0], dif[1] * end[1]) / dd).clamp(0.0, 1.0);
    let tmp = [a.mul_add(dif[0], end[0]), a.mul_add(dif[1], end[1])];
    if tmp[0].mul_add(tmp[0], tmp[1] * tmp[1]) > sqrad && side.is_none_or(|s| dot2(s, &tmp) >= 0.0) {
        return None;
    }
    let sqrt0 = (sqlen0 - sqrad).sqrt();
    let sqrt1 = (sqlen1 - sqrad).sqrt();
    let mut sol = [[[0.0f64; 2]; 2]; 2];
    let mut good = [0.0f64; 2];
    for i in 0..2 {
        let sgn = if i == 0 { 1.0 } else { -1.0 };
        sol[i][0][0] = end[0].mul_add(sqrad, sgn * radius * end[1] * sqrt0) / sqlen0;
        sol[i][0][1] = end[1].mul_add(sqrad, -(sgn * radius * end[0] * sqrt0)) / sqlen0;
        sol[i][1][0] = end[2].mul_add(sqrad, -(sgn * radius * end[3] * sqrt1)) / sqlen1;
        sol[i][1][1] = end[3].mul_add(sqrad, sgn * radius * end[2] * sqrt1) / sqlen1;
        good[i] = match side {
            Some(s) => {
                let mut t = [sol[i][0][0] + sol[i][1][0], sol[i][0][1] + sol[i][1][1]];
                normalize2(&mut t);
                dot2(&t, s)
            }
            None => {
                let t = [sol[i][0][0] - sol[i][1][0], sol[i][0][1] - sol[i][1][1]];
                -dot2(&t, &t)
            }
        };
        if is_intersect(&end[0..2], &sol[i][0], &end[2..4], &sol[i][1]) {
            good[i] = -10000.0;
        }
    }
    let i = if good[0] > good[1] { 0 } else { 1 };
    let pnt = [sol[i][0][0], sol[i][0][1], sol[i][1][0], sol[i][1][1]];
    if is_intersect(&end[0..2], &pnt[0..2], &end[2..4], &pnt[2..4]) {
        return None;
    }
    Some((pnt, length_circle(&sol[i][0], &sol[i][1], i, radius)))
}

/// `wrap_inside`: the side site inside the obstacle; both points are one, the arc zero.
fn wrap_inside(end: &[f64; 4], radius: f64) -> Option<([f64; 4], f64)> {
    const MAXITER: usize = 20;
    const ZINIT: f64 = 1.0 - 1e-7;
    const TOLERANCE: f64 = 1e-6;
    let len0 = dot2(&end[0..2], &end[0..2]).sqrt();
    let len1 = dot2(&end[2..4], &end[2..4]).sqrt();
    let dif = [end[2] - end[0], end[3] - end[1]];
    let dd = dif[0].mul_add(dif[0], dif[1] * dif[1]);
    if len0 <= radius || len1 <= radius || radius < MJ_MINVAL || len0 < MJ_MINVAL || len1 < MJ_MINVAL {
        return None;
    }
    if dd > MJ_MINVAL {
        let a = -dif[0].mul_add(end[0], dif[1] * end[1]) / dd;
        if a > 0.0 && a < 1.0 {
            // `mju_addScl`: end + dif·a
            let t = [dif[0].mul_add(a, end[0]), dif[1].mul_add(a, end[1])];
            if dot2(&t, &t).sqrt() <= radius {
                return None;
            }
        }
    }
    let mut mid = [0.5 * (end[0] + end[2]), 0.5 * (end[1] + end[3])];
    normalize2(&mut mid);
    let fallback = [mid[0] * radius, mid[1] * radius, mid[0] * radius, mid[1] * radius];
    let (a, b) = (radius / len0, radius / len1);
    let cos_g = (len0.mul_add(len0, len1 * len1) - dd) / (2.0 * len0 * len1);
    if cos_g < -1.0 + MJ_MINVAL {
        return None;
    } else if cos_g > 1.0 - MJ_MINVAL {
        return Some((fallback, 0.0));
    }
    let g = cos_g.acos();
    let fz = |z: f64| (a * z).asin() + (b * z).asin() - 2.0 * z.asin() + g;
    let mut z = ZINIT;
    let mut f = fz(z);
    if f > 0.0 {
        return Some((fallback, 0.0));
    }
    let mut iter = 0;
    while iter < MAXITER && f.abs() > TOLERANCE {
        let fmax = |x: f64| if MJ_MINVAL >= x { MJ_MINVAL } else { x };
        let df = a / fmax((-(z * z * a)).mul_add(a, 1.0).sqrt()) + b / fmax((-(z * z * b)).mul_add(b, 1.0).sqrt()) - 2.0 / fmax((-z).mul_add(z, 1.0).sqrt());
        if df > -MJ_MINVAL {
            return Some((fallback, 0.0));
        }
        let z1 = z - f / df;
        if z1 > z {
            return Some((fallback, 0.0));
        }
        z = z1;
        f = fz(z);
        if f > TOLERANCE {
            return Some((fallback, 0.0));
        }
        iter += 1;
    }
    if iter >= MAXITER {
        return Some((fallback, 0.0));
    }
    let (mut vec, ang) = if end[0].mul_add(end[3], -(end[1] * end[2])) > 0.0 { ([end[0], end[1]], z.asin() - (a * z).asin()) } else { ([end[2], end[3]], z.asin() - (b * z).asin()) };
    normalize2(&mut vec);
    let (sn, cs) = crate::mujoco_kinematics::sincos(ang);
    let p = [radius * cs.mul_add(vec[0], -(sn * vec[1])), radius * sn.mul_add(vec[0], cs * vec[1])];
    Some(([p[0], p[1], p[0], p[1]], 0.0))
}

/// **`mju_wrap`**: the two 3-D wrap points and the length around the geom, or `None` for no wrap.
fn wrap(x0: &[f64; 3], x1: &[f64; 3], xpos: &[f64; 3], xmat: &[f64; 9], radius: f64, cylinder: bool, side: Option<&[f64; 3]>) -> Option<([f64; 6], f64)> {
    use crate::mujoco_efc::{mat_t_vec3, mat_vec3};
    let p0 = mat_t_vec3(xmat, &[x0[0] - xpos[0], x0[1] - xpos[1], x0[2] - xpos[2]]);
    let p1 = mat_t_vec3(xmat, &[x1[0] - xpos[0], x1[1] - xpos[1], x1[2] - xpos[2]]);
    if norm3(&p0) < MJ_MINVAL || norm3(&p1) < MJ_MINVAL {
        return None;
    }
    let (ax0, ax1) = if !cylinder {
        let mut a0 = p0;
        normalize3(&mut a0);
        let mut normal = cross(&p0, &p1);
        let nrm = normalize3(&mut normal);
        if nrm < MJ_MINVAL {
            let mut i = 0;
            if a0[1].abs() > a0[0].abs() && a0[1].abs() > a0[2].abs() {
                i = 1;
            }
            if a0[2].abs() > a0[0].abs() && a0[2].abs() > a0[1].abs() {
                i = 2;
            }
            let mut a1 = [1.0, 1.0, 1.0];
            a1[i] = 0.0;
            normal = cross(&a0, &a1);
            normalize3(&mut normal);
        }
        let mut a1 = cross(&normal, &a0);
        normalize3(&mut a1);
        (a0, a1)
    } else {
        ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    };
    let d = [dot3(&p0, &ax0), dot3(&p0, &ax1), dot3(&p1, &ax0), dot3(&p1, &ax1)];
    let s = side.map(|sp| mat_t_vec3(xmat, &[sp[0] - xpos[0], sp[1] - xpos[1], sp[2] - xpos[2]]));
    let sd = s.map(|s| {
        let mut v = [dot3(&s, &ax0), dot3(&s, &ax1)];
        normalize2(&mut v);
        [v[0] * radius, v[1] * radius]
    });
    let (pnt, mut wlen) = match s {
        Some(s) if norm3(&s) < radius => wrap_inside(&d, radius)?,
        _ => wrap_circle(&d, sd.as_ref(), radius)?,
    };
    let mut res = [0.0f64; 6];
    for i in 0..2 {
        for k in 0..3 {
            res[3 * i + k] = ax0[k] * pnt[2 * i];
        }
        for k in 0..3 {
            res[3 * i + k] += ax1[k] * pnt[2 * i + 1];
        }
    }
    if cylinder {
        let l0 = (p0[0] - res[0]).mul_add(p0[0] - res[0], (p0[1] - res[1]) * (p0[1] - res[1])).sqrt();
        let l1 = (p1[0] - res[3]).mul_add(p1[0] - res[3], (p1[1] - res[4]) * (p1[1] - res[4])).sqrt();
        res[2] = p0[2] + (p1[2] - p0[2]) * l0 / (l0 + wlen + l1);
        res[5] = p0[2] + (p1[2] - p0[2]) * (l0 + wlen) / (l0 + wlen + l1);
        let height = (res[5] - res[2]).abs();
        wlen = wlen.mul_add(wlen, height * height).sqrt();
    }
    let w0 = mat_vec3(xmat, &[res[0], res[1], res[2]]);
    let w1 = mat_vec3(xmat, &[res[3], res[4], res[5]]);
    Some(([w0[0] + xpos[0], w0[1] + xpos[1], w0[2] + xpos[2], w1[0] + xpos[0], w1[1] + xpos[1], w1[2] + xpos[2]], wlen))
}

impl SmoothModel {
    /// `makeTendonSparse` for a spatial tendon: every dof of every body its sites and wrap geoms ride on and
    /// of their ancestors, sorted.
    pub(crate) fn spatial_colind(&self, path: &[TendonObj]) -> Vec<usize> {
        let mut cols: Vec<usize> = Vec::new();
        for o in path {
            let body = match o {
                TendonObj::Site(s) => s.body,
                TendonObj::Wrap { geom, .. } => geom.body,
            };
            let mut b = body;
            while b > 0 {
                for k in 0..self.body_dofnum[b] {
                    let dof = self.body_dofadr[b] + k;
                    if !cols.contains(&dof) {
                        cols.push(dof);
                    }
                }
                b = self.kin.bodies[b].parent;
            }
        }
        cols.sort_unstable();
        cols
    }

    /// The world frame of a site or geom.
    fn obj_frame(&self, k: &KinState, g: &KinGeom) -> ([f64; 3], [f64; 9]) {
        let b = g.body;
        mj_local2global((&k.xpos[b], &k.xquat[b], &k.xmat[b]), &(k.xipos[b], k.ximat[b]), &g.pos, &g.quat, g.sameframe)
    }

    /// **The spatial half of `mj_tendon`** for one tendon: its length, and its Jacobian's values on
    /// `colind`. (Pulleys, which divide what follows them, are not carried by the loader: the divisor is 1.)
    pub(crate) fn spatial_tendon(&self, path: &[TendonObj], colind: &[usize], k: &KinState, cdof: &[[f64; 6]], subtree_com: &[[f64; 3]]) -> (f64, Vec<f64>) {
        let mut len = 0.0f64;
        let mut jrow = vec![0.0f64; colind.len()];
        let divisor = 1.0f64;
        let n = path.len();
        let mut j = 0;
        while j + 1 < n {
            let (o0, o1) = (&path[j], &path[j + 1]);
            let TendonObj::Site(s0) = o0 else {
                // a path that does not start a segment at a site; MuJoCo's compiler refuses it
                j += 1;
                continue;
            };
            let x0 = self.obj_frame(k, s0).0;
            let mut wpnt = [[0.0f64; 3]; 4];
            let mut wbody = [0usize; 4];
            wpnt[0] = x0;
            wbody[0] = s0.body;
            let (wrapped, s1, step) = match o1 {
                TendonObj::Wrap { geom, radius, cylinder, side } => {
                    let TendonObj::Site(s1) = &path[j + 2] else { unreachable!("a wrap geom is followed by a site") };
                    let x1 = self.obj_frame(k, s1).0;
                    let (gp, gm) = self.obj_frame(k, geom);
                    let sp = side.as_ref().map(|s| self.obj_frame(k, s).0);
                    (wrap(&x0, &x1, &gp, &gm, *radius, *cylinder, sp.as_ref()).map(|w| (w, geom.body)), s1, 2)
                }
                TendonObj::Site(s1) => (None, s1, 1),
            };
            let x1 = self.obj_frame(k, s1).0;
            let nseg = match wrapped {
                None => {
                    wpnt[1] = x1;
                    wbody[1] = s1.body;
                    len += dist3(&wpnt[0], &wpnt[1]) / divisor;
                    1
                }
                Some(((w, wlen), gbody)) => {
                    wpnt[1] = [w[0], w[1], w[2]];
                    wpnt[2] = [w[3], w[4], w[5]];
                    wpnt[3] = x1;
                    wbody[1] = gbody;
                    wbody[2] = gbody;
                    wbody[3] = s1.body;
                    len += (dist3(&wpnt[0], &wpnt[1]) + wlen + dist3(&wpnt[2], &wpnt[3])) / divisor;
                    3
                }
            };
            for s in 0..nseg {
                if wbody[s] == wbody[s + 1] {
                    continue;
                }
                let mut dif = [wpnt[s + 1][0] - wpnt[s][0], wpnt[s + 1][1] - wpnt[s][1], wpnt[s + 1][2] - wpnt[s][2]];
                normalize3(&mut dif);
                let (chain, jd) = self.jac_dif_pair_points(cdof, subtree_com, wbody[s], wbody[s + 1], &wpnt[s], &wpnt[s + 1]);
                if chain.is_empty() {
                    continue;
                }
                // `mju_mulMatTVec(tmp, jacdif, dif, 3, NV)`
                let mut tmp = vec![0.0f64; chain.len()];
                for r in 0..3 {
                    if dif[r] != 0.0 {
                        for (t, row) in tmp.iter_mut().zip(&jd[r]) {
                            *t = row.mul_add(dif[r], *t);
                        }
                    }
                }
                // `mju_combineSparseInc(J, tmp, nv, 1, 1/divisor, …)`
                let b = 1.0 / divisor;
                if chain.as_slice() == colind {
                    for (x, t) in jrow.iter_mut().zip(&tmp) {
                        *x = x.mul_add(1.0, t * b);
                    }
                } else {
                    let mut si = 0;
                    for (di, &c) in colind.iter().enumerate() {
                        while si < chain.len() && chain[si] < c {
                            si += 1;
                        }
                        if si < chain.len() && chain[si] == c {
                            jrow[di] = b.mul_add(tmp[si], jrow[di]);
                            si += 1;
                        }
                    }
                }
            }
            j += step;
        }
        (len, jrow)
    }
}
