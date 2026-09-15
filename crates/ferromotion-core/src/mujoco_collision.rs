//! **MuJoCo's collision pipeline for primitives, as MuJoCo computes it** — the contacts that feed
//! [`crate::solve_contacts_mujoco`], ported routine for routine from `engine_collision_primitive.c`,
//! `engine_collision_box.c` and `engine_collision_driver.c` (MuJoCo 3.13), and pinned against `mj_forward`.
//!
//! What a MuJoCo contact is: a point `pos` **halfway between the two surfaces**, a signed `dist` (negative
//! when penetrating), a frame whose first axis is the normal **from geom 1 toward geom 2**, and the solver
//! parameters of the pair. The pair's parameters come from the two geoms by [`contact_param`]: the higher
//! `priority` wins outright; at equal priority `condim` is the max, `friction` the element-wise max, and
//! `solref`/`solimp` are mixed by `solmix` weights (direct negative `solref` takes the min instead).
//! A pair is a candidate only if its `contype`/`conaffinity` bitmasks cross ([`can_collide`]), the bodies
//! are not welded together, not both dof-less, and not parent and child unless `filterparent` is disabled
//! ([`filter_body_pair`]). Detection uses `margin + gap`; the contact is then **excluded** from the solver
//! when `dist ≥ margin` ([`set_contact`]), so `gap` is a band in which contacts are found but not acted on.
//!
//! Routines carried: plane–{sphere, capsule, cylinder, box}, sphere–{sphere, capsule, cylinder, box},
//! capsule–capsule and MuJoCo's own SAT box–box with its clipped-face multi-contact (up to 8) and its
//! numerical tolerances. Not carried, by name: capsule–box, ellipsoid pairs, height fields, and every
//! mesh pair — MuJoCo runs those through its native GJK/EPA with multi-contact perturbation, whose exact
//! contact sets this crate's own GJK/EPA do not reproduce; a plane–mesh first contact IS carried, because it
//! is a support query any correct hull gives the same answer to.
//!
//! Every expected number in the tests is MuJoCo 3.13.0's (`scripts/mujoco_collision_probe.py`).

use nalgebra::{Matrix3, Vector3};

const MINVAL: f64 = 1e-15;
const MAXVAL: f64 = 1e10;

/// A geom's world pose as MuJoCo holds it: `geom_xpos` and `geom_xmat` (columns are the geom's axes).
#[derive(Clone, Copy, Debug)]
pub struct GeomPose {
    pub pos: Vector3<f64>,
    pub mat: Matrix3<f64>,
}

impl GeomPose {
    /// The geom's local z axis in the world (a plane's normal, a capsule's or cylinder's axis).
    pub fn axis(&self) -> Vector3<f64> {
        self.mat.column(2).into()
    }
}

/// MuJoCo geom types, in MuJoCo's enum order — the pair routines take the lower type first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GeomType {
    Plane = 0,
    HField = 1,
    Sphere = 2,
    Capsule = 3,
    Ellipsoid = 4,
    Cylinder = 5,
    Box = 6,
    Mesh = 7,
}

/// A raw contact before the pair's parameters are attached (`mjPreContact`).
#[derive(Clone, Copy, Debug)]
pub struct PreContact {
    /// Signed distance, negative when penetrating.
    pub dist: f64,
    /// Halfway between the two surfaces.
    pub pos: Vector3<f64>,
    /// From geom 1 toward geom 2.
    pub normal: Vector3<f64>,
    /// A preferred first tangent (a capsule's axis); zero when the routine states none.
    pub tangent: Vector3<f64>,
}

fn pre(dist: f64, pos: Vector3<f64>, normal: Vector3<f64>, tangent: Vector3<f64>) -> PreContact {
    PreContact { dist, pos, normal, tangent }
}

// ------------------------------------------------------------------------------------------------
// primitive pair routines (engine_collision_primitive.c)
// ------------------------------------------------------------------------------------------------

/// `mjraw_PlaneSphere`: `r2` is the sphere's radius.
pub fn plane_sphere(margin: f64, p1: &GeomPose, p2: &GeomPose, r2: f64) -> Option<PreContact> {
    let n = p1.axis();
    let cdist = (p2.pos - p1.pos).dot(&n);
    if cdist > margin + r2 {
        return None;
    }
    let dist = cdist - r2;
    Some(pre(dist, p2.pos + n * (-dist / 2.0 - r2), n, Vector3::zeros()))
}

/// `mjraw_SphereSphere`.
pub fn sphere_sphere(margin: f64, p1: &GeomPose, r1: f64, p2: &GeomPose, r2: f64) -> Option<PreContact> {
    let dif = p1.pos - p2.pos;
    let cdist_sqr = dif.dot(&dif);
    let min_dist = margin + r1 + r2;
    if cdist_sqr > min_dist * min_dist {
        return None;
    }
    let dist = cdist_sqr.sqrt() - r1 - r2;
    let mut normal = p2.pos - p1.pos;
    let len = normal.norm();
    if len < MINVAL {
        normal = p1.axis().cross(&p2.axis());
        let l = normal.norm();
        if l >= MINVAL {
            normal /= l;
        }
    } else {
        normal /= len;
    }
    Some(pre(dist, p1.pos + normal * (r1 + dist / 2.0), normal, Vector3::zeros()))
}

/// `mjc_PlaneCapsule`: `size2 = (radius, half-length)`; each end cap is a plane–sphere test and the
/// capsule's axis is the preferred tangent.
pub fn plane_capsule(margin: f64, p1: &GeomPose, p2: &GeomPose, size2: [f64; 2]) -> Vec<PreContact> {
    let axis = p2.axis();
    let seg = axis * size2[1];
    let mut out = Vec::with_capacity(2);
    for end in [p2.pos + seg, p2.pos - seg] {
        let ep = GeomPose { pos: end, mat: p2.mat };
        if let Some(mut c) = plane_sphere(margin, p1, &ep, size2[0]) {
            c.tangent = axis;
            out.push(c);
        }
    }
    out
}

/// `mjraw_SphereCapsule`: the sphere against the nearest point of the capsule's segment.
pub fn sphere_capsule(margin: f64, p1: &GeomPose, r1: f64, p2: &GeomPose, size2: [f64; 2]) -> Option<PreContact> {
    let axis = p2.axis();
    let x = axis.dot(&(p1.pos - p2.pos)).clamp(-size2[1], size2[1]);
    let near = GeomPose { pos: p2.pos + axis * x, mat: p2.mat };
    sphere_sphere(margin, p1, r1, &near, size2[0])
}

/// `mjraw_CapsuleCapsule`: closest points of the two segments (two contacts when parallel).
pub fn capsule_capsule(margin: f64, p1: &GeomPose, size1: [f64; 2], p2: &GeomPose, size2: [f64; 2]) -> Vec<PreContact> {
    let axis1 = p1.axis() * size1[1];
    let axis2 = p2.axis() * size2[1];
    let dif = p1.pos - p2.pos;
    let ma = axis1.dot(&axis1);
    let mb = -axis1.dot(&axis2);
    let mc = axis2.dot(&axis2);
    let u = -axis1.dot(&dif);
    let v = axis2.dot(&dif);
    let det = ma * mc - mb * mb;
    let ss = |p: Vector3<f64>, q: Vector3<f64>| sphere_sphere(margin, &GeomPose { pos: p, mat: p1.mat }, size1[0], &GeomPose { pos: q, mat: p2.mat }, size2[0]);
    if det.abs() >= MINVAL {
        let mut x1 = (mc * u - mb * v) / det;
        let mut x2 = (ma * v - mb * u) / det;
        if x1 > 1.0 {
            x1 = 1.0;
            x2 = (v - mb) / mc;
        } else if x1 < -1.0 {
            x1 = -1.0;
            x2 = (v + mb) / mc;
        }
        if x2 > 1.0 {
            x2 = 1.0;
            x1 = ((u - mb) / ma).clamp(-1.0, 1.0);
        } else if x2 < -1.0 {
            x2 = -1.0;
            x1 = ((u + mb) / ma).clamp(-1.0, 1.0);
        }
        return ss(p1.pos + axis1 * x1, p2.pos + axis2 * x2).into_iter().collect();
    }
    // parallel: both ends of segment 1 against segment 2, then segment 2's ends against segment 1
    let mut out = Vec::with_capacity(2);
    let x2 = ((v - mb) / mc).clamp(-1.0, 1.0);
    out.extend(ss(p1.pos + axis1, p2.pos + axis2 * x2));
    let x2 = ((v + mb) / mc).clamp(-1.0, 1.0);
    out.extend(ss(p1.pos - axis1, p2.pos + axis2 * x2));
    if out.len() >= 2 {
        return out;
    }
    let x1 = ((u - mb) / ma).clamp(-1.0, 1.0);
    out.extend(ss(p1.pos + axis1 * x1, p2.pos + axis2));
    if out.len() >= 2 {
        return out;
    }
    let x1 = ((u + mb) / ma).clamp(-1.0, 1.0);
    out.extend(ss(p1.pos + axis1 * x1, p2.pos - axis2));
    out
}

/// `mjc_PlaneCylinder`: `size2 = (radius, half-height)`; up to four contacts on the lower rim.
pub fn plane_cylinder(margin: f64, p1: &GeomPose, p2: &GeomPose, size2: [f64; 2]) -> Vec<PreContact> {
    let normal = p1.axis();
    let mut axis = p2.axis();
    let mut prjaxis = normal.dot(&axis);
    if prjaxis > 0.0 {
        axis = -axis;
        prjaxis = -prjaxis;
    }
    let dist0 = (p2.pos - p1.pos).dot(&normal);
    // vec = the rim direction most aligned with −normal
    let mut vec = axis * prjaxis - normal;
    let len_sqr = vec.dot(&vec);
    if len_sqr >= MINVAL * MINVAL {
        vec *= size2[0] / len_sqr.sqrt();
    } else {
        vec = p2.mat.column(0) * size2[0];
    }
    let prjvec = vec.dot(&normal);
    let axis = axis * size2[1];
    prjaxis *= size2[1];
    let mut out = Vec::with_capacity(4);
    let d = dist0 + prjaxis + prjvec;
    if d <= margin {
        out.push(pre(d, p2.pos + vec + axis - normal * (d * 0.5), normal, Vector3::zeros()));
    } else {
        return out;
    }
    let d = dist0 - prjaxis + prjvec;
    if d <= margin {
        out.push(pre(d, p2.pos + vec - axis - normal * (d * 0.5), normal, Vector3::zeros()));
    }
    let prjvec1 = -prjvec * 0.5;
    let d = dist0 + prjaxis + prjvec1;
    if d <= margin {
        let mut vec1 = vec.cross(&axis);
        let l = vec1.norm();
        if l >= MINVAL {
            vec1 /= l;
        }
        vec1 *= size2[0] * 3.0f64.sqrt() / 2.0;
        out.push(pre(d, p2.pos + vec1 + axis - vec * 0.5 - normal * (d * 0.5), normal, Vector3::zeros()));
        out.push(pre(d, p2.pos - vec1 + axis - vec * 0.5 - normal * (d * 0.5), normal, Vector3::zeros()));
    }
    out
}

/// `mjc_SphereCylinder`: side, cap or rim, whichever the sphere's centre is nearest.
pub fn sphere_cylinder(margin: f64, p1: &GeomPose, r1: f64, p2: &GeomPose, size2: [f64; 2]) -> Option<PreContact> {
    let (radius, height) = (size2[0], size2[1]);
    let axis = p2.axis();
    let vec = p1.pos - p2.pos;
    let x = axis.dot(&vec);
    let a_proj = axis * x;
    let p_proj = vec - a_proj;
    let p_proj_sqr = p_proj.dot(&p_proj);
    let mut side = x.abs() < height;
    let mut cap = p_proj_sqr < radius * radius;
    if side && cap {
        // sphere centre inside the cylinder: keep the nearer exit
        if height - x.abs() < radius - p_proj_sqr.sqrt() {
            side = false;
        } else {
            cap = false;
        }
    }
    if side {
        return sphere_sphere(margin, p1, r1, &GeomPose { pos: p2.pos + a_proj, mat: p2.mat }, radius);
    }
    if cap {
        // the cap as a plane facing the sphere, then flip the normal back to geom1 → geom2
        let (pos_cap, mat_cap) = if x > 0.0 {
            (p2.pos + axis * height, p2.mat)
        } else {
            let m = p2.mat;
            (p2.pos - axis * height, Matrix3::new(-m[(0, 0)], m[(0, 1)], -m[(0, 2)], -m[(1, 0)], m[(1, 1)], -m[(1, 2)], -m[(2, 0)], m[(2, 1)], -m[(2, 2)]))
        };
        let mut c = plane_sphere(margin, &GeomPose { pos: pos_cap, mat: mat_cap }, p1, r1)?;
        c.normal = -c.normal;
        return Some(c);
    }
    // rim: the nearest point of the circular edge, as a zero-radius sphere
    let rim = p2.pos + p_proj * (radius / p_proj_sqr.sqrt()) + axis * if x > 0.0 { height } else { -height };
    sphere_sphere(margin, p1, r1, &GeomPose { pos: rim, mat: p2.mat }, 0.0)
}

/// `mjc_PlaneBox`: every corner below the plane (at most four), `size2` the half-extents.
pub fn plane_box(margin: f64, p1: &GeomPose, p2: &GeomPose, size2: [f64; 3]) -> Vec<PreContact> {
    let norm = p1.axis();
    let dist = (p2.pos - p1.pos).dot(&norm);
    let mut out = Vec::with_capacity(4);
    for i in 0..8 {
        let v = Vector3::new(if i & 1 != 0 { size2[0] } else { -size2[0] }, if i & 2 != 0 { size2[1] } else { -size2[1] }, if i & 4 != 0 { size2[2] } else { -size2[2] });
        let corner = p2.mat * v;
        let ldist = norm.dot(&corner);
        if dist + ldist > margin || ldist > 0.0 {
            continue;
        }
        let d = dist + ldist;
        out.push(pre(d, corner + p2.pos + norm * (-d / 2.0), norm, Vector3::zeros()));
        if out.len() >= 4 {
            break;
        }
    }
    out
}

/// `mjraw_SphereBox`: the sphere against the nearest point of the box, with MuJoCo's nearest-face rule when
/// the centre is inside.
pub fn sphere_box(margin: f64, p1: &GeomPose, r1: f64, p2: &GeomPose, size2: [f64; 3]) -> Option<PreContact> {
    let center = p2.mat.transpose() * (p1.pos - p2.pos);
    let mut clamped = center;
    for i in 0..3 {
        if size2[i] > 0.0 {
            clamped[i] = clamped[i].clamp(-size2[i], size2[i]);
        }
    }
    let mut tmp = clamped - center;
    let mut dist = tmp.norm();
    if dist - r1 > margin {
        return None;
    }
    let (pos, normal_local);
    if dist <= MINVAL {
        // centre inside: the nearest face
        let mut closest = (size2[0] + size2[1] + size2[2]) * 2.0;
        let mut k = 0;
        for i in 0..6 {
            let s = if i % 2 == 1 { 1.0 } else { -1.0 };
            let c = (s * size2[i / 2] - center[i / 2]).abs();
            if closest > c {
                closest = c;
                k = i;
            }
        }
        let mut nearest = Vector3::zeros();
        nearest[k / 2] = if k % 2 == 1 { -1.0 } else { 1.0 };
        pos = center + nearest * ((r1 - closest) / 2.0);
        normal_local = nearest;
        dist = -closest;
    } else {
        tmp /= dist;
        let deepest = center + tmp * r1;
        pos = clamped * 0.5 + deepest * 0.5;
        normal_local = tmp;
    }
    Some(pre(dist - r1, p2.mat * pos + p2.pos, p2.mat * normal_local, Vector3::zeros()))
}

// --- box–box (engine_collision_box.c), MuJoCo's double-precision tolerances
const BOXBOX_SEPEPS: f64 = 1e-13;
const BOXBOX_PAREPS: f64 = 1e-16;
const BOXBOX_SGNEPS: f64 = 1e-9;
const BOXBOX_DUPEPS: f64 = 1e-14;
const BOXBOX_EDGEBIAS: f64 = 1e-6;
const BOXBOX_MAXVERT: usize = 12;

fn clip_half_plane(poly: &mut Vec<[f64; 3]>, coord: usize, sign: f64, limit: f64) {
    let d: Vec<f64> = poly.iter().map(|p| sign * p[coord] - limit).collect();
    if d.iter().all(|&x| x <= 0.0) {
        return;
    }
    let mut out: Vec<[f64; 3]> = Vec::with_capacity(BOXBOX_MAXVERT);
    let n = poly.len();
    for k in 0..n {
        let k1 = if k + 1 == n { 0 } else { k + 1 };
        let (dp, dq) = (d[k], d[k1]);
        if dp <= 0.0 && out.len() < BOXBOX_MAXVERT {
            out.push(poly[k]);
        }
        if ((dp < 0.0 && dq > 0.0) || (dp > 0.0 && dq < 0.0)) && out.len() < BOXBOX_MAXVERT {
            let t = dp / (dp - dq);
            let (p, q) = (poly[k], poly[k1]);
            out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])]);
        }
    }
    *poly = out;
}

/// `mjc_BoxBox`: separating-axis test over 15 axes, then either one edge–edge contact or the incident
/// face clipped to the reference face (up to 8 contacts).
pub fn box_box(margin: f64, p1: &GeomPose, size1: [f64; 3], p2: &GeomPose, size2: [f64; 3]) -> Vec<PreContact> {
    let pos21 = p1.mat.transpose() * (p2.pos - p1.pos);
    let pos12 = p2.mat.transpose() * (p1.pos - p2.pos);
    let rot = p1.mat.transpose() * p2.mat;
    let r = |i: usize, j: usize| rot[(i, j)];
    let rotabs = rot.abs();
    let septol = margin + BOXBOX_SEPEPS * (size1[0] + size1[1] + size1[2] + size2[0] + size2[1] + size2[2]);
    let mut sep_best = -MAXVAL;
    let mut code: i32 = -1;
    for i in 0..3 {
        let radius2 = rotabs[(i, 0)] * size2[0] + rotabs[(i, 1)] * size2[1] + rotabs[(i, 2)] * size2[2];
        let sep = pos21[i].abs() - size1[i] - radius2;
        if sep > septol {
            return Vec::new();
        }
        if sep > sep_best {
            sep_best = sep;
            code = i as i32;
        }
    }
    for j in 0..3 {
        let radius1 = rotabs[(0, j)] * size1[0] + rotabs[(1, j)] * size1[1] + rotabs[(2, j)] * size1[2];
        let sep = pos12[j].abs() - size2[j] - radius1;
        if sep > septol {
            return Vec::new();
        }
        if sep > sep_best {
            sep_best = sep;
            code = 3 + j as i32;
        }
    }
    let sep_face = sep_best;
    let code_face = code;
    for i in 0..3 {
        for j in 0..3 {
            let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
            let mut ax1 = -r(i2, j);
            let mut ax2 = r(i1, j);
            let norm2 = ax1 * ax1 + ax2 * ax2;
            if norm2 < BOXBOX_PAREPS {
                continue;
            }
            let inv = 1.0 / norm2.sqrt();
            ax1 *= inv;
            ax2 *= inv;
            let radius1 = size1[i1] * ax1.abs() + size1[i2] * ax2.abs();
            let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
            let a2_1 = ax1 * r(i1, j1) + ax2 * r(i2, j1);
            let a2_2 = ax1 * r(i1, j2) + ax2 * r(i2, j2);
            let radius2 = size2[j1] * a2_1.abs() + size2[j2] * a2_2.abs();
            let sep = (ax1 * pos21[i1] + ax2 * pos21[i2]).abs() - radius1 - radius2;
            if sep > septol {
                return Vec::new();
            }
            if sep - BOXBOX_EDGEBIAS * sep.abs() > sep_best && sep > sep_face {
                sep_best = sep;
                code = 6 + 3 * i as i32 + j as i32;
            }
        }
    }
    if code < 0 {
        return Vec::new();
    }
    if code >= 6 {
        // an edge axis nearly parallel to the best face normal, and barely better: prefer the face
        let i = ((code - 6) / 3) as usize;
        let j = ((code - 6) % 3) as usize;
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let mut axis = Vector3::zeros();
        axis[i1] = -r(i2, j);
        axis[i2] = r(i1, j);
        axis /= axis.norm();
        let face_dot = if code_face < 3 {
            axis[code_face as usize].abs()
        } else {
            let f = (code_face - 3) as usize;
            (axis[0] * r(0, f) + axis[1] * r(1, f) + axis[2] * r(2, f)).abs()
        };
        if face_dot > 0.99 && sep_best < sep_face + 0.05 * sep_face.abs() + MINVAL {
            code = code_face;
        }
    }
    if code >= 6 {
        let i = ((code - 6) / 3) as usize;
        let j = ((code - 6) % 3) as usize;
        let (i1, i2) = ((i + 1) % 3, (i + 2) % 3);
        let (j1, j2) = ((j + 1) % 3, (j + 2) % 3);
        let mut axis = Vector3::zeros();
        axis[i1] = -r(i2, j);
        axis[i2] = r(i1, j);
        axis /= axis.norm();
        if axis.dot(&pos21) < 0.0 {
            axis = -axis;
        }
        let a2 = Vector3::new(
            axis[0] * r(0, 0) + axis[1] * r(1, 0) + axis[2] * r(2, 0),
            axis[0] * r(0, 1) + axis[1] * r(1, 1) + axis[2] * r(2, 1),
            axis[0] * r(0, 2) + axis[1] * r(1, 2) + axis[2] * r(2, 2),
        );
        let ambig = BOXBOX_SGNEPS;
        let amb1: Option<usize> = if axis[i1].abs() < ambig { Some(i1) } else if axis[i2].abs() < ambig { Some(i2) } else { None };
        let amb2: Option<usize> = if a2[j1].abs() < ambig { Some(j1) } else if a2[j2].abs() < ambig { Some(j2) } else { None };
        let d2 = Vector3::new(r(0, j), r(1, j), r(2, j));
        let b = d2[i];
        let denom = 1.0 - b * b;
        let (mut w1, mut w2) = (Vector3::zeros(), Vector3::zeros());
        let mut best_d2 = MAXVAL;
        for v1 in 0..if amb1.is_some() { 2 } else { 1 } {
            for v2 in 0..if amb2.is_some() { 2 } else { 1 } {
                let mut c1 = Vector3::zeros();
                c1[i1] = if axis[i1] >= 0.0 { size1[i1] } else { -size1[i1] };
                c1[i2] = if axis[i2] >= 0.0 { size1[i2] } else { -size1[i2] };
                if let (Some(a), 1) = (amb1, v1) {
                    c1[a] = -c1[a];
                }
                let mut cc = Vector3::zeros();
                cc[j1] = if a2[j1] >= 0.0 { -size2[j1] } else { size2[j1] };
                cc[j2] = if a2[j2] >= 0.0 { -size2[j2] } else { size2[j2] };
                if let (Some(a), 1) = (amb2, v2) {
                    cc[a] = -cc[a];
                }
                let c2 = rot * cc + pos21;
                let e = c2 - c1;
                let d1e = e[i];
                let d2e = d2.dot(&e);
                let mut s = if denom < MINVAL { 0.0 } else { (d1e - b * d2e) / denom };
                s = s.clamp(-size1[i], size1[i]);
                let t = (b * s - d2e).clamp(-size2[j], size2[j]);
                s = (d1e + b * t).clamp(-size1[i], size1[i]);
                let mut p1c = c1;
                p1c[i] += s;
                let p2c = c2 + d2 * t;
                let gap = p2c - p1c;
                let gap2 = gap.dot(&gap);
                if gap2 < best_d2 {
                    best_d2 = gap2;
                    w1 = p1c;
                    w2 = p2c;
                }
            }
        }
        let dist = (w2 - w1).dot(&axis);
        if dist > septol {
            return Vec::new();
        }
        let mid = (w1 + w2) * 0.5;
        return vec![pre(dist, p1.mat * mid + p1.pos, p1.mat * axis, Vector3::zeros())];
    }
    // face contact: clip the incident face of the other box against the reference face
    let ref1 = code < 3;
    let a = if ref1 { code } else { code - 3 } as usize;
    let sizeref = if ref1 { size1 } else { size2 };
    let sizeinc = if ref1 { size2 } else { size1 };
    let (posref, matref) = if ref1 { (p1.pos, p1.mat) } else { (p2.pos, p2.mat) };
    let posoi = if ref1 { pos21 } else { pos12 };
    let rinc = if ref1 { rot } else { rot.transpose() };
    let sgn = if posoi[a] >= 0.0 { 1.0 } else { -1.0 };
    let mut binc = 0;
    for k in 1..3 {
        if rinc[(a, k)].abs() > rinc[(a, binc)].abs() {
            binc = k;
        }
    }
    let tinc = if sgn * rinc[(a, binc)] > 0.0 { -1.0 } else { 1.0 };
    let (ax, ay) = ((a + 1) % 3, (a + 2) % 3);
    let (bu, bv) = ((binc + 1) % 3, (binc + 2) % 3);
    let (mut cx, mut du, mut dv) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    for rr in 0..3 {
        let c = if rr == 0 { ax } else if rr == 1 { ay } else { a };
        cx[rr] = posoi[c] + tinc * sizeinc[binc] * rinc[(c, binc)];
        du[rr] = sizeinc[bu] * rinc[(c, bu)];
        dv[rr] = sizeinc[bv] * rinc[(c, bv)];
    }
    cx[2] = sgn * cx[2] - sizeref[a];
    du[2] *= sgn;
    dv[2] *= sgn;
    let corner_sign = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];
    let mut poly: Vec<[f64; 3]> = corner_sign.iter().map(|[su, sv]| [cx[0] + su * du[0] + sv * dv[0], cx[1] + su * du[1] + sv * dv[1], cx[2] + su * du[2] + sv * dv[2]]).collect();
    clip_half_plane(&mut poly, 0, 1.0, sizeref[ax]);
    clip_half_plane(&mut poly, 0, -1.0, sizeref[ax]);
    clip_half_plane(&mut poly, 1, 1.0, sizeref[ay]);
    clip_half_plane(&mut poly, 1, -1.0, sizeref[ay]);
    let dupe2 = BOXBOX_DUPEPS * (sizeref[ax] * sizeref[ax] + sizeref[ay] * sizeref[ay]);
    let mut accepted: Vec<[f64; 3]> = Vec::new();
    for v in &poly {
        if v[2] > margin {
            continue;
        }
        if accepted.iter().any(|q| (q[0] - v[0]).powi(2) + (q[1] - v[1]).powi(2) < dupe2) {
            continue;
        }
        accepted.push(*v);
    }
    let nsign = if ref1 { sgn } else { -sgn };
    let normal = matref.column(a) * nsign;
    accepted
        .iter()
        .map(|v| {
            let mut posc = Vector3::zeros();
            posc[ax] = v[0];
            posc[ay] = v[1];
            posc[a] = sgn * (sizeref[a] + 0.5 * v[2]);
            pre(v[2], matref * posc + posref, normal, Vector3::zeros())
        })
        .collect()
}

/// `mjc_PlaneConvex`, first contact only: the vertex of a convex set deepest below the plane. Any
/// correct hull gives the same support vertex, so the mesh's own vertices serve (MuJoCo's
/// `mjc_meshSupport` scans them all when no hull graph is available). MuJoCo's additional face contacts
/// from the hull polygon are not carried.
pub fn plane_convex_first(margin: f64, p1: &GeomPose, p2: &GeomPose, local_verts: &[Vector3<f64>]) -> Option<PreContact> {
    let normal = p1.axis();
    let local_dir = p2.mat.transpose() * (-normal);
    let mut best = f64::NEG_INFINITY;
    let mut v = Vector3::zeros();
    for lv in local_verts {
        let d = local_dir.dot(lv);
        if d > best {
            best = d;
            v = *lv;
        }
    }
    let vw = p2.mat * v + p2.pos;
    let dist = normal.dot(&(vw - p1.pos));
    if dist > margin {
        return None;
    }
    Some(pre(dist, vw - normal * (0.5 * dist), normal, Vector3::zeros()))
}

// ------------------------------------------------------------------------------------------------
// frame, pair parameters, filtering (engine_collision_driver.c)
// ------------------------------------------------------------------------------------------------

/// `mju_makeFrame`: complete `[normal, tangent]` into an orthonormal frame; a zero tangent is replaced by
/// `y` unless the normal is mostly `y`, in which case `z`.
pub fn make_frame(normal: Vector3<f64>, tangent: Vector3<f64>) -> [Vector3<f64>; 3] {
    let n = normal / normal.norm();
    let mut t = tangent;
    if t.dot(&t) < 0.25 {
        t = if n[1] < 0.5 && n[1] > -0.5 { Vector3::y() } else { Vector3::z() };
    }
    t -= n * n.dot(&t);
    t /= t.norm();
    [n, t, n.cross(&t)]
}

/// A geom's contact parameters as MJCF states them (defaults: condim 3, priority 0, solmix 1,
/// solref `0.02 1`, solimp `0.9 0.95 0.001 0.5 2`, friction `1 0.005 0.0001`, margin 0, gap 0,
/// contype 1, conaffinity 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomParams {
    pub condim: usize,
    pub priority: i32,
    pub solmix: f64,
    pub solref: [f64; 2],
    pub solimp: [f64; 5],
    /// sliding, torsional, rolling
    pub friction: [f64; 3],
    pub margin: f64,
    pub gap: f64,
    pub contype: u32,
    pub conaffinity: u32,
}

impl Default for GeomParams {
    fn default() -> Self {
        Self { condim: 3, priority: 0, solmix: 1.0, solref: [0.02, 1.0], solimp: [0.9, 0.95, 0.001, 0.5, 2.0], friction: [1.0, 0.005, 0.0001], margin: 0.0, gap: 0.0, contype: 1, conaffinity: 1 }
    }
}

/// The parameters a contact between two geoms carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairParams {
    pub condim: usize,
    pub solref: [f64; 2],
    pub solimp: [f64; 5],
    /// MuJoCo's unpacked 5-vector: `(slide, slide, spin, roll, roll)`.
    pub friction: [f64; 5],
}

/// `mj_contactParam`: the higher priority wins outright; otherwise condim max, friction max, solref/solimp
/// mixed by `solmix` (direct negative solref: min).
pub fn contact_param(a: &GeomParams, b: &GeomParams) -> PairParams {
    let (condim, solref, solimp, fri) = if a.priority > b.priority {
        (a.condim, a.solref, a.solimp, a.friction)
    } else if a.priority < b.priority {
        (b.condim, b.solref, b.solimp, b.friction)
    } else {
        let mix = if a.solmix >= MINVAL && b.solmix >= MINVAL {
            a.solmix / (a.solmix + b.solmix)
        } else if a.solmix < MINVAL && b.solmix < MINVAL {
            0.5
        } else if a.solmix < MINVAL {
            0.0
        } else {
            1.0
        };
        let solref = if a.solref[0] > 0.0 && b.solref[0] > 0.0 {
            [mix * a.solref[0] + (1.0 - mix) * b.solref[0], mix * a.solref[1] + (1.0 - mix) * b.solref[1]]
        } else {
            [a.solref[0].min(b.solref[0]), a.solref[1].min(b.solref[1])]
        };
        let mut solimp = [0.0; 5];
        for i in 0..5 {
            solimp[i] = mix * a.solimp[i] + (1.0 - mix) * b.solimp[i];
        }
        let fri = [a.friction[0].max(b.friction[0]), a.friction[1].max(b.friction[1]), a.friction[2].max(b.friction[2])];
        (a.condim.max(b.condim), solref, solimp, fri)
    };
    PairParams { condim, solref, solimp, friction: [fri[0], fri[0], fri[1], fri[2], fri[2]] }
}

/// `filterBitmask` inverted: two geoms (or bodies) can collide when either's `contype` meets the other's
/// `conaffinity`.
pub fn can_collide(contype1: u32, conaffinity1: u32, contype2: u32, conaffinity2: u32) -> bool {
    (contype1 & conaffinity2) != 0 || (contype2 & conaffinity1) != 0
}

/// `filterBodyPair` (awake bodies): `true` when the pair is skipped — same weld body, both dof-less, or
/// parent and child (by weld body) unless `filterparent` is disabled.
pub fn filter_body_pair(weldbody1: usize, weldparent1: usize, dofnum1: usize, weldbody2: usize, weldparent2: usize, dofnum2: usize, filterparent: bool) -> bool {
    if weldbody1 == weldbody2 {
        return true;
    }
    if dofnum1 == 0 && dofnum2 == 0 {
        return true;
    }
    if filterparent && weldbody1 != 0 && weldbody2 != 0 && (weldbody1 == weldparent2 || weldbody2 == weldparent1) {
        return true;
    }
    false
}

/// The pair's detection margin and gap: the sum of the two geoms' (an explicit `<pair>` states its own).
pub fn margin_and_gap(a: &GeomParams, b: &GeomParams) -> (f64, f64) {
    (a.margin + b.margin, a.gap + b.gap)
}

/// A contact as `mjContact` carries it into the solver.
#[derive(Clone, Debug)]
pub struct ContactRecord {
    pub dist: f64,
    pub pos: Vector3<f64>,
    /// `[normal, t1, t2]`, normal from geom 1 toward geom 2.
    pub frame: [Vector3<f64>; 3],
    pub dim: usize,
    pub includemargin: f64,
    pub friction: [f64; 5],
    pub solref: [f64; 2],
    pub solimp: [f64; 5],
    /// `dist ≥ includemargin`: found within the gap band but not handed to the solver.
    pub exclude: bool,
}

/// `mj_setContact`: attach the pair's parameters and complete the frame; `includemargin` is the pair's
/// margin (without the gap).
pub fn set_contact(c: &PreContact, params: &PairParams, includemargin: f64) -> ContactRecord {
    ContactRecord {
        dist: c.dist,
        pos: c.pos,
        frame: make_frame(c.normal, c.tangent),
        dim: params.condim,
        includemargin,
        friction: params.friction,
        solref: params.solref,
        solimp: params.solimp,
        exclude: c.dist >= includemargin,
    }
}

/// One geom for the pair dispatcher: type, pose, MuJoCo `size` (half-extents / radius+half-length), and
/// for a mesh its local vertices.
#[derive(Clone, Debug)]
pub struct CollisionGeom<'a> {
    pub kind: GeomType,
    pub pose: GeomPose,
    pub size: [f64; 3],
    pub verts: Option<&'a [Vector3<f64>]>,
}

/// `mjCOLLISIONFUNC[type1][type2]` for the pairs carried, with the lower type first (as MuJoCo orders
/// them); an `Err` names a pair this port does not carry. The returned contacts have their normal from the
/// FIRST argument toward the second, so a swapped pair has its normals flipped back.
pub fn collide_pair(margin: f64, g1: &CollisionGeom, g2: &CollisionGeom) -> Result<Vec<PreContact>, String> {
    use GeomType::*;
    let (a, b, swapped) = if g1.kind <= g2.kind { (g1, g2, false) } else { (g2, g1, true) };
    let cs = match (a.kind, b.kind) {
        (Plane, Sphere) => plane_sphere(margin, &a.pose, &b.pose, b.size[0]).into_iter().collect(),
        (Plane, Capsule) => plane_capsule(margin, &a.pose, &b.pose, [b.size[0], b.size[1]]),
        (Plane, Cylinder) => plane_cylinder(margin, &a.pose, &b.pose, [b.size[0], b.size[1]]),
        (Plane, Box) => plane_box(margin, &a.pose, &b.pose, b.size),
        (Plane, Mesh) => plane_convex_first(margin, &a.pose, &b.pose, b.verts.ok_or("mesh geom without vertices")?).into_iter().collect(),
        (Sphere, Sphere) => sphere_sphere(margin, &a.pose, a.size[0], &b.pose, b.size[0]).into_iter().collect(),
        (Sphere, Capsule) => sphere_capsule(margin, &a.pose, a.size[0], &b.pose, [b.size[0], b.size[1]]).into_iter().collect(),
        (Sphere, Cylinder) => sphere_cylinder(margin, &a.pose, a.size[0], &b.pose, [b.size[0], b.size[1]]).into_iter().collect(),
        (Sphere, Box) => sphere_box(margin, &a.pose, a.size[0], &b.pose, b.size).into_iter().collect(),
        (Capsule, Capsule) => capsule_capsule(margin, &a.pose, [a.size[0], a.size[1]], &b.pose, [b.size[0], b.size[1]]),
        (Box, Box) => box_box(margin, &a.pose, a.size, &b.pose, b.size),
        (x, y) => return Err(format!("{x:?}–{y:?} is not carried by this port (MuJoCo runs it through its native CCD)")),
    };
    Ok(if swapped {
        cs.into_iter()
            .map(|mut c| {
                c.normal = -c.normal;
                c
            })
            .collect()
    } else {
        cs
    })
}

#[cfg(test)]
// the 7-digit quaternions are the probe's verbatim inputs (MuJoCo normalises them, as nalgebra does here), and
// the expected normals are MuJoCo's printed doubles; replacing either with `FRAC_1_SQRT_2` would test a
// different pose or a different oracle
#[allow(clippy::approx_constant)]
mod tests {
    //! Every expected number is MuJoCo 3.13.0's `mjContact` (`dist`, `pos`, `frame`) from
    //! `scripts/mujoco_collision_probe.py`, with the bodies placed by `pos`/`quat` as stated there.
    use super::*;
    use nalgebra::{Quaternion, UnitQuaternion};

    fn pose(p: [f64; 3], q: [f64; 4]) -> GeomPose {
        let uq = UnitQuaternion::from_quaternion(Quaternion::new(q[0], q[1], q[2], q[3]));
        GeomPose { pos: Vector3::new(p[0], p[1], p[2]), mat: *uq.to_rotation_matrix().matrix() }
    }
    const ID: [f64; 4] = [1.0, 0.0, 0.0, 0.0];
    fn floor() -> GeomPose {
        pose([0.0, 0.0, 0.0], ID)
    }

    fn check(c: &PreContact, dist: f64, pos: [f64; 3], normal: [f64; 3], tol: f64) {
        assert!((c.dist - dist).abs() < tol, "dist {} vs MuJoCo {dist}", c.dist);
        let dp = (c.pos - Vector3::new(pos[0], pos[1], pos[2])).norm();
        assert!(dp < tol, "pos {:?} vs MuJoCo {pos:?} ({dp:.2e})", c.pos);
        let dn = (c.normal - Vector3::new(normal[0], normal[1], normal[2])).norm();
        assert!(dn < tol, "normal {:?} vs MuJoCo {normal:?}", c.normal);
    }

    #[test]
    fn plane_pairs_match_mujoco() {
        let c = plane_sphere(0.0, &floor(), &pose([0.2, 0.3, 0.095], ID), 0.1).unwrap();
        check(&c, -0.0050000000000000044, [0.2, 0.3, -0.0025000000000000022], [0.0, 0.0, 1.0], 1e-15);
        let cap = pose([0.0, 0.0, 0.12], [0.9659258, 0.0, 0.2588190, 0.0]);
        let cs = plane_capsule(0.0, &floor(), &cap, [0.05, 0.2]);
        assert_eq!(cs.len(), 1);
        check(&cs[0], -0.10320508810920316, [-0.09999998726541509, 0.0, -0.05160254405460158], [0.0, 0.0, 1.0], 1e-12);
        assert!((cs[0].tangent - cap.axis()).norm() < 1e-12, "capsule axis is the preferred tangent");
        let cs = plane_cylinder(0.0, &floor(), &pose([0.1, 0.0, 0.14], [0.9848078, 0.1736482, 0.0, 0.0]), [0.1, 0.15]);
        assert_eq!(cs.len(), 1);
        check(&cs[0], -0.0351559086309797, [0.1, -0.04266623573337449, -0.01757795431548985], [0.0, 0.0, 1.0], 1e-12);
        let cs = plane_box(0.0, &floor(), &pose([0.0, 0.0, 0.06], [0.9961947, 0.0871557, 0.0, 0.0]), [0.1, 0.15, 0.05]);
        assert_eq!(cs.len(), 2);
        check(&cs[0], -0.015287602412473095, [-0.1, -0.13903876050577182, -0.007643801206236547], [0.0, 0.0, 1.0], 1e-12);
        check(&cs[1], -0.015287602412473095, [0.1, -0.13903876050577182, -0.007643801206236547], [0.0, 0.0, 1.0], 1e-12);
    }

    #[test]
    fn sphere_pairs_match_mujoco() {
        let c = sphere_sphere(0.0, &pose([0.0, 0.0, 1.0], ID), 0.1, &pose([0.2, 0.1, 1.05], ID), 0.15).unwrap();
        check(&c, -0.020871215252207975, [0.07817821097640076, 0.03908910548820038, 1.0195445527441003], [0.8728715609439697, 0.43643578047198484, 0.2182178902359926], 1e-14);
        let cap = pose([0.0, 0.0, 1.0], [0.7071068, 0.0, 0.7071068, 0.0]);
        let c = sphere_capsule(0.0, &pose([0.1, 0.03, 1.12], ID), 0.1, &cap, [0.05, 0.3]).unwrap();
        check(&c, -0.026306831231470096, [0.10000000000000005, 0.00893660937409168, 1.0357464374963667], [4.487810586786504e-16, -0.24253562503633277, -0.970142500145332], 1e-12);
        let c = sphere_capsule(0.0, &pose([0.36, 0.02, 1.1], ID), 0.1, &cap, [0.05, 0.3]).unwrap();
        check(&c, -0.03167840433800764, [0.3173226861790723, 0.00577422872635742, 1.028871143631787], [-0.5070925528371093, -0.16903085094570325, -0.845154254728517], 1e-12);
        let cyl = pose([0.0, 0.0, 1.0], ID);
        let c = sphere_cylinder(0.0, &pose([0.18, 0.02, 1.05], ID), 0.1, &cyl, [0.1, 0.15]).unwrap();
        check(&c, -0.018892297237251693, [0.09, 0.01, 1.05], [-0.993883734673619, -0.11043152607484656, 0.0], 1e-14);
        let c = sphere_cylinder(0.0, &pose([0.02, 0.03, 1.24], ID), 0.1, &cyl, [0.1, 0.15]).unwrap();
        check(&c, -0.009999999999999926, [0.02, 0.03, 1.145], [0.0, 0.0, -1.0], 1e-14);
        let bx = pose([0.0, 0.0, 1.0], ID);
        let c = sphere_box(0.0, &pose([0.05, 0.02, 1.24], ID), 0.1, &bx, [0.2, 0.3, 0.15]).unwrap();
        check(&c, -0.010000000000000009, [0.05, 0.02, 1.145], [0.0, 0.0, -1.0], 1e-14);
        let c = sphere_box(0.0, &pose([0.26, 0.36, 1.0], ID), 0.1, &bx, [0.2, 0.3, 0.15]).unwrap();
        check(&c, -0.015147186257614298, [0.19464466094067262, 0.29464466094067265, 1.0], [-0.7071067811865476, -0.7071067811865476, 0.0], 1e-14);
    }

    #[test]
    fn capsule_capsule_matches_mujoco_skew_and_parallel() {
        let c1 = pose([0.0, 0.0, 1.0], ID);
        let cs = capsule_capsule(0.0, &c1, [0.05, 0.3], &pose([0.05, 0.02, 1.1], [0.7071068, 0.7071068, 0.0, 0.0]), [0.04, 0.25]);
        assert_eq!(cs.len(), 1);
        check(&cs[0], -0.04, [0.030000000000000002, 0.0, 1.1], [1.0, 0.0, 0.0], 1e-12);
        let cs = capsule_capsule(0.0, &c1, [0.05, 0.3], &pose([0.08, 0.0, 1.1], ID), [0.04, 0.25]);
        assert_eq!(cs.len(), 2, "parallel capsules give two contacts");
        check(&cs[0], -0.010000000000000002, [0.045, 0.0, 1.3], [1.0, 0.0, 0.0], 1e-14);
        check(&cs[1], -0.010000000000000002, [0.045, 0.0, 0.8500000000000001], [1.0, 0.0, 0.0], 1e-14);
    }

    #[test]
    fn box_box_matches_mujoco_face_and_edge() {
        let b1 = pose([0.0, 0.0, 1.0], ID);
        let cs = box_box(0.0, &b1, [0.2, 0.3, 0.15], &pose([0.05, 0.1, 1.24], [0.9961947, 0.0, 0.0, 0.0871557]), [0.1, 0.1, 0.1]);
        assert_eq!(cs.len(), 4, "face contact: the incident face clipped to the reference face");
        let expect = [
            [0.13111596743962825, 0.21584558613228721, 1.145],
            [-0.06584558613228722, 0.18111596743962827, 1.145],
            [-0.03111596743962825, -0.015845586132287218, 1.145],
            [0.1658455861322872, 0.018884032560371758, 1.145],
        ];
        for (c, e) in cs.iter().zip(expect) {
            check(c, -0.010000000000000009, e, [0.0, 0.0, 1.0], 1e-12);
        }
        let cs = box_box(0.0, &b1, [0.2, 0.3, 0.15], &pose([0.25, 0.0, 1.2], [0.9238795, 0.0, 0.3826834, 0.0]), [0.1, 0.1, 0.1]);
        assert_eq!(cs.len(), 2, "an edge resting on a face");
        check(&cs[0], -0.029289321881345337, [0.18964466130227925, -0.1, 1.139644660579066], [0.7071067564945002, 0.0, 0.7071068058785942], 1e-9);
        check(&cs[1], -0.029289321881345337, [0.18964466130227925, 0.10000000000000005, 1.139644660579066], [0.7071067564945002, 0.0, 0.7071068058785942], 1e-9);
    }

    #[test]
    fn frame_completion_is_mujocos() {
        let f = make_frame(Vector3::z(), Vector3::zeros());
        assert!((f[1] - Vector3::y()).norm() < 1e-15 && (f[2] - Vector3::new(-1.0, 0.0, 0.0)).norm() < 1e-15);
        let f = make_frame(Vector3::new(0.8728715609439697, 0.43643578047198484, 0.2182178902359926), Vector3::zeros());
        assert!((f[1] - Vector3::new(-0.4234048992199707, 0.8997354108424372, -0.10585122480499276)).norm() < 1e-14);
        let f = make_frame(Vector3::new(4.487810586786504e-16, -0.24253562503633277, -0.970142500145332), Vector3::zeros());
        assert!((f[1] - Vector3::new(1.121952646696625e-16, 0.970142500145332, -0.2425356250363328)).norm() < 1e-14);
    }

    #[test]
    fn pair_parameters_mix_as_mujoco_mixes_them() {
        let floor = GeomParams { friction: [0.3, 0.005, 0.0001], solref: [0.05, 0.8], solimp: [0.8, 0.9, 0.002, 0.4, 3.0], solmix: 2.0, condim: 3, ..Default::default() };
        let ball = GeomParams { friction: [0.7, 0.005, 0.0001], solref: [0.01, 1.2], solimp: [0.95, 0.99, 0.005, 0.6, 1.0], solmix: 0.5, condim: 4, margin: 0.01, gap: 0.002, ..Default::default() };
        let p = contact_param(&floor, &ball);
        assert_eq!(p.condim, 4);
        assert_eq!(p.friction, [0.7, 0.7, 0.005, 0.0001, 0.0001]);
        assert!((p.solref[0] - 0.042).abs() < 1e-15 && (p.solref[1] - 0.88).abs() < 1e-15);
        let want = [0.8300000000000001, 0.918, 0.0026, 0.44, 2.6];
        for i in 0..5 {
            assert!((p.solimp[i] - want[i]).abs() < 1e-14, "solimp[{i}] {} vs {}", p.solimp[i], want[i]);
        }
        let (margin, gap) = margin_and_gap(&floor, &ball);
        assert_eq!((margin, gap), (0.01, 0.002));
        // detection at margin + gap, inclusion at margin: a ball 5 mm up is found and kept (dist < 0.01)
        let c = plane_sphere(margin + gap, &GeomPose { pos: Vector3::zeros(), mat: Matrix3::identity() }, &GeomPose { pos: Vector3::new(0.0, 0.0, 0.105), mat: Matrix3::identity() }, 0.1).unwrap();
        let rec = set_contact(&c, &p, margin);
        assert!((rec.dist - 0.0049999999999999906).abs() < 1e-15 && !rec.exclude && rec.includemargin == 0.01);
        // priority wins outright, condim included
        let floor = GeomParams { friction: [0.3, 0.005, 0.0001], priority: 1, solref: [0.05, 0.8], condim: 1, ..Default::default() };
        let ball = GeomParams { friction: [0.7, 0.005, 0.0001], solref: [0.01, 1.2], condim: 6, ..Default::default() };
        let p = contact_param(&floor, &ball);
        assert_eq!((p.condim, p.friction[0], p.solref), (1, 0.3, [0.05, 0.8]));
    }

    #[test]
    fn filters_are_mujocos() {
        assert!(can_collide(1, 1, 1, 1));
        assert!(!can_collide(0, 0, 1, 1), "a visual geom (0/0) never collides");
        assert!(can_collide(2, 0, 0, 2), "one side's contype meeting the other's conaffinity is enough");
        assert!(filter_body_pair(3, 1, 1, 3, 1, 1, true), "same weld body");
        assert!(filter_body_pair(0, 0, 0, 4, 0, 0, true), "both dof-less");
        assert!(filter_body_pair(2, 1, 1, 3, 2, 1, true), "parent and child");
        assert!(!filter_body_pair(2, 1, 1, 3, 2, 1, false), "unless filterparent is disabled");
        assert!(!filter_body_pair(0, 0, 0, 3, 2, 1, true), "the world against a moving body");
    }

    #[test]
    fn the_dispatcher_orders_types_and_flips_a_swapped_normal() {
        let floor = CollisionGeom { kind: GeomType::Plane, pose: GeomPose { pos: Vector3::zeros(), mat: Matrix3::identity() }, size: [1.0, 1.0, 0.1], verts: None };
        let ball = CollisionGeom { kind: GeomType::Sphere, pose: pose([0.2, 0.3, 0.095], ID), size: [0.1, 0.0, 0.0], verts: None };
        let a = collide_pair(0.0, &floor, &ball).unwrap();
        let b = collide_pair(0.0, &ball, &floor).unwrap();
        assert_eq!(a.len(), 1);
        assert!((a[0].normal + b[0].normal).norm() < 1e-15 && (a[0].pos - b[0].pos).norm() < 1e-15);
        let mesh = [Vector3::new(-0.1, -0.1, -0.1), Vector3::new(0.1, -0.1, -0.1), Vector3::new(0.0, 0.1, -0.1), Vector3::new(0.0, 0.0, 0.1)];
        let tet = CollisionGeom { kind: GeomType::Mesh, pose: pose([0.0, 0.0, 0.05], ID), size: [0.0; 3], verts: Some(&mesh) };
        let c = collide_pair(0.0, &floor, &tet).unwrap();
        assert_eq!(c.len(), 1);
        assert!((c[0].dist + 0.05).abs() < 1e-15 && (c[0].pos.z + 0.025).abs() < 1e-15);
        assert!(collide_pair(0.0, &tet, &tet).is_err(), "mesh–mesh names itself as not carried");
    }
}
