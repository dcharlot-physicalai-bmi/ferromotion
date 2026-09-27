//! **MuJoCo's native convex collision detection, ported line for line** — the GJK distance query, the EPA
//! penetration depth on an expanding polytope, and the multi-contact face clipping that MuJoCo 3.x runs for
//! every pair its closed-form primitive routines do not cover: meshes, cylinders, ellipsoids and boxes
//! against each other (`engine_collision_gjk.c`, `engine_collision_convex.c`).
//!
//! The algorithm as MuJoCo runs it:
//!
//! 1. **GJK** on the Minkowski difference, seeded at the two geoms' centres, with the barycentric
//!    sub-distance routines `S1D`/`S2D`/`S3D` written out (no matrix solves), MuJoCo's own termination rules
//!    (`ccd_tolerance`, `ccd_iterations`, an exact-arithmetic `epsilon` of zero for pairs of discrete geoms),
//!    the `gjkIntersect` fallback when only a yes/no is needed, and the sphere/capsule shortcut that runs
//!    GJK on a point or a segment and inflates the answer by the radius afterwards.
//! 2. **EPA** when GJK ends touching: the simplex is grown into a polytope (`polytope2/3/4`, with MuJoCo's
//!    120°-rotated seed directions and its degenerate-face fallbacks), then the face nearest the origin is
//!    expanded through its horizon until the upper and lower bounds meet within tolerance; discrete pairs stop
//!    as soon as a support vertex repeats.
//! 3. **Multi-contact**: the EPA face's three Minkowski vertices identify, on each geom, a face, an edge or a
//!    vertex; aligned faces (or an edge on a face) are clipped against each other (`polygonClip`) and up to
//!    four witness pairs are kept (`hull4`, the largest-area quadrilateral). Boxes, cylinders (a 16-gon cap)
//!    and hull polygons all take part. When `multiccd` is off, or the pair is not eligible, `mjc_Convex`
//!    instead perturbs the pair by ±1e-3 rad about the contact frame's tangents and keeps distinct extra
//!    contacts.
//!
//! Every number in the tests is MuJoCo 3.13.0's (`scripts/mujoco_ccd_probe.py`, and
//! `scripts/mujoco_ccd_harness.c` for the probes pinned bit for bit). Where MuJoCo's answer is itself only
//! tolerance-exact (a curved surface, an iterative distance) the tests say so with the tolerance MuJoCo's own
//! `ccd_tolerance` implies; where the answer is combinatorial (which face, which corners) it is exact to
//! floating-point.
//!
//! ⭐ The arithmetic is contracted exactly as clang contracts MuJoCo's C on Apple silicon (see below), and
//! the port is **bit-identical to MuJoCo's own `engine_collision_gjk.c` built that way on all 3,614 geom
//! pairs** that reach this code across Menagerie (`MENAGERIE_CCD_DUMP` in `examples/menagerie_contacts`,
//! rerun by `scripts/mujoco_ccd_pairs.c`, compared by `scripts/mujoco_ccd_pairs_compare.py`). The same
//! source built without contraction — the arithmetic of MuJoCo's x86-64 wheels — differs from it on 939 of
//! those pairs, 40 of them beyond 1e-9 (reversed normals at equal depth, other corners of a tied face).

use crate::mujoco_collision::{GeomPose, GeomType, PreContact};
use crate::mujoco_hull::MeshHull;
use nalgebra::Vector3;

const MINVAL: f64 = 1e-15;
const MINVAL2: f64 = MINVAL * MINVAL;
const MAXVAL: f64 = 1e10;
const MAXVAL2: f64 = MAXVAL * MAXVAL;
const MAX_LIMIT: f64 = f64::MAX;
const MINDIST2: f64 = MINVAL2;
const MINDIST3: f64 = MINVAL2;
const MINDIST4: f64 = MINVAL2;
const MINEPATOL: f64 = MINVAL;
const FACE_TOL: f64 = 0.996;
const EDGE_TOL: f64 = 0.0888;
const MAXCONPAIR: usize = 50;

type V3 = Vector3<f64>;

// ⛔ MuJoCo's Apple-silicon build contracts `a*b + c` into one fused multiply–add wherever clang can, and the
// CCD's ties (which face, which corner) follow the last bit. Every expression below is written the way clang
// contracts MuJoCo's C: in `a*b + c*d` the LEFT product is fused and the right one rounded; a chain
// `x + y + z` of products fuses each later product into the running sum; `s += a*b` is one fma.

/// `dot3` / `mju_dot3`.
#[inline]
fn dot3(a: &V3, b: &V3) -> f64 {
    a.z.mul_add(b.z, a.x.mul_add(b.x, a.y * b.y))
}

/// `norm3` / `mju_norm3`.
#[inline]
fn norm3(v: &V3) -> f64 {
    dot3(v, v).sqrt()
}

/// `cross3` / `mji_cross` / `mju_cross`.
#[inline]
fn cross3(a: &V3, b: &V3) -> V3 {
    V3::new(a.y.mul_add(b.z, -(a.z * b.y)), a.z.mul_add(b.x, -(a.x * b.z)), a.x.mul_add(b.y, -(a.y * b.x)))
}

/// `addScl3` / `mji_addToScl3`: `v1 + s·v2`, fused.
#[inline]
fn add_scl3(v1: &V3, v2: &V3, s: f64) -> V3 {
    V3::new(s.mul_add(v2.x, v1.x), s.mul_add(v2.y, v1.y), s.mul_add(v2.z, v1.z))
}

/// One row of a 3×3 product, `a·x + b·y + c·z`, as clang contracts it.
#[inline]
fn row3(a: f64, b: f64, c: f64, v: &V3) -> f64 {
    c.mul_add(v.z, a.mul_add(v.x, b * v.y))
}

/// `mulMatTVec3` / `mju_mulMatTVec3`: `matᵀ · v`.
#[inline]
fn mat_t_vec(m: &nalgebra::Matrix3<f64>, v: &V3) -> V3 {
    V3::new(row3(m[(0, 0)], m[(1, 0)], m[(2, 0)], v), row3(m[(0, 1)], m[(1, 1)], m[(2, 1)], v), row3(m[(0, 2)], m[(1, 2)], m[(2, 2)], v))
}

/// `mju_mulMatVec3` / `localToGlobal` / `globalcoord` without the translation: `mat · v`.
#[inline]
fn mat_vec(m: &nalgebra::Matrix3<f64>, v: &V3) -> V3 {
    V3::new(row3(m[(0, 0)], m[(0, 1)], m[(0, 2)], v), row3(m[(1, 0)], m[(1, 1)], m[(1, 2)], v), row3(m[(2, 0)], m[(2, 1)], m[(2, 2)], v))
}

/// `localToGlobal`: `mat · v`, then `+ pos` as a separate statement.
#[inline]
fn local_to_global(m: &nalgebra::Matrix3<f64>, v: &V3, pos: &V3) -> V3 {
    mat_vec(m, v) + pos
}

/// The minors `M_i4` of `S2D` / `triAffineCoord`: `a₀a₁ − b₀b₁ − c₀c₁ + d₀d₁ + e₀e₁ − f₀f₁`, left to right.
#[inline]
fn minor6(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64), e: (f64, f64), f: (f64, f64)) -> f64 {
    let t = a.0.mul_add(a.1, -(b.0 * b.1));
    let t = (-c.0).mul_add(c.1, t);
    let t = d.0.mul_add(d.1, t);
    let t = e.0.mul_add(e.1, t);
    (-f.0).mul_add(f.1, t)
}

/// The cofactors `C3i`: `a₀a₁ + b₀b₁ + c₀c₁ − d₀d₁ − e₀e₁ − f₀f₁`, left to right.
#[inline]
fn cofactor6(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64), e: (f64, f64), f: (f64, f64)) -> f64 {
    let t = a.0.mul_add(a.1, b.0 * b.1);
    let t = c.0.mul_add(c.1, t);
    let t = (-d.0).mul_add(d.1, t);
    let t = (-e.0).mul_add(e.1, t);
    (-f.0).mul_add(f.1, t)
}

/// One convex geom as the CCD sees it (`mjCCDObj`): type, pose, MuJoCo size, hull data for a mesh.
#[derive(Clone)]
pub struct CcdObj<'a> {
    pub kind: GeomType,
    pub pose: GeomPose,
    pub size: [f64; 3],
    pub hull: Option<&'a MeshHull>,
    /// Half the pair's margin is added to each support point (`obj->margin` in MuJoCo is the pair margin).
    pub margin: f64,
    /// The last support vertex index (box corner code, cylinder cap, mesh vertex) — MuJoCo caches it in the
    /// object and the multi-contact stage reads it back through the polytope vertices.
    vertindex: i64,
    /// The hill-climbing support's cached hull vertex (`obj->meshindex`, a hull-local index).
    meshindex: i64,
    /// Sphere/capsule shortcut: the support is a point or a segment while this is set.
    reduced: Option<Reduced>,
}

#[derive(Clone, Copy, PartialEq)]
enum Reduced {
    Point,
    Line,
}

impl<'a> CcdObj<'a> {
    pub fn new(kind: GeomType, pose: GeomPose, size: [f64; 3], hull: Option<&'a MeshHull>, margin: f64) -> Self {
        Self { kind, pose, size, hull, margin, vertindex: -1, meshindex: -1, reduced: None }
    }

    fn center(&self) -> V3 {
        self.pose.pos
    }

    /// `mjCGeom::GetRBound`: the bounding-sphere radius MuJoCo stores as `geom_rbound`.
    pub fn rbound(&self) -> f64 {
        let s = self.size;
        match self.kind {
            GeomType::Sphere => s[0],
            GeomType::Capsule => s[0] + s[1],
            GeomType::Cylinder => (s[0] * s[0] + s[1] * s[1]).sqrt(),
            GeomType::Ellipsoid => s[0].max(s[1]).max(s[2]),
            GeomType::Box => (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt(),
            GeomType::Mesh => self.hull.map(|h| h.rbound()).unwrap_or(0.0),
            GeomType::Plane | GeomType::HField => 0.0,
        }
    }

    /// The support point in the world along `dir` (unit), with MuJoCo's per-type conventions and the
    /// `vertindex` side effect.
    fn support(&mut self, dir: &V3) -> V3 {
        let mat = self.pose.mat;
        let pos = self.pose.pos;
        match self.reduced {
            // `mjc_pointSupport`
            Some(Reduced::Point) => return pos,
            // `mjc_lineSupport`
            Some(Reduced::Line) => {
                let dot = row3(mat[(0, 2)], mat[(1, 2)], mat[(2, 2)], dir);
                let scl = if dot >= 0.0 { self.size[1] } else { -self.size[1] };
                return V3::new(mat[(0, 2)].mul_add(scl, pos.x), mat[(1, 2)].mul_add(scl, pos.y), mat[(2, 2)].mul_add(scl, pos.z));
            }
            None => {}
        }
        if self.kind == GeomType::Sphere {
            let r = self.size[0];
            return V3::new(r.mul_add(dir.x, pos.x), r.mul_add(dir.y, pos.y), r.mul_add(dir.z, pos.z));
        }
        let local_dir = mat_t_vec(&mat, dir);
        match self.kind {
            GeomType::Capsule => {
                let mut ls = V3::new(local_dir.x * self.size[0], local_dir.y * self.size[0], local_dir.z * self.size[0]);
                ls.z += if local_dir.z >= 0.0 { self.size[1] } else { -self.size[1] };
                local_to_global(&mat, &ls, &pos)
            }
            GeomType::Ellipsoid => {
                let mut ls = Vector3::new(local_dir.x * self.size[0], local_dir.y * self.size[1], local_dir.z * self.size[2]);
                let n2 = dot3(&ls, &ls);
                if n2 < MINVAL2 {
                    return V3::new(mat[(0, 0)].mul_add(self.size[0], pos.x), mat[(1, 0)].mul_add(self.size[0], pos.y), mat[(2, 0)].mul_add(self.size[0], pos.z));
                }
                let inv = 1.0 / n2.sqrt();
                ls.x *= inv * self.size[0];
                ls.y *= inv * self.size[1];
                ls.z *= inv * self.size[2];
                local_to_global(&mat, &ls, &pos)
            }
            GeomType::Cylinder => {
                let n2 = local_dir.x.mul_add(local_dir.x, local_dir.y * local_dir.y);
                let scl = if n2 >= MINVAL2 { self.size[0] / n2.sqrt() } else { 0.0 };
                let ls = Vector3::new(scl * local_dir.x, scl * local_dir.y, if local_dir.z >= 0.0 { self.size[1] } else { -self.size[1] });
                self.vertindex = if local_dir.z >= 0.0 { 0 } else { 1 };
                local_to_global(&mat, &ls, &pos)
            }
            GeomType::Box => {
                let ls = Vector3::new(
                    if local_dir.x >= 0.0 { self.size[0] } else { -self.size[0] },
                    if local_dir.y >= 0.0 { self.size[1] } else { -self.size[1] },
                    if local_dir.z >= 0.0 { self.size[2] } else { -self.size[2] },
                );
                self.vertindex = (ls.x > 0.0) as i64 | ((ls.y > 0.0) as i64) << 1 | ((ls.z > 0.0) as i64) << 2;
                local_to_global(&mat, &ls, &pos)
            }
            GeomType::Mesh => {
                let hull = self.hull.expect("mesh geom without hull data");
                let v = hull.support_mujoco(&local_dir, &mut self.vertindex, &mut self.meshindex);
                local_to_global(&mat, &v, &pos)
            }
            GeomType::Sphere | GeomType::Plane | GeomType::HField => pos,
        }
    }
}

/// A vertex of the Minkowski difference (`Vertex`).
#[derive(Clone, Copy, Debug, Default)]
struct Vertex {
    vert: V3,
    vert1: V3,
    vert2: V3,
    index1: i64,
    index2: i64,
}

/// `support`: the Minkowski-difference vertex along `dir` (obj1) and `dir_neg` (obj2), margins included.
fn mink_support(obj1: &mut CcdObj, obj2: &mut CcdObj, dir: &V3, dir_neg: &V3) -> Vertex {
    let mut v1 = obj1.support(dir);
    if obj1.margin > 0.0 {
        let m = 0.5 * obj1.margin;
        v1 = V3::new(dir.x.mul_add(m, v1.x), dir.y.mul_add(m, v1.y), dir.z.mul_add(m, v1.z));
    }
    let mut v2 = obj2.support(dir_neg);
    if obj2.margin > 0.0 {
        let m = 0.5 * obj2.margin;
        v2 = V3::new(dir_neg.x.mul_add(m, v2.x), dir_neg.y.mul_add(m, v2.y), dir_neg.z.mul_add(m, v2.z));
    }
    Vertex { vert: v1 - v2, vert1: v1, vert2: v2, index1: obj1.vertindex, index2: obj2.vertindex }
}

/// `mju_normalize3`: multiply by the reciprocal norm (not divide — the last bit decides exact ties in the
/// polygon pruning below), `(1, 0, 0)` for a vector shorter than `mjMINVAL`.
fn normalize3(v: &V3) -> V3 {
    let norm = norm3(v);
    if norm < MINVAL {
        Vector3::x()
    } else {
        let inv = 1.0 / norm;
        Vector3::new(v.x * inv, v.y * inv, v.z * inv)
    }
}

/// `det3`: `v1 · (v2 × v3)` as MuJoCo writes it out, contracted.
fn det3(v1: &V3, v2: &V3, v3: &V3) -> f64 {
    let a = v2.y.mul_add(v3.z, -(v2.z * v3.y));
    let b = v2.z.mul_add(v3.x, -(v2.x * v3.z));
    let c = v2.x.mul_add(v3.y, -(v2.y * v3.x));
    v1.z.mul_add(c, v1.x.mul_add(a, v1.y * b))
}

fn same_sign2(a: f64, b: f64) -> i32 {
    if a > 0.0 && b > 0.0 {
        1
    } else if a < 0.0 && b < 0.0 {
        -1
    } else {
        0
    }
}

/// `lincomb`: `Σ coefᵢ·vᵢ` for up to four vectors, contracted as MuJoCo's unrolled sums are.
fn lincomb(coef: &[f64], vs: &[V3]) -> V3 {
    let comp = |k: usize| -> f64 {
        match coef.len() {
            0 => 0.0,
            1 => coef[0] * vs[0][k],
            2 => coef[0].mul_add(vs[0][k], coef[1] * vs[1][k]),
            3 => coef[2].mul_add(vs[2][k], coef[0].mul_add(vs[0][k], coef[1] * vs[1][k])),
            _ => coef[3].mul_add(vs[3][k], coef[2].mul_add(vs[2][k], coef[0].mul_add(vs[0][k], coef[1] * vs[1][k]))),
        }
    };
    V3::new(comp(0), comp(1), comp(2))
}

/// `projectOriginPlane`: the origin's projection on the plane of three points; `None` if degenerate.
fn project_origin_plane(v1: &V3, v2: &V3, v3: &V3) -> Option<V3> {
    let diff21 = v2 - v1;
    let diff31 = v3 - v1;
    let diff32 = v3 - v2;
    let n = cross3(&diff32, &diff21);
    let (nv, nn) = (dot3(&n, v2), dot3(&n, &n));
    if nn == 0.0 {
        return None;
    }
    if nv != 0.0 && nn > MINVAL {
        return Some(n * (nv / nn));
    }
    let n = cross3(&diff21, &diff31);
    let (nv, nn) = (dot3(&n, v1), dot3(&n, &n));
    if nn == 0.0 {
        return None;
    }
    if nv != 0.0 && nn > MINVAL {
        return Some(n * (nv / nn));
    }
    let n = cross3(&diff31, &diff32);
    let (nv, nn) = (dot3(&n, v3), dot3(&n, &n));
    Some(n * (nv / nn))
}

fn project_origin_line(v1: &V3, v2: &V3) -> V3 {
    let diff = v2 - v1;
    let scl = -(dot3(v2, &diff) / dot3(&diff, &diff));
    add_scl3(v2, &diff, scl)
}

fn s1d(s1: &V3, s2: &V3) -> [f64; 2] {
    let p_o = project_origin_line(s1, s2);
    let mut mu_max = s1.x - s2.x;
    let mut index = 0;
    let mu = s1.y - s2.y;
    if mu.abs() >= mu_max.abs() {
        mu_max = mu;
        index = 1;
    }
    let mu = s1.z - s2.z;
    if mu.abs() >= mu_max.abs() {
        mu_max = mu;
        index = 2;
    }
    let c1 = p_o[index] - s2[index];
    let c2 = s1[index] - p_o[index];
    let same = same_sign2(mu_max, c1) != 0 && same_sign2(mu_max, c2) != 0;
    if same {
        [c1 / mu_max, c2 / mu_max]
    } else {
        [0.0, 1.0]
    }
}

fn s2d(s1: &V3, s2: &V3, s3: &V3) -> [f64; 3] {
    let Some(p_o) = project_origin_plane(s1, s2, s3) else {
        let l = s1d(s1, s2);
        return [l[0], l[1], 0.0];
    };
    let m14 = minor6((s2.y, s3.z), (s2.z, s3.y), (s1.y, s3.z), (s1.z, s3.y), (s1.y, s2.z), (s1.z, s2.y));
    let m24 = minor6((s2.x, s3.z), (s2.z, s3.x), (s1.x, s3.z), (s1.z, s3.x), (s1.x, s2.z), (s1.z, s2.x));
    let m34 = minor6((s2.x, s3.y), (s2.y, s3.x), (s1.x, s3.y), (s1.y, s3.x), (s1.x, s2.y), (s1.y, s2.x));
    let (mu1, mu2, mu3) = (m14.abs(), m24.abs(), m34.abs());
    let (m_max, x, y) = if mu1 >= mu2 && mu1 >= mu3 {
        (m14, 1, 2)
    } else if mu2 >= mu3 {
        (m24, 0, 2)
    } else {
        (m34, 0, 1)
    };
    let (s1_2, s2_2, s3_2, p_2) = ([s1[x], s1[y]], [s2[x], s2[y]], [s3[x], s3[y]], [p_o[x], p_o[y]]);
    let c31 = cofactor6((p_2[0], s2_2[1]), (p_2[1], s3_2[0]), (s2_2[0], s3_2[1]), (p_2[0], s3_2[1]), (p_2[1], s2_2[0]), (s3_2[0], s2_2[1]));
    let c32 = cofactor6((p_2[0], s3_2[1]), (p_2[1], s1_2[0]), (s3_2[0], s1_2[1]), (p_2[0], s1_2[1]), (p_2[1], s3_2[0]), (s1_2[0], s3_2[1]));
    let c33 = cofactor6((p_2[0], s1_2[1]), (p_2[1], s2_2[0]), (s1_2[0], s2_2[1]), (p_2[0], s2_2[1]), (p_2[1], s1_2[0]), (s2_2[0], s1_2[1]));
    let (comp1, comp2, comp3) = (same_sign2(m_max, c31) != 0, same_sign2(m_max, c32) != 0, same_sign2(m_max, c33) != 0);
    if comp1 && comp2 && comp3 {
        return [c31 / m_max, c32 / m_max, c33 / m_max];
    }
    let mut lambda = [0.0; 3];
    let mut dmin = MAX_LIMIT;
    if !comp1 {
        let l = s1d(s2, s3);
        let x = lincomb(&l, &[*s2, *s3]);
        lambda = [0.0, l[0], l[1]];
        dmin = dot3(&x, &x);
    }
    if !comp2 {
        let l = s1d(s1, s3);
        let x = lincomb(&l, &[*s1, *s3]);
        let d = dot3(&x, &x);
        if d < dmin {
            lambda = [l[0], 0.0, l[1]];
            dmin = d;
        }
    }
    if !comp3 {
        let l = s1d(s1, s2);
        let x = lincomb(&l, &[*s1, *s2]);
        let d = dot3(&x, &x);
        if d < dmin {
            lambda = [l[0], l[1], 0.0];
        }
    }
    lambda
}

fn s3d(s1: &V3, s2: &V3, s3: &V3, s4: &V3) -> [f64; 4] {
    let c41 = -det3(s2, s3, s4);
    let c42 = det3(s1, s3, s4);
    let c43 = -det3(s1, s2, s4);
    let c44 = det3(s1, s2, s3);
    let m_det = c41 + c42 + c43 + c44;
    let (comp1, comp2, comp3, comp4) = (same_sign2(m_det, c41) != 0, same_sign2(m_det, c42) != 0, same_sign2(m_det, c43) != 0, same_sign2(m_det, c44) != 0);
    if comp1 && comp2 && comp3 && comp4 {
        return [c41 / m_det, c42 / m_det, c43 / m_det, c44 / m_det];
    }
    let mut lambda = [0.0; 4];
    let mut dmin = MAX_LIMIT;
    if !comp1 {
        let l = s2d(s2, s3, s4);
        let x = lincomb(&l, &[*s2, *s3, *s4]);
        lambda = [0.0, l[0], l[1], l[2]];
        dmin = dot3(&x, &x);
    }
    if !comp2 {
        let l = s2d(s1, s3, s4);
        let x = lincomb(&l, &[*s1, *s3, *s4]);
        let d = dot3(&x, &x);
        if d < dmin {
            lambda = [l[0], 0.0, l[1], l[2]];
            dmin = d;
        }
    }
    if !comp3 {
        let l = s2d(s1, s2, s4);
        let x = lincomb(&l, &[*s1, *s2, *s4]);
        let d = dot3(&x, &x);
        if d < dmin {
            lambda = [l[0], l[1], 0.0, l[2]];
            dmin = d;
        }
    }
    if !comp4 {
        let l = s2d(s1, s2, s3);
        let x = lincomb(&l, &[*s1, *s2, *s3]);
        let d = dot3(&x, &x);
        if d < dmin {
            lambda = [l[0], l[1], l[2], 0.0];
        }
    }
    lambda
}

fn subdistance(n: usize, simplex: &[Vertex; 4]) -> [f64; 4] {
    let (s1, s2, s3, s4) = (&simplex[0].vert, &simplex[1].vert, &simplex[2].vert, &simplex[3].vert);
    match n {
        4 => s3d(s1, s2, s3, s4),
        3 => {
            let l = s2d(s1, s2, s3);
            [l[0], l[1], l[2], 0.0]
        }
        2 => {
            let l = s1d(s1, s2);
            [l[0], l[1], 0.0, 0.0]
        }
        _ => [1.0, 0.0, 0.0, 0.0],
    }
}

/// `mjCCDStatus`: what a query leaves behind.
#[derive(Clone, Debug)]
pub struct CcdStatus {
    pub separated: bool,
    pub dist: Vec<f64>,
    pub x1: Vec<V3>,
    pub x2: Vec<V3>,
    pub gjk_iterations: usize,
    pub epa_iterations: usize,
    /// `mjEPAStatus`: 0 success, −1 no contact, otherwise the polytope-construction failure code.
    pub epa_status: i32,
    simplex: [Vertex; 4],
    nsimplex: usize,
    tolerance: f64,
    max_iterations: usize,
    max_contacts: usize,
    dist_cutoff: f64,
}

/// `mjCCDConfig`.
#[derive(Clone, Copy, Debug)]
pub struct CcdConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub max_contacts: usize,
    pub dist_cutoff: f64,
}

fn discrete_geoms(o1: &CcdObj, o2: &CcdObj) -> bool {
    if o1.margin != 0.0 || o2.margin != 0.0 {
        return false;
    }
    let d = |k: GeomType| matches!(k, GeomType::Mesh | GeomType::Box | GeomType::HField);
    d(o1.kind) && d(o2.kind)
}

fn signed_distance(v1: &Vertex, v2: &Vertex, v3: &Vertex) -> (V3, f64) {
    let normal = cross3(&(v3.vert - v1.vert), &(v2.vert - v1.vert));
    let norm2 = dot3(&normal, &normal);
    if norm2 > MINVAL2 && norm2 < MAXVAL2 {
        let n = normal * (1.0 / norm2.sqrt());
        return (n, dot3(&n, &v1.vert));
    }
    (normal, MAX_LIMIT)
}

/// `gjkIntersect`: −1 = undecided, 1 = origin inside the tetrahedron, 0 = separated.
fn gjk_intersect(st: &mut CcdStatus, o1: &mut CcdObj, o2: &mut CcdObj) -> i32 {
    let mut simplex = st.simplex;
    let mut s = [0usize, 1, 2, 3];
    let mut k = st.gjk_iterations;
    let kmax = st.max_iterations;
    while k < kmax {
        let (n0, d0) = signed_distance(&simplex[s[2]], &simplex[s[1]], &simplex[s[3]]);
        let (n1, d1) = signed_distance(&simplex[s[0]], &simplex[s[2]], &simplex[s[3]]);
        let (n2, d2) = signed_distance(&simplex[s[1]], &simplex[s[0]], &simplex[s[3]]);
        let (n3, d3) = signed_distance(&simplex[s[0]], &simplex[s[1]], &simplex[s[2]]);
        let dist = [d0, d1, d2, d3];
        let normals = [n0, n1, n2, n3];
        if dist.contains(&0.0) {
            st.gjk_iterations = k;
            return -1;
        }
        let i = if dist[0] < dist[1] { 0 } else { 1 };
        let j = if dist[2] < dist[3] { 2 } else { 3 };
        let index = if dist[i] < dist[j] { i } else { j };
        if dist[index] > 0.0 {
            st.nsimplex = 4;
            st.simplex = [simplex[s[0]], simplex[s[1]], simplex[s[2]], simplex[s[3]]];
            st.gjk_iterations = k;
            return 1;
        }
        let dir = normals[index];
        simplex[s[index]] = mink_support(o1, o2, &dir, &(-dir));
        if dot3(&dir, &simplex[s[index]].vert) < 0.0 {
            st.nsimplex = 0;
            st.gjk_iterations = k;
            return 0;
        }
        let i = (index + 1) & 3;
        let j = (index + 2) & 3;
        s.swap(i, j);
        k += 1;
    }
    st.gjk_iterations = k;
    -1
}

/// `gjk`: the distance query; leaves the simplex and witness points in `st`.
fn gjk(st: &mut CcdStatus, o1: &mut CcdObj, o2: &mut CcdObj) {
    let get_dist = st.dist_cutoff > 0.0;
    let mut backup_gjk = !get_dist;
    let mut simplex = [Vertex::default(); 4];
    let mut n = 0usize;
    let mut k = 0usize;
    let kmax = st.max_iterations;
    let mut x1_k = st.x1[0];
    let mut x2_k = st.x2[0];
    let mut lambda = [1.0, 0.0, 0.0, 0.0];
    let tol2 = st.tolerance * st.tolerance;
    st.separated = false;
    let discrete = discrete_geoms(o1, o2);
    let epsilon = if discrete { 0.0 } else { 0.5 * tol2 };
    let min_norm = if discrete { MINVAL } else { st.tolerance };
    let mut x_k = x1_k - x2_k;
    let mut x_norm = norm3(&x_k);
    let mut x_norm_prev = 0.0;
    while k < kmax {
        if x_norm < min_norm || (x_norm_prev - x_norm).abs() < MINVAL {
            break;
        }
        let dir_neg = x_k * (1.0 / x_norm);
        simplex[n] = mink_support(o1, o2, &(-dir_neg), &dir_neg);
        let s_k = simplex[n].vert;
        let diff = x_k - s_k;
        if dot3(&x_k, &diff) < epsilon {
            break;
        }
        let lower = dot3(&x_k, &s_k);
        if !get_dist {
            if lower > 0.0 {
                st.separated = true;
                st.gjk_iterations = k;
                st.nsimplex = 0;
                st.x1.truncate(1);
                st.x2.truncate(1);
                st.dist = vec![MAX_LIMIT];
                return;
            }
        } else if st.dist_cutoff < MAX_LIMIT && lower > 0.0 && lower >= st.dist_cutoff * x_norm {
            st.separated = true;
            st.gjk_iterations = k;
            st.nsimplex = 0;
            st.dist = vec![MAX_LIMIT];
            return;
        }
        if n == 3 && backup_gjk {
            st.gjk_iterations = k;
            st.simplex = simplex;
            let ret = gjk_intersect(st, o1, o2);
            if ret != -1 {
                st.separated = ret == 0;
                st.dist = vec![if ret > 0 { 0.0 } else { MAX_LIMIT }];
                return;
            }
            k = st.gjk_iterations;
            backup_gjk = false;
        }
        let lam = subdistance(n + 1, &simplex);
        let mut m = 0;
        let mut kept = [Vertex::default(); 4];
        let mut lk = [0.0; 4];
        for i in 0..4 {
            if lam[i] == 0.0 {
                continue;
            }
            kept[m] = simplex[i];
            lk[m] = lam[i];
            m += 1;
        }
        n = m;
        simplex = kept;
        lambda = lk;
        if n < 1 {
            st.gjk_iterations = k;
            st.nsimplex = 0;
            st.dist = vec![MAX_LIMIT];
            st.separated = true;
            return;
        }
        x_k = lincomb(&lambda[..n], &[simplex[0].vert, simplex[1].vert, simplex[2].vert, simplex[3].vert][..n]);
        x_norm_prev = x_norm;
        x_norm = norm3(&x_k);
        k += 1;
        if n == 4 {
            break;
        }
    }
    if n > 0 {
        x1_k = lincomb(&lambda[..n], &[simplex[0].vert1, simplex[1].vert1, simplex[2].vert1, simplex[3].vert1][..n]);
        x2_k = lincomb(&lambda[..n], &[simplex[0].vert2, simplex[1].vert2, simplex[2].vert2, simplex[3].vert2][..n]);
    }
    // mark separation (`gjkSupport` divides by `x_norm` even when it is zero, as MuJoCo does)
    let dir_neg = x_k * (1.0 / x_norm);
    let tmp = mink_support(o1, o2, &(-dir_neg), &dir_neg);
    if dot3(&x_k, &tmp.vert) > 0.0 {
        st.separated = true;
    }
    st.x1 = vec![x1_k];
    st.x2 = vec![x2_k];
    st.gjk_iterations = k;
    st.nsimplex = n;
    st.simplex = simplex;
    st.dist = vec![if n == 4 && !st.separated { 0.0 } else { x_norm }];
}

// ------------------------------------------------------------------------------------------------
// EPA
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Face {
    verts: [usize; 3],
    adj: [usize; 3],
    v: V3,
    dist2: f64,
    /// index in the map; −1 not in map, −2 deleted
    index: i64,
}

struct Polytope {
    verts: Vec<Vertex>,
    faces: Vec<Face>,
    maxfaces: usize,
    center: V3,
    map: Vec<usize>,
    horizon_indices: Vec<usize>,
    horizon_edges: Vec<usize>,
    horizon_w: V3,
}

impl Polytope {
    fn new(max_iterations: usize) -> Self {
        Self { verts: Vec::with_capacity(5 + max_iterations), faces: Vec::with_capacity(6 * max_iterations), maxfaces: 6 * max_iterations, center: V3::zeros(), map: Vec::new(), horizon_indices: Vec::new(), horizon_edges: Vec::new(), horizon_w: V3::zeros() }
    }

    fn insert_vertex(&mut self, v: Vertex) -> usize {
        self.verts.push(v);
        self.verts.len() - 1
    }

    fn epa_support(&mut self, o1: &mut CcdObj, o2: &mut CcdObj, d: &V3, dnorm: f64) -> usize {
        let (dir, dir_neg) = if dnorm > MINVAL {
            let dir = V3::new(d.x / dnorm, d.y / dnorm, d.z / dnorm);
            (dir, dir * -1.0)
        } else {
            (Vector3::x(), -Vector3::x())
        };
        let v = mink_support(o1, o2, &dir, &dir_neg);
        self.insert_vertex(v)
    }

    fn delete_face(&mut self, fi: usize) {
        let idx = self.faces[fi].index;
        if idx >= 0 {
            let last = self.map.pop().unwrap();
            if (idx as usize) < self.map.len() {
                self.map[idx as usize] = last;
                self.faces[last].index = idx;
            } else if idx as usize == self.map.len() {
                // the deleted face was the last map entry; nothing to move
            }
        }
        self.faces[fi].index = -2;
    }

    fn attach_face(&mut self, v1: usize, v2: usize, v3: usize, adj1: usize, adj2: usize, adj3: usize) -> f64 {
        let proj = project_origin_plane(&self.verts[v3].vert, &self.verts[v2].vert, &self.verts[v1].vert);
        let mut face = Face { verts: [v1, v2, v3], adj: [adj1, adj2, adj3], v: V3::zeros(), dist2: 0.0, index: -1 };
        let Some(mut v) = proj else {
            self.faces.push(face);
            return 0.0;
        };
        let outward = self.verts[v1].vert - self.center;
        if dot3(&v, &outward) < 0.0 {
            v *= -1.0;
        }
        face.v = v;
        face.dist2 = dot3(&v, &v);
        self.faces.push(face);
        face.dist2
    }

    fn add_to_map(&mut self, fi: usize) {
        self.faces[fi].index = self.map.len() as i64;
        self.map.push(fi);
    }

    fn get_edge(&self, fi: usize, vertex: usize) -> usize {
        let v = self.faces[fi].verts;
        if v[0] == vertex {
            0
        } else if v[1] == vertex {
            1
        } else {
            2
        }
    }

    fn horizon_rec(&mut self, fi: usize, e: usize) -> bool {
        let face = self.faces[fi];
        if dot3(&face.v, &self.horizon_w) - face.dist2 > MINVAL {
            self.delete_face(fi);
            for k in 1..3 {
                let i = (e + k) % 3;
                let adj = face.adj[i];
                if self.faces[adj].index > -2 {
                    let adj_edge = self.get_edge(adj, face.verts[(i + 1) % 3]);
                    if !self.horizon_rec(adj, adj_edge) {
                        self.horizon_indices.push(adj);
                        self.horizon_edges.push(adj_edge);
                    }
                }
            }
            return true;
        }
        false
    }

    fn horizon(&mut self, fi: usize) {
        self.delete_face(fi);
        let face = self.faces[fi];
        let adj = face.adj[0];
        let adj_edge = self.get_edge(adj, face.verts[1]);
        if !self.horizon_rec(adj, adj_edge) {
            self.horizon_indices.push(adj);
            self.horizon_edges.push(adj_edge);
        }
        let adj = face.adj[1];
        let adj_edge = self.get_edge(adj, face.verts[2]);
        if self.faces[adj].index > -2 && !self.horizon_rec(adj, adj_edge) {
            self.horizon_indices.push(adj);
            self.horizon_edges.push(adj_edge);
        }
        let adj = face.adj[2];
        let adj_edge = self.get_edge(adj, face.verts[0]);
        if self.faces[adj].index > -2 && !self.horizon_rec(adj, adj_edge) {
            self.horizon_indices.push(adj);
            self.horizon_edges.push(adj_edge);
        }
    }

    fn replace_simplex3(&mut self, st: &mut CcdStatus, v1: usize, v2: usize, v3: usize) {
        st.nsimplex = 3;
        st.simplex[0] = self.verts[v1];
        st.simplex[1] = self.verts[v2];
        st.simplex[2] = self.verts[v3];
        self.faces.clear();
        self.verts.clear();
        self.map.clear();
    }
}

fn tri_affine_coord(v1: &V3, v2: &V3, v3: &V3, p: &V3) -> [f64; 3] {
    let m14 = minor6((v2.y, v3.z), (v2.z, v3.y), (v1.y, v3.z), (v1.z, v3.y), (v1.y, v2.z), (v1.z, v2.y));
    let m24 = minor6((v2.x, v3.z), (v2.z, v3.x), (v1.x, v3.z), (v1.z, v3.x), (v1.x, v2.z), (v1.z, v2.x));
    let m34 = minor6((v2.x, v3.y), (v2.y, v3.x), (v1.x, v3.y), (v1.y, v3.x), (v1.x, v2.y), (v1.y, v2.x));
    let (mu1, mu2, mu3) = (m14.abs(), m24.abs(), m34.abs());
    let (m_max, x, y) = if mu1 >= mu2 && mu1 >= mu3 {
        (m14, 1, 2)
    } else if mu2 >= mu3 {
        (m24, 0, 2)
    } else {
        (m34, 0, 1)
    };
    let c31 = cofactor6((p[x], v2[y]), (p[y], v3[x]), (v2[x], v3[y]), (p[x], v3[y]), (p[y], v2[x]), (v3[x], v2[y]));
    let c32 = cofactor6((p[x], v3[y]), (p[y], v1[x]), (v3[x], v1[y]), (p[x], v1[y]), (p[y], v3[x]), (v1[x], v3[y]));
    let c33 = cofactor6((p[x], v1[y]), (p[y], v2[x]), (v1[x], v2[y]), (p[x], v2[y]), (p[y], v1[x]), (v2[x], v1[y]));
    [c31 / m_max, c32 / m_max, c33 / m_max]
}

fn tri_point_intersect(v1: &V3, v2: &V3, v3: &V3, p: &V3) -> bool {
    let l = tri_affine_coord(v1, v2, v3, p);
    if l[0] < 0.0 || l[1] < 0.0 || l[2] < 0.0 {
        return false;
    }
    let pr = V3::new(
        v3.x.mul_add(l[2], v1.x.mul_add(l[0], v2.x * l[1])),
        v3.y.mul_add(l[2], v1.y.mul_add(l[0], v2.y * l[1])),
        v3.z.mul_add(l[2], v1.z.mul_add(l[0], v2.z * l[1])),
    );
    norm3(&(pr - p)) < MINVAL
}

fn same_side(p0: &V3, p1: &V3, p2: &V3, p3: &V3) -> bool {
    let n = cross3(&(p1 - p0), &(p2 - p0));
    let dot1 = dot3(&n, &(p3 - p0));
    let dot2 = dot3(&n, &(p0 * -1.0));
    (dot1 > 0.0 && dot2 > 0.0) || (dot1 < 0.0 && dot2 < 0.0)
}

fn test_tetra(p0: &V3, p1: &V3, p2: &V3, p3: &V3) -> bool {
    same_side(p0, p1, p2, p3) && same_side(p1, p2, p3, p0) && same_side(p2, p3, p0, p1) && same_side(p3, p0, p1, p2)
}

/// `rotmat`: 120° about `axis`. `u·u·(1 − cos)` rounds `u·u` and fuses the second product into the sum.
fn rotmat120(axis: &V3) -> nalgebra::Matrix3<f64> {
    let n = norm3(axis);
    let (u1, u2, u3) = (axis.x / n, axis.y / n, axis.z / n);
    let s = 0.86602540378;
    let c = -0.5;
    let k = 1.0 - c;
    nalgebra::Matrix3::new(
        (u1 * u1).mul_add(k, c), (u1 * u2).mul_add(k, -(u3 * s)), (u1 * u3).mul_add(k, u2 * s),
        (u2 * u1).mul_add(k, u3 * s), (u2 * u2).mul_add(k, c), (u2 * u3).mul_add(k, -(u1 * s)),
        (u1 * u3).mul_add(k, -(u2 * s)), (u2 * u3).mul_add(k, u1 * s), (u3 * u3).mul_add(k, c),
    )
}

fn ray_triangle(v1: &V3, v2: &V3, v3: &V3, v4: &V3, v5: &V3) -> i32 {
    let d12 = v2 - v1;
    let d13 = v3 - v1;
    let d14 = v4 - v1;
    let d15 = v5 - v1;
    let vol1 = det3(&d13, &d14, &d12);
    let vol2 = det3(&d14, &d15, &d12);
    let vol3 = det3(&d15, &d13, &d12);
    if vol1 >= 0.0 && vol2 >= 0.0 && vol3 >= 0.0 {
        return 1;
    }
    if vol1 <= 0.0 && vol2 <= 0.0 && vol3 <= 0.0 {
        return -1;
    }
    0
}

const EPA_SUCCESS: i32 = 0;
const EPA_P2_NONCONVEX: i32 = 2;
const EPA_P3_BAD_NORMAL: i32 = 4;
const EPA_P3_INVALID_V4: i32 = 5;
const EPA_P3_INVALID_V5: i32 = 6;
const EPA_P3_MISSING_ORIGIN: i32 = 7;
const EPA_P3_ORIGIN_ON_FACE: i32 = 8;
const EPA_P4_MISSING_ORIGIN: i32 = 9;

fn polytope2(pt: &mut Polytope, st: &mut CcdStatus, o1: &mut CcdObj, o2: &mut CcdObj) -> i32 {
    let v1 = st.simplex[0].vert;
    let v2 = st.simplex[1].vert;
    pt.center = (v1 + v2) * 0.5;
    let diff = v2 - v1;
    let mut value = MAX_LIMIT;
    let mut index = 0;
    for i in 0..3 {
        if diff[i].abs() < value {
            value = diff[i].abs();
            index = i;
        }
    }
    let mut e = V3::zeros();
    e[index] = 1.0;
    let d1 = cross3(&e, &diff);
    let r = rotmat120(&diff);
    let d2 = mat_vec(&r, &d1);
    let d3 = mat_vec(&r, &d2);
    let v1i = pt.insert_vertex(st.simplex[0]);
    let v2i = pt.insert_vertex(st.simplex[1]);
    let v3i = pt.epa_support(o1, o2, &d1, norm3(&d1));
    let v4i = pt.epa_support(o1, o2, &d2, norm3(&d2));
    let v5i = pt.epa_support(o1, o2, &d3, norm3(&d3));
    let (v3, v4, v5) = (pt.verts[v3i].vert, pt.verts[v4i].vert, pt.verts[v5i].vert);
    let tri = [(v1i, v3i, v4i, 1, 3, 2), (v1i, v5i, v3i, 2, 4, 0), (v1i, v4i, v5i, 0, 5, 1), (v2i, v4i, v3i, 5, 0, 4), (v2i, v3i, v5i, 3, 1, 5), (v2i, v5i, v4i, 4, 2, 3)];
    for (a, b, c, x, y, z) in tri {
        if pt.attach_face(a, b, c, x, y, z) < MINDIST2 {
            pt.replace_simplex3(st, a, b, c);
            return polytope3(pt, st, o1, o2);
        }
    }
    if ray_triangle(&v1, &v2, &v3, &v4, &v5) == 0 {
        return EPA_P2_NONCONVEX;
    }
    for i in 0..6 {
        pt.add_to_map(i);
    }
    EPA_SUCCESS
}

fn polytope3(pt: &mut Polytope, st: &mut CcdStatus, o1: &mut CcdObj, o2: &mut CcdObj) -> i32 {
    let (v1, v2, v3) = (st.simplex[0].vert, st.simplex[1].vert, st.simplex[2].vert);
    // `scl3(center, center, 1.0/3.0)`: a multiply by the rounded third, not a division by three
    pt.center = (v1 + v2 + v3) * (1.0 / 3.0);
    let n = cross3(&(v2 - v1), &(v3 - v1));
    let n_norm = norm3(&n);
    if n_norm < MINVAL {
        return EPA_P3_BAD_NORMAL;
    }
    let v1i = pt.insert_vertex(st.simplex[0]);
    let v2i = pt.insert_vertex(st.simplex[1]);
    let v3i = pt.insert_vertex(st.simplex[2]);
    let v5i = pt.epa_support(o1, o2, &(n * -1.0), n_norm);
    let v4i = pt.epa_support(o1, o2, &n, n_norm);
    let (v4, v5) = (pt.verts[v4i].vert, pt.verts[v5i].vert);
    if tri_point_intersect(&v1, &v2, &v3, &v4) {
        return EPA_P3_INVALID_V4;
    }
    if tri_point_intersect(&v1, &v2, &v3, &v5) {
        return EPA_P3_INVALID_V5;
    }
    if st.dist[0] > 10.0 * MINVAL && !test_tetra(&v1, &v2, &v3, &v4) && !test_tetra(&v1, &v2, &v3, &v5) {
        return EPA_P3_MISSING_ORIGIN;
    }
    let tri = [(v4i, v1i, v2i, 1, 3, 2), (v4i, v3i, v1i, 2, 4, 0), (v4i, v2i, v3i, 0, 5, 1), (v5i, v2i, v1i, 5, 0, 4), (v5i, v1i, v3i, 3, 1, 5), (v5i, v3i, v2i, 4, 2, 3)];
    for (a, b, c, x, y, z) in tri {
        if pt.attach_face(a, b, c, x, y, z) < MINDIST3 {
            return EPA_P3_ORIGIN_ON_FACE;
        }
    }
    for i in 0..6 {
        pt.add_to_map(i);
    }
    EPA_SUCCESS
}

fn polytope4(pt: &mut Polytope, st: &mut CcdStatus, o1: &mut CcdObj, o2: &mut CcdObj) -> i32 {
    let v1 = pt.insert_vertex(st.simplex[0]);
    let v2 = pt.insert_vertex(st.simplex[1]);
    let v3 = pt.insert_vertex(st.simplex[2]);
    let v4 = pt.insert_vertex(st.simplex[3]);
    pt.center = (pt.verts[v1].vert + pt.verts[v2].vert + pt.verts[v3].vert + pt.verts[v4].vert) * 0.25;
    let tri = [(v1, v2, v3, 1, 3, 2), (v1, v4, v2, 2, 3, 0), (v1, v3, v4, 0, 3, 1), (v4, v3, v2, 2, 0, 1)];
    for (a, b, c, x, y, z) in tri {
        if pt.attach_face(a, b, c, x, y, z) < MINDIST4 {
            pt.replace_simplex3(st, a, b, c);
            return polytope3(pt, st, o1, o2);
        }
    }
    if !test_tetra(&pt.verts[v1].vert, &pt.verts[v2].vert, &pt.verts[v3].vert, &pt.verts[v4].vert) {
        return EPA_P4_MISSING_ORIGIN;
    }
    for i in 0..4 {
        pt.add_to_map(i);
    }
    EPA_SUCCESS
}

fn epa_witness(pt: &Polytope, fi: usize) -> (V3, V3, f64) {
    let f = &pt.faces[fi];
    let (a, b, c) = (&pt.verts[f.verts[0]], &pt.verts[f.verts[1]], &pt.verts[f.verts[2]]);
    let l = tri_affine_coord(&a.vert, &b.vert, &c.vert, &f.v);
    (lincomb(&l, &[a.vert1, b.vert1, c.vert1]), lincomb(&l, &[a.vert2, b.vert2, c.vert2]), -f.dist2.sqrt())
}

/// `epa`: returns the index of the final face, if any.
fn epa(st: &mut CcdStatus, pt: &mut Polytope, o1: &mut CcdObj, o2: &mut CcdObj) -> Option<usize> {
    let mut upper = MAX_LIMIT;
    let mut upper2 = MAX_LIMIT;
    let mut face: Option<usize> = None;
    let discrete = discrete_geoms(o1, o2);
    let tolerance = if discrete { MINEPATOL } else { st.tolerance };
    let kmax = st.max_iterations.min(1000);
    let mut k = 0;
    while k < kmax {
        let pface = face;
        let mut lower2 = MAX_LIMIT;
        let mut best: Option<usize> = None;
        for &fi in &pt.map {
            if pt.faces[fi].dist2 < lower2 {
                best = Some(fi);
                lower2 = pt.faces[fi].dist2;
            }
        }
        face = best;
        if lower2 > upper2 || face.is_none() {
            face = pface;
            break;
        }
        let fi = face.unwrap();
        if lower2 <= 0.0 {
            break;
        }
        let lower = lower2.sqrt();
        let fv = pt.faces[fi].v;
        let wi = pt.epa_support(o1, o2, &fv, lower);
        let w = pt.verts[wi];
        let upper_k = dot3(&fv, &w.vert) / lower;
        if upper_k < upper {
            upper = upper_k;
            upper2 = upper * upper;
        }
        if upper - lower < tolerance {
            if k == 0 && upper < lower - 1e-10 {
                face = None;
            }
            break;
        }
        if discrete {
            let nverts = pt.verts.len() - 1;
            if (0..nverts).any(|i| w.index1 == pt.verts[i].index1 && w.index2 == pt.verts[i].index2) {
                break;
            }
        }
        pt.horizon_w = w.vert;
        pt.horizon_indices.clear();
        pt.horizon_edges.clear();
        pt.horizon(fi);
        if pt.horizon_indices.len() < 3 {
            face = None;
            break;
        }
        let nfaces = pt.faces.len();
        let nedges = pt.horizon_indices.len();
        if nedges > pt.maxfaces - nfaces {
            break;
        }
        let mut broke = false;
        for i in 0..nedges {
            let cur = nfaces + i;
            let next = nfaces + (i + 1) % nedges;
            let prev = if i == 0 { nfaces + nedges - 1 } else { cur - 1 };
            let hzn_index = pt.horizon_indices[i];
            let hzn_edge = pt.horizon_edges[i];
            let hv = pt.faces[hzn_index].verts;
            let v1 = hv[hzn_edge];
            let v2 = hv[(hzn_edge + 1) % 3];
            pt.faces[hzn_index].adj[hzn_edge] = cur;
            let dist2 = pt.attach_face(wi, v2, v1, prev, hzn_index, next);
            if dist2 == 0.0 {
                face = None;
                broke = true;
                break;
            }
            if dist2 >= lower2 && dist2 <= upper2 {
                let last = pt.faces.len() - 1;
                pt.add_to_map(last);
            }
        }
        pt.horizon_indices.clear();
        pt.horizon_edges.clear();
        if broke || pt.map.is_empty() {
            break;
        }
        k += 1;
    }
    st.epa_iterations = k;
    match face {
        Some(fi) => {
            let (x1, x2, d) = epa_witness(pt, fi);
            st.x1 = vec![x1];
            st.x2 = vec![x2];
            st.dist = vec![d];
            Some(fi)
        }
        None => {
            st.x1.clear();
            st.x2.clear();
            st.dist = vec![0.0];
            None
        }
    }
}

// ------------------------------------------------------------------------------------------------
// multi-contact
// ------------------------------------------------------------------------------------------------

fn area4(hull: &[V3], a: usize, b: usize, c: usize, d: usize) -> f64 {
    0.5 * norm3(&cross3(&(hull[a] - hull[c]), &(hull[b] - hull[d])))
}

fn hull4(hull: &[V3]) -> [usize; 4] {
    let n = hull.len();
    let (mut a, mut b, mut c, mut d) = (0usize, 1usize, 2usize, 3usize);
    let mut res = [0, 1, 2, 3];
    let mut m = area4(hull, a, b, c, d);
    while a < n {
        loop {
            let d_next = (d + 1) % n;
            let m_next = area4(hull, a, b, c, d_next);
            if m_next <= m {
                break;
            }
            d = d_next;
            m = m_next;
            res = [a, b, c, d];
            loop {
                let c_next = (c + 1) % n;
                let m_next = area4(hull, a, b, c_next, d);
                if m_next <= m {
                    break;
                }
                c = c_next;
                m = m_next;
                res = [a, b, c, d];
            }
            loop {
                let b_next = (b + 1) % n;
                let m_next = area4(hull, a, b_next, c, d);
                if m_next <= m {
                    break;
                }
                b = b_next;
                m = m_next;
                res = [a, b, c, d];
            }
        }
        if b == a {
            b = (b + 1) % n;
            if c == b {
                c = (c + 1) % n;
                if d == c {
                    d = (d + 1) % n;
                }
            }
        }
        a += 1;
    }
    res
}

fn plane_normal(v1: &V3, v2: &V3, n: &V3) -> (V3, f64) {
    let v3 = v1 + n;
    let res = normalize3(&cross3(&(v2 - v1), &(v3 - v1)));
    (res, dot3(&res, v1))
}

fn halfspace(a: &V3, n: &V3, p: &V3) -> bool {
    dot3(&(p - a), n) > -MINVAL
}

fn witness_on_face(v: &V3, p: &V3, n: &V3, dir: &V3) -> (V3, V3, f64) {
    let dist = dot3(&(v - p), n);
    (add_scl3(v, dir, -dist.abs()), *v, dist)
}

/// `polygonClip`: clip `face2` against the prism of `face1`, keep points below `face1`'s plane, reduce to
/// the four of largest area when only four are wanted.
///
/// Returns whether it produced witness points; `false` leaves `st` holding EPA's single pair untouched.
fn polygon_clip(st: &mut CcdStatus, face1: &[V3], face2: &[V3], n: &V3, dir: &V3) -> bool {
    let nface1 = face1.len();
    if nface1 < 3 {
        return false;
    }
    let mut pn = Vec::with_capacity(nface1);
    let mut pd = Vec::with_capacity(nface1);
    for i in 0..nface1 {
        let (a, b) = plane_normal(&face1[i], &face1[(i + 1) % nface1], n);
        pn.push(a);
        pd.push(b);
    }
    let mut polygon: Vec<V3> = face2.to_vec();
    for e in 0..nface1 {
        let mut clipped: Vec<V3> = Vec::with_capacity(polygon.len() + 4);
        let np = polygon.len();
        for i in 0..np {
            let p = polygon[i];
            let q = polygon[(i + 1) % np];
            let pq = q - p;
            let inside1 = halfspace(&face1[e], &pn[e], &p);
            let inside2 = halfspace(&face1[e], &pn[e], &q);
            if !inside1 && !inside2 {
                continue;
            }
            if inside1 && inside2 {
                clipped.push(q);
                continue;
            }
            let tmp = dot3(&pn[e], &pq);
            if tmp != 0.0 {
                let t = (pd[e] - dot3(&pn[e], &p)) / tmp;
                if (0.0..=1.0).contains(&t) {
                    clipped.push(add_scl3(&p, &pq, t));
                }
            }
            if inside2 {
                clipped.push(q);
            }
        }
        polygon = clipped;
    }
    let polygon: Vec<V3> = polygon.into_iter().filter(|p| dot3(&(p - face1[0]), n) <= 0.0).collect();
    if polygon.is_empty() {
        return false;
    }
    st.x1.clear();
    st.x2.clear();
    st.dist.clear();
    if st.max_contacts < 5 && polygon.len() > 4 {
        let idx = hull4(&polygon);
        for &i in &idx {
            let (w1, w2, d) = witness_on_face(&polygon[i], &face1[0], n, dir);
            st.x1.push(w1);
            st.x2.push(w2);
            st.dist.push(d);
        }
        return true;
    }
    if face2.len() == 2 && polygon.len() > 2 {
        let (mut best1, mut best2, mut dd) = (0, 1, 0.0);
        for i in 0..polygon.len() {
            for j in i + 1..polygon.len() {
                let diff = polygon[j] - polygon[i];
                let d2 = dot3(&diff, &diff);
                if d2 > dd {
                    dd = d2;
                    best1 = i;
                    best2 = j;
                }
            }
        }
        for &i in &[best1, best2] {
            let (w1, w2, d) = witness_on_face(&polygon[i], &face1[0], n, dir);
            st.x1.push(w1);
            st.x2.push(w2);
            st.dist.push(d);
        }
        return true;
    }
    for p in polygon.iter().take(MAXCONPAIR) {
        let (w1, w2, d) = witness_on_face(p, &face1[0], n, dir);
        st.x1.push(w1);
        st.x2.push(w2);
        st.dist.push(d);
    }
    true
}

fn globalcoord(pose: &GeomPose, l: V3, with_pos: bool) -> V3 {
    let r = mat_vec(&pose.mat, &l);
    if with_pos {
        r + pose.pos
    } else {
        r
    }
}

/// Normals of the faces incident on the simplex feature (`dim` = 1 vertex, 2 edge, 3 face) of a geom, with
/// the face indices they belong to.
fn feature_normals(obj: &CcdObj, dim: usize, vi: [i64; 3], dir: &V3) -> Vec<(V3, usize)> {
    match obj.kind {
        GeomType::Box => box_normals(obj, dim, vi, dir),
        GeomType::Mesh => mesh_normals(obj, dim, vi),
        GeomType::Cylinder => {
            if dim == 1 {
                vec![(globalcoord(&obj.pose, Vector3::new(0.0, 0.0, if vi[0] != 0 { -1.0 } else { 1.0 }), false), vi[0] as usize)]
            } else {
                Vec::new()
            }
        }
        _ => Vec::new(),
    }
}

fn box_normals2(pose: &GeomPose, n: &V3) -> Option<(V3, usize)> {
    let normals = [Vector3::x(), -Vector3::x(), Vector3::y(), -Vector3::y(), Vector3::z(), -Vector3::z()];
    let mut local = mat_t_vec(&pose.mat, n);
    local *= 1.0 / dot3(&local, &local).sqrt();
    for (i, nn) in normals.iter().enumerate() {
        if dot3(&local, nn) > FACE_TOL {
            return Some((globalcoord(pose, *nn, false), i));
        }
    }
    None
}

fn box_normals(obj: &CcdObj, dim: usize, vi: [i64; 3], dir: &V3) -> Vec<(V3, usize)> {
    let (v1, v2, v3) = (vi[0], vi[1], vi[2]);
    let pose = &obj.pose;
    let bit = |v: i64, m: i64| (v & m) != 0;
    if dim == 3 {
        let x = ((bit(v1, 1) && bit(v2, 1) && bit(v3, 1)) as i32) - ((!bit(v1, 1) && !bit(v2, 1) && !bit(v3, 1)) as i32);
        let y = ((bit(v1, 2) && bit(v2, 2) && bit(v3, 2)) as i32) - ((!bit(v1, 2) && !bit(v2, 2) && !bit(v3, 2)) as i32);
        let z = ((bit(v1, 4) && bit(v2, 4) && bit(v3, 4)) as i32) - ((!bit(v1, 4) && !bit(v2, 4) && !bit(v3, 4)) as i32);
        let normal = globalcoord(pose, Vector3::new(x as f64, y as f64, z as f64), false);
        let mut ind = Vec::new();
        if x != 0 {
            ind.push(0);
        }
        if y != 0 {
            ind.push(2);
        }
        if z != 0 {
            ind.push(4);
        }
        if x + y + z == -1 && !ind.is_empty() {
            ind[0] += 1;
        }
        if ind.len() == 1 {
            return vec![(normal, ind[0])];
        }
        return box_normals2(pose, dir).into_iter().collect();
    }
    if dim == 2 {
        let x = ((bit(v1, 1) && bit(v2, 1)) as i32) - ((!bit(v1, 1) && !bit(v2, 1)) as i32);
        let y = ((bit(v1, 2) && bit(v2, 2)) as i32) - ((!bit(v1, 2) && !bit(v2, 2)) as i32);
        let z = ((bit(v1, 4) && bit(v2, 4)) as i32) - ((!bit(v1, 4) && !bit(v2, 4)) as i32);
        let mut out = Vec::new();
        if x != 0 {
            out.push((globalcoord(pose, Vector3::new(x as f64, 0.0, 0.0), false), if x > 0 { 0 } else { 1 }));
        }
        if y != 0 {
            out.push((globalcoord(pose, Vector3::new(0.0, y as f64, 0.0), false), if y > 0 { 2 } else { 3 }));
        }
        if z != 0 {
            // (MuJoCo writes this normal into slot 1 regardless; with all three set it then falls through
            // to `boxNormals2` anyway, as the count check below does)
            out.push((globalcoord(pose, Vector3::new(0.0, 0.0, z as f64), false), if z > 0 { 4 } else { 5 }));
        }
        if out.len() == 2 {
            return out;
        }
        return box_normals2(pose, dir).into_iter().collect();
    }
    if dim == 1 {
        let x = if bit(v1, 1) { 1.0 } else { -1.0 };
        let y = if bit(v1, 2) { 1.0 } else { -1.0 };
        let z = if bit(v1, 4) { 1.0 } else { -1.0 };
        return vec![
            (globalcoord(pose, Vector3::new(x, 0.0, 0.0), false), if x > 0.0 { 0 } else { 1 }),
            (globalcoord(pose, Vector3::new(0.0, y, 0.0), false), if y > 0.0 { 2 } else { 3 }),
            (globalcoord(pose, Vector3::new(0.0, 0.0, z), false), if z > 0.0 { 4 } else { 5 }),
        ];
    }
    Vec::new()
}

fn mesh_normals(obj: &CcdObj, dim: usize, vi: [i64; 3]) -> Vec<(V3, usize)> {
    let hull = obj.hull.expect("mesh geom without hull data");
    let pm = |v: i64| -> &[usize] { &hull.polymap[v as usize] };
    let intersect = |a: &[usize], b: &[usize]| -> Vec<usize> {
        let mut out = Vec::new();
        for &x in a {
            for &y in b {
                if x == y {
                    out.push(x);
                    if out.len() == 2 {
                        return out;
                    }
                }
            }
        }
        out
    };
    let g = |i: usize| (globalcoord(&obj.pose, hull.polygons[i].normal, false), i);
    match dim {
        3 => {
            let e = intersect(pm(vi[0]), pm(vi[1]));
            if e.is_empty() {
                return Vec::new();
            }
            let f = intersect(&e, pm(vi[2]));
            if f.is_empty() {
                return Vec::new();
            }
            vec![g(f[0])]
        }
        2 => intersect(pm(vi[0]), pm(vi[1])).into_iter().map(g).collect(),
        1 => pm(vi[0]).iter().map(|&i| g(i)).collect(),
        _ => Vec::new(),
    }
}

/// Edge directions (unit) and their far endpoints for the edge/vertex feature of a geom.
fn edge_normals(obj: &CcdObj, dim: usize, v: &[V3; 3], v1i: i64) -> Vec<(V3, V3)> {
    let pose = &obj.pose;
    match obj.kind {
        GeomType::Box => {
            if dim == 2 {
                let d = normalize3(&(v[1] - v[0]));
                return vec![(d, v[1])];
            }
            if dim == 1 {
                let s = obj.size;
                let x = if v1i & 1 != 0 { s[0] } else { -s[0] };
                let y = if v1i & 2 != 0 { s[1] } else { -s[1] };
                let z = if v1i & 4 != 0 { s[2] } else { -s[2] };
                let mut out = Vec::new();
                for end in [Vector3::new(-x, y, z), Vector3::new(x, -y, z), Vector3::new(x, y, -z)] {
                    let e = globalcoord(pose, end, true);
                    out.push((normalize3(&(e - v[0])), e));
                }
                return out;
            }
            Vec::new()
        }
        GeomType::Mesh => {
            let hull = obj.hull.expect("mesh geom without hull data");
            if dim == 2 {
                let d = normalize3(&(v[1] - v[0]));
                return vec![(d, v[1])];
            }
            if dim == 1 {
                let mut out = Vec::new();
                for &pi in &hull.polymap[v1i as usize] {
                    let poly = &hull.polygons[pi];
                    let nvert = poly.verts.len();
                    if let Some(j) = poly.verts.iter().position(|&x| x == v1i as usize) {
                        let k = if j == 0 { nvert - 1 } else { j - 1 };
                        let e = globalcoord(pose, hull.verts[poly.verts[k]], true);
                        out.push((normalize3(&(e - v[0])), e));
                    }
                }
                return out;
            }
            Vec::new()
        }
        GeomType::Cylinder => {
            if dim == 1 || dim == 2 {
                let sgn = if v1i != 0 { 1.0 } else { -1.0 };
                let m = &obj.pose.mat;
                let d = V3::new(sgn * m[(0, 2)], sgn * m[(1, 2)], sgn * m[(2, 2)]);
                return vec![(d, add_scl3(&v[0], &d, 2.0 * obj.size[1]))];
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

// the 16-gon tables are MuJoCo's verbatim 15-digit constants (note the 547/548 last digits), which decide the
// last-bit ties in the polygon pruning; the library constants would change the answer
#[allow(clippy::approx_constant)]
fn face_vertices(obj: &CcdObj, idx: usize) -> Vec<V3> {
    let pose = &obj.pose;
    match obj.kind {
        GeomType::Cylinder => {
            let cos16 = [1.0, 0.923879532511287, 0.707106781186548, 0.382683432365090, 0.0, -0.382683432365090, -0.707106781186547, -0.923879532511287, -1.0, -0.923879532511287, -0.707106781186548, -0.382683432365090, 0.0, 0.382683432365090, 0.707106781186547, 0.923879532511287];
            let sin16 = [0.0, 0.382683432365090, 0.707106781186547, 0.923879532511287, 1.0, 0.923879532511287, 0.707106781186548, 0.382683432365090, 0.0, -0.382683432365090, -0.707106781186547, -0.923879532511287, -1.0, -0.923879532511287, -0.707106781186548, -0.382683432365090];
            let sgn = if idx != 0 { -1.0 } else { 1.0 };
            (0..16).map(|i| globalcoord(pose, Vector3::new(cos16[i] * obj.size[0], -sin16[i] * obj.size[0] * sgn, sgn * obj.size[1]), true)).collect()
        }
        GeomType::Box => {
            let s = obj.size;
            let corners: [[V3; 4]; 6] = [
                [Vector3::new(s[0], s[1], s[2]), Vector3::new(s[0], s[1], -s[2]), Vector3::new(s[0], -s[1], -s[2]), Vector3::new(s[0], -s[1], s[2])],
                [Vector3::new(-s[0], s[1], -s[2]), Vector3::new(-s[0], s[1], s[2]), Vector3::new(-s[0], -s[1], s[2]), Vector3::new(-s[0], -s[1], -s[2])],
                [Vector3::new(-s[0], s[1], -s[2]), Vector3::new(s[0], s[1], -s[2]), Vector3::new(s[0], s[1], s[2]), Vector3::new(-s[0], s[1], s[2])],
                [Vector3::new(-s[0], -s[1], s[2]), Vector3::new(s[0], -s[1], s[2]), Vector3::new(s[0], -s[1], -s[2]), Vector3::new(-s[0], -s[1], -s[2])],
                [Vector3::new(-s[0], s[1], s[2]), Vector3::new(s[0], s[1], s[2]), Vector3::new(s[0], -s[1], s[2]), Vector3::new(-s[0], -s[1], s[2])],
                [Vector3::new(s[0], s[1], -s[2]), Vector3::new(-s[0], s[1], -s[2]), Vector3::new(-s[0], -s[1], -s[2]), Vector3::new(s[0], -s[1], -s[2])],
            ];
            corners[idx].iter().map(|c| globalcoord(pose, *c, true)).collect()
        }
        GeomType::Mesh => {
            let hull = obj.hull.expect("mesh geom without hull data");
            let poly = &hull.polygons[idx];
            poly.verts.iter().rev().map(|&v| globalcoord(pose, hull.verts[v], true)).collect()
        }
        _ => Vec::new(),
    }
}

fn aligned_faces(n1: &[(V3, usize)], n2: &[(V3, usize)]) -> Option<(usize, usize)> {
    for (i, a) in n1.iter().enumerate() {
        for (j, b) in n2.iter().enumerate() {
            if dot3(&a.0, &b.0) < -FACE_TOL {
                return Some((i, j));
            }
        }
    }
    None
}

fn aligned_face_edge(edges: &[(V3, V3)], faces: &[(V3, usize)], dir: &V3) -> Option<(usize, usize)> {
    for (i, f) in faces.iter().enumerate() {
        if dot3(&f.0, dir) <= MINVAL {
            continue;
        }
        for (j, e) in edges.iter().enumerate() {
            if dot3(&e.0, &f.0).abs() < EDGE_TOL {
                return Some((j, i));
            }
        }
    }
    None
}

fn simplex_dim(vi: &mut [i64; 3], v: &mut [V3; 3]) -> usize {
    if vi[0] == vi[1] {
        if vi[0] == vi[2] {
            return 1;
        }
        vi[1] = vi[2];
        v[1] = v[2];
        return 2;
    }
    if vi[2] == vi[0] || vi[2] == vi[1] {
        2
    } else {
        3
    }
}

fn multicontact(pt: &Polytope, fi: usize, st: &mut CcdStatus, o1: &CcdObj, o2: &CcdObj) {
    if o1.kind == GeomType::Mesh && o1.hull.map(|h| h.polygons.is_empty()).unwrap_or(true) {
        return;
    }
    if o2.kind == GeomType::Mesh && o2.hull.map(|h| h.polygons.is_empty()).unwrap_or(true) {
        return;
    }
    let f = &pt.faces[fi];
    let vs = [&pt.verts[f.verts[0]], &pt.verts[f.verts[1]], &pt.verts[f.verts[2]]];
    let mut v1i = [vs[0].index1, vs[1].index1, vs[2].index1];
    let mut v2i = [vs[0].index2, vs[1].index2, vs[2].index2];
    let mut v1 = [vs[0].vert1, vs[1].vert1, vs[2].vert1];
    let mut v2 = [vs[0].vert2, vs[1].vert2, vs[2].vert2];
    let nface1 = simplex_dim(&mut v1i, &mut v1);
    let nface2 = simplex_dim(&mut v2i, &mut v2);
    let dir = st.x2[0] - st.x1[0];
    let dir_neg = st.x1[0] - st.x2[0];
    let mut n1 = feature_normals(o1, nface1, v1i, &dir_neg);
    let mut n2 = feature_normals(o2, nface2, v2i, &dir);
    let mut edgecon1 = false;
    let mut edgecon2 = false;
    let mut e1: Vec<(V3, V3)> = Vec::new();
    let mut e2: Vec<(V3, V3)> = Vec::new();
    let (i, j) = match aligned_faces(&n1, &n2) {
        Some(r) => r,
        None => {
            if nface1 < 3 && nface1 <= nface2 {
                e1 = edge_normals(o1, nface1, &v1, v1i[0]);
                n1.clear();
                let Some(r) = aligned_face_edge(&e1, &n2, &dir) else { return };
                edgecon1 = true;
                r
            } else if nface2 < 3 {
                e2 = edge_normals(o2, nface2, &v2, v2i[0]);
                n2.clear();
                let Some(r) = aligned_face_edge(&e2, &n1, &dir_neg) else { return };
                edgecon2 = true;
                r
            } else {
                return;
            }
        }
    };
    let face1: Vec<V3> = if edgecon1 { vec![v1[0], e1[i].1] } else { face_vertices(o1, if edgecon2 { n1[j].1 } else { n1[i].1 }) };
    let face2: Vec<V3> = if edgecon2 { vec![v2[0], e2[i].1] } else { face_vertices(o2, n2[j].1) };
    if edgecon1 {
        // the faces go in reversed, so the witness points come out belonging to the other geom.
        //
        // ⛔ MuJoCo swaps `status->nx` pairs unconditionally, and when the clip produced nothing that count is
        // still the 1 left by EPA: it swaps EPA's own witness pair, which inverts the contact normal (on
        // Menagerie, one cylinder–mesh pair in `unitree_g1/g1_with_hands.xml`, depth right, normal exactly
        // backwards). This is a port of MuJoCo, so it does the same.
        let wit_dir = n2[j].0 * -1.0;
        let nrm = n2[j].0;
        polygon_clip(st, &face2, &face1, &nrm, &wit_dir);
        std::mem::swap(&mut st.x1, &mut st.x2);
        return;
    }
    if edgecon2 {
        let wit_dir = n1[j].0 * -1.0;
        let nrm = n1[j].0;
        polygon_clip(st, &face1, &face2, &nrm, &wit_dir);
        return;
    }
    let wit_dir = n2[j].0;
    let nrm = n1[i].0;
    polygon_clip(st, &face1, &face2, &nrm, &wit_dir);
}

fn inflate(st: &mut CcdStatus, margin1: f64, margin2: f64) {
    let n = normalize3(&(st.x2[0] - st.x1[0]));
    if margin1 != 0.0 {
        st.x1[0] = add_scl3(&st.x1[0], &n, margin1);
    }
    if margin2 != 0.0 {
        st.x2[0] = add_scl3(&st.x2[0], &n, -margin2);
    }
    st.dist[0] -= margin1 + margin2;
}

/// `mjc_ccd`: distance (positive), or the penetration and witness points (negative), for two convex geoms.
pub fn ccd(config: &CcdConfig, o1: &mut CcdObj, o2: &mut CcdObj) -> CcdStatus {
    let mut st = CcdStatus {
        separated: false,
        dist: vec![0.0],
        x1: vec![o1.center()],
        x2: vec![o2.center()],
        gjk_iterations: 0,
        epa_iterations: 0,
        epa_status: -1,
        simplex: [Vertex::default(); 4],
        nsimplex: 0,
        tolerance: config.tolerance,
        max_iterations: config.max_iterations,
        max_contacts: config.max_contacts,
        dist_cutoff: config.dist_cutoff,
    };
    let is_round = |k: GeomType| matches!(k, GeomType::Sphere | GeomType::Capsule);
    if is_round(o1.kind) || is_round(o2.kind) {
        let (m1, m2) = (o1.margin, o2.margin);
        let (mut full1, mut full2) = (0.0, 0.0);
        if o1.kind == GeomType::Sphere {
            full1 = o1.size[0] + 0.5 * m1;
            o1.reduced = Some(Reduced::Point);
            o1.margin = 0.0;
        } else if o1.kind == GeomType::Capsule {
            full1 = o1.size[0] + 0.5 * m1;
            o1.reduced = Some(Reduced::Line);
            o1.margin = 0.0;
        }
        if o2.kind == GeomType::Sphere {
            full2 = o2.size[0] + 0.5 * m2;
            o2.reduced = Some(Reduced::Point);
            o2.margin = 0.0;
        } else if o2.kind == GeomType::Capsule {
            full2 = o2.size[0] + 0.5 * m2;
            o2.reduced = Some(Reduced::Line);
            o2.margin = 0.0;
        }
        st.dist_cutoff += full1 + full2;
        gjk(&mut st, o1, o2);
        st.dist_cutoff = config.dist_cutoff;
        o1.margin = m1;
        o2.margin = m2;
        o1.reduced = None;
        o2.reduced = None;
        if st.dist[0] > st.tolerance {
            inflate(&mut st, full1, full2);
            if st.dist[0] > st.dist_cutoff {
                st.dist[0] = MAX_LIMIT;
            }
            return st;
        }
        if config.max_contacts == 0 {
            st.x1.clear();
            st.x2.clear();
            st.dist = vec![0.0];
            return st;
        }
        st.gjk_iterations = 0;
        st.x1 = vec![o1.center()];
        st.x2 = vec![o2.center()];
    }
    gjk(&mut st, o1, o2);
    if config.max_contacts == 0 {
        return st;
    }
    if st.dist[0] <= config.tolerance && st.nsimplex > 1 && !st.separated {
        st.dist[0] = 0.0;
        let mut pt = Polytope::new(config.max_iterations);
        let ret = match st.nsimplex {
            2 => polytope2(&mut pt, &mut st, o1, o2),
            3 => polytope3(&mut pt, &mut st, o1, o2),
            _ => polytope4(&mut pt, &mut st, o1, o2),
        };
        st.epa_status = ret;
        if ret == EPA_SUCCESS {
            let face = epa(&mut st, &mut pt, o1, o2);
            if let (Some(fi), true) = (face, config.max_contacts > 1) {
                multicontact(&pt, fi, &mut st, o1, o2);
            }
        }
    }
    st
}


/// `mjc_PlaneConvex` (MuJoCo 3.13): the deepest support vertex of a convex geom below a plane, then — for a
/// mesh — the other corners of the hull polygon at that vertex whose normal is most anti-aligned with the
/// plane, pruned to the largest-area quadrilateral anchored at the deepest vertex (`hull4f`) and kept only
/// when below the margin and below the mesh centre. An ellipsoid or other convex geom yields one contact.
pub fn plane_convex(margin: f64, plane: &GeomPose, obj: &mut CcdObj) -> Vec<PreContact> {
    let normal = plane.axis();
    let dir = -normal;
    obj.vertindex = -1;
    obj.meshindex = -1;
    let v = obj.support(&dir);
    let dist = dot3(&normal, &(v - plane.pos));
    if dist > margin {
        return Vec::new();
    }
    let mut out = vec![PreContact { dist, pos: add_scl3(&v, &normal, -0.5 * dist), normal, tangent: V3::zeros() }];
    let Some(hull) = obj.hull.filter(|_| obj.kind == GeomType::Mesh) else { return out };
    if hull.polygons.is_empty() || obj.vertindex < 0 {
        return out;
    }
    let vi = obj.vertindex as usize;
    let local_normal = mat_t_vec(&obj.pose.mat, &normal);
    let mut best_poly: Option<usize> = None;
    let mut best_dot = 1.0;
    for &pi in &hull.polymap[vi] {
        let nd = dot3(&hull.polygons[pi].normal, &local_normal);
        if nd < best_dot {
            best_dot = nd;
            best_poly = Some(pi);
        }
    }
    let Some(bp) = best_poly else { return out };
    let face = &hull.polygons[bp].verts;
    let a = face.iter().position(|&x| x == vi).unwrap_or(0);
    let idx = hull4f(&hull.verts, face, a);
    for &i in idx.iter().skip(1) {
        let pnt = local_to_global(&obj.pose.mat, &hull.verts[face[i]], &obj.pose.pos);
        let vdist = dot3(&normal, &(pnt - plane.pos));
        if vdist > margin || dot3(&normal, &(pnt - obj.pose.pos)) > 0.0 {
            continue;
        }
        out.push(PreContact { dist: vdist, pos: add_scl3(&pnt, &normal, -0.5 * vdist), normal, tangent: V3::zeros() });
    }
    out
}

/// `hull4f`: prune a polygon (given by vertex indices) to the largest-area quadrilateral that keeps vertex
/// `a`; a polygon of at most four vertices is returned whole, starting at `a`.
fn hull4f(verts: &[V3], idx: &[usize], a: usize) -> Vec<usize> {
    let n = idx.len();
    let area = |a: usize, b: usize, c: usize, d: usize| 0.5 * norm3(&cross3(&(verts[idx[a]] - verts[idx[c]]), &(verts[idx[b]] - verts[idx[d]])));
    let (mut b, mut c, mut d) = ((a + 1) % n, (a + 2) % n, (a + 3) % n);
    let mut res = vec![a, b, c, d];
    if n <= 4 {
        res.truncate(n);
        return res;
    }
    let mut m = area(a, b, c, d);
    loop {
        let d_next = (d + 1) % n;
        let m_next = area(a, b, c, d_next);
        if m_next <= m {
            break;
        }
        d = d_next;
        m = m_next;
        res = vec![a, b, c, d];
        loop {
            let c_next = (c + 1) % n;
            let m_next = area(a, b, c_next, d);
            if m_next <= m {
                break;
            }
            c = c_next;
            m = m_next;
            res = vec![a, b, c, d];
        }
        loop {
            let b_next = (b + 1) % n;
            let m_next = area(a, b_next, c, d);
            if m_next <= m {
                break;
            }
            b = b_next;
            m = m_next;
            res = vec![a, b, c, d];
        }
    }
    res
}

// ------------------------------------------------------------------------------------------------
// the driver's use of it (mjc_Convex / mjc_penetration / maxContacts)
// ------------------------------------------------------------------------------------------------

/// `maxContacts`: how many contacts MuJoCo asks the CCD for.
pub fn max_contacts(k1: GeomType, k2: GeomType, margin: f64, multiccd: bool) -> usize {
    if margin > 0.0 {
        return 1;
    }
    if k1 == GeomType::Box && k2 == GeomType::Box {
        return 8;
    }
    let poly = |k: GeomType| matches!(k, GeomType::Box | GeomType::Mesh | GeomType::Cylinder);
    if poly(k1) && poly(k2) {
        return if multiccd { 4 } else { 1 };
    }
    1
}

/// `mjc_penetration` with the native CCD: the contacts from one query, margin added back to the distances.
fn penetration(config: &CcdConfig, o1: &mut CcdObj, o2: &mut CcdObj, margin: f64) -> Vec<PreContact> {
    let st = ccd(config, o1, o2);
    let mut out = Vec::new();
    let min = st.dist.iter().cloned().fold(f64::INFINITY, f64::min);
    if min < 0.0 {
        for i in 0..st.x1.len() {
            let normal = normalize3(&(st.x1[i] - st.x2[i]));
            out.push(PreContact { dist: margin + st.dist[i], pos: (st.x1[i] + st.x2[i]) * 0.5, normal, tangent: V3::zeros() });
        }
    }
    out
}

/// `mjc_Convex`: MuJoCo's contact set for a pair the primitive routines do not cover. `ccd_tolerance` and
/// `ccd_iterations` are `<option>` values (defaults 1e-6, 35); `multiccd` is the `multiccd` flag.
pub fn convex_pair(o1: &mut CcdObj, o2: &mut CcdObj, margin: f64, ccd_tolerance: f64, ccd_iterations: usize, multiccd: bool) -> Vec<PreContact> {
    let (rbound1, rbound2) = (o1.rbound(), o2.rbound());
    o1.margin = margin;
    o2.margin = margin;
    let maxc = max_contacts(o1.kind, o2.kind, margin, multiccd);
    let config = CcdConfig { max_iterations: ccd_iterations, tolerance: ccd_tolerance, max_contacts: maxc, dist_cutoff: 0.0 };
    let mut con = penetration(&config, o1, o2, margin);
    if maxc > 1 {
        return con;
    }
    let no_round = |k: GeomType| !matches!(k, GeomType::Ellipsoid | GeomType::Sphere);
    if con.len() == 1 && multiccd && no_round(o1.kind) && no_round(o2.kind) {
        let relative_tolerance = 1e-3;
        let perturbation_angle = 1e-3;
        let frame = crate::mujoco_collision::make_frame(con[0].normal, V3::zeros());
        let tolerance = relative_tolerance * rbound1.min(rbound2);
        let (pose1, pose2) = (o1.pose, o2.pose);
        for axis in [frame[1], frame[2]] {
            for angle in [-perturbation_angle, perturbation_angle] {
                let rot = quat_to_mat(&axis_angle_to_quat(&axis, angle));
                let origin = con[0].pos;
                o1.pose = rotate_frame(&pose1, &origin, &rot);
                o2.pose = rotate_frame(&pose2, &origin, &rot.transpose());
                let cfg1 = CcdConfig { max_contacts: 1, ..config };
                let extra = penetration(&cfg1, o1, o2, margin);
                if let Some(mut c) = extra.into_iter().next().filter(|c| con.iter().all(|k| norm3(&(k.pos - c.pos)) > tolerance)) {
                    c.dist = con[0].dist;
                    con.push(c);
                }
                o1.pose = pose1;
                o2.pose = pose2;
            }
        }
    }
    con
}

/// `mju_rotateFrame`: rotate a pose about `origin`, with MuJoCo's own operation order.
fn rotate_frame(pose: &GeomPose, origin: &V3, rot: &nalgebra::Matrix3<f64>) -> GeomPose {
    // `mju_mulMatMat3(mat, rot, xmat)`: each entry a contracted row·column
    let mut mat = nalgebra::Matrix3::zeros();
    for i in 0..3 {
        for j in 0..3 {
            mat[(i, j)] = rot[(i, 2)].mul_add(pose.mat[(2, j)], rot[(i, 0)].mul_add(pose.mat[(0, j)], rot[(i, 1)] * pose.mat[(1, j)]));
        }
    }
    let rel = origin - pose.pos;
    let vec = mat_vec(rot, &rel) - rel;
    GeomPose { pos: pose.pos - vec, mat }
}

/// `mju_axisAngle2Quat`.
fn axis_angle_to_quat(axis: &V3, angle: f64) -> [f64; 4] {
    if angle == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let s = (angle * 0.5).sin();
    [(angle * 0.5).cos(), axis.x * s, axis.y * s, axis.z * s]
}

/// `mju_quat2Mat`.
fn quat_to_mat(q: &[f64; 4]) -> nalgebra::Matrix3<f64> {
    if q[0] == 1.0 && q[1] == 0.0 && q[2] == 0.0 && q[3] == 0.0 {
        return nalgebra::Matrix3::identity();
    }
    let (q00, q01, q02, q03) = (q[0] * q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3]);
    let (q11, q12, q13) = (q[1] * q[1], q[1] * q[2], q[1] * q[3]);
    let (q22, q23, q33) = (q[2] * q[2], q[2] * q[3], q[3] * q[3]);
    nalgebra::Matrix3::new(
        q00 + q11 - q22 - q33, 2.0 * (q12 - q03), 2.0 * (q13 + q02),
        2.0 * (q12 + q03), q00 - q11 + q22 - q33, 2.0 * (q23 - q01),
        2.0 * (q13 - q02), 2.0 * (q23 + q01), q00 - q11 - q22 + q33,
    )
}

#[cfg(test)]
// the 7-digit quaternions are the probe's verbatim inputs (MuJoCo normalises them, as `rot` does here); the
// 17-significant-digit expectations are `%.17g` prints of MuJoCo's doubles, kept verbatim
#[allow(clippy::approx_constant, clippy::excessive_precision)]
mod tests {
    //! Every expected number is MuJoCo 3.13.0's `mjContact` from `scripts/mujoco_ccd_probe.py`: a box of
    //! half-sizes (0.2, 0.3, 0.15) at z = 1, a box-shaped OBJ (half-sizes 0.1, 0.15, 0.075), an
    //! octahedron OBJ of radius 0.12, and the primitives placed as stated in each test. Meshes are prepared
    //! exactly as MuJoCo's compiler stores them: `f32` vertices, centred at the legacy-inertia CoM in the
    //! `mjuu_eig3` principal frame, the geom's pose compensated.
    use super::*;
    use crate::mjcf_tree::{eig3_mujoco, mesh_inertia_mujoco, obj_as_mujoco_reads_it, quat_to_rotation, MeshInertia};
    use crate::mujoco_collision::{collide_pair, collide_pair_with, CollideOptions, CollisionGeom};
    use nalgebra::Matrix3;

    const BOXMESH: &str = "v -0.1 -0.15 -0.075\nv -0.1 -0.15 0.075\nv -0.1 0.15 -0.075\nv -0.1 0.15 0.075\nv 0.1 -0.15 -0.075\nv 0.1 -0.15 0.075\nv 0.1 0.15 -0.075\nv 0.1 0.15 0.075\nf 1 2 4\nf 1 4 3\nf 5 7 8\nf 5 8 6\nf 1 5 6\nf 1 6 2\nf 3 4 8\nf 3 8 7\nf 1 3 7\nf 1 7 5\nf 2 6 8\nf 2 8 4\n";
    const OCTA: &str = "v 0.12 0 0\nv -0.12 0 0\nv 0 0.12 0\nv 0 -0.12 0\nv 0 0 0.12\nv 0 0 -0.12\nf 1 3 5\nf 3 2 5\nf 2 4 5\nf 4 1 5\nf 3 1 6\nf 2 3 6\nf 4 2 6\nf 1 4 6\n";
    const ID: [f64; 4] = [1.0, 0.0, 0.0, 0.0];

    /// `mjCMesh::Process`: the hull data plus the CoM and principal frame the geom's pose absorbs.
    fn mujoco_mesh(obj: &str) -> (MeshHull, V3, Matrix3<f64>) {
        let mut mesh = obj_as_mujoco_reads_it(obj).unwrap();
        for v in &mut mesh.verts {
            *v = v.map(|x| x as f32 as f64);
        }
        let (_, com, unit) = mesh_inertia_mujoco(&mesh, MeshInertia::Legacy).unwrap();
        let (_, quat) = eig3_mujoco(&unit);
        let r = quat_to_rotation(&quat);
        let stored = crate::TriMesh3 { verts: mesh.verts.iter().map(|v| r.transpose() * (v - com)).collect(), tris: mesh.tris.clone() };
        (MeshHull::from_frames(&mesh.verts, &stored, &stored.verts, None).unwrap(), com, r)
    }

    /// `mju_normalize4` (its sum of squares contracted, as the arm64 build computes it) then `mju_quat2Mat`:
    /// how the harness poses every probe, so the collider sees MuJoCo's own rotation to the bit.
    fn rot(q: [f64; 4]) -> Matrix3<f64> {
        let n = q[3].mul_add(q[3], q[2].mul_add(q[2], q[0].mul_add(q[0], q[1] * q[1]))).sqrt();
        let q = if n < MINVAL {
            [1.0, 0.0, 0.0, 0.0]
        } else if (n - 1.0).abs() > MINVAL {
            let inv = 1.0 / n;
            q.map(|x| x * inv)
        } else {
            q
        };
        quat_to_mat(&q)
    }

    fn pose(p: [f64; 3], q: [f64; 4]) -> GeomPose {
        GeomPose { pos: Vector3::new(p[0], p[1], p[2]), mat: rot(q) }
    }

    /// A mesh geom at a body's pose: the body carries the mesh's CoM offset and principal rotation.
    fn mesh_geom<'a>(hull: &'a MeshHull, com: &V3, r: &Matrix3<f64>, p: [f64; 3], q: [f64; 4]) -> CollisionGeom<'a> {
        let rb = rot(q);
        CollisionGeom { kind: GeomType::Mesh, pose: GeomPose { pos: Vector3::new(p[0], p[1], p[2]) + rb * com, mat: rb * r }, size: [0.0; 3], hull: Some(hull) }
    }

    fn prim(kind: GeomType, size: [f64; 3], p: [f64; 3], q: [f64; 4]) -> CollisionGeom<'static> {
        CollisionGeom { kind, pose: pose(p, q), size, hull: None }
    }

    fn big_box() -> CollisionGeom<'static> {
        prim(GeomType::Box, [0.2, 0.3, 0.15], [0.0, 0.0, 1.0], ID)
    }

    /// The contact SET matches: every expected position is met by exactly one contact, every contact has
    /// the stated distance and normal. Order is not compared — it follows hull vertex order, which is
    /// qhull's in MuJoCo and ours here.
    fn check_set(got: &[PreContact], dist: f64, normal: [f64; 3], positions: &[[f64; 3]], tol: f64) {
        assert_eq!(got.len(), positions.len(), "{} contacts, MuJoCo has {}: {got:?}", got.len(), positions.len());
        let n = Vector3::new(normal[0], normal[1], normal[2]);
        let mut used = vec![false; got.len()];
        for p in positions {
            let p = Vector3::new(p[0], p[1], p[2]);
            let (i, d) = got.iter().enumerate().filter(|(i, _)| !used[*i]).map(|(i, c)| (i, (c.pos - p).norm())).fold((usize::MAX, f64::INFINITY), |a, b| if b.1 < a.1 { b } else { a });
            assert!(d <= tol, "no contact within {tol:.1e} of MuJoCo's {p:?}; nearest {d:.2e}: {got:?}");
            used[i] = true;
        }
        for c in got {
            assert!((c.dist - dist).abs() <= tol, "dist {} vs MuJoCo {dist}", c.dist);
            assert!((c.normal - n).norm() <= tol, "normal {:?} vs MuJoCo {normal:?}", c.normal);
        }
    }

    #[test]
    fn a_tilted_box_mesh_resting_on_a_box_yields_mujocos_four_clipped_corners() {
        let (hull, com, r) = mujoco_mesh(BOXMESH);
        let m = mesh_geom(&hull, &com, &r, [0.05, 0.1, 1.215], [0.9961947, 0.0, 0.0, 0.0871557]);
        let c = collide_pair(0.0, &big_box(), &m).unwrap();
        check_set(
            &c,
            -0.010000002980232292,
            [0.0, 0.0, 1.0],
            &[
                [-0.02243356319891833, -0.06508598065393406, 1.1449999985098838],
                [0.17452799330795354, -0.030356361443763358, 1.1449999985098838],
                [0.12243356319891834, 0.2650859806539341, 1.1449999985098838],
                [-0.07452799330795352, 0.23035636144376337, 1.1449999985098838],
            ],
            1e-9,
        );
    }

    #[test]
    fn with_multiccd_disabled_the_same_pair_yields_mujocos_single_epa_witness() {
        let (hull, com, r) = mujoco_mesh(BOXMESH);
        let m = mesh_geom(&hull, &com, &r, [0.05, 0.1, 1.215], [0.9961947, 0.0, 0.0, 0.0871557]);
        let opts = CollideOptions { multiccd: false, ..Default::default() };
        let c = collide_pair_with(&opts, 0.0, &big_box(), &m).unwrap();
        check_set(&c, -0.01000000298023229, [0.0, 0.0, 1.0], &[[0.03727506225771292, 0.06483742636809203, 1.1449999985098838]], 1e-9);
    }

    #[test]
    fn an_octahedron_vertex_into_a_box_face_yields_mujocos_single_contact() {
        let (hull, com, r) = mujoco_mesh(OCTA);
        let m = mesh_geom(&hull, &com, &r, [0.03, 0.02, 1.26], ID);
        let c = collide_pair(0.0, &big_box(), &m).unwrap();
        check_set(&c, -0.009999997317790887, [0.0, 0.0, 1.0], &[[0.03, 0.020000000000000004, 1.1450000013411044]], 1e-9);
    }

    #[test]
    fn a_tilted_octahedron_edge_on_a_box_mesh_yields_mujocos_two_edge_contacts() {
        let (h1, c1, r1) = mujoco_mesh(BOXMESH);
        let (h2, c2, r2) = mujoco_mesh(OCTA);
        let m1 = mesh_geom(&h1, &c1, &r1, [0.0, 0.0, 1.0], ID);
        let m2 = mesh_geom(&h2, &c2, &r2, [0.02, 0.03, 1.15], [0.9238795, 0.3826834, 0.0, 0.0]);
        let c = collide_pair(0.0, &m1, &m2).unwrap();
        assert_eq!(c.len(), 2, "{c:?}");
        let expected = [[0.02, 0.1148528088827319, 1.0700735940857047], [0.02, -0.05485281480882305, 1.0700735970487503]];
        let dists = [-0.009852817789055583, -0.00985281186296441];
        for (p, d) in expected.iter().zip(dists) {
            let p = Vector3::new(p[0], p[1], p[2]);
            let k = c.iter().min_by(|a, b| (a.pos - p).norm().partial_cmp(&(b.pos - p).norm()).unwrap()).unwrap();
            assert!((k.pos - p).norm() < 1e-9, "{:?} vs {p:?}", k.pos);
            assert!((k.dist - d).abs() < 1e-9, "{} vs {d}", k.dist);
            assert!((k.normal - Vector3::z()).norm() < 1e-9);
        }
    }

    /// ⚠ A regular 16-gon makes `hull4`'s greedy area comparisons EXACT TIES, so the chosen quadrilateral is
    /// decided by the last bit, and MuJoCo's own answer is build-dependent. The arm64 macOS wheel (clang
    /// contracts `a*b + c` into fused multiply-adds — 13,528 `fmadd` instructions in its arm64 slice, none in
    /// its x86-64 slice) returns the square at 112.5°, 22.5°, −67.5°, 202.5° below, and so does this port,
    /// which fuses at the same sites; the same C source built with `-ffp-contract=off` (what the x86-64
    /// wheels compute) returns the square at 90°, 0°, −90°, 180° instead. Both are
    /// `scripts/mujoco_ccd_harness.c` built around MuJoCo's own `engine_collision_gjk.c`
    /// (`scripts/mujoco_ccd_harness_build.sh`); these are `ccd_on`'s numbers, compared bit for bit.
    #[test]
    fn a_cylinder_standing_on_a_box_yields_mujocos_four_sixteen_gon_contacts() {
        let cy = prim(GeomType::Cylinder, [0.05, 0.1, 0.0], [0.04, 0.02, 1.24], ID);
        let c = collide_pair(0.0, &cy, &big_box()).unwrap();
        check_set(
            &c,
            -0.010000000000000009,
            [0.0, 0.0, -1.0],
            &[
                [-0.0061939766255643203, 0.039134171618254485, 1.145],
                [0.059134171618254461, 0.066193976625564352, 1.145],
                [0.086193976625564342, 0.00086582838174548171, 1.145],
                [0.020865828381745527, -0.026193976625564355, 1.145],
            ],
            0.0,
        );
        // every contact sits on the cylinder's rim, whichever quadrilateral a build picks
        for k in &c {
            let r = ((k.pos.x - 0.04).powi(2) + (k.pos.y - 0.02).powi(2)).sqrt();
            assert!((r - 0.05).abs() < 1e-12, "{r}");
        }
    }

    /// ⚠ MuJoCo's dispatcher sends capsule–box to its own `mjc_CapsuleBox`, not here (that routine is
    /// `mujoco_collision::capsule_box`, pinned separately and exact). This pins the CCD path itself, which
    /// the dispatcher reaches for every other capsule pair, against MuJoCo's own `engine_collision_gjk.c`
    /// driven directly by `scripts/mujoco_ccd_harness.c`.
    ///
    /// A segment parallel to a face is a degenerate closest-feature pair, so GJK's witness point is decided
    /// by the last bit. Here both builds of MuJoCo's source land mid-segment and the two perturbations then
    /// add both capsule ends — three contacts — and differ only in the last digits; the numbers below are
    /// the contracted build's (`ccd_on`), bit for bit.
    #[test]
    fn a_capsule_lying_on_a_box_yields_mujocos_perturbation_contacts() {
        let mut cp = CcdObj::new(GeomType::Capsule, pose([0.04, 0.02, 1.18], [0.7071068, 0.0, 0.7071068, 0.0]), [0.04, 0.1, 0.0], None, 0.0);
        let mut bx = CcdObj::new(GeomType::Box, pose([0.0, 0.0, 1.0], ID), [0.2, 0.3, 0.15], None, 0.0);
        let c = convex_pair(&mut cp, &mut bx, 0.0, 1e-6, 35, true);
        assert_eq!(c.len(), 3, "{c:?}");
        let expected = [
            ([0.026666666666666668, 0.020000000000000021, 1.1449999999999998], [0.0, 6.9388939039070686e-16, -1.0]),
            ([-0.060069869953409211, 0.020000000000000039, 1.1450000349133043], [-0.00099999983333316994, 1.0230933182414512e-15, -0.99999950000004167]),
            ([0.14006982995343253, 0.020000000000000039, 1.1450000348866374], [0.00099999983333323196, 1.017757799866334e-15, -0.99999950000004167]),
        ];
        for (k, (p, n)) in c.iter().zip(expected) {
            assert_eq!(k.dist, -0.0099999999999999777);
            assert_eq!(k.pos, Vector3::new(p[0], p[1], p[2]));
            assert_eq!(k.normal, Vector3::new(n[0], n[1], n[2]));
        }
    }

    #[test]
    fn crossed_cylinders_yield_mujocos_iterated_epa_answer() {
        let c1 = prim(GeomType::Cylinder, [0.05, 0.2, 0.0], [0.0, 0.0, 1.0], ID);
        let c2 = prim(GeomType::Cylinder, [0.05, 0.2, 0.0], [0.02, 0.01, 1.095], [0.7071068, 0.0, 0.7071068, 0.0]);
        let c = collide_pair(0.0, &c1, &c2).unwrap();
        // MuJoCo's source built without contraction differs only in the last digits (pos x
        // −1.4796140527732876e-05, normal x −2.824882700680563e-07); the contracted build's, bit for bit:
        check_set(&c, -0.089999910950677928, [-2.8248827006248473e-07, 0.99999999999995637, -8.719299348732596e-08], &[[-1.4796140527733129e-05, 0.0050000005607276354, 1.0949930830747925]], 0.0);
    }

    /// `fourier_n1/n1.xml` geoms 1 and 16 at MuJoCo's own `geom_xpos`/`geom_xmat`: two cylinders overlapping
    /// by 31 µm at a generic orientation, where every product in GJK and EPA reaches the answer. MuJoCo's
    /// source built without contraction reports this contact with the normal REVERSED at the same depth;
    /// the arm64 wheel, `ccd_on`, and this port report the numbers below, bit for bit.
    #[test]
    fn fourier_n1s_overlapping_cylinders_match_the_contracted_build_bit_for_bit() {
        let cyl = |size: [f64; 3], p: [f64; 3], m: [f64; 9]| CollisionGeom {
            kind: GeomType::Cylinder,
            pose: GeomPose { pos: Vector3::new(p[0], p[1], p[2]), mat: Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]) },
            size,
            hull: None,
        };
        let a = cyl(
            [0.065, 0.0675, 0.0],
            [0.08539340246023025, 0.032896559837938814, 0.6641187211978501],
            [-0.49418139395483024, -0.8109762031412986, 0.31321294323093446, 0.6602653884717504, -0.11574723429074663, 0.7420594279033802, -0.5655390054232421, 0.5735156281153694, 0.592659647396652],
        );
        let b = cyl(
            [0.06, 0.095, 0.0],
            [0.04788949966356997, 0.23417202888574384, 0.6552473273724194],
            [-0.6018919636550226, -0.46927452071099596, 0.6461481937597335, -0.1511126975750603, -0.7275770566496756, -0.6691760450498682, 0.7841498687872679, -0.5004128803782532, 0.3670094446093328],
        );
        let c = collide_pair(0.0, &a, &b).unwrap();
        check_set(&c, -3.1287806572612574e-05, [0.052617795152431056, -0.99686384009728823, -0.059108814400097937], &[[0.077371193662543206, 0.12623679201342905, 0.66536192686712903]], 0.0);
    }

    #[test]
    fn an_ellipsoid_on_a_box_yields_mujocos_iterated_epa_answer() {
        let el = prim(GeomType::Ellipsoid, [0.06, 0.04, 0.03], [0.03, 0.02, 1.175], [0.9238795, 0.0, 0.3826834, 0.0]);
        let c = collide_pair(0.0, &el, &big_box()).unwrap();
        check_set(&c, -0.022434120907788884, [-6.8874450054395757e-08, -3.4741962506832099e-08, -0.999999999999997], &[[0.058423792189869435, 0.020006847523118557, 1.1387829395461053]], 0.0);
    }

    #[test]
    fn a_sphere_on_a_box_mesh_takes_the_point_shortcut_and_inflates() {
        let (hull, com, r) = mujoco_mesh(BOXMESH);
        let m = mesh_geom(&hull, &com, &r, [0.0, 0.0, 1.0], ID);
        let s = prim(GeomType::Sphere, [0.05, 0.0, 0.0], [0.03, 0.04, 1.12], ID);
        let c = collide_pair(0.0, &s, &m).unwrap();
        check_set(&c, -0.005000002980232357, [0.0, 0.0, -1.0], &[[0.030000000000000002, 0.040000000000000015, 1.0725000014901163]], 1e-9);
    }

    /// A clip that writes nothing must SAY so and leave EPA's witness pair alone. (MuJoCo's edge-on-face
    /// branch swaps the pair whether or not the clip wrote anything, and this port does the same.)
    #[test]
    fn a_clip_that_produces_nothing_reports_it() {
        let mut st = CcdStatus {
            separated: false,
            dist: vec![-1.0],
            x1: vec![Vector3::new(0.0, 0.0, 1.0)],
            x2: vec![Vector3::new(0.0, 0.0, -1.0)],
            gjk_iterations: 0,
            epa_iterations: 0,
            epa_status: 0,
            simplex: [Vertex::default(); 4],
            nsimplex: 4,
            tolerance: 1e-6,
            max_iterations: 35,
            max_contacts: 4,
            dist_cutoff: 0.0,
        };
        let (x1, x2) = (st.x1[0], st.x2[0]);
        // a unit square in the z = 0 plane, wound so its side planes face inward for n = +z (the winding
        // `boxFace` and `meshFace` produce), and an edge that sits far outside its prism
        let face1 = [Vector3::new(-1.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0), Vector3::new(1.0, -1.0, 0.0), Vector3::new(-1.0, -1.0, 0.0)];
        let far = [Vector3::new(50.0, 50.0, -1.0), Vector3::new(60.0, 50.0, -1.0)];
        let n = Vector3::z();
        assert!(!polygon_clip(&mut st, &face1, &far, &n, &(-n)), "a clip with nothing inside must report false");
        assert_eq!((st.x1[0], st.x2[0]), (x1, x2), "and must leave EPA's witness pair untouched");

        // the same edge over the square does produce points, and says so
        let over = [Vector3::new(-0.5, 0.0, -0.1), Vector3::new(0.5, 0.0, -0.1)];
        assert!(polygon_clip(&mut st, &face1, &over, &n, &(-n)), "a clip with points inside must report true");
        assert_eq!(st.x1.len(), 2, "an edge clipped against a face gives its two ends");
    }

    #[test]
    fn a_tilted_box_mesh_on_a_plane_yields_mujocos_two_face_corners() {
        let (hull, com, r) = mujoco_mesh(BOXMESH);
        let m = mesh_geom(&hull, &com, &r, [0.0, 0.0, 0.07], [0.9961947, 0.0871557, 0.0, 0.0]);
        let floor = prim(GeomType::Plane, [1.0, 1.0, 0.1], [0.0, 0.0, 0.0], ID);
        let c = collide_pair(0.0, &floor, &m).unwrap();
        check_set(
            &c,
            -0.029907800578941948,
            [0.0, 0.0, 1.0],
            &[[-0.10000000149011641, -0.13469756352158993, -0.014953900289470974], [0.10000000149011641, -0.13469756352158993, -0.014953900289470974]],
            1e-9,
        );
    }
}
