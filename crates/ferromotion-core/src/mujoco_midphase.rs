//! **`mj_collision` above the colliders**: which geom pairs a collider is ever handed. MuJoCo 3.13.0 decides
//! it in two gates (`engine_collision_driver.c`), ported here in its own arithmetic:
//!
//! * **the broadphase** (`mj_broadphase`) — the world's geoms and any static plane are paired with every
//!   body outright; every other pair of bodies must overlap in a sweep-and-prune over boxes laid along the
//!   principal axes of the geom cloud, swept in `float`;
//! * **the midphase** (`mj_collideTree`) — two bodies that are not both single-geom descend their compiled
//!   bounding-volume hierarchies together (boxes over the geoms, in each body's INERTIAL frame, built by
//!   the compiler's `mjCBoundingVolumeHierarchy`), and a geom pair reaches its collider only if every node
//!   pair on the way down overlaps and the two geoms' own boxes do.
//!
//! Both gates are conservative, so they move the contact set only at the edge — and the edge is exactly
//! where a port with the same colliders still reports a different contact: franka's finger pads touch at
//! rest 7.8e-18 apart, and MuJoCo calls them separated at a BVH node.
//!
//! ⛔ Contracted as clang contracts MuJoCo's C and C++ on arm64 (see [`crate::mujoco_ccd`]).

use crate::mujoco_kinematics::XFrame;
use std::ops::Range;

/// `mjEPS`
const MJ_EPS: f64 = 1e-14;
/// `mjMAXVAL`: a box half-size at least this large is infinite (a plane)
const MJ_MAXVAL: f64 = 1e10;
/// `eigEPS` in `mju_eig3`: `mjMINVAL * 1000`
const EIG_EPS: f64 = 1e-12;

/// `std::min` / `mju_min`: the FIRST argument on a tie (`f64::min` promises neither on signed zeros).
fn cmin(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `std::max` / `mju_max`: the FIRST argument on a tie.
fn cmax(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `mju_dot3`, contracted.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
}

/// `mji_mulMatVec3`, contracted.
fn mul_mat_vec3(m: &[f64; 9], v: &[f64; 3]) -> [f64; 3] {
    [m[2].mul_add(v[2], m[0].mul_add(v[0], m[1] * v[1])), m[5].mul_add(v[2], m[3].mul_add(v[0], m[4] * v[1])), m[8].mul_add(v[2], m[6].mul_add(v[0], m[7] * v[1]))]
}

/// **One body's bounding-volume hierarchy as MuJoCo compiles it** (`bvh_aabb`, `bvh_child`, `bvh_nodeid`,
/// `bvh_depth` from `body_bvhadr`): node 0 is the root, nodes are numbered depth-first with the lower half
/// along the split axis first, and every box is in the body's inertial frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MjBvh {
    /// each node's box: centre, then half-sizes
    pub aabb: Vec<[f64; 6]>,
    /// each node's two children, `-1` for none
    pub child: Vec<[i32; 2]>,
    /// the geom a leaf holds (MuJoCo's geom id), `-1` on an inner node
    pub nodeid: Vec<i32>,
    /// each node's depth, the root 0
    pub depth: Vec<i32>,
}

/// One collidable geom as `mjCBody::ComputeBVH` hands it to the hierarchy: its id, its pose in the body frame
/// (`geom_pos`, `geom_quat`) and its own box (`geom_aabb`).
pub(crate) struct BvLeaf {
    pub(crate) id: usize,
    pub(crate) pos: [f64; 3],
    pub(crate) quat: [f64; 4],
    pub(crate) aabb: [f64; 6],
}

impl MjBvh {
    /// `mjCBoundingVolumeHierarchy::CreateBVH` over `leaves` (a body's geoms with a nonzero `contype` or
    /// `conaffinity`, in geom order), in the body's inertial frame `ipos`/`iquat`. `rot` is the compiler's
    /// `mjuu_rotVecQuat`.
    pub(crate) fn build(ipos: &[f64; 3], iquat: &[f64; 4], leaves: &[BvLeaf], rot: fn(&[f64; 3], &[f64; 4]) -> [f64; 3]) -> MjBvh {
        let qinv = [iquat[0], -iquat[1], -iquat[2], -iquat[3]];
        let lpos: Vec<[f64; 3]> = leaves.iter().map(|l| rot(&[l.pos[0] - ipos[0], l.pos[1] - ipos[1], l.pos[2] - ipos[2]], &qinv)).collect();
        let mut out = MjBvh::default();
        let mut elements: Vec<usize> = (0..leaves.len()).collect();
        out.make(&mut elements, 0, leaves, &lpos, ipos, &qinv, rot);
        out
    }

    /// `MakeBVH`: the node over `elements`, then its two halves.
    #[allow(clippy::too_many_arguments)]
    fn make(&mut self, elements: &mut [usize], lev: i32, leaves: &[BvLeaf], lpos: &[[f64; 3]], ipos: &[f64; 3], qinv: &[f64; 4], rot: fn(&[f64; 3], &[f64; 4]) -> [f64; 3]) -> i32 {
        let n = elements.len();
        if n == 0 {
            return -1;
        }
        let mut aamm = [f64::MAX, f64::MAX, f64::MAX, -f64::MAX, -f64::MAX, -f64::MAX];
        for &e in elements.iter() {
            let (l, a) = (&leaves[e], &leaves[e].aabb);
            let b = [a[0] - a[3], a[1] - a[4], a[2] - a[5], a[0] + a[3], a[1] + a[4], a[2] + a[5]];
            for v in 0..8 {
                let corner = [if v & 1 != 0 { b[3] } else { b[0] }, if v & 2 != 0 { b[4] } else { b[1] }, if v & 4 != 0 { b[5] } else { b[2] }];
                // to the body's inertial frame
                let mut bx = rot(&corner, &l.quat);
                for k in 0..3 {
                    bx[k] += l.pos[k] - ipos[k];
                }
                let vert = rot(&bx, qinv);
                for k in 0..3 {
                    aamm[k] = cmin(aamm[k], vert[k]);
                    aamm[k + 3] = cmax(aamm[k + 3], vert[k]);
                }
            }
        }
        // inflate flat boxes
        for k in 0..3 {
            if (aamm[k] - aamm[k + 3]).abs() < MJ_EPS {
                aamm[k] -= MJ_EPS;
                aamm[k + 3] += MJ_EPS;
            }
        }
        let index = self.aabb.len();
        self.child.push([-1, -1]);
        self.nodeid.push(-1);
        self.depth.push(lev);
        self.aabb.push([(aamm[3] + aamm[0]) / 2.0, (aamm[4] + aamm[1]) / 2.0, (aamm[5] + aamm[2]) / 2.0, (aamm[3] - aamm[0]) / 2.0, (aamm[4] - aamm[1]) / 2.0, (aamm[5] - aamm[2]) / 2.0]);
        if n == 1 {
            self.nodeid[index] = leaves[elements[0]].id as i32;
            return index as i32;
        }
        // the longest axis, by a margin of at least mjEPS
        let edges = [aamm[3] - aamm[0], aamm[4] - aamm[1], aamm[5] - aamm[2]];
        let mut axis = 0;
        if edges[1] >= edges[0] + MJ_EPS {
            axis = 1;
        }
        if edges[2] >= edges[axis] + MJ_EPS {
            axis = 2;
        }
        // `std::nth_element` at the median: which elements land in each half is fixed by the comparator
        // (positions within mjEPS tie, and a tie goes to the earlier geom), whatever order each half is in.
        // An insertion sort, because the comparator is not a total order Rust's sort could be promised.
        let less = |a: usize, b: usize| if (lpos[a][axis] - lpos[b][axis]).abs() > MJ_EPS { lpos[a][axis] < lpos[b][axis] } else { a < b };
        for i in 1..n {
            let mut j = i;
            while j > 0 && less(elements[j], elements[j - 1]) {
                elements.swap(j, j - 1);
                j -= 1;
            }
        }
        let m = n / 2;
        let (lo, hi) = elements.split_at_mut(m);
        let c0 = self.make(lo, lev + 1, leaves, lpos, ipos, qinv, rot);
        let c1 = self.make(hi, lev + 1, leaves, lpos, ipos, qinv, rot);
        self.child[index] = [c0, c1];
        index as i32
    }

    fn is_leaf(&self, node: usize) -> bool {
        self.child[node][0] < 0 && self.child[node][1] < 0
    }
}

/// One geom as the driver reads it from `mjModel`.
#[derive(Clone, Debug)]
pub(crate) struct DriverGeom {
    pub(crate) body: usize,
    pub(crate) plane: bool,
    /// `geom_rbound`
    pub(crate) rbound: f64,
    /// `geom_aabb`: centre and half-sizes in the geom frame
    pub(crate) aabb: [f64; 6],
    pub(crate) margin: f64,
    pub(crate) gap: f64,
    pub(crate) contype: u32,
    pub(crate) conaffinity: u32,
}

/// What `mj_broadphase` and `mj_collideTree` read from the compiled model, bodies in MuJoCo's order.
#[derive(Clone, Debug, Default)]
pub(crate) struct Driver {
    pub(crate) parent: Vec<usize>,
    /// `body_weldid`
    pub(crate) weld: Vec<usize>,
    /// `body_dofnum[b] > 0`
    pub(crate) has_dofs: Vec<bool>,
    /// `body_contype` / `body_conaffinity`: the OR over the body's geoms
    pub(crate) contype: Vec<u32>,
    pub(crate) conaffinity: Vec<u32>,
    /// `body_geomadr .. + body_geomnum`
    pub(crate) geoms: Vec<Range<usize>>,
    /// `body_margin`: the largest `margin + gap` of the body's geoms
    pub(crate) margin: Vec<f64>,
    /// `body_bvhadr >= 0`: the body's hierarchy, `None` for a body with no collidable geom
    pub(crate) bvh: Vec<Option<MjBvh>>,
    pub(crate) geom: Vec<DriverGeom>,
}

/// `filterBitmask` inverted.
fn compatible(contype1: u32, conaffinity1: u32, contype2: u32, conaffinity2: u32) -> bool {
    (contype1 & conaffinity2) != 0 || (contype2 & conaffinity1) != 0
}

impl Driver {
    /// Everything but the hierarchies, from the bodies' parents and joint counts and the geoms.
    pub(crate) fn new(parent: Vec<usize>, has_joints: Vec<bool>, geom: Vec<DriverGeom>) -> Driver {
        let nbody = parent.len();
        let mut weld = vec![0; nbody];
        for b in 1..nbody {
            weld[b] = if has_joints[b] { b } else { weld[parent[b]] };
        }
        let mut geoms = vec![0..0; nbody];
        let (mut contype, mut conaffinity, mut margin) = (vec![0u32; nbody], vec![0u32; nbody], vec![0.0f64; nbody]);
        for (g, dg) in geom.iter().enumerate() {
            let b = dg.body;
            geoms[b] = if geoms[b].is_empty() { g..g + 1 } else { geoms[b].start..g + 1 };
            contype[b] |= dg.contype;
            conaffinity[b] |= dg.conaffinity;
            margin[b] = margin[b].max(dg.margin + dg.gap);
        }
        Driver { parent, weld, has_dofs: has_joints, contype, conaffinity, geoms, margin, bvh: vec![None; nbody], geom }
    }

    fn can_collide(&self, b: usize) -> bool {
        self.contype[b] != 0 || self.conaffinity[b] != 0
    }

    /// `canCollide2`
    pub(crate) fn can_collide2(&self, b1: usize, b2: usize) -> bool {
        compatible(self.contype[b1], self.conaffinity[b1], self.contype[b2], self.conaffinity[b2])
    }

    /// `filterBodyPair` for awake bodies: `true` discards.
    pub(crate) fn filter_body_pair(&self, b1: usize, b2: usize, filterparent: bool) -> bool {
        let (w1, w2) = (self.weld[b1], self.weld[b2]);
        let (pw1, pw2) = (self.weld[self.parent[w1]], self.weld[self.parent[w2]]);
        if w1 == w2 || (!self.has_dofs[w1] && !self.has_dofs[w2]) {
            return true;
        }
        filterparent && w1 != 0 && w2 != 0 && (w1 == pw2 || w2 == pw1)
    }

    /// `add_pair`: a body pair whose geoms could ever meet, `(lower id, higher id)`.
    fn add_pair(&self, b1: usize, b2: usize, out: &mut Vec<(usize, usize)>) {
        if compatible(self.contype[b1], self.conaffinity[b1], self.contype[b2], self.conaffinity[b2]) {
            out.push((b1.min(b2), b1.max(b2)));
        }
    }

    /// **`mj_broadphase`**: the body pairs `mj_collision` visits, sorted by signature and without repeats.
    /// `gx` is every geom's world frame (`geom_xpos`, row-major `geom_xmat`).
    pub(crate) fn broadphase(&self, gx: &[XFrame], filterparent: bool) -> Vec<(usize, usize)> {
        let nbody = self.parent.len();
        let mut pairs = Vec::new();
        // the world's geoms, and a static body's plane, meet every body outright
        for b1 in 0..nbody {
            if !self.can_collide(b1) {
                continue;
            }
            let has_plane = self.geoms[b1].clone().any(|g| self.geom[g].plane);
            if (b1 == 0 && !self.geoms[0].is_empty()) || (!self.has_dofs[self.weld[b1]] && has_plane) {
                for b2 in 0..nbody {
                    if self.can_collide(b2) && !self.filter_body_pair(b1, b2, filterparent) {
                        self.add_pair(b1, b2, &mut pairs);
                    }
                }
            }
        }
        // the principal frame of the non-world geoms' centres
        let moving: Vec<usize> = (0..self.geom.len()).filter(|&g| self.geom[g].body != 0).collect();
        if !moving.is_empty() {
            let mut cen = [0.0f64; 3];
            for &g in &moving {
                for k in 0..3 {
                    cen[k] += gx[g].0[k];
                }
            }
            let inv = 1.0 / moving.len() as f64;
            cen = cen.map(|c| c * inv);
            let mut cov = [0.0f64; 9];
            for &g in &moving {
                let d = [gx[g].0[0] - cen[0], gx[g].0[1] - cen[1], gx[g].0[2] - cen[2]];
                let (d00, d01, d02, d11, d12, d22) = (d[0] * d[0], d[0] * d[1], d[0] * d[2], d[1] * d[1], d[1] * d[2], d[2] * d[2]);
                for (c, v) in cov.iter_mut().zip([d00, d01, d02, d01, d11, d12, d02, d12, d22]) {
                    *c += v;
                }
            }
            let cov = cov.map(|c| c * inv);
            let frame = eig3(&cov).1;
            let ids: Vec<usize> = (1..nbody).filter(|&b| self.can_collide(b)).collect();
            if ids.len() > 1 {
                let aamm: Vec<[f64; 6]> = ids.iter().map(|&b| self.make_aamm(b, gx, &frame)).collect();
                for (i, j) in sweep_and_prune(&aamm) {
                    let (b1, b2) = (ids[i], ids[j]);
                    if !self.filter_body_pair(b1, b2, filterparent) {
                        self.add_pair(b1, b2, &mut pairs);
                    }
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// `makeAAMM`: a body's box along the rows of `frame`, each geom bounded by the tighter of its
    /// bounding sphere and its own box, widened by its `margin + gap`.
    fn make_aamm(&self, b: usize, gx: &[XFrame], frame: &[f64; 9]) -> [f64; 6] {
        let mut aamm = [0.0; 6];
        for (i, g) in self.geoms[b].clone().enumerate() {
            let dg = &self.geom[g];
            let margin = dg.margin + dg.gap;
            let (xpos, xmat) = (&gx[g].0, &gx[g].1);
            let c = mul_mat_vec3(xmat, &[dg.aabb[0], dg.aabb[1], dg.aabb[2]]);
            let pos = [c[0] + xpos[0], c[1] + xpos[1], c[2] + xpos[2]];
            let axis = |k: usize| [xmat[k], xmat[3 + k], xmat[6 + k]];
            let mut own = [0.0; 6];
            for j in 0..3 {
                let fj = [frame[3 * j], frame[3 * j + 1], frame[3 * j + 2]];
                let aabb_cen = dot3(&pos, &fj);
                let aabb_half = (dg.aabb[3] * dot3(&axis(0), &fj)).abs() + (dg.aabb[4] * dot3(&axis(1), &fj)).abs() + (dg.aabb[5] * dot3(&axis(2), &fj)).abs();
                let r_cen = dot3(xpos, &fj);
                own[j] = cmax(r_cen - dg.rbound, aabb_cen - aabb_half) - margin;
                own[j + 3] = cmin(r_cen + dg.rbound, aabb_cen + aabb_half) + margin;
            }
            if i == 0 {
                aamm = own;
            } else {
                for j in 0..3 {
                    aamm[j] = cmin(aamm[j], own[j]);
                    aamm[j + 3] = cmax(aamm[j + 3], own[j + 3]);
                }
            }
        }
        aamm
    }

    /// **`mj_collideTree`** for bodies `b1 < b2`: every geom pair whose leaves are reached — every node pair
    /// on the way down overlapping, then the two geoms' bounding spheres and their own boxes — in the order
    /// the traversal reaches them. The caller still applies the rest of `filterCollisionPair`. Every node
    /// MuJoCo's `bvh_active` would mark goes into `active` as `(body, node)`.
    pub(crate) fn collide_tree(&self, b1: usize, b2: usize, gx: &[XFrame], xi: &[XFrame], out: &mut Vec<(usize, usize)>, active: &mut Vec<(usize, usize)>) {
        let (Some(t1), Some(t2)) = (&self.bvh[b1], &self.bvh[b2]) else { return };
        if !self.can_collide2(b1, b2) {
            return;
        }
        let mut product: Option<([f64; 36], [f64; 12])> = None;
        let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
        while let Some((n1, n2)) = stack.pop() {
            let (leaf1, leaf2) = (t1.is_leaf(n1), t2.is_leaf(n2));
            if leaf1 && leaf2 {
                let (g1, g2) = (t1.nodeid[n1] as usize, t2.nodeid[n2] as usize);
                let (d1, d2) = (&self.geom[g1], &self.geom[g2]);
                let mg = (d1.margin + d2.margin) + (d1.gap + d2.gap);
                if !filter_sphere(d1, d2, &gx[g1].0, &gx[g2].0, &gx[g1].1, &gx[g2].1, mg) && obb_leaf(&d1.aabb, &gx[g1], &d2.aabb, &gx[g2], mg) {
                    out.push((g1, g2));
                    active.push((b1, n1));
                    active.push((b2, n2));
                }
                continue;
            }
            let margin = self.margin[b1] + self.margin[b2];
            let pr = product.get_or_insert_with(|| obb_products(&xi[b1], &xi[b2]));
            if !obb_node(&t1.aabb[n1], &t2.aabb[n2], pr, margin) {
                continue;
            }
            active.push((b1, n1));
            active.push((b2, n2));
            let children = |t: &MjBvh, n: usize| t.child[n].into_iter().filter(|&c| c != -1).map(|c| c as usize).collect::<Vec<_>>();
            if !leaf1 && leaf2 {
                stack.extend(children(t1, n1).into_iter().map(|c| (c, n2)));
            } else if leaf1 && !leaf2 {
                stack.extend(children(t2, n2).into_iter().map(|c| (n1, c)));
            } else {
                // ⛔ MuJoCo's "surface area" is of `half-size - centre`, not of the box — ported as written,
                // because which tree it descends decides which node pairs are tested on the way down
                let surface = |a: &[f64; 6]| {
                    let (x, y, z) = (a[3] - a[0], a[4] - a[1], a[5] - a[2]);
                    z.mul_add(x, x.mul_add(y, y * z))
                };
                if surface(&t1.aabb[n1]) > surface(&t2.aabb[n2]) {
                    stack.extend(children(t1, n1).into_iter().map(|c| (c, n2)));
                } else {
                    stack.extend(children(t2, n2).into_iter().map(|c| (n1, c)));
                }
            }
        }
    }
}

/// `mj_filterSphere`: `true` when two geoms cannot touch within `margin`.
pub(crate) fn filter_sphere(g1: &DriverGeom, g2: &DriverGeom, p1: &[f64; 3], p2: &[f64; 3], m1: &[f64; 9], m2: &[f64; 9], margin: f64) -> bool {
    if g1.rbound > 0.0 && g2.rbound > 0.0 {
        let d = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
        let bound = g1.rbound + g2.rbound + margin;
        return dot3(&d, &d) > bound * bound;
    }
    let plane_dist = |pp: &[f64; 3], pm: &[f64; 9], q: &[f64; 3]| dot3(&[q[0] - pp[0], q[1] - pp[1], q[2] - pp[2]], &[pm[2], pm[5], pm[8]]);
    if g1.plane && g2.rbound > 0.0 && plane_dist(p1, m1, p2) > margin + g2.rbound {
        return true;
    }
    g2.plane && g1.rbound > 0.0 && plane_dist(p2, m2, p1) > margin + g1.rbound
}

/// `mj_collideOBB` without precomputed products — the leaf test, on the two geoms' own boxes.
fn obb_leaf(a1: &[f64; 6], f1: &XFrame, a2: &[f64; 6], f2: &XFrame, margin: f64) -> bool {
    let aabb = [a1, a2];
    let frames = [f1, f2];
    let inf = |a: &[f64; 6]| [a[3] >= MJ_MAXVAL, a[4] >= MJ_MAXVAL, a[5] >= MJ_MAXVAL];
    let (inf1, inf2) = (inf(a1), inf(a2));
    if inf1.iter().all(|&x| x) || inf2.iter().all(|&x| x) {
        return true;
    }
    let infinite = [inf1.iter().any(|&x| x), inf2.iter().any(|&x| x)];
    let xcenter: [[f64; 3]; 2] = std::array::from_fn(|i| {
        let c = mul_mat_vec3(&frames[i].1, &[aabb[i][0], aabb[i][1], aabb[i][2]]);
        [c[0] + frames[i].0[0], c[1] + frames[i].0[1], c[2] + frames[i].0[2]]
    });
    let normal = |i: usize, j: usize| [frames[i].1[j], frames[i].1[3 + j], frames[i].1[6 + j]];
    for j in 0..2 {
        if infinite[1 - j] {
            continue;
        }
        for k in 0..3 {
            let njk = normal(j, k);
            let (mut proj, mut radius) = ([0.0; 2], [0.0; 2]);
            for i in 0..2 {
                proj[i] = dot3(&xcenter[i], &njk);
                radius[i] = (aabb[i][3] * dot3(&normal(i, 0), &njk)).abs() + (aabb[i][4] * dot3(&normal(i, 1), &njk)).abs() + (aabb[i][5] * dot3(&normal(i, 2), &njk)).abs();
            }
            if radius[0] + radius[1] + margin < (proj[1] - proj[0]).abs() {
                return false;
            }
        }
    }
    true
}

/// The products `mj_collideOBB` precomputes once per body pair, from the two inertial frames.
fn obb_products(f1: &XFrame, f2: &XFrame) -> ([f64; 36], [f64; 12]) {
    let frames = [f1, f2];
    let normal = |i: usize, j: usize| [frames[i].1[j], frames[i].1[3 + j], frames[i].1[6 + j]];
    let (mut product, mut offset) = ([0.0; 36], [0.0; 12]);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                for l in 0..3 {
                    product[18 * i + 9 * j + 3 * k + l] = dot3(&normal(i, l), &normal(j, k));
                }
                offset[6 * i + 3 * j + k] = dot3(&frames[i].0, &normal(j, k));
            }
        }
    }
    (product, offset)
}

/// `mj_collideOBB` with precomputed products — the test at an inner node pair, on the nodes' boxes.
fn obb_node(a1: &[f64; 6], a2: &[f64; 6], pr: &([f64; 36], [f64; 12]), margin: f64) -> bool {
    let aabb = [a1, a2];
    let (product, offset) = pr;
    let inf = |a: &[f64; 6]| [a[3] >= MJ_MAXVAL, a[4] >= MJ_MAXVAL, a[5] >= MJ_MAXVAL];
    let (inf1, inf2) = (inf(a1), inf(a2));
    if inf1.iter().all(|&x| x) || inf2.iter().all(|&x| x) {
        return true;
    }
    let infinite = [inf1.iter().any(|&x| x), inf2.iter().any(|&x| x)];
    for j in 0..2 {
        if infinite[1 - j] {
            continue;
        }
        for k in 0..3 {
            let (mut proj, mut radius) = ([0.0; 2], [0.0; 2]);
            for i in 0..2 {
                let adr = 18 * i + 9 * j + 3 * k;
                let a = aabb[i];
                proj[i] = a[2].mul_add(product[adr + 2], a[0].mul_add(product[adr], a[1] * product[adr + 1])) + offset[6 * i + 3 * j + k];
                radius[i] = (a[3] * product[adr]).abs() + (a[4] * product[adr + 1]).abs() + (a[5] * product[adr + 2]).abs();
            }
            if radius[0] + radius[1] + margin < (proj[1] - proj[0]).abs() {
                return false;
            }
        }
    }
    true
}

/// `mj_SAP` along the first row of the frame: pairs `(i, j)` of boxes (indices into `aamm`) that overlap on
/// all three axes, the sweep axis compared in `float`.
fn sweep_and_prune(aamm: &[[f64; 6]]) -> Vec<(usize, usize)> {
    // (value, id, is_max), sorted stably by value as `mjSORT` does
    let mut sorted: Vec<(f32, usize, bool)> = Vec::with_capacity(2 * aamm.len());
    for (i, a) in aamm.iter().enumerate() {
        sorted.push((a[0] as f32, i, false));
        sorted.push((a[3] as f32, i, true));
    }
    // `SAPcmp`: below, equal, else above — so a NaN is "above" whichever side it is on
    mj_sort(&mut sorted, |a, b| if a.0 < b.0 { -1 } else if a.0 == b.0 { 0 } else { 1 });
    let mut active: Vec<usize> = Vec::new();
    let mut pairs = Vec::new();
    for &(_, id, is_max) in &sorted {
        if !is_max {
            for &other in &active {
                let (a, b) = (&aamm[other], &aamm[id]);
                if a[1] > b[4] || b[1] > a[4] || a[2] > b[5] || b[2] > a[5] {
                    continue;
                }
                pairs.push((other, id));
            }
            active.push(id);
        } else if let Some(p) = active.iter().position(|&x| x == id) {
            active.remove(p);
        }
    }
    pairs
}

/// `mjSORT` (`engine_sort.h`): insertion sort over runs of 32, then bottom-up merges taking the LEFT element
/// on a tie — stable, and ported as written so that a comparator a `NaN` makes inconsistent still sorts as
/// MuJoCo's does.
fn mj_sort<T: Copy>(arr: &mut [T], cmp: impl Fn(&T, &T) -> i32) {
    const RUN: usize = 32;
    let n = arr.len();
    for start in (0..n).step_by(RUN) {
        let end = (start + RUN).min(n);
        for j in start + 1..end {
            let tmp = arr[j];
            let mut k = j;
            while k > start && cmp(&arr[k - 1], &tmp) > 0 {
                arr[k] = arr[k - 1];
                k -= 1;
            }
            arr[k] = tmp;
        }
    }
    let mut src = arr.to_vec();
    let mut dest = arr.to_vec();
    let mut len = RUN;
    while len < n {
        for start in (0..n).step_by(2 * len) {
            let mid = start + len;
            let end = (start + 2 * len).min(n);
            if mid < end {
                let (mut i, mut j, mut k) = (start, mid, start);
                while i < mid && j < end {
                    if cmp(&src[i], &src[j]) <= 0 {
                        dest[k] = src[i];
                        i += 1;
                    } else {
                        dest[k] = src[j];
                        j += 1;
                    }
                    k += 1;
                }
                dest[k..k + (mid - i)].copy_from_slice(&src[i..mid]);
                let k = k + (mid - i);
                dest[k..k + (end - j)].copy_from_slice(&src[j..end]);
            } else {
                dest[start..end].copy_from_slice(&src[start..end]);
            }
        }
        std::mem::swap(&mut src, &mut dest);
        len *= 2;
    }
    arr.copy_from_slice(&src);
}

/// **`mju_eig3`** (the engine's, not the compiler's `mjuu_eig3`): eigenvalues, the eigenvector matrix
/// (`mju_quat2Mat` of the returned quaternion) and that quaternion, of a symmetric 3×3 matrix — a Jacobi
/// iteration from the identity, then sorted in decreasing order by quarter turns.
pub(crate) fn eig3(mat: &[f64; 9]) -> ([f64; 3], [f64; 9], [f64; 4]) {
    use crate::mujoco_kinematics::{mul_quat, normalize4, quat2mat};
    // `mji_mulMatTMat3` and `mji_mulMatMat3`, contracted
    let mat_t_mat = |a: &[f64; 9], b: &[f64; 9]| -> [f64; 9] { std::array::from_fn(|k| (k / 3, k % 3)).map(|(r, c)| a[6 + r].mul_add(b[6 + c], a[r].mul_add(b[c], a[3 + r] * b[3 + c]))) };
    let mat_mat = |a: &[f64; 9], b: &[f64; 9]| -> [f64; 9] { std::array::from_fn(|k| (k / 3, k % 3)).map(|(r, c)| a[3 * r + 2].mul_add(b[6 + c], a[3 * r].mul_add(b[c], a[3 * r + 1] * b[3 + c]))) };
    let mut quat = [1.0, 0.0, 0.0, 0.0];
    let mut eigval = [0.0; 3];
    for _ in 0..500 {
        let eigvec = quat2mat(&quat);
        let d = mat_mat(&mat_t_mat(&eigvec, mat), &eigvec);
        eigval = [d[0], d[4], d[8]];
        let (rk, ck, rotk) = if d[1].abs() > d[2].abs() && d[1].abs() > d[5].abs() {
            (0, 1, 2)
        } else if d[2].abs() > d[5].abs() {
            (0, 2, 1)
        } else {
            (1, 2, 0)
        };
        if d[3 * rk + ck].abs() < EIG_EPS {
            break;
        }
        let tau = (d[4 * ck] - d[4 * rk]) / (2.0 * d[3 * rk + ck]);
        let t = if tau >= 0.0 { 1.0 / (tau + tau.mul_add(tau, 1.0).sqrt()) } else { -1.0 / (-tau + tau.mul_add(tau, 1.0).sqrt()) };
        let c = 1.0 / t.mul_add(t, 1.0).sqrt();
        if c > 1.0 - EIG_EPS {
            break;
        }
        let mut tmp = [0.0; 4];
        let s = (-0.5f64).mul_add(c, 0.5).sqrt();
        tmp[rotk + 1] = if tau >= 0.0 { -s } else { s };
        if rotk == 1 {
            tmp[rotk + 1] = -tmp[rotk + 1];
        }
        tmp[0] = (-tmp[rotk + 1]).mul_add(tmp[rotk + 1], 1.0).sqrt();
        normalize4(&mut tmp);
        quat = mul_quat(&quat, &tmp);
        normalize4(&mut quat);
    }
    for j in 0..3 {
        let j1 = j % 2;
        if eigval[j1] + EIG_EPS < eigval[j1 + 1] {
            eigval.swap(j1, j1 + 1);
            #[allow(clippy::approx_constant)]
            let h = 0.707106781186548;
            let mut tmp = [h, 0.0, 0.0, 0.0];
            tmp[(j1 + 2) % 3 + 1] = h;
            quat = mul_quat(&quat, &tmp);
            normalize4(&mut quat);
        }
    }
    (eigval, quat2mat(&quat), quat)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `mjSORT` is a stable sort: on a total order it agrees with the standard library's, at sizes on both
    /// sides of its 32-element runs.
    #[test]
    fn mjsort_is_stable() {
        for n in [0usize, 1, 5, 32, 33, 64, 100, 257] {
            let v: Vec<(u32, usize)> = (0..n).map(|i| (((i * 7919) % 13) as u32, i)).collect();
            let mut ours = v.clone();
            mj_sort(&mut ours, |a, b| (a.0 as i64 - b.0 as i64).signum() as i32);
            let mut std = v;
            std.sort_by_key(|x| x.0);
            assert_eq!(ours, std, "n = {n}");
        }
    }

    /// The sweep compares the sweep axis in `float`, stably: a box whose `float` minimum equals an earlier
    /// box's `float` maximum comes AFTER it and is never paired with it, however the doubles overlap.
    #[test]
    fn the_sweep_compares_in_float_and_keeps_ties_in_order() {
        let a = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        // overlaps `a` by 1e-12 in double, not at all in float
        let b = [1.0 - 1e-12, 0.0, 0.0, 2.0, 1.0, 1.0];
        assert_eq!(sweep_and_prune(&[a, b]), vec![]);
        // listed the other way round, b's minimum ties a's maximum AHEAD of it in the stable order, and the
        // same two boxes are a pair
        assert_eq!(sweep_and_prune(&[b, a]), vec![(1, 0)]);
        let c = [0.5, 0.0, 0.0, 2.0, 1.0, 1.0];
        assert_eq!(sweep_and_prune(&[a, c]), vec![(0, 1)]);
    }
}
