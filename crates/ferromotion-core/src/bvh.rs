//! **Bounding-volume hierarchy (AABB tree)** — the broadphase the collision layer lacked (it had
//! kd-tree / voxel-hash acceleration but no BVH). A binary tree of axis-aligned boxes, built
//! top-down by splitting the longest axis at the centroid median, culls the O(n²) all-pairs overlap
//! test to near O(n log n) by pruning disjoint subtrees. It also answers box- and ray-queries. The
//! set of candidate pairs it reports is *exactly* the brute-force set — a broadphase must never miss
//! a real overlap. Pure `nalgebra` → WASM-clean.

use nalgebra::Vector3;

/// An axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vector3<f64>,
    pub max: Vector3<f64>,
}

impl Aabb {
    pub fn new(min: Vector3<f64>, max: Vector3<f64>) -> Self {
        Aabb { min, max }
    }
    pub fn center(&self) -> Vector3<f64> {
        (self.min + self.max) * 0.5
    }
    fn union(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.inf(&o.min), max: self.max.sup(&o.max) }
    }
    /// Overlap (touching counts) with another box.
    pub fn overlaps(&self, o: &Aabb) -> bool {
        self.min.x <= o.max.x && self.max.x >= o.min.x && self.min.y <= o.max.y && self.max.y >= o.min.y && self.min.z <= o.max.z && self.max.z >= o.min.z
    }
    /// Slab ray test: does the ray `origin + t·dir` (`t ≥ 0`) intersect this box?
    pub fn ray_hits(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> bool {
        let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
        for a in 0..3 {
            let inv = 1.0 / dir[a];
            let mut ta = (self.min[a] - origin[a]) * inv;
            let mut tb = (self.max[a] - origin[a]) * inv;
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t1 < t0 {
                return false;
            }
        }
        true
    }
}

enum Node {
    Leaf { prim: usize },
    Inner { bounds: Aabb, left: usize, right: usize },
}

/// A BVH over a fixed set of primitive AABBs (indexed 0..n).
pub struct Bvh {
    nodes: Vec<Node>,
    root: usize,
    aabbs: Vec<Aabb>,
}

impl Bvh {
    /// Build the tree over `aabbs` (top-down, longest-axis centroid-median split).
    /// A box with a non-finite bound is left **out of the tree**. Ordering it late is not enough: its
    /// bounds `union` into every ancestor node, and nothing overlaps a `NaN`, so entire subtrees of
    /// real geometry are pruned from every query. Measured before this guard, a query covering two
    /// real boxes and one `NaN` box returned **one** of the two. Such a box keeps its index in
    /// `aabbs` and is simply never reported as a hit, which is what an invalid box should do.
    pub fn build(aabbs: &[Aabb]) -> Bvh {
        let mut idx: Vec<usize> =
            (0..aabbs.len()).filter(|&i| aabbs[i].min.iter().chain(aabbs[i].max.iter()).all(|v| v.is_finite())).collect();
        let mut nodes = Vec::new();
        let root = if idx.is_empty() { usize::MAX } else { build_rec(aabbs, &mut idx, &mut nodes) };
        Bvh { nodes, root, aabbs: aabbs.to_vec() }
    }

    fn bounds(&self, n: usize) -> Aabb {
        match &self.nodes[n] {
            Node::Leaf { prim } => self.aabbs[*prim],
            Node::Inner { bounds, .. } => *bounds,
        }
    }

    /// All candidate colliding pairs `(i, j)` with `i < j` — primitives whose AABBs overlap. Reported
    /// set is exactly the brute-force set.
    pub fn potential_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if self.root != usize::MAX {
            self.pairs_between(self.root, self.root, &mut out);
        }
        out.sort_unstable();
        out
    }

    fn pairs_within(&self, n: usize, out: &mut Vec<(usize, usize)>) {
        if let Node::Inner { left, right, .. } = &self.nodes[n] {
            self.pairs_within(*left, out);
            self.pairs_within(*right, out);
            self.pairs_between(*left, *right, out);
        }
    }

    fn pairs_between(&self, a: usize, b: usize, out: &mut Vec<(usize, usize)>) {
        if a == b {
            self.pairs_within(a, out);
            return;
        }
        if !self.bounds(a).overlaps(&self.bounds(b)) {
            return;
        }
        match (&self.nodes[a], &self.nodes[b]) {
            (Node::Leaf { prim: pa }, Node::Leaf { prim: pb }) => {
                if self.aabbs[*pa].overlaps(&self.aabbs[*pb]) {
                    let (i, j) = (*pa.min(pb), *pa.max(pb));
                    out.push((i, j));
                }
            }
            (Node::Inner { left, right, .. }, Node::Leaf { .. }) => {
                self.pairs_between(*left, b, out);
                self.pairs_between(*right, b, out);
            }
            (Node::Leaf { .. }, Node::Inner { left, right, .. }) => {
                self.pairs_between(a, *left, out);
                self.pairs_between(a, *right, out);
            }
            (Node::Inner { left: al, right: ar, .. }, Node::Inner { left: bl, right: br, .. }) => {
                let (al, ar, bl, br) = (*al, *ar, *bl, *br);
                self.pairs_between(al, bl, out);
                self.pairs_between(al, br, out);
                self.pairs_between(ar, bl, out);
                self.pairs_between(ar, br, out);
            }
        }
    }

    /// Primitive indices whose AABB overlaps `q`.
    pub fn query(&self, q: &Aabb) -> Vec<usize> {
        let mut out = Vec::new();
        if self.root != usize::MAX {
            self.query_rec(self.root, q, &mut out);
        }
        out.sort_unstable();
        out
    }

    fn query_rec(&self, n: usize, q: &Aabb, out: &mut Vec<usize>) {
        if !self.bounds(n).overlaps(q) {
            return;
        }
        match &self.nodes[n] {
            Node::Leaf { prim } => out.push(*prim),
            Node::Inner { left, right, .. } => {
                self.query_rec(*left, q, out);
                self.query_rec(*right, q, out);
            }
        }
    }

    /// Primitive indices whose AABB the ray `origin + t·dir` intersects.
    pub fn ray_query(&self, origin: Vector3<f64>, dir: Vector3<f64>) -> Vec<usize> {
        let mut out = Vec::new();
        if self.root != usize::MAX {
            self.ray_rec(self.root, origin, dir, &mut out);
        }
        out.sort_unstable();
        out
    }

    fn ray_rec(&self, n: usize, o: Vector3<f64>, d: Vector3<f64>, out: &mut Vec<usize>) {
        if !self.bounds(n).ray_hits(o, d) {
            return;
        }
        match &self.nodes[n] {
            Node::Leaf { prim } => out.push(*prim),
            Node::Inner { left, right, .. } => {
                self.ray_rec(*left, o, d, out);
                self.ray_rec(*right, o, d, out);
            }
        }
    }
}

fn build_rec(aabbs: &[Aabb], idx: &mut [usize], nodes: &mut Vec<Node>) -> usize {
    if idx.len() == 1 {
        nodes.push(Node::Leaf { prim: idx[0] });
        return nodes.len() - 1;
    }
    // total bounds + centroid bounds
    let mut bounds = aabbs[idx[0]];
    let mut cmin = aabbs[idx[0]].center();
    let mut cmax = cmin;
    for &i in idx.iter() {
        bounds = bounds.union(&aabbs[i]);
        let c = aabbs[i].center();
        cmin = cmin.inf(&c);
        cmax = cmax.sup(&c);
    }
    let ext = cmax - cmin;
    let axis = if ext.x >= ext.y && ext.x >= ext.z {
        0
    } else if ext.y >= ext.z {
        1
    } else {
        2
    };
    // Ordering only: `Bvh::build` has already excluded non-finite boxes, so every centre here is real.
    // A total order is used rather than an unwrapped comparison so this cannot panic if that ever slips.
    idx.sort_by(|&a, &b| aabbs[a].center()[axis].total_cmp(&aabbs[b].center()[axis]));
    let mid = idx.len() / 2;
    let (l, r) = idx.split_at_mut(mid);
    let left = build_rec(aabbs, l, nodes);
    let right = build_rec(aabbs, r, nodes);
    nodes.push(Node::Inner { bounds, left, right });
    nodes.len() - 1
}

#[cfg(test)]
mod verification {
    use super::*;

    fn boxes(n: usize, seed: u64) -> Vec<Aabb> {
        let mut s = seed;
        let mut rnd = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            ((z ^ (z >> 31)) as f64) / (u64::MAX as f64)
        };
        (0..n)
            .map(|_| {
                let c = Vector3::new(rnd() * 10.0, rnd() * 10.0, rnd() * 10.0);
                let h = Vector3::new(0.2 + rnd() * 0.8, 0.2 + rnd() * 0.8, 0.2 + rnd() * 0.8);
                Aabb::new(c - h, c + h)
            })
            .collect()
    }

    /// The BVH's candidate pair set is exactly the brute-force overlapping-pair set — no misses, no
    /// spurious pairs — while pruning most of the O(n²) tests.
    #[test]
    fn bvh_pairs_match_brute_force() {
        let ab = boxes(300, 0x99);
        let mut brute = Vec::new();
        for i in 0..ab.len() {
            for j in (i + 1)..ab.len() {
                if ab[i].overlaps(&ab[j]) {
                    brute.push((i, j));
                }
            }
        }
        brute.sort_unstable();
        let bvh = Bvh::build(&ab);
        let pairs = bvh.potential_pairs();
        eprintln!("BVH pairs {} == brute-force {} (of {} possible)", pairs.len(), brute.len(), ab.len() * (ab.len() - 1) / 2);
        assert_eq!(pairs, brute, "BVH pair set differs from brute force");
    }

    /// Box- and ray-queries return exactly the brute-force overlapping/intersecting sets.
    #[test]
    fn bvh_queries_match_brute_force() {
        let ab = boxes(200, 0x1234);
        let bvh = Bvh::build(&ab);
        let q = Aabb::new(Vector3::new(3.0, 3.0, 3.0), Vector3::new(6.0, 6.0, 6.0));
        let mut brute: Vec<usize> = (0..ab.len()).filter(|&i| ab[i].overlaps(&q)).collect();
        brute.sort_unstable();
        assert_eq!(bvh.query(&q), brute, "box query differs");

        let (o, d) = (Vector3::new(-1.0, 5.0, 5.0), Vector3::new(1.0, 0.0, 0.0));
        let mut rbrute: Vec<usize> = (0..ab.len()).filter(|&i| ab[i].ray_hits(o, d)).collect();
        rbrute.sort_unstable();
        assert_eq!(bvh.ray_query(o, d), rbrute, "ray query differs");
        eprintln!("BVH box query {} hits, ray query {} hits (match brute force)", brute.len(), rbrute.len());
    }
}
