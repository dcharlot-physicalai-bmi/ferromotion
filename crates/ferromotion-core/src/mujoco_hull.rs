//! **A mesh as MuJoCo's collision pipeline sees it** — its convex hull with the adjacency and the merged
//! coplanar polygons that MuJoCo's native CCD walks (`mjCMesh::MakeGraph`, `MakePolygons`,
//! `MakePolygonNormals` in `user_mesh.cc`).
//!
//! MuJoCo hulls with qhull and keeps: the hull's vertices (as indices into the mesh's own vertex list), the
//! hull triangles, and per hull vertex its neighbours (the "graph" its hill-climbing support walks). It then
//! merges hull triangles whose normals round to the same `(θ, φ)` at a 0.01 rad resolution into polygons,
//! traces each polygon's boundary path, and stores the polygon normal (from its first three vertices) and a
//! vertex → polygons map. Those polygons are what `multicontact` clips against each other to produce up to
//! four contact points from a face–face collision, and what `mjc_PlaneConvex` reads to add a face's other
//! corners to the plane's first contact.
//!
//! ⭐ THREE frames, as `mjCMesh::Process` has them. The hull and the polygon merge run on the vertices AS
//! THE FILE STATES THEM (`f32`, before `refpos`, `refquat` and `scale`: `MakeGraph` and `MakePolygons` come
//! first), the polygon normals are recomputed from the FINAL `f64` vertices, CoM-centred in the principal
//! frame (`MakePolygonNormals`, from each polygon's first three vertices, never reoriented), and the
//! collider reads the final vertices rounded to `f32`. On a MIRRORED mesh (negative scale product) a
//! polygon wound outward in the file is wound inward once mirrored, and MuJoCo keeps it that way.
//!
//! ⭐ The hull is qhull's own ([`crate::qhull::mesh_graph`], a port verified integer for integer against
//! MuJoCo's `mesh_graph` on all 794 Menagerie collision meshes): the same vertices in the same order, each
//! vertex's neighbours in the same order, the same triangles in the same order and orientation — so the
//! polygons merge from the same faces in the same sequence, and the support function climbs the same graph
//! from the same seeds (`mesh_extrema`). A mesh the port refuses (a qhull path it does not carry) falls back
//! to [`crate::try_convex_hull_3d`]: the same polytope, not the same order.

use crate::TriMesh3;
use nalgebra::Vector3;
use std::collections::HashMap;

/// One merged coplanar polygon of the hull: its boundary vertex indices (into the mesh's vertices, in
/// order) and its normal.
#[derive(Clone, Debug)]
pub struct HullPolygon {
    pub verts: Vec<usize>,
    pub normal: Vector3<f64>,
}

/// The hull data MuJoCo's collision code reads from a mesh geom.
#[derive(Clone, Debug)]
pub struct MeshHull {
    /// Every mesh vertex, rounded to `f32` (MuJoCo stores mesh vertices as float).
    pub verts: Vec<Vector3<f64>>,
    /// Indices of the mesh vertices that lie on the hull.
    pub hull_verts: Vec<usize>,
    /// Hull triangles as mesh-vertex indices, wound outward in the FILE's frame — which is inward in the
    /// stored one on a mirrored mesh.
    pub hull_faces: Vec<[usize; 3]>,
    /// `<mesh maxhullvert>` if the model set one. A capped hull does NOT contain every mesh vertex — that is
    /// what the cap buys — so anything checking containment has to know.
    pub max_verts: Option<usize>,
    /// Merged coplanar polygons of the hull.
    pub polygons: Vec<HullPolygon>,
    /// For each mesh vertex, the polygons it belongs to (`mesh_polymap`).
    pub polymap: Vec<Vec<usize>>,
    /// qhull's hull graph as MuJoCo stores it (`mesh_graph`) and the 27 seeds its hill-climbing support
    /// starts from (`mesh_extrema`); `None` when the hull did not come from the qhull port.
    pub graph: Option<MeshGraph>,
}

/// `mesh_graph` split into its arrays, with `mesh_extrema`.
#[derive(Clone, Debug)]
pub struct MeshGraph {
    /// Per hull vertex, where its neighbour list starts in `edge_localid`.
    pub vert_edgeadr: Vec<usize>,
    /// Per hull vertex, its index in the mesh's vertex list.
    pub vert_globalid: Vec<usize>,
    /// Neighbour lists (hull-local indices), each ended by −1.
    pub edge_localid: Vec<i32>,
    /// For the 27 features of a unit cube (`cx, cy, cz ∈ {−1, 0, 1}`, x slowest), the hull vertex furthest
    /// along it, in `f32` as MuJoCo computes it from the stored vertices.
    pub extrema: [usize; 27],
}

/// `mjMESH_HILLCLIMB_MIN`: below this many mesh vertices MuJoCo scans them all instead of climbing.
const HILLCLIMB_MIN: usize = 10;

/// `dot3f` as clang contracts it: `a·b` with the first product fused.
fn dot3f(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.z.mul_add(b.z, a.x.mul_add(b.x, a.y * b.y))
}

const MINVAL: f64 = 1e-15;
/// `kAngleTol` in `mjCMesh::MakePolygons`.
const ANGLE_TOL: f64 = 0.01;

impl MeshHull {
    /// Build from a mesh in its own frame (after `scale`/`refpos`/`refquat`). `None` when the vertices have
    /// no 3-D hull (fewer than four, collinear, coplanar — MuJoCo refuses those meshes at compile time).
    pub fn new(mesh: &TriMesh3) -> Option<Self> {
        Self::with_max_verts(mesh, None)
    }

    /// The same, with MJCF's `<mesh maxhullvert>` — the hull qhull would build under `Q9 TA<n−4>`. A capped
    /// hull is an under-approximation of the mesh and MuJoCo collides against it anyway. The mesh is taken
    /// to be in all three frames at once; see [`MeshHull::from_frames`] for a mesh that is not.
    pub fn with_max_verts(mesh: &TriMesh3, max_verts: Option<usize>) -> Option<Self> {
        Self::from_frames(&mesh.verts, mesh, &mesh.verts, max_verts)
    }

    /// **As `mjCMesh::Process` builds it**, from the vertex list in its three frames (same indices in all
    /// three): `file`, as the file states it, where the hull is built and the polygons merged; `stored`, the
    /// processed mesh MuJoCo keeps (rounded to `f32` here), which the collider reads; and `final_f64`, the
    /// same vertices before that rounding, which the polygon normals are computed from.
    pub fn from_frames(file: &[Vector3<f64>], stored: &TriMesh3, final_f64: &[Vector3<f64>], max_verts: Option<usize>) -> Option<Self> {
        let key_verts: Vec<Vector3<f64>> = file.iter().map(|v| v.map(|x| x as f32 as f64)).collect();
        let verts: Vec<Vector3<f64>> = stored.verts.iter().map(|v| v.map(|x| x as f32 as f64)).collect();
        if key_verts.len() != verts.len() || final_f64.len() != verts.len() {
            return None;
        }
        let pts: Vec<[f64; 3]> = key_verts.iter().map(|v| [v.x, v.y, v.z]).collect();
        let (hull_faces, hull_verts, graph) = match crate::qhull::mesh_graph(&pts, max_verts) {
            Ok(g) => {
                let nv = g[0] as usize;
                let nf = g[1] as usize;
                let vert_edgeadr: Vec<usize> = g[2..2 + nv].iter().map(|&x| x as usize).collect();
                let vert_globalid: Vec<usize> = g[2 + nv..2 + 2 * nv].iter().map(|&x| x as usize).collect();
                let edge_localid: Vec<i32> = g[2 + 2 * nv..2 + 3 * nv + 3 * nf].to_vec();
                let faces: Vec<[usize; 3]> = g[2 + 3 * nv + 3 * nf..].chunks(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect();
                // `mesh_extrema`, from the stored float vertices in float arithmetic
                let mut max_val = [f32::MIN; 27];
                let mut extrema = [0usize; 27];
                for (local, &global) in vert_globalid.iter().enumerate() {
                    let (x, y, z) = (verts[global].x as f32, verts[global].y as f32, verts[global].z as f32);
                    let mut k = 0;
                    for cx in -1..=1 {
                        for cy in -1..=1 {
                            for cz in -1..=1 {
                                let dot = x * cx as f32 + y * cy as f32 + z * cz as f32;
                                if dot > max_val[k] {
                                    max_val[k] = dot;
                                    extrema[k] = local;
                                }
                                k += 1;
                            }
                        }
                    }
                }
                let hull_verts = vert_globalid.clone();
                (faces, hull_verts, Some(MeshGraph { vert_edgeadr, vert_globalid, edge_localid, extrema }))
            }
            Err(_) => {
                let hull = crate::try_convex_hull_3d_capped(&key_verts, max_verts)?;
                // the hull mesh's vertices are a subset of the file's (same coordinates); map each back to its index
                let mut index_of: HashMap<[u64; 3], usize> = HashMap::new();
                for (i, v) in key_verts.iter().enumerate() {
                    index_of.entry([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]).or_insert(i);
                }
                let back: Vec<usize> = hull.verts.iter().map(|v| index_of[&[v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]]).collect();
                let hull_faces: Vec<[usize; 3]> = hull.tris.iter().map(|t| [back[t[0]], back[t[1]], back[t[2]]]).collect();
                let mut hull_verts: Vec<usize> = back.clone();
                hull_verts.sort_unstable();
                hull_verts.dedup();
                (hull_faces, hull_verts, None)
            }
        };
        let polygons = make_polygons(&key_verts, &hull_faces, final_f64);
        let mut polymap = vec![Vec::new(); verts.len()];
        for (pi, p) in polygons.iter().enumerate() {
            for &v in &p.verts {
                polymap[v].push(pi);
            }
        }
        Some(Self { verts, hull_verts, hull_faces, polygons, polymap, max_verts, graph })
    }

    /// `mjc_meshSupport`: the vertex most along `local_dir` (mesh frame), scanning the hull's vertices from
    /// `-FLT_MAX`, with MuJoCo's tie rule: a cached previous answer (`start`) is compared first and kept on
    /// an exact tie, otherwise the lowest index wins. MuJoCo hill-climbs its vertex graph instead once a mesh
    /// has ten or more vertices; that reaches the same support VALUE and can differ only in which of several
    /// tied vertices it names. Returns `(index, vertex)`.
    pub fn support(&self, local_dir: &Vector3<f64>, start: Option<usize>) -> (usize, Vector3<f64>) {
        let mut best = -(f32::MAX as f64);
        let mut imax = 0;
        if let Some(i) = start.filter(|&i| i < self.verts.len()) {
            imax = i;
            best = local_dir.dot(&self.verts[i]);
        }
        // only hull vertices can be extreme; scanning them (not every mesh vertex) reaches the same value —
        // Menagerie collision meshes run to 23k vertices, of which a few hundred are on the hull
        for &i in &self.hull_verts {
            let d = local_dir.dot(&self.verts[i]);
            if d > best {
                best = d;
                imax = i;
            }
        }
        (imax, self.verts[imax])
    }

    /// **MuJoCo's mesh support, exactly**: `mjc_hillclimbSupport` on the hull graph for a mesh of at least
    /// ten vertices — seeded at `mesh_extrema` for the direction's cube feature, or at the cached `meshindex`
    /// when that is at least as far along — and `mjc_meshSupport`'s scan of every mesh vertex otherwise (from
    /// the cached `vertindex`, first maximum wins). Updates both caches as MuJoCo does; returns the vertex in
    /// the mesh frame.
    pub fn support_mujoco(&self, local_dir: &Vector3<f64>, vertindex: &mut i64, meshindex: &mut i64) -> Vector3<f64> {
        match &self.graph {
            Some(g) if self.verts.len() >= HILLCLIMB_MIN => {
                let c = |x: f64| (x > 0.4) as usize + 1 - (x < -0.4) as usize;
                let grid_idx = g.extrema[c(local_dir.x) * 9 + c(local_dir.y) * 3 + c(local_dir.z)];
                let at = |local: usize| dot3f(local_dir, &self.verts[g.vert_globalid[local]]);
                let mut imax = if *meshindex >= 0 {
                    let cached = at(*meshindex as usize);
                    let seed = at(grid_idx);
                    if seed > cached {
                        grid_idx
                    } else {
                        *meshindex as usize
                    }
                } else {
                    grid_idx
                };
                let mut max = at(imax);
                let mut prev = usize::MAX;
                while imax != prev {
                    prev = imax;
                    let mut i = g.vert_edgeadr[imax];
                    while g.edge_localid[i] >= 0 {
                        let sub = g.edge_localid[i] as usize;
                        let vdot = at(sub);
                        if vdot > max {
                            max = vdot;
                            imax = sub;
                        }
                        i += 1;
                    }
                }
                *meshindex = imax as i64;
                *vertindex = g.vert_globalid[imax] as i64;
                self.verts[g.vert_globalid[imax]]
            }
            _ => {
                let mut max = -(f32::MAX as f64);
                let mut imax = 0usize;
                if *vertindex >= 0 {
                    imax = *vertindex as usize;
                    max = dot3f(local_dir, &self.verts[imax]);
                }
                for (i, v) in self.verts.iter().enumerate() {
                    let vdot = dot3f(local_dir, v);
                    if vdot > max {
                        max = vdot;
                        imax = i;
                    }
                }
                *vertindex = imax as i64;
                self.verts[imax]
            }
        }
    }

    /// `mjCGeom::GetRBound` for a mesh geom: the norm of the half-extents of the vertex bounds, which are
    /// symmetric about the origin only by accident — MuJoCo takes the larger magnitude per axis.
    pub fn rbound(&self) -> f64 {
        let mut h = [0.0f64; 3];
        for v in &self.verts {
            for k in 0..3 {
                h[k] = h[k].max(v[k].abs());
            }
        }
        (h[0] * h[0] + h[1] * h[1] + h[2] * h[2]).sqrt()
    }
}

/// `mjuu_crossvec` as clang contracts it: each component's first product fused.
fn crossvec(b: &Vector3<f64>, c: &Vector3<f64>) -> Vector3<f64> {
    Vector3::new(b.y.mul_add(c.z, -(b.z * c.y)), b.z.mul_add(c.x, -(b.x * c.z)), b.x.mul_add(c.y, -(b.y * c.x)))
}

/// `mjuu_dot3` as clang contracts it.
fn dot3(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.z.mul_add(b.z, a.x.mul_add(b.x, a.y * b.y))
}

/// `MeshPolygonKey`: the rounded spherical angles of a face normal, or `None` for a degenerate face.
fn polygon_key(v1: &Vector3<f64>, v2: &Vector3<f64>, v3: &Vector3<f64>) -> Option<(i64, i64)> {
    let n = crossvec(&(v2 - v1), &(v3 - v1));
    let norm = dot3(&n, &n).sqrt();
    if norm < MINVAL {
        return None;
    }
    // `+ 0.0` turns a negative zero positive: atan2 is sensitive to the sign of zero (MuJoCo does the same)
    let n = Vector3::new(n.x / norm + 0.0, n.y / norm + 0.0, n.z / norm + 0.0);
    if n.z.abs() > 1.0 - 1e-7 {
        let rphi = if n.z < 0.0 { (std::f64::consts::PI / ANGLE_TOL).round() } else { 0.0 };
        return Some((0, rphi as i64));
    }
    let rtheta = (n.y.atan2(n.x) / ANGLE_TOL).round();
    let rphi = (n.z.acos() / ANGLE_TOL).round();
    Some((rtheta as i64, rphi as i64))
}

/// `MeshPolygon`: directed boundary edges of a growing coplanar polygon, with island bookkeeping.
/// (The rounded-angle normal MuJoCo's constructor also sets is dropped: `MakePolygonNormals` overwrites it.)
struct MeshPolygon {
    edges: Vec<(usize, usize)>,
    islands: Vec<usize>,
    nisland: usize,
}

impl MeshPolygon {
    fn new(v1: usize, v2: usize, v3: usize) -> Self {
        Self { edges: vec![(v1, v2), (v2, v3), (v3, v1)], islands: vec![0, 0, 0], nisland: 1 }
    }

    fn combine_islands(&mut self, a: &mut usize, b: &mut usize) {
        if *b < *a {
            std::mem::swap(a, b);
        }
        for k in self.islands.iter_mut() {
            if *k == *b {
                *k = *a;
            } else if *k > *b {
                *k -= 1;
            }
        }
    }

    fn insert_face(&mut self, v1: usize, v2: usize, v3: usize) {
        let (mut add1, mut add2, mut add3) = (true, true, true);
        let mut island: Option<usize> = None;
        if let Some(i) = self.edges.iter().position(|e| *e == (v2, v1)) {
            add1 = false;
            island = Some(self.islands[i]);
            self.edges.remove(i);
            self.islands.remove(i);
        }
        if let Some(i) = self.edges.iter().position(|e| *e == (v3, v2)) {
            let mut island2 = self.islands[i];
            match island {
                None => island = Some(island2),
                Some(mut isl) if isl != island2 => {
                    self.nisland -= 1;
                    self.combine_islands(&mut isl, &mut island2);
                    island = Some(isl);
                }
                _ => {}
            }
            add2 = false;
            self.edges.remove(i);
            self.islands.remove(i);
        }
        if let Some(i) = self.edges.iter().position(|e| *e == (v1, v3)) {
            let mut island3 = self.islands[i];
            match island {
                None => island = Some(island3),
                Some(mut isl) if isl != island3 => {
                    self.nisland -= 1;
                    self.combine_islands(&mut isl, &mut island3);
                    island = Some(isl);
                }
                _ => {}
            }
            add3 = false;
            self.edges.remove(i);
            self.islands.remove(i);
        }
        let island = match island {
            Some(i) => i,
            None => {
                self.nisland += 1;
                self.nisland - 1
            }
        };
        if add1 {
            self.edges.push((v1, v2));
            self.islands.push(island);
        }
        if add2 {
            self.edges.push((v2, v3));
            self.islands.push(island);
        }
        if add3 {
            self.edges.push((v3, v1));
            self.islands.push(island);
        }
    }

    fn paths(&self) -> Vec<Vec<usize>> {
        if self.edges.len() == 3 {
            return vec![vec![self.edges[0].0, self.edges[1].0, self.edges[2].0]];
        }
        let mut paths = Vec::new();
        for i in 0..self.nisland {
            let mut path: Vec<usize> = Vec::new();
            for (j, e) in self.edges.iter().enumerate() {
                if self.islands[j] == i {
                    path.push(e.0);
                    path.push(e.1);
                    break;
                }
            }
            if path.is_empty() {
                continue;
            }
            let mut next = *path.last().unwrap();
            for _ in 0..self.edges.len() {
                let mut finished = false;
                for k in 1..self.edges.len() {
                    if self.islands[k] == i && self.edges[k].0 == next {
                        next = self.edges[k].1;
                        if next == path[0] {
                            paths.push(path.clone());
                            finished = true;
                            break;
                        }
                        path.push(next);
                        break;
                    }
                }
                if finished {
                    break;
                }
            }
        }
        paths
    }
}

/// `mjCMesh::MakePolygons` + `MakePolygonNormals`: merge hull triangles by rounded normal and trace paths
/// in the file's frame (`verts`), then take each polygon's normal from its first three vertices in the
/// final one (`normal_verts`).
fn make_polygons(verts: &[Vector3<f64>], faces: &[[usize; 3]], normal_verts: &[Vector3<f64>]) -> Vec<HullPolygon> {
    // insertion order matters for the output order; keep it as a Vec keyed by the rounded angles
    let mut keys: Vec<(i64, i64)> = Vec::new();
    let mut polys: Vec<MeshPolygon> = Vec::new();
    for f in faces {
        let (v1, v2, v3) = (f[0], f[1], f[2]);
        let Some(key) = polygon_key(&verts[v1], &verts[v2], &verts[v3]) else { continue };
        match keys.iter().position(|k| *k == key) {
            Some(i) => polys[i].insert_face(v1, v2, v3),
            None => {
                keys.push(key);
                polys.push(MeshPolygon::new(v1, v2, v3));
            }
        }
    }
    let mut out = Vec::new();
    for p in &polys {
        for path in p.paths() {
            if path.len() < 3 {
                continue;
            }
            out.push(HullPolygon { normal: make_normal(&normal_verts[path[0]], &normal_verts[path[1]], &normal_verts[path[2]]), verts: path });
        }
    }
    out
}

/// `mjuu_makenormal`: the unit normal of `(b − a) × (c − a)`, or `(1, 0, 0)` below `mjEPS` = 1e-14. The
/// winding is whatever the three vertices carry; nothing here reorients it.
fn make_normal(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Vector3<f64> {
    let n = crossvec(&(b - a), &(c - a));
    let nrm = dot3(&n, &n).sqrt();
    if nrm < 1e-14 {
        return Vector3::x();
    }
    n / nrm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_mesh(hx: f64, hy: f64, hz: f64) -> TriMesh3 {
        let mut verts = Vec::new();
        for x in [-hx, hx] {
            for y in [-hy, hy] {
                for z in [-hz, hz] {
                    verts.push(Vector3::new(x, y, z));
                }
            }
        }
        let tris = vec![[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]];
        TriMesh3 { verts, tris }
    }

    #[test]
    fn a_box_mesh_merges_into_six_quads_with_outward_normals() {
        let h = MeshHull::new(&box_mesh(0.1, 0.15, 0.075)).unwrap();
        assert_eq!(h.hull_verts.len(), 8);
        assert_eq!(h.hull_faces.len(), 12);
        assert_eq!(h.polygons.len(), 6, "twelve triangles merge into six faces");
        for p in &h.polygons {
            assert_eq!(p.verts.len(), 4);
            // outward: the normal points away from the centre through the face's first vertex
            assert!(p.normal.dot(&h.verts[p.verts[0]]) > 0.0);
            // an axis normal
            assert!((p.normal.abs().max() - 1.0).abs() < 1e-12);
        }
        assert!(h.polymap.iter().all(|m| m.len() == 3), "every corner sits on three faces");
        // the support along +z is a top vertex, at MuJoCo's f32-rounded height
        let (_, v) = h.support(&Vector3::z(), None);
        assert_eq!(v.z, 0.075f32 as f64);
    }

    #[test]
    fn an_octahedron_keeps_its_eight_triangles() {
        let r = 0.12;
        let verts = vec![Vector3::new(r, 0.0, 0.0), Vector3::new(-r, 0.0, 0.0), Vector3::new(0.0, r, 0.0), Vector3::new(0.0, -r, 0.0), Vector3::new(0.0, 0.0, r), Vector3::new(0.0, 0.0, -r)];
        let tris = vec![[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]];
        let h = MeshHull::new(&TriMesh3 { verts, tris }).unwrap();
        assert_eq!(h.polygons.len(), 8);
        assert!(h.polygons.iter().all(|p| p.verts.len() == 3));
        assert!(h.polymap.iter().all(|m| m.len() == 4));
    }

    #[test]
    fn a_flat_mesh_is_refused() {
        let verts = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0), Vector3::new(1.0, 1.0, 0.0)];
        assert!(MeshHull::new(&TriMesh3 { verts, tris: vec![[0, 1, 2], [1, 3, 2]] }).is_none());
    }
}
