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
//! The hull itself is computed here by [`crate::try_convex_hull_3d`]; a convex hull's vertex set and its
//! face set are unique up to coplanar-face triangulation, and the polygon merge removes exactly that
//! ambiguity, so the polygons MuJoCo ends up with are reproducible without qhull. Vertex coordinates are
//! rounded to `f32` as MuJoCo stores them.

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
    /// Hull triangles as mesh-vertex indices, outward wound.
    pub hull_faces: Vec<[usize; 3]>,
    /// Merged coplanar polygons of the hull.
    pub polygons: Vec<HullPolygon>,
    /// For each mesh vertex, the polygons it belongs to (`mesh_polymap`).
    pub polymap: Vec<Vec<usize>>,
}

const MINVAL: f64 = 1e-15;
/// `kAngleTol` in `mjCMesh::MakePolygons`.
const ANGLE_TOL: f64 = 0.01;

impl MeshHull {
    /// Build from a mesh in its own frame (after `scale`/`refpos`/`refquat`). `None` when the vertices have
    /// no 3-D hull (fewer than four, collinear, coplanar — MuJoCo refuses those meshes at compile time).
    pub fn new(mesh: &TriMesh3) -> Option<Self> {
        let verts: Vec<Vector3<f64>> = mesh.verts.iter().map(|v| v.map(|x| x as f32 as f64)).collect();
        let hull = crate::try_convex_hull_3d(&verts)?;
        // the hull mesh's vertices are a subset of `verts` (same coordinates); map each back to its index
        let mut index_of: HashMap<[u64; 3], usize> = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            index_of.entry([v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]).or_insert(i);
        }
        let back: Vec<usize> = hull.verts.iter().map(|v| index_of[&[v.x.to_bits(), v.y.to_bits(), v.z.to_bits()]]).collect();
        let hull_faces: Vec<[usize; 3]> = hull.tris.iter().map(|t| [back[t[0]], back[t[1]], back[t[2]]]).collect();
        let mut hull_verts: Vec<usize> = back.clone();
        hull_verts.sort_unstable();
        hull_verts.dedup();
        let polygons = make_polygons(&verts, &hull_faces);
        let mut polymap = vec![Vec::new(); verts.len()];
        for (pi, p) in polygons.iter().enumerate() {
            for &v in &p.verts {
                polymap[v].push(pi);
            }
        }
        Some(Self { verts, hull_verts, hull_faces, polygons, polymap })
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

/// `MeshPolygonKey`: the rounded spherical angles of a face normal, or `None` for a degenerate face.
fn polygon_key(v1: &Vector3<f64>, v2: &Vector3<f64>, v3: &Vector3<f64>) -> Option<(i64, i64)> {
    let n = (v2 - v1).cross(&(v3 - v1));
    let norm = n.norm();
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
struct MeshPolygon {
    edges: Vec<(usize, usize)>,
    islands: Vec<usize>,
    nisland: usize,
    normal: Vector3<f64>,
}

impl MeshPolygon {
    fn new(v1: usize, v2: usize, v3: usize, theta: f64, phi: f64) -> Self {
        Self { edges: vec![(v1, v2), (v2, v3), (v3, v1)], islands: vec![0, 0, 0], nisland: 1, normal: Vector3::new(theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()) }
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

/// `mjCMesh::MakePolygons` + `MakePolygonNormals`: merge hull triangles by rounded normal, trace paths, and
/// take each polygon's normal from its first three vertices.
fn make_polygons(verts: &[Vector3<f64>], faces: &[[usize; 3]]) -> Vec<HullPolygon> {
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
                polys.push(MeshPolygon::new(v1, v2, v3, ANGLE_TOL * key.0 as f64, ANGLE_TOL * key.1 as f64));
            }
        }
    }
    let mut out = Vec::new();
    for p in &polys {
        for path in p.paths() {
            if path.len() < 3 {
                continue;
            }
            let n = (verts[path[1]] - verts[path[0]]).cross(&(verts[path[2]] - verts[path[0]]));
            let l = n.norm();
            let normal = if l >= MINVAL { n / l } else { p.normal };
            out.push(HullPolygon { verts: path, normal });
        }
    }
    out
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
