//! **Contact parity with MuJoCo on all of MuJoCo Menagerie: the collision pipeline, geom pair by geom pair.**
//!
//! `scripts/menagerie_contacts_oracle.py` runs MuJoCo 3.13.0 on every compilable Menagerie model at K sampled
//! states and records, per sample, every geom's world pose and every `mjContact` MuJoCo produced (geoms,
//! `dist`, `pos`, `frame`, `dim`, `includemargin`, `friction`, `solref`, `solimp`, `exclude`);
//! `scripts/menagerie_contacts_flat.py` flattens that to text. This example reads the model through
//! [`ferromotion_core::tree_from_mjcf`] (for body parents and welds, contact excludes and pairs, and the
//! collision meshes' hull data), places every geom at MuJoCo's recorded pose, runs ferromotion's port of
//! MuJoCo's collision pipeline over every geom pair MuJoCo would consider — `contype`/`conaffinity`,
//! weld/parent filtering, `<exclude>`, `<pair>`, margins and gaps, bounding-sphere pre-check, the primitive
//! routines, the native GJK/EPA with multi-contact — and compares the contact SET with MuJoCo's.
//!
//! Poses come from the oracle rather than from our kinematics on purpose: kinematics are measured separately
//! (`menagerie_parity`), and this sweep isolates the collision pipeline.
//!
//! ```text
//! cargo run --release --example menagerie_contacts -- <menagerie root> <flat oracle dir> [substring filter]
//! ```
//!
//! A contact **matches** when the same unordered geom pair has a contact within 1e-6 m (nearest is taken),
//! and its `dist`, `pos`, normal (sign-corrected for pair order), `dim`, `includemargin`, `friction`,
//! `solref`, `solimp` and `exclude` agree; the worst residuals among matches are reported. A contact is
//! **near** when the same pair has a contact at the same depth (2e-5) and normal (1e-3) but the witness sits
//! elsewhere on the face — the same manifold sampled at different corners. Since the hulls are qhull's own
//! ([`ferromotion_core::qhull`]) and the polygons come in libc++'s order, that is rare. Height-field pairs
//! are not carried and are counted separately.

use ferromotion_core::{
    can_collide, collide_pair_with, contact_param, filter_body_pair, set_contact, tree_from_mjcf, CollideOptions, CollisionGeom, ContactRecord, GeomParams, GeomPose, GeomType, MeshHull, MjcfTree, PairParams,
};
use nalgebra::{Matrix3, Vector3};
use std::collections::{BTreeMap, HashMap, HashSet};

struct OGeom {
    name: String,
    ty: String,
    body: String,
    size: [f64; 3],
    params: GeomParams,
    rbound: f64,
    mesh: Option<String>,
    /// `geom_aabb`: centre and half-sizes in the geom frame (a plane's is the half-space below it)
    aabb: [f64; 6],
}

struct OContact {
    g1: usize,
    g2: usize,
    dist: f64,
    pos: Vector3<f64>,
    normal: Vector3<f64>,
    dim: usize,
    includemargin: f64,
    friction: [f64; 5],
    solref: [f64; 2],
    solimp: [f64; 5],
    exclude: bool,
}

struct OSample {
    poses: Vec<GeomPose>,
    contacts: Vec<OContact>,
}

struct Oracle {
    rel: String,
    filterparent: bool,
    nativeccd: bool,
    multiccd: bool,
    ccdtol: f64,
    ccditer: usize,
    override_margin: Option<f64>,
    geoms: Vec<OGeom>,
    /// MuJoCo's stored mesh frame per mesh name: `(mesh_pos, mesh_quat wxyz)`
    meshes: BTreeMap<String, (Vector3<f64>, [f64; 4])>,
    samples: Vec<OSample>,
}

fn parse_oracle(text: &str) -> Oracle {
    let mut o = Oracle { rel: String::new(), filterparent: true, nativeccd: true, multiccd: true, ccdtol: 1e-6, ccditer: 35, override_margin: None, geoms: Vec::new(), meshes: BTreeMap::new(), samples: Vec::new() };
    let f = |s: &str| s.parse::<f64>().unwrap_or_else(|_| panic!("bad number {s:?}"));
    let u = |s: &str| s.parse::<usize>().unwrap_or_else(|_| panic!("bad integer {s:?}"));
    for line in text.lines() {
        let t: Vec<&str> = line.split('\t').collect();
        match t[0] {
            "model" => {
                o.rel = t[1].to_string();
                o.filterparent = t[3] == "1";
                o.nativeccd = t[4] == "1";
                o.multiccd = t[5] == "1";
                o.ccdtol = f(t[6]);
                o.ccditer = u(t[7]);
                if t[10] == "1" {
                    o.override_margin = Some(f(t[11]));
                }
            }
            "geom" => {
                let params = GeomParams {
                    condim: u(t[10]),
                    priority: t[13].parse().unwrap(),
                    solmix: f(t[14]),
                    solref: [f(t[18]), f(t[19])],
                    solimp: [f(t[20]), f(t[21]), f(t[22]), f(t[23]), f(t[24])],
                    friction: [f(t[15]), f(t[16]), f(t[17])],
                    margin: f(t[11]),
                    gap: f(t[12]),
                    contype: t[8].parse::<i64>().unwrap() as u32,
                    conaffinity: t[9].parse::<i64>().unwrap() as u32,
                    adhesion: t.get(27).map(|x| f(x)).unwrap_or(0.0),
                };
                let aabb = if t.len() > 33 { [f(t[28]), f(t[29]), f(t[30]), f(t[31]), f(t[32]), f(t[33])] } else { [0.0; 6] };
                o.geoms.push(OGeom { name: t[2].to_string(), ty: t[3].to_string(), body: t[4].to_string(), size: [f(t[5]), f(t[6]), f(t[7])], params, rbound: f(t[25]), mesh: if t[26] == "-" { None } else { Some(t[26].to_string()) }, aabb });
            }
            "mesh" => {
                o.meshes.insert(t[1].to_string(), (Vector3::new(f(t[2]), f(t[3]), f(t[4])), [f(t[5]), f(t[6]), f(t[7]), f(t[8])]));
            }
            "sample" => o.samples.push(OSample { poses: Vec::new(), contacts: Vec::new() }),
            "gpose" => {
                let s = o.samples.last_mut().unwrap();
                let m: Vec<f64> = t[5..14].iter().map(|x| f(x)).collect();
                s.poses.push(GeomPose { pos: Vector3::new(f(t[2]), f(t[3]), f(t[4])), mat: Matrix3::from_row_slice(&m) });
            }
            "contact" => {
                let s = o.samples.last_mut().unwrap();
                s.contacts.push(OContact {
                    g1: u(t[1]),
                    g2: u(t[2]),
                    dist: f(t[3]),
                    pos: Vector3::new(f(t[4]), f(t[5]), f(t[6])),
                    normal: Vector3::new(f(t[7]), f(t[8]), f(t[9])),
                    dim: u(t[16]),
                    includemargin: f(t[17]),
                    friction: [f(t[18]), f(t[19]), f(t[20]), f(t[21]), f(t[22])],
                    solref: [f(t[23]), f(t[24])],
                    solimp: [f(t[25]), f(t[26]), f(t[27]), f(t[28]), f(t[29])],
                    exclude: t[30] != "0",
                });
            }
            _ => {}
        }
    }
    o
}

fn geom_type(s: &str) -> Option<GeomType> {
    Some(match s {
        "plane" => GeomType::Plane,
        "hfield" => GeomType::HField,
        "sphere" => GeomType::Sphere,
        "capsule" => GeomType::Capsule,
        "ellipsoid" => GeomType::Ellipsoid,
        "cylinder" => GeomType::Cylinder,
        "box" => GeomType::Box,
        "mesh" => GeomType::Mesh,
        _ => return None,
    })
}

/// `mj_collideOBB` for two geoms: separating-axis test of their oriented bounding boxes (`geom_aabb` in
/// each geom's frame) over the six face normals, inflated by `margin`; an axis marked infinite
/// (≥ `mjMAXVAL`) skips the box on the other side. This is the midphase's leaf test — the one place MuJoCo's
/// pairing is NOT conservative, because a plane's box is the half-space below it.
fn collide_obb(a1: &[f64; 6], p1: &GeomPose, a2: &[f64; 6], p2: &GeomPose, margin: f64) -> bool {
    const MAXVAL: f64 = 1e10;
    let inf = |a: &[f64; 6]| [a[3] >= MAXVAL, a[4] >= MAXVAL, a[5] >= MAXVAL];
    let (inf1, inf2) = (inf(a1), inf(a2));
    if (inf1[0] && inf1[1] && inf1[2]) || (inf2[0] && inf2[1] && inf2[2]) {
        return true;
    }
    let boxes = [(a1, p1), (a2, p2)];
    let infinite = [inf1.iter().any(|&b| b), inf2.iter().any(|&b| b)];
    let centers: Vec<Vector3<f64>> = boxes.iter().map(|(a, p)| p.mat * Vector3::new(a[0], a[1], a[2]) + p.pos).collect();
    let normal = |i: usize, j: usize| -> Vector3<f64> { boxes[i].1.mat.column(j).into() };
    for j in 0..2 {
        if infinite[1 - j] {
            continue;
        }
        for k in 0..3 {
            let n = normal(j, k);
            let mut proj = [0.0; 2];
            let mut radius = [0.0; 2];
            for i in 0..2 {
                proj[i] = centers[i].dot(&n);
                radius[i] = (boxes[i].0[3] * normal(i, 0).dot(&n)).abs() + (boxes[i].0[4] * normal(i, 1).dot(&n)).abs() + (boxes[i].0[5] * normal(i, 2).dot(&n)).abs();
            }
            if radius[0] + radius[1] + margin < (proj[1] - proj[0]).abs() {
                return false;
            }
        }
    }
    true
}

/// `body_weldid` as an id: 0 for the world and for bodies fixed to it, else 1 + the tree joint the body rides on.
fn weld_of(t: &MjcfTree, body: &str) -> usize {
    if body == "world" {
        return 0;
    }
    t.body_frames.get(body).map(|(j, _)| j + 1).unwrap_or(0)
}

/// The body that carries the weld: the nearest jointed ancestor-or-self, or the world.
fn weld_root<'a>(t: &'a MjcfTree, body: &'a str) -> &'a str {
    let mut b = body;
    loop {
        if b == "world" || t.tree.link_names.contains_key(b) {
            return b;
        }
        b = t.body_parent.get(b).map(|s| s.as_str()).unwrap_or("world");
    }
}

#[derive(Default)]
struct Tally {
    oracle: usize,
    matched: usize,
    /// same geom pair, same normal and depth within the CCD tolerance, witness elsewhere on the face
    near: usize,
    missing: usize,
    extra: usize,
    hfield: usize,
    /// per GEOM PAIR, not per contact: both sides found a manifold there
    pair_both: usize,
    pair_only_mujoco: usize,
    pair_only_ours: usize,
    /// over pairs both found: how many produced the SAME NUMBER of contact points, and the histogram of
    /// (MuJoCo's count, ours) for the ones that did not — a manifold of 4 answered with 1 is a different
    /// failure from a witness in the wrong place, and the totals cannot tell them apart
    pair_count_same: usize,
    counts: BTreeMap<(usize, usize), usize>,
    /// ⭐ pairs whose MANIFOLD agrees — same deepest penetration and same normal — at 1e-9 and at 1e-6.
    /// This is the quantity the solver actually integrates; which corner of a clipped polygon each witness
    /// landed on is a different question, and reporting only the contact-by-contact match conflates them.
    /// how far our nearest contact on the same pair was, for the ones that did not match
    miss_dist: BTreeMap<&'static str, usize>,
    pair_manifold_tight: usize,
    pair_manifold_loose: usize,
    /// worst disagreement, over pairs both found, in the deepest penetration and its normal
    pair_worst_depth: f64,
    pair_worst_normal: f64,
    worst_dist: f64,
    worst_pos: f64,
    worst_normal: f64,
    param_mismatch: usize,
}

/// One CCD pair for `scripts/mujoco_ccd_pairs.c`: the two objects in the order the CCD sees them (lower
/// geom type first, as `mj_collision` orders them), and our contacts with normals in that order.
#[allow(clippy::too_many_arguments)]
fn dump_ccd_pair(w: &mut impl std::io::Write, rel: &str, sample: usize, i: usize, j: usize, margin: f64, opts: &CollideOptions, ci: &CollisionGeom, cj: &CollisionGeom, cs: &[ferromotion_core::PreContact]) {
    use GeomType::*;
    let (a, b, swapped) = if ci.kind <= cj.kind { (ci, cj, false) } else { (cj, ci, true) };
    let analytic = matches!(
        (a.kind, b.kind),
        (Plane, _) | (HField, _) | (_, HField) | (Sphere, Sphere) | (Sphere, Capsule) | (Sphere, Cylinder) | (Sphere, Box) | (Capsule, Capsule) | (Capsule, Box) | (Box, Box)
    );
    if analytic {
        return;
    }
    let f = |x: f64| format!("{x:e}");
    let _ = writeln!(w, "pair\t{rel}\t{sample}\t{i}\t{j}\t{}\t{}\t{}\t{}", f(margin), f(opts.ccd_tolerance), opts.ccd_iterations, opts.multiccd as i32);
    for g in [a, b] {
        let m = &g.pose.mat;
        let mat: Vec<String> = (0..3).flat_map(|r| (0..3).map(move |c| (r, c))).map(|(r, c)| f(m[(r, c)])).collect();
        let _ = writeln!(w, "obj\t{}\t{} {} {}\t{} {} {}\t{}", g.kind as i32, f(g.size[0]), f(g.size[1]), f(g.size[2]), f(g.pose.pos.x), f(g.pose.pos.y), f(g.pose.pos.z), mat.join(" "));
        if let Some(h) = g.hull {
            let verts: Vec<String> = h.verts.iter().flat_map(|v| [f(v.x), f(v.y), f(v.z)]).collect();
            let (graph, extrema) = match &h.graph {
                Some(gr) => {
                    let nv = gr.vert_globalid.len();
                    let nf = h.hull_faces.len();
                    let mut ints: Vec<i64> = vec![nv as i64, nf as i64];
                    ints.extend(gr.vert_edgeadr.iter().map(|&x| x as i64));
                    ints.extend(gr.vert_globalid.iter().map(|&x| x as i64));
                    ints.extend(gr.edge_localid.iter().map(|&x| x as i64));
                    ints.extend(h.hull_faces.iter().flat_map(|t| t.iter().map(|&x| x as i64)));
                    (ints.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "), gr.extrema.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "))
                }
                None => ("-".to_string(), "-".to_string()),
            };
            let polys: Vec<String> = h.polygons.iter().map(|p| format!("{} {} {} {} {}", p.verts.len(), p.verts.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "), f(p.normal.x), f(p.normal.y), f(p.normal.z))).collect();
            let _ = writeln!(w, "mesh\t{}\t{}\t{graph}\t{extrema}\t{}\t{}", h.verts.len(), verts.join(" "), h.polygons.len(), polys.join(" "));
        }
    }
    let ours: Vec<String> = cs
        .iter()
        .map(|c| {
            let n = if swapped { -c.normal } else { c.normal };
            format!("{} {} {} {} {} {} {}", f(c.dist), f(c.pos.x), f(c.pos.y), f(c.pos.z), f(n.x), f(n.y), f(n.z))
        })
        .collect();
    let _ = writeln!(w, "ours\t{}\t{}", cs.len(), ours.join("\t"));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_contacts <menagerie root> <flat oracle dir> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let mut files: Vec<_> = std::fs::read_dir(&args[2]).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().map(|x| x == "txt").unwrap_or(false)).collect();
    files.sort();
    let mut grand = Tally::default();
    let mut by_type: BTreeMap<String, Tally> = BTreeMap::new();
    let mut refused: Vec<(String, String)> = Vec::new();
    let mut frames_differ = 0usize;
    let mut imperfect: Vec<(String, String)> = Vec::new();
    let mut dumped = false;
    let mut not_carried: BTreeMap<String, usize> = BTreeMap::new();
    let mut models = 0;
    // MENAGERIE_NOTES=<n> lists up to n discrepancies per model (default 3)
    let max_notes: usize = std::env::var("MENAGERIE_NOTES").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    // MENAGERIE_CCD_DUMP=<file>: every pair that goes through the native CCD, with its inputs and our
    // contacts, for `scripts/mujoco_ccd_pairs.c` to rerun through MuJoCo's own engine_collision_gjk.c
    let mut ccd_dump: Option<std::io::BufWriter<std::fs::File>> = std::env::var("MENAGERIE_CCD_DUMP").ok().map(|p| std::io::BufWriter::new(std::fs::File::create(p).unwrap()));
    for file in &files {
        let o = parse_oracle(&std::fs::read_to_string(file).unwrap());
        if filter.as_ref().is_some_and(|f| !o.rel.contains(f.as_str())) {
            continue;
        }
        let model = root.join(&o.rel);
        let dir = model.parent().unwrap().to_path_buf();
        let xml = std::fs::read_to_string(&model).unwrap();
        let resolve = |p: &str| std::fs::read(dir.join(p)).ok();
        let t = match tree_from_mjcf(&xml, &resolve) {
            Ok(t) => t,
            Err(e) => {
                refused.push((o.rel.clone(), e));
                continue;
            }
        };
        models += 1;
        // the loader's own hulls: its mesh frames (CoM and principal axes) are MuJoCo's bit for bit
        // (`examples/menagerie_mesh_store`), so a mesh geom's recorded `geom_xmat` places OUR stored vertices
        // exactly where it places MuJoCo's. A frame that is not bit-identical is counted, not papered over.
        let hulls: BTreeMap<String, MeshHull> = t.mesh_hulls.clone();
        for (name, (pos, q)) in &o.meshes {
            if let Some(d) = t.mesh_props.get(name) {
                let ours = [d.com.x, d.com.y, d.com.z, d.quat[0], d.quat[1], d.quat[2], d.quat[3]];
                let theirs = [pos.x, pos.y, pos.z, q[0], q[1], q[2], q[3]];
                if ours.iter().zip(&theirs).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    frames_differ += 1;
                }
            }
        }
        // welds
        let welds: Vec<(usize, usize, usize)> = o
            .geoms
            .iter()
            .map(|g| {
                let w = weld_of(&t, &g.body);
                let root_body = weld_root(&t, &g.body);
                let pw = if root_body == "world" { 0 } else { weld_of(&t, t.body_parent.get(root_body).map(|s| s.as_str()).unwrap_or("world")) };
                (w, pw, if w == 0 { 0 } else { 1 })
            })
            .collect();
        let excludes: HashSet<(String, String)> = t.contact_excludes.iter().flat_map(|(a, b)| [(a.clone(), b.clone()), (b.clone(), a.clone())]).collect();
        let pairs: HashMap<(String, String), usize> = t.contact_pairs.iter().enumerate().flat_map(|(i, p)| [((p.geom1.clone(), p.geom2.clone()), i), ((p.geom2.clone(), p.geom1.clone()), i)]).collect();
        let opts = CollideOptions { ccd_tolerance: o.ccdtol, ccd_iterations: o.ccditer, multiccd: o.multiccd };
        let mut tally = Tally::default();
        let mut model_notes: Vec<String> = Vec::new();
        for (si, s) in o.samples.iter().enumerate() {
            // every geom at MuJoCo's pose
            let geoms: Vec<Option<CollisionGeom>> = o
                .geoms
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    let kind = geom_type(&g.ty)?;
                    let hull = match (&g.mesh, kind) {
                        (Some(m), GeomType::Mesh) => Some(hulls.get(m)?),
                        _ => None,
                    };
                    Some(CollisionGeom { kind, pose: s.poses[i], size: g.size, hull })
                })
                .collect();
            let mut mine: Vec<(usize, usize, ContactRecord)> = Vec::new();
            let n = o.geoms.len();
            for i in 0..n {
                for j in i + 1..n {
                    let (gi, gj) = (&o.geoms[i], &o.geoms[j]);
                    let (Some(ci), Some(cj)) = (&geoms[i], &geoms[j]) else { continue };
                    let pair = pairs.get(&(gi.name.clone(), gj.name.clone())).map(|&k| &t.contact_pairs[k]);
                    let (margin, gap, params) = if let Some(p) = pair {
                        (p.margin, p.gap, PairParams { condim: p.condim, solref: p.solref, solimp: p.solimp, friction: p.friction, adhesion: p.adhesion })
                    } else {
                        if !can_collide(gi.params.contype, gi.params.conaffinity, gj.params.contype, gj.params.conaffinity) {
                            continue;
                        }
                        let (w1, w2) = (welds[i], welds[j]);
                        if filter_body_pair(w1.0, w1.1, w1.2, w2.0, w2.1, w2.2, o.filterparent) {
                            continue;
                        }
                        if excludes.contains(&(gi.body.clone(), gj.body.clone())) {
                            continue;
                        }
                        let (m, g) = ferromotion_core::margin_and_gap(&gi.params, &gj.params);
                        (m, g, contact_param(&gi.params, &gj.params))
                    };
                    let margin = o.override_margin.unwrap_or(margin);
                    // MuJoCo detects with margin + gap and records includemargin = margin
                    let detect = margin + gap;
                    if ci.kind == GeomType::HField || cj.kind == GeomType::HField {
                        continue;
                    }
                    // midphase leaf: oriented bounding boxes (a plane's is the half-space below it)
                    if !collide_obb(&gi.aabb, &ci.pose, &gj.aabb, &cj.pose, detect) {
                        continue;
                    }
                    // mj_collideGeoms: bounding-sphere pre-check (planes: the centre's height above the plane)
                    if ci.kind != GeomType::Plane && cj.kind != GeomType::Plane {
                        if (ci.pose.pos - cj.pose.pos).norm() > gi.rbound + gj.rbound + detect {
                            continue;
                        }
                    } else {
                        let (plane, other, rb) = if ci.kind == GeomType::Plane { (ci, cj, gj.rbound) } else { (cj, ci, gi.rbound) };
                        if rb > 0.0 && plane.pose.axis().dot(&(other.pose.pos - plane.pose.pos)) > detect + rb {
                            continue;
                        }
                    }
                    let dbg = std::env::var("MENAGERIE_PAIR").ok().map(|v| v == format!("{i},{j}")).unwrap_or(false);
                    match collide_pair_with(&opts, detect, ci, cj) {
                        Ok(cs) => {
                            if let Some(w) = ccd_dump.as_mut() {
                                dump_ccd_pair(w, &o.rel, si, i, j, detect, &opts, ci, cj, &cs);
                            }
                            if dbg {
                                eprintln!("pair {i},{j} ({}–{}): {} contacts {cs:?}", gi.name, gj.name, cs.len());
                            }
                            for c in cs {
                                mine.push((i, j, set_contact(&c, &params, margin)));
                            }
                        }
                        Err(e) => {
                            *not_carried.entry(e).or_default() += 1;
                        }
                    }
                }
            }
            // Per-pair agreement. A multi-contact manifold is sampled at points MuJoCo itself chooses with
            // a greedy over its polygon's vertex order, and that order is build-dependent — so the invariant
            // to compare is the manifold: does the same geom pair touch, how deep, and along which normal.
            {
                let key = |g1: usize, g2: usize| if g1 < g2 { (g1, g2) } else { (g2, g1) };
                let (mut theirs_n, mut ours_n): (BTreeMap<(usize, usize), usize>, BTreeMap<(usize, usize), usize>) = (BTreeMap::new(), BTreeMap::new());
                for c in &s.contacts {
                    *theirs_n.entry(key(oc_g1(c), oc_g2(c))).or_default() += 1;
                }
                for (i, j, _) in &mine {
                    *ours_n.entry((*i, *j)).or_default() += 1;
                }
                let mut theirs: BTreeMap<(usize, usize), (f64, Vector3<f64>)> = BTreeMap::new();
                for c in &s.contacts {
                    let k = key(oc_g1(c), oc_g2(c));
                    let n = if c.g1 == k.0 { c.normal } else { -c.normal };
                    let e = theirs.entry(k).or_insert((f64::INFINITY, n));
                    if c.dist < e.0 {
                        *e = (c.dist, n);
                    }
                }
                let mut ours: BTreeMap<(usize, usize), (f64, Vector3<f64>)> = BTreeMap::new();
                for (i, j, c) in &mine {
                    let e = ours.entry((*i, *j)).or_insert((f64::INFINITY, c.frame[0]));
                    if c.dist < e.0 {
                        *e = (c.dist, c.frame[0]);
                    }
                }
                for (k, (d, n)) in &theirs {
                    if o.geoms[k.0].ty == "hfield" || o.geoms[k.1].ty == "hfield" {
                        continue;
                    }
                    let bt = by_type.entry(pair_key(&o.geoms[k.0].ty, &o.geoms[k.1].ty)).or_default();
                    match ours.get(k) {
                        Some((d2, n2)) => {
                            let (dd, dn) = ((d2 - d).abs(), (n2 - n).norm());
                            let (tn, on) = (theirs_n.get(k).copied().unwrap_or(0), ours_n.get(k).copied().unwrap_or(0));
                            // DUMP_PAIR=<type> prints the first disagreeing pair of that type as a ready
                            // `ccd_one_pair` command line: the instrument that turned every earlier
                            // collision defect from a statistic into a reproduction takes explicit poses,
                            // and assembling them by hand from the oracle is where the mistakes were made
                            if std::env::var("DUMP_PAIR").is_ok_and(|w| w == pair_key(&o.geoms[k.0].ty, &o.geoms[k.1].ty)) && tn != on && !dumped {
                                dumped = true;
                                let g = |i: usize| {
                                    let gm = &o.geoms[i];
                                    let p = &s.poses[i];
                                    format!(
                                        "{} {:.17e} {:.17e} {:.17e} {}",
                                        gm.ty,
                                        gm.size[0],
                                        gm.size[1],
                                        gm.size[2],
                                        gm.mesh.clone().unwrap_or_else(|| "-".into())
                                    ) + &format!(
                                        " # pos {:.17e} {:.17e} {:.17e} mat {}",
                                        p.pos.x,
                                        p.pos.y,
                                        p.pos.z,
                                        p.mat.iter().map(|x| format!("{x:.17e}")).collect::<Vec<_>>().join(" ")
                                    )
                                };
                                let pose = |i: usize| {
                                    let p = &s.poses[i];
                                    format!("{:.17e} {:.17e} {:.17e} {}", p.pos.x, p.pos.y, p.pos.z, p.mat.transpose().iter().map(|x| format!("{x:.17e}")).collect::<Vec<_>>().join(" "))
                                };
                                let spec = |i: usize| {
                                    let gm = &o.geoms[i];
                                    format!("{} {:.17e} {:.17e} {:.17e} {}", gm.ty, gm.size[0], gm.size[1], gm.size[2], gm.mesh.clone().unwrap_or_else(|| "-".into()))
                                };
                                let _ = g;
                                println!(
                                    "DUMP {} {}–{}: MuJoCo {tn} contacts, ours {on}\n  cargo run --release --example ccd_one_pair -- <menagerie> {} {} {} {} {}",
                                    o.rel, o.geoms[k.0].name, o.geoms[k.1].name, o.rel, spec(k.0), spec(k.1), pose(k.0), pose(k.1)
                                );
                                for c in s.contacts.iter().filter(|c| key(oc_g1(c), oc_g2(c)) == *k) {
                                    println!("  MuJoCo: dist {} pos {:?} normal {:?}", c.dist, c.pos.as_slice(), c.normal.as_slice());
                                }
                            }
                            for t in [&mut tally, bt] {
                                t.pair_both += 1;
                                if tn == on {
                                    t.pair_count_same += 1;
                                } else {
                                    *t.counts.entry((tn, on)).or_default() += 1;
                                }
                                if dd < 1e-9 && dn < 1e-9 {
                                    t.pair_manifold_tight += 1;
                                }
                                if dd < 1e-6 && dn < 1e-6 {
                                    t.pair_manifold_loose += 1;
                                }
                                t.pair_worst_depth = t.pair_worst_depth.max(dd);
                                t.pair_worst_normal = t.pair_worst_normal.max(dn);
                            }
                            if (dn > 1e-3 || dd > 1e-5) && model_notes.len() < max_notes {
                                model_notes.push(format!(
                                    "PAIR {}–{} ({}): deepest MuJoCo {d} along {n:?}, ours {d2} along {n2:?}",
                                    o.geoms[k.0].name, o.geoms[k.1].name, pair_key(&o.geoms[k.0].ty, &o.geoms[k.1].ty)
                                ));
                            }
                        }
                        None => {
                            tally.pair_only_mujoco += 1;
                            bt.pair_only_mujoco += 1;
                        }
                    }
                }
                for k in ours.keys() {
                    if !theirs.contains_key(k) {
                        let bt = by_type.entry(pair_key(&o.geoms[k.0].ty, &o.geoms[k.1].ty)).or_default();
                        tally.pair_only_ours += 1;
                        bt.pair_only_ours += 1;
                    }
                }
            }

            // match
            let mut used = vec![false; mine.len()];
            for oc in &s.contacts {
                tally.oracle += 1;
                let key = pair_key(&o.geoms[oc.g1].ty, &o.geoms[oc.g2].ty);
                let bt = by_type.entry(key.clone()).or_default();
                bt.oracle += 1;
                if o.geoms[oc.g1].ty == "hfield" || o.geoms[oc.g2].ty == "hfield" {
                    tally.hfield += 1;
                    bt.hfield += 1;
                    continue;
                }
                let (a, b) = if oc.g1 < oc.g2 { (oc.g1, oc.g2) } else { (oc.g2, oc.g1) };
                let mut best: Option<(usize, f64)> = None;
                for (k, (i, j, c)) in mine.iter().enumerate() {
                    if used[k] || *i != a || *j != b {
                        continue;
                    }
                    let d = (c.pos - oc.pos).norm();
                    if best.map(|(_, bd)| d < bd).unwrap_or(true) {
                        best = Some((k, d));
                    }
                }
                let exact = matches!(best, Some((_, d)) if d < 1e-6);
                // second tier: the same face pair at the same depth and normal, the witness elsewhere on the
                // face — MuJoCo's own choice among a clipped polygon's corners (`hull4`) starts from qhull's
                // vertex order, and a curved mesh's near-coplanar hull vertices differ between hull builders
                let near = if exact {
                    None
                } else {
                    mine.iter()
                        .enumerate()
                        .filter(|(k, (i, j, c))| !used[*k] && *i == a && *j == b && (c.dist - oc.dist).abs() < 2e-5 && ((if oc.g1 == a { c.frame[0] } else { -c.frame[0] }) - oc.normal).norm() < 1e-3)
                        .map(|(k, (_, _, c))| (k, (c.pos - oc.pos).norm()))
                        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
                };
                match (exact, near) {
                    (true, _) | (false, Some(_)) => {
                        let (k, d) = if exact { best.unwrap() } else { near.unwrap() };
                        used[k] = true;
                        let c = &mine[k].2;
                        // my normal runs from geom i (lower index) to j; MuJoCo's from its g1 to g2
                        let n_mine = if oc.g1 == a { c.frame[0] } else { -c.frame[0] };
                        let dd = (c.dist - oc.dist).abs();
                        let dn = (n_mine - oc.normal).norm();
                        for t in [&mut tally, bt] {
                            if exact {
                                t.matched += 1;
                                t.worst_dist = t.worst_dist.max(dd);
                                t.worst_pos = t.worst_pos.max(d);
                                t.worst_normal = t.worst_normal.max(dn);
                            } else {
                                t.near += 1;
                            }
                        }
                        let params_ok = c.dim == oc.dim
                            && (c.includemargin - oc.includemargin).abs() < 1e-12
                            && c.friction.iter().zip(oc.friction).all(|(x, y)| (x - y).abs() < 1e-12)
                            && c.solref.iter().zip(oc.solref).all(|(x, y)| (x - y).abs() < 1e-12)
                            && c.solimp.iter().zip(oc.solimp).all(|(x, y)| (x - y).abs() < 1e-12)
                            && c.exclude == oc.exclude;
                        if !params_ok {
                            tally.param_mismatch += 1;
                            bt.param_mismatch += 1;
                            if model_notes.len() < max_notes {
                                model_notes.push(format!("params {}–{}: dim {} vs {}, includemargin {} vs {}, friction {:?} vs {:?}, solref {:?} vs {:?}, exclude {} vs {}", o.geoms[a].name, o.geoms[b].name, c.dim, oc.dim, c.includemargin, oc.includemargin, c.friction, oc.friction, c.solref, oc.solref, c.exclude, oc.exclude));
                            }
                        }
                        if !exact && model_notes.len() < max_notes {
                            model_notes.push(format!("near {}–{} ({}): witness {:.2e} away on the same face, dist {:.2e} normal {:.2e}", o.geoms[a].name, o.geoms[b].name, key, d, dd, dn));
                        } else if (dd > 1e-9 || dn > 1e-9 || d > 1e-9) && model_notes.len() < max_notes {
                            model_notes.push(format!("residual {}–{} ({}): dist {:.2e} pos {:.2e} normal {:.2e}", o.geoms[a].name, o.geoms[b].name, key, dd, d, dn));
                        }
                    }
                    _ => {
                        tally.missing += 1;
                        bt.missing += 1;
                        // ⭐ how far away OUR nearest contact of the same pair is. A miss at a tenth of a
                        // millimetre is a witness on the wrong corner of the right face; a miss at a
                        // centimetre is the wrong face. The totals cannot tell those apart, and they need
                        // completely different work.
                        let mag = best.map(|(_, d)| d).unwrap_or(f64::INFINITY);
                        let bucket = match mag {
                            d if d < 1e-5 => "<10um",
                            d if d < 1e-4 => "<0.1mm",
                            d if d < 1e-3 => "<1mm",
                            d if d < 1e-2 => "<1cm",
                            d if d.is_finite() => ">=1cm",
                            _ => "no contact of ours on that pair",
                        };
                        for t in [&mut tally, bt] {
                            *t.miss_dist.entry(bucket).or_default() += 1;
                        }
                        if model_notes.len() < max_notes {
                            model_notes.push(format!("missing {}–{} ({}): MuJoCo dist {} pos {:?}{}", o.geoms[a].name, o.geoms[b].name, key, oc.dist, oc.pos, best.map(|(k, d)| format!(", nearest of ours {d:.2e} away at dist {}", mine[k].2.dist)).unwrap_or_default()));
                        }
                    }
                }
            }
            for (k, (i, j, c)) in mine.iter().enumerate() {
                if !used[k] {
                    tally.extra += 1;
                    let key = pair_key(&o.geoms[*i].ty, &o.geoms[*j].ty);
                    by_type.entry(key.clone()).or_default().extra += 1;
                    if model_notes.len() < max_notes {
                        model_notes.push(format!("extra {}–{} ({}): dist {} pos {:?} exclude {}", o.geoms[*i].name, o.geoms[*j].name, key, c.dist, c.pos, c.exclude));
                    }
                }
            }
        }
        let perfect = tally.missing == 0 && tally.extra == 0 && tally.param_mismatch == 0 && tally.worst_dist < 1e-9 && tally.worst_pos < 1e-9 && tally.worst_normal < 1e-9;
        println!(
            "{:<60} oracle {:>5} matched {:>5} near {:>4} missing {:>4} extra {:>4} hfield {:>2} params✗ {:>3} | worst dist {:.1e} pos {:.1e} normal {:.1e}{}",
            o.rel,
            tally.oracle,
            tally.matched,
            tally.near,
            tally.missing,
            tally.extra,
            tally.hfield,
            tally.param_mismatch,
            tally.worst_dist,
            tally.worst_pos,
            tally.worst_normal,
            if perfect { "  ✓" } else { "" }
        );
        for n in &model_notes {
            println!("      {n}");
        }
        if !perfect {
            imperfect.push((o.rel.clone(), model_notes.first().cloned().unwrap_or_default()));
        }
        grand.oracle += tally.oracle;
        grand.matched += tally.matched;
        grand.pair_both += tally.pair_both;
        grand.pair_count_same += tally.pair_count_same;
        grand.pair_manifold_tight += tally.pair_manifold_tight;
        for (k, v) in &tally.miss_dist {
            *grand.miss_dist.entry(k).or_default() += v;
        }
        grand.pair_manifold_loose += tally.pair_manifold_loose;
        for (k, v) in &tally.counts {
            *grand.counts.entry(*k).or_default() += v;
        }
        grand.pair_only_mujoco += tally.pair_only_mujoco;
        grand.pair_only_ours += tally.pair_only_ours;
        grand.pair_worst_depth = grand.pair_worst_depth.max(tally.pair_worst_depth);
        grand.pair_worst_normal = grand.pair_worst_normal.max(tally.pair_worst_normal);
        grand.near += tally.near;
        grand.missing += tally.missing;
        grand.extra += tally.extra;
        grand.hfield += tally.hfield;
        grand.param_mismatch += tally.param_mismatch;
        grand.worst_dist = grand.worst_dist.max(tally.worst_dist);
        grand.worst_pos = grand.worst_pos.max(tally.worst_pos);
        grand.worst_normal = grand.worst_normal.max(tally.worst_normal);
        if !o.nativeccd {
            println!("      (model disables nativeccd — MuJoCo used libccd here)");
        }
    }
    println!();
    println!("models {models} (loader refused {}); mesh frames not bit-identical to MuJoCo's: {frames_differ}", refused.len());
    for (r, e) in &refused {
        println!("  refused {r}: {e}");
    }
    println!(
        "contacts: oracle {} matched {} near {} missing {} extra {} hfield(not carried) {} param-mismatch {} | worst dist {:.2e} pos {:.2e} normal {:.2e}",
        grand.oracle, grand.matched, grand.near, grand.missing, grand.extra, grand.hfield, grand.param_mismatch, grand.worst_dist, grand.worst_pos, grand.worst_normal
    );
    println!(
        "geom PAIRS that touch: both {} | only MuJoCo {} | only ours {} | worst depth {:.2e} m, worst normal {:.2e}",
        grand.pair_both, grand.pair_only_mujoco, grand.pair_only_ours, grand.pair_worst_depth, grand.pair_worst_normal
    );
    println!(
        "  of those, the SAME NUMBER of contact points: {} of {}; the same MANIFOLD (depth and normal) to 1e-9: {}, to 1e-6: {}",
        grand.pair_count_same, grand.pair_both, grand.pair_manifold_tight, grand.pair_manifold_loose
    );
    {
        // the ten commonest disagreements, so "missing 583" can be read as manifolds rather than points
        let mut h: Vec<_> = grand.counts.iter().collect();
        h.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
        let line: Vec<String> = h.iter().take(10).map(|((t, o), v)| format!("MJ {t}/ours {o}: {v}")).collect();
        println!("  contact-count disagreements: {}", line.join(", "));
    }
    {
        let order = ["<10um", "<0.1mm", "<1mm", "<1cm", ">=1cm", "no contact of ours on that pair"];
        let line: Vec<String> = order.iter().filter_map(|k| grand.miss_dist.get(k).map(|v| format!("{k}: {v}"))).collect();
        println!("  how far our nearest contact was, for the misses: {}", line.join(", "));
    }
    println!("by pair type:");
    for (k, t) in &by_type {
        println!(
            "  {:<22} contacts {:>5}/{:<5} near {:>4} | PAIRS both {:>5} onlyMJ {:>4} onlyOurs {:>4} depth {:.1e} normal {:.1e}",
            k, t.matched, t.oracle, t.near, t.pair_both, t.pair_only_mujoco, t.pair_only_ours, t.pair_worst_depth, t.pair_worst_normal
        );
        if !t.miss_dist.is_empty() {
            let order = ["<10um", "<0.1mm", "<1mm", "<1cm", ">=1cm", "no contact of ours on that pair"];
            let line: Vec<String> = order.iter().filter_map(|k| t.miss_dist.get(k).map(|v| format!("{k}: {v}"))).collect();
            println!("  {:<22} misses by distance: {}", "", line.join(", "));
        }
        if t.pair_manifold_loose < t.pair_both || t.pair_count_same < t.pair_both {
            println!("  {:<22} manifold agrees on {} of {} pairs at 1e-9, {} at 1e-6", "", t.pair_manifold_tight, t.pair_both, t.pair_manifold_loose);
        }
        if t.pair_count_same < t.pair_both {
            let mut h: Vec<_> = t.counts.iter().collect();
            h.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
            let line: Vec<String> = h.iter().take(6).map(|((a, b), v)| format!("MJ {a}/ours {b}: {v}")).collect();
            println!("  {:<22} same contact COUNT on {} of {} pairs; {}", "", t.pair_count_same, t.pair_both, line.join(", "));
        }
    }
    if !not_carried.is_empty() {
        println!("pairs not carried:");
        for (k, v) in &not_carried {
            println!("  {v:>6} × {k}");
        }
    }
    println!("models with any discrepancy: {}", imperfect.len());
    for (r, n) in &imperfect {
        println!("  {r}: {n}");
    }
}

fn oc_g1(c: &OContact) -> usize {
    c.g1
}

fn oc_g2(c: &OContact) -> usize {
    c.g2
}

fn pair_key(a: &str, b: &str) -> String {
    if a <= b {
        format!("{a}+{b}")
    } else {
        format!("{b}+{a}")
    }
}
