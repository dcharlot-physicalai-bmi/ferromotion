//! **qhull, the part MuJoCo runs — ported so a mesh's convex hull is MuJoCo's, facet for facet.**
//!
//! `mjCMesh::MakeGraph` (MuJoCo 3.13.0) hands a mesh's vertices to qhull (`d1c2fc0`, 2020.2) with the
//! options `qhull Qt` — and `Q9 TA<maxhullvert − 4>` under `<mesh maxhullvert>` — then `qh_triangulate` and
//! `qh_vertexneighbors`, and stores the result as `mesh_graph`: the hull's vertices in qhull's vertex-list
//! order, each vertex's neighbours in the order its facet set holds them, and the triangles in qhull's
//! facet-list order. The contact code walks that graph — the support function climbs it, and
//! `MakePolygons` merges faces in its order — so a hull with the same vertices in a different order is a
//! different collider. [`mesh_graph`] reproduces it: the same incremental build (initial simplex from the
//! extreme points, furthest-point partitioning, the pre-merge of non-convex and coplanar facets that the
//! default `C-0` asks for, vertex renaming, and triangulation), with every set operation's ordering kept —
//! `qh_setdel` moves the last element into the hole; `qh_setdelsorted` shifts; `FOREACH` stops at the first
//! null — because the output order IS those operations' history.
//!
//! ⛔ **Two things outside qhull's source decide the answer, and both are reproduced.**
//! * **Fused multiply–add.** The Apple-silicon MuJoCo wheel's qhull is compiled with floating-point
//!   contraction, so `a*b + c` is one rounding wherever clang fused it. Built without contraction, qhull
//!   returns a different graph for 120 of 794 Menagerie meshes. Every fused site here was read off the
//!   compiled assembly (`qh_distplane`, `qh_sethyperplane_det`, `qh_normalize2`, `qh_projectpoint`,
//!   `qh_getangle`, `qh_determinant`, `qh_distround`, `qh_detroundoff`, `qh_pointdist`,
//!   `qh_findbesthorizon`, `qh_gausselim`, `qh_backnormal`, `qh_sethyperplane_gauss`), and is written as
//!   [`f64::mul_add`] — which is the same single rounding on every platform.
//! * **`qsort` on ties.** qhull sorts pending merges and candidate vertices with comparators that never
//!   return 0 (`qh_compare_facetmerge`, `qh_comparevisit`), so which of two equal keys comes first is
//!   decided by the C library's sort. `apple_qsort` here is macOS libc's: Bentley–McIlroy with insertion sort
//!   below 8 elements, a swap-budgeted insertion sort when a partition moved nothing, and a heapsort once
//!   the depth passes `2·(fls(n) − 1)` — matched against the system `qsort` on 200,000 tie-heavy arrays.
//!
//! What qhull does with a pointer's VALUE (`qh_gethash` hashes addresses) cannot change the answer: the
//! hash only places a facet in an open-addressed table whose matches are decided by vertex sets. Paths no
//! Menagerie mesh reaches — duplicate ridges, multi-facet coplanar-horizon cycles, twisted and pinched
//! merges — return an error rather than a guess, so a caller can fall back to another hull.

const NULL: usize = usize::MAX;
/// `qh_MERGEridge` — `(facetT *)2` in a neighbour set.
const MERGERIDGE: usize = usize::MAX - 2;
const REALMAX: f64 = f64::MAX;
const REALMIN: f64 = f64::MIN_POSITIVE;
const REALEPS: f64 = f64::EPSILON;
const DIM: usize = 3;

// merge types (`mergeType`)
const MRG_COPLANAR: u8 = 1;
const MRG_ANGLECOPLANAR: u8 = 2;
const MRG_CONCAVE: u8 = 3;
const MRG_CONCAVECOPLANAR: u8 = 4;
const MRG_TWISTED: u8 = 5;
const MRG_FLIP: u8 = 6;
const MRG_DUPRIDGE: u8 = 7;
const MRG_DEGEN: u8 = 10;
const MRG_REDUNDANT: u8 = 11;
const MRG_MIRROR: u8 = 12;
const MRG_COPLANARHORIZON: u8 = 13;

// user_r.h
const QH_RATIOCOPLANAROUTSIDE: f64 = 30.0;
const QH_RATIOMAXSIMPLEX: f64 = 1.0e-3;
const QH_RATIONEARINSIDE: f64 = 5.0;
const QH_WIDECOPLANAR: f64 = 6.0;
const QH_MAXNARROW: f64 = -0.99999999;
const QH_BESTCENTRUM: usize = 20;
const QH_BESTCENTRUM2: usize = 2;
const QH_BESTNONCONVEX: usize = 15;
const QH_MAXNEWCENTRUM: usize = 5;
const QH_MAXNUMMERGE: u32 = 511;

/// **MuJoCo's `mesh_graph` for a set of points**: `[numvert, numface, vert_edgeadr[numvert],
/// vert_globalid[numvert], edge_localid[numvert + 3·numface], face_globalid[3·numface]]`, exactly as
/// `mjCMesh::MakeGraph` writes it. `max_verts` is `<mesh maxhullvert>`.
pub fn mesh_graph(points: &[[f64; 3]], max_verts: Option<usize>) -> Result<Vec<i32>, String> {
    if points.len() < 4 {
        return Err("fewer than four points".into());
    }
    let mut qh = Qh::new(points.to_vec(), max_verts);
    qh.qhull()?;
    qh.triangulate()?;
    qh.vertexneighbors();
    Ok(qh.graph())
}

#[derive(Clone, Default)]
struct Facet {
    furthestdist: f64,
    maxoutside: f64,
    offset: f64,
    normal: Option<[f64; 3]>,
    center: Option<[f64; 3]>,
    /// the union `f`: replace / samecycle / newcycle / trivisible / triowner
    f: usize,
    previous: usize,
    next: usize,
    vertices: Vec<usize>,
    ridges: Option<Vec<usize>>,
    neighbors: Vec<usize>,
    outsideset: Option<Vec<usize>>,
    coplanarset: Option<Vec<usize>>,
    visitid: u32,
    id: u32,
    nummerge: u32,
    tricoplanar: bool,
    newfacet: bool,
    visible: bool,
    toporient: bool,
    simplicial: bool,
    seen: bool,
    flipped: bool,
    notfurthest: bool,
    good: bool,
    dupridge: bool,
    mergeridge: bool,
    mergeridge2: bool,
    coplanarhorizon: bool,
    mergehorizon: bool,
    tested: bool,
    keepcentrum: bool,
    newmerge: bool,
    degenerate: bool,
    redundant: bool,
}

#[derive(Clone, Default)]
struct Vertex {
    next: usize,
    previous: usize,
    point: usize,
    neighbors: Vec<usize>,
    id: u32,
    visitid: u32,
    seen: bool,
    deleted: bool,
    delridge: bool,
    newfacet: bool,
    partitioned: bool,
}

#[derive(Clone, Default)]
struct Ridge {
    vertices: Vec<usize>,
    top: usize,
    bottom: usize,
    tested: bool,
    nonconvex: bool,
    simplicialtop: bool,
    simplicialbot: bool,
}

#[derive(Clone, Copy)]
struct Merge {
    angle: f64,
    distance: f64,
    facet1: usize,
    facet2: usize,
    mergetype: u8,
}

// ---------------------------------------------------------------------------------------------------
// qset: sets are Vecs; `NULL` entries are kept where qhull keeps them, and FOREACH stops at the first.

fn at(s: &[usize], i: usize) -> usize {
    s.get(i).copied().unwrap_or(NULL)
}

/// `qh_setappend` (a null element is not appended).
fn set_append(s: &mut Vec<usize>, e: usize) {
    if e != NULL {
        s.push(e);
    }
}

/// `qh_setappend` onto a set that may not exist yet.
fn set_append_opt(s: &mut Option<Vec<usize>>, e: usize) {
    if e != NULL {
        s.get_or_insert_with(Vec::new).push(e);
    }
}

/// `qh_setdel`: finds `e` (scanning up to the first null), moves the LAST element into its place.
fn set_del(s: &mut Vec<usize>, e: usize) -> bool {
    for i in 0..s.len() {
        if s[i] == NULL {
            return false;
        }
        if s[i] == e {
            s.swap_remove(i);
            return true;
        }
    }
    false
}

/// `qh_setdelsorted`: removes `e`, keeping the order.
fn set_delsorted(s: &mut Vec<usize>, e: usize) -> bool {
    for i in 0..s.len() {
        if s[i] == NULL {
            return false;
        }
        if s[i] == e {
            s.remove(i);
            return true;
        }
    }
    false
}

/// `qh_setdellast`: pops — or returns null when the set's FIRST element is null.
fn set_dellast(s: &mut Vec<usize>) -> usize {
    if at(s, 0) == NULL {
        return NULL;
    }
    s.pop().unwrap_or(NULL)
}

/// `qh_setreplace`.
fn set_replace(s: &mut [usize], old: usize, new: usize) -> Result<(), String> {
    for x in s.iter_mut() {
        if *x == NULL {
            break;
        }
        if *x == old {
            *x = new;
            return Ok(());
        }
    }
    Err("qhull internal error (qh_setreplace): element not found".into())
}

/// `qh_setin`.
fn set_in(s: &[usize], e: usize) -> bool {
    for &x in s {
        if x == NULL {
            return false;
        }
        if x == e {
            return true;
        }
    }
    false
}

/// `qh_setindex` (scans the whole set, nulls included).
fn set_index(s: &[usize], e: usize) -> Option<usize> {
    s.iter().position(|&x| x == e)
}

/// `qh_setunique`.
fn set_unique(s: &mut Vec<usize>, e: usize) -> bool {
    if !set_in(s, e) {
        set_append(s, e);
        true
    } else {
        false
    }
}

/// `qh_setnew_delnthsorted`: `prepend` nulls, then `s[..size]` without its `nth`.
fn set_new_delnthsorted(s: &[usize], size: usize, nth: usize, prepend: usize) -> Vec<usize> {
    let mut v = vec![NULL; prepend];
    v.extend_from_slice(&s[..nth]);
    v.extend_from_slice(&s[nth + 1..size]);
    v
}

/// `qh_setequal_except`.
fn set_equal_except(a: &[usize], skip_a: usize, b: &[usize], mut skip_b: usize) -> bool {
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut skip = 0;
    loop {
        if at(a, ia) == skip_a {
            skip += 1;
            ia += 1;
        }
        if skip_b != NULL {
            if at(b, ib) == skip_b {
                skip += 1;
                ib += 1;
            }
        } else if at(a, ia) != at(b, ib) {
            skip += 1;
            skip_b = at(b, ib);
            ib += 1;
            if skip_b == NULL {
                return false;
            }
        }
        if at(a, ia) == NULL {
            break;
        }
        if at(a, ia) != at(b, ib) {
            return false;
        }
        ia += 1;
        ib += 1;
    }
    !(skip != 2 || at(b, ib) != NULL)
}

/// macOS libc `qsort` on a set of handles, with a comparator returning 1 or −1 (never 0).
///
/// Bentley–McIlroy ("Engineering a Sort Function") as Apple's libc runs it: insertion sort below eight
/// elements; median of three (ninther above 40); when a partition made no swap, an insertion sort that gives
/// up after `1 + n/4` swaps; and heapsort (FreeBSD's) once `2·(fls(n) − 1)` levels are spent.
pub(crate) fn apple_qsort(a: &mut [usize], cmp: &mut dyn FnMut(usize, usize) -> i32) {
    let n = a.len();
    if n == 0 {
        return;
    }
    let depth = 2 * (usize::BITS - n.leading_zeros()) as i32 - 2;
    qsort_rec(a, cmp, depth);
}

fn qsort_rec(mut a: &mut [usize], cmp: &mut dyn FnMut(usize, usize) -> i32, mut depth: i32) {
    loop {
        let n = a.len();
        if depth <= 0 {
            heapsort(a, cmp);
            return;
        }
        depth -= 1;
        if n < 8 {
            for pm in 1..n {
                let mut pl = pm;
                while pl > 0 && cmp(a[pl - 1], a[pl]) > 0 {
                    a.swap(pl, pl - 1);
                    pl -= 1;
                }
            }
            return;
        }
        let mut pm = n / 2;
        {
            let mut pl = 0;
            let mut pn = n - 1;
            let med3 = |a: &[usize], x: usize, y: usize, z: usize, cmp: &mut dyn FnMut(usize, usize) -> i32| -> usize {
                if cmp(a[x], a[y]) < 0 {
                    if cmp(a[y], a[z]) < 0 {
                        y
                    } else if cmp(a[x], a[z]) < 0 {
                        z
                    } else {
                        x
                    }
                } else if cmp(a[y], a[z]) > 0 {
                    y
                } else if cmp(a[x], a[z]) < 0 {
                    x
                } else {
                    z
                }
            };
            if n > 40 {
                let d = n / 8;
                pl = med3(a, pl, pl + d, pl + 2 * d, cmp);
                pm = med3(a, pm - d, pm, pm + d, cmp);
                pn = med3(a, pn - 2 * d, pn - d, pn, cmp);
            }
            pm = med3(a, pl, pm, pn, cmp);
        }
        a.swap(0, pm);
        let (mut pa, mut pb) = (1usize, 1usize);
        let (mut pc, mut pd) = (n as isize - 1, n as isize - 1);
        let mut swap_cnt = 0usize;
        loop {
            while pb as isize <= pc {
                let r = cmp(a[pb], a[0]);
                if r > 0 {
                    break;
                }
                if r == 0 {
                    swap_cnt = 1;
                    a.swap(pa, pb);
                    pa += 1;
                }
                pb += 1;
            }
            while pb as isize <= pc {
                let r = cmp(a[pc as usize], a[0]);
                if r < 0 {
                    break;
                }
                if r == 0 {
                    swap_cnt = 1;
                    a.swap(pc as usize, pd as usize);
                    pd -= 1;
                }
                pc -= 1;
            }
            if pb as isize > pc {
                break;
            }
            a.swap(pb, pc as usize);
            swap_cnt = 1;
            pb += 1;
            pc -= 1;
        }
        let pn = n;
        let r = pa.min(pb - pa);
        for i in 0..r {
            a.swap(i, pb - r + i);
        }
        let r = ((pd - pc) as usize).min(pn - pd as usize - 1);
        for i in 0..r {
            a.swap(pb + i, pn - r + i);
        }
        if swap_cnt == 0 {
            let budget = 1 + n / 4;
            let mut swaps = 0usize;
            let mut gave_up = false;
            'ins: for pm in 1..n {
                let mut pl = pm;
                while pl > 0 && cmp(a[pl - 1], a[pl]) > 0 {
                    a.swap(pl, pl - 1);
                    pl -= 1;
                    swaps += 1;
                    if swaps > budget {
                        gave_up = true;
                        break 'ins;
                    }
                }
            }
            if !gave_up {
                return;
            }
        }
        let left = pb - pa;
        let right = (pd - pc) as usize;
        if left > 1 {
            qsort_rec(&mut a[..left], cmp, depth);
        }
        if right > 1 {
            let tmp = a;
            a = &mut tmp[pn - right..];
            continue;
        }
        return;
    }
}

/// FreeBSD's `heapsort` (Knuth's Algorithm H with the bottom-up select), as libc's `qsort` falls back to.
fn heapsort(a: &mut [usize], cmp: &mut dyn FnMut(usize, usize) -> i32) {
    let nmemb0 = a.len();
    if nmemb0 <= 1 {
        return;
    }
    // 1-based: base[i] = a[i - 1]
    let mut nmemb = nmemb0;
    let mut l = nmemb / 2 + 1;
    loop {
        l -= 1;
        if l == 0 {
            break;
        }
        let mut i = l;
        loop {
            let mut j = i * 2;
            if j > nmemb {
                break;
            }
            if j < nmemb && cmp(a[j - 1], a[j]) < 0 {
                j += 1;
            }
            if cmp(a[j - 1], a[i - 1]) <= 0 {
                break;
            }
            a.swap(i - 1, j - 1);
            i = j;
        }
    }
    while nmemb > 1 {
        let k = a[nmemb - 1];
        a[nmemb - 1] = a[0];
        nmemb -= 1;
        let mut i = 1;
        loop {
            let mut j = i * 2;
            if j > nmemb {
                break;
            }
            if j < nmemb && cmp(a[j - 1], a[j]) < 0 {
                j += 1;
            }
            a[i - 1] = a[j - 1];
            i = j;
        }
        loop {
            let j = i;
            i = j / 2;
            if j == 1 || cmp(k, a[i - 1]) < 0 {
                a[j - 1] = k;
                break;
            }
            a[j - 1] = a[i - 1];
        }
    }
}

/// qhull's `qhT`, the part this subset reads.
struct Qh {
    points: Vec<[f64; 3]>,
    interior_point: [f64; 3],
    facets: Vec<Facet>,
    vertices: Vec<Vertex>,
    ridges: Vec<Ridge>,
    merges: Vec<Merge>,
    facet_list: usize,
    facet_tail: usize,
    facet_next: usize,
    newfacet_list: usize,
    visible_list: usize,
    vertex_list: usize,
    vertex_tail: usize,
    newvertex_list: usize,
    num_facets: i64,
    num_vertices: i64,
    num_visible: i64,
    num_outside: i64,
    facet_id: u32,
    vertex_id: u32,
    visit_id: u32,
    vertex_visit: u32,
    first_newfacet: u32,
    repart_facetid: u32,
    /// `zzval_(Ztotmerge)` — a statistic that qhull's own heuristics read (`qh_USEfindbestnew`)
    totmerge: u64,
    stopadd: i64,
    pickfurthest: bool,
    zeroall_ok: bool,
    narrowhull: bool,
    vertexneighbors: bool,
    newfacets: bool,
    findbestnew: bool,
    findbest_notsharp: bool,
    postmerging: bool,
    has_triangulation: bool,
    max_outside: f64,
    max_vertex: f64,
    min_vertex: f64,
    maxabs_coord: f64,
    maxsumcoord: f64,
    maxwidth: f64,
    mindenom_1: f64,
    mindenom: f64,
    mindenom_1_2: f64,
    mindenom_2: f64,
    distround: f64,
    anglereound: f64,
    premerge_centrum: f64,
    premerge_cos: f64,
    one_merge: f64,
    near_inside: f64,
    min_visible: f64,
    max_coplanar: f64,
    min_outside: f64,
    wide_facet: f64,
    max_outside_limit: f64,
    centrum_radius: f64,
    cos_max: f64,
    near_zero: [f64; 3],
    del_vertices: Vec<usize>,
    coplanarfacetset: Vec<usize>,
    hash_table: Vec<usize>,
    facet_mergeset: Option<Vec<usize>>,
    degen_mergeset: Option<Vec<usize>>,
    vertex_mergeset: Option<Vec<usize>>,
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    // `a[0]*b[0] + a[1]*b[1] + a[2]*b[2]` as clang contracts it: the right product of the first sum rounds
    a[2].mul_add(b[2], a[0].mul_add(b[0], a[1] * b[1]))
}

impl Qh {
    fn new(points: Vec<[f64; 3]>, max_verts: Option<usize>) -> Self {
        Qh {
            points,
            interior_point: [0.0; 3],
            facets: Vec::new(),
            vertices: Vec::new(),
            ridges: Vec::new(),
            merges: Vec::new(),
            facet_list: NULL,
            facet_tail: NULL,
            facet_next: NULL,
            newfacet_list: NULL,
            visible_list: NULL,
            vertex_list: NULL,
            vertex_tail: NULL,
            newvertex_list: NULL,
            num_facets: 0,
            num_vertices: 0,
            num_visible: 0,
            num_outside: 0,
            facet_id: 0,
            vertex_id: 0,
            visit_id: 0,
            vertex_visit: 0,
            first_newfacet: 0,
            repart_facetid: 0,
            totmerge: 0,
            // "TA<n>": stop after n points past the initial simplex; qh_initflags stores n + 1
            stopadd: max_verts.map_or(0, |m| m as i64 - 4 + 1),
            pickfurthest: max_verts.is_some(),
            zeroall_ok: true,
            narrowhull: false,
            vertexneighbors: false,
            newfacets: false,
            findbestnew: false,
            findbest_notsharp: false,
            postmerging: false,
            has_triangulation: false,
            max_outside: 0.0,
            max_vertex: 0.0,
            min_vertex: 0.0,
            maxabs_coord: 0.0,
            maxsumcoord: 0.0,
            maxwidth: -REALMAX,
            mindenom_1: if 1.0 / REALMAX < REALMIN { REALMIN } else { 1.0 / REALMAX },
            mindenom: 0.0,
            mindenom_1_2: 0.0,
            mindenom_2: 0.0,
            distround: 0.0,
            anglereound: 0.0,
            premerge_centrum: 0.0,
            premerge_cos: REALMAX,
            one_merge: 0.0,
            near_inside: 0.0,
            min_visible: REALMAX,
            max_coplanar: REALMAX,
            min_outside: 0.0,
            wide_facet: 0.0,
            max_outside_limit: 0.0,
            centrum_radius: 0.0,
            cos_max: 0.0,
            near_zero: [0.0; 3],
            del_vertices: Vec::new(),
            coplanarfacetset: Vec::new(),
            hash_table: Vec::new(),
            facet_mergeset: None,
            degen_mergeset: None,
            vertex_mergeset: None,
        }
    }

    // ------------------------------------------------------------------------------------ geometry

    /// `qh_distplane`, 3-d: `offset + p0·n0 + p1·n1 + p2·n2`, each product fused into the running sum.
    fn distplane(&self, point: &[f64; 3], facet: usize) -> f64 {
        let f = &self.facets[facet];
        let n = f.normal.as_ref().expect("qh_distplane: facet without a normal");
        point[2].mul_add(n[2], point[1].mul_add(n[1], point[0].mul_add(n[0], f.offset)))
    }

    /// `qh_getcenter`: the mean of the facet's vertices, summed in set order.
    fn getcenter(&self, vertices: &[usize]) -> [f64; 3] {
        let count = vertices.len() as f64;
        let mut c = [0.0; 3];
        for (k, ck) in c.iter_mut().enumerate() {
            for &v in vertices {
                if v == NULL {
                    break;
                }
                *ck += self.points[self.vertices[v].point][k];
            }
            *ck /= count;
        }
        c
    }

    /// `qh_getcentrum`: the centre projected onto the facet's plane (`qh_projectpoint`: `p − d·n`, fused).
    fn getcentrum(&self, facet: usize) -> [f64; 3] {
        let point = self.getcenter(&self.facets[facet].vertices);
        let dist = self.distplane(&point, facet);
        let n = self.facets[facet].normal.unwrap();
        [(-dist).mul_add(n[0], point[0]), (-dist).mul_add(n[1], point[1]), (-dist).mul_add(n[2], point[2])]
    }

    /// `qh_getangle`: the dot product accumulated from zero, each product fused.
    fn getangle(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let mut angle = 0.0f64;
        for k in 0..DIM {
            angle = a[k].mul_add(b[k], angle);
        }
        angle
    }

    /// `qh_normalize2` for 3-d, `minnorm` unused.
    fn normalize2(&self, normal: &mut [f64; 3], toporient: bool) {
        let mut norm = normal[2].mul_add(normal[2], normal[0].mul_add(normal[0], normal[1] * normal[1])).sqrt();
        if norm > self.mindenom {
            if !toporient {
                norm = -norm;
            }
            normal[0] /= norm;
            normal[1] /= norm;
            normal[2] /= norm;
        } else if norm == 0.0 {
            let temp = (1.0f64 / DIM as f64).sqrt();
            *normal = [temp; 3];
        } else {
            if !toporient {
                norm = -norm;
            }
            for k in 0..DIM {
                let (temp, zerodiv) = divzero(normal[k], norm, self.mindenom_1);
                if !zerodiv {
                    normal[k] = temp;
                } else {
                    let maxp = maxabsval(normal);
                    let temp = if normal[maxp] * norm >= 0.0 { 1.0 } else { -1.0 };
                    *normal = [0.0; 3];
                    normal[maxp] = temp;
                    return;
                }
            }
        }
    }

    /// `qh_sethyperplane_det`, 3-d. `rows` are the facet's vertex points in set order; returns nearzero.
    fn sethyperplane_det(&self, rows: &[[f64; 3]; 3], toporient: bool, normal: &mut [f64; 3], offset: &mut f64) -> bool {
        let dx = |i: usize, j: usize| rows[i][0] - rows[j][0];
        let dy = |i: usize, j: usize| rows[i][1] - rows[j][1];
        let dz = |i: usize, j: usize| rows[i][2] - rows[j][2];
        // det2_(a1, a2, b1, b2) = a1·b2 − a2·b1: the left product fused, the right one rounded
        let det2 = |a1: f64, a2: f64, b1: f64, b2: f64| a1.mul_add(b2, -(a2 * b1));
        normal[0] = det2(dy(2, 0), dz(2, 0), dy(1, 0), dz(1, 0));
        normal[1] = det2(dx(1, 0), dz(1, 0), dx(2, 0), dz(2, 0));
        normal[2] = det2(dx(2, 0), dy(2, 0), dx(1, 0), dy(1, 0));
        self.normalize2(normal, toporient);
        let point0 = &rows[0];
        let sum0 = dot3(point0, normal);
        *offset = -sum0;
        let maxround = self.distround;
        for i in (0..DIM).rev() {
            let point = &rows[i];
            // `point != point0` compares POINTERS: only row 0 is skipped
            if i != 0 {
                let dist = dot3(point, normal) - sum0;
                if dist > maxround || dist < -maxround {
                    return true;
                }
            }
        }
        false
    }

    /// `qh_sethyperplane_gauss`, 3-d, from the rows `v − point0` of the two other vertices.
    fn sethyperplane_gauss(&self, mut rows: [[f64; 3]; 2], point0: &[f64; 3], toporient: bool, normal: &mut [f64; 3], offset: &mut f64) -> bool {
        let mut sign = toporient;
        let mut nearzero = self.gausselim(&mut rows, &mut sign);
        for k in (0..DIM - 1).rev() {
            if rows[k][k] < 0.0 {
                sign = !sign;
            }
        }
        let nearzero2 = self.backnormal(&rows, sign, normal);
        if nearzero2 {
            nearzero = true;
        }
        self.normalize2(normal, true);
        *offset = -(point0[0] * normal[0]);
        for k in 1..DIM {
            *offset = (-point0[k]).mul_add(normal[k], *offset);
        }
        nearzero
    }

    /// `qh_gausselim` on a 2×3 system (rows swap by pointer, which here swaps the arrays).
    fn gausselim(&self, rows: &mut [[f64; 3]; 2], sign: &mut bool) -> bool {
        let numrow = 2;
        let numcol = 3;
        let mut nearzero = false;
        for k in 0..numrow {
            let mut pivot_abs = rows[k][k].abs();
            let mut pivoti = k;
            for (i, row) in rows.iter().enumerate().take(numrow).skip(k + 1) {
                let temp = row[k].abs();
                if temp > pivot_abs {
                    pivot_abs = temp;
                    pivoti = i;
                }
            }
            if pivoti != k {
                rows.swap(pivoti, k);
                *sign = !*sign;
            }
            if pivot_abs <= self.near_zero[k] {
                nearzero = true;
                if pivot_abs == 0.0 {
                    continue;
                }
            }
            let pivot = rows[k][k];
            for i in k + 1..numrow {
                let n = rows[i][k] / pivot;
                for j in k + 1..numcol {
                    rows[i][j] = (-n).mul_add(rows[k][j], rows[i][j]);
                }
            }
        }
        nearzero
    }

    /// `qh_backnormal` for 2 rows, 3 columns; returns nearzero.
    fn backnormal(&self, rows: &[[f64; 3]; 2], sign: bool, normal: &mut [f64; 3]) -> bool {
        let numrow = 2usize;
        let numcol = 3usize;
        let mut zerocol: i32 = -1;
        let mut np = numcol as isize - 1;
        normal[np as usize] = if sign { -1.0 } else { 1.0 };
        np -= 1;
        for i in (0..numrow).rev() {
            let npu = np as usize;
            normal[npu] = 0.0;
            for j in i + 1..numcol {
                normal[npu] = (-rows[i][j]).mul_add(normal[npu + j - i], normal[npu]);
            }
            let diagonal = rows[i][i];
            if diagonal.abs() > self.mindenom_2 {
                normal[npu] /= diagonal;
                np -= 1;
            } else {
                let (v, waszero) = divzero(normal[npu], diagonal, self.mindenom_1_2);
                normal[npu] = v;
                if waszero {
                    zerocol = i as i32;
                    normal[npu] = if sign { -1.0 } else { 1.0 };
                    np -= 1;
                    for t in (np + 2) as usize..numcol {
                        normal[t] = 0.0;
                    }
                } else {
                    np -= 1;
                }
            }
        }
        zerocol != -1
    }

    /// `qh_determinant` for 2-d and 3-d.
    fn determinant(&self, rows: &[[f64; 3]], dim: usize) -> (f64, bool) {
        let det2 = |a1: f64, a2: f64, b1: f64, b2: f64| a1.mul_add(b2, -(a2 * b1));
        if dim == 2 {
            let det = det2(rows[0][0], rows[0][1], rows[1][0], rows[1][1]);
            (det, det.abs() < 10.0 * self.near_zero[1])
        } else {
            let (a, b, c) = (&rows[0], &rows[1], &rows[2]);
            let d1 = det2(b[1], b[2], c[1], c[2]);
            let d2 = det2(a[1], a[2], c[1], c[2]);
            let d3 = det2(a[1], a[2], b[1], b[2]);
            // a1·D1 − b1·D2 + c1·D3, contracted as fma(c1, D3, fma(a1, D1, −(b1·D2)))
            let det = c[0].mul_add(d3, a[0].mul_add(d1, -(b[0] * d2)));
            (det, det.abs() < 10.0 * self.near_zero[2])
        }
    }

    /// `qh_detsimplex`: the determinant of `points[..dim] − apex`.
    fn detsimplex(&self, apex: usize, simplex: &[usize], dim: usize) -> Result<(f64, bool), String> {
        let a = self.points[apex];
        let mut rows = [[0.0; 3]; 3];
        let mut i = 0;
        for &p in simplex {
            if i == dim {
                break;
            }
            let q = self.points[p];
            rows[i] = [q[0] - a[0], q[1] - a[1], q[2] - a[2]];
            i += 1;
        }
        if i < dim {
            return Err("qhull internal error (qh_detsimplex): #points < dimension".into());
        }
        Ok(self.determinant(&rows, dim))
    }

    /// `qh_orientoutside`.
    fn orientoutside(&mut self, facet: usize) -> bool {
        let dist = self.distplane(&self.interior_point.clone(), facet);
        if dist > 0.0 {
            let f = &mut self.facets[facet];
            let n = f.normal.as_mut().unwrap();
            for x in n.iter_mut() {
                *x = -*x;
            }
            f.offset = -f.offset;
            return true;
        }
        false
    }

    /// `qh_setfacetplane`.
    fn setfacetplane(&mut self, facet: usize) {
        let verts: Vec<usize> = self.facets[facet].vertices.iter().copied().take_while(|&v| v != NULL).collect();
        let rows: [[f64; 3]; 3] = [self.points[self.vertices[verts[0]].point], self.points[self.vertices[verts[1]].point], self.points[self.vertices[verts[2]].point]];
        let toporient = self.facets[facet].toporient;
        let mut normal = [0.0; 3];
        let mut offset = 0.0;
        let nearzero = self.sethyperplane_det(&rows, toporient, &mut normal, &mut offset);
        self.facets[facet].normal = Some(normal);
        self.facets[facet].offset = offset;
        if nearzero {
            let p0 = rows[0];
            let mut g = [[0.0; 3]; 2];
            let mut i = 0;
            for (k, v) in verts.iter().enumerate() {
                // `vertex->point != point0` by pointer: skips the first vertex only
                if k != 0 && self.vertices[*v].point != self.vertices[verts[0]].point {
                    let p = self.points[self.vertices[*v].point];
                    g[i] = [p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]];
                    i += 1;
                }
            }
            let mut normal = [0.0; 3];
            let mut offset = 0.0;
            let nearzero = self.sethyperplane_gauss(g, &p0, toporient, &mut normal, &mut offset);
            self.facets[facet].normal = Some(normal);
            self.facets[facet].offset = offset;
            if nearzero {
                self.orientoutside(facet);
            }
        }
        self.checkflipped(facet, true);
    }

    /// `qh_checkflipped(facet, NULL, allerror)`: returns false (and sets `flipped`) when flipped.
    fn checkflipped(&mut self, facet: usize, allerror: bool) -> bool {
        if self.facets[facet].flipped {
            return false;
        }
        let dist = self.distplane(&self.interior_point.clone(), facet);
        if (allerror && dist >= -self.distround) || (!allerror && dist > 0.0) {
            self.facets[facet].flipped = true;
            return false;
        }
        true
    }
}

/// `qh_divzero`: `numer/denom` unless that would overflow.
fn divzero(numer: f64, denom: f64, mindenom1: f64) -> (f64, bool) {
    if numer < mindenom1 && numer > -mindenom1 {
        let numerx = numer.abs();
        let denomx = denom.abs();
        if numerx < denomx {
            (numer / denom, false)
        } else {
            (0.0, true)
        }
    } else {
        let temp = denom / numer;
        if temp > mindenom1 || temp < -mindenom1 {
            (numer / denom, false)
        } else {
            (0.0, true)
        }
    }
}

/// `qh_maxabsval`: index of the coordinate of largest magnitude (first on ties).
fn maxabsval(normal: &[f64; 3]) -> usize {
    let mut maxval = -REALMAX;
    let mut maxp = 0;
    for (k, &x) in normal.iter().enumerate() {
        let absval = if x < 0.0 { -x } else { x };
        if absval > maxval {
            maxval = absval;
            maxp = k;
        }
    }
    maxp
}

/// Which list head `qh_prependfacet` updates.
#[derive(Clone, Copy)]
enum Head {
    FacetNext,
    VisibleList,
}

impl Qh {
    // ------------------------------------------------------------------------- lists and objects

    fn otherfacet(&self, ridge: usize, facet: usize) -> usize {
        let r = &self.ridges[ridge];
        if r.top == facet {
            r.bottom
        } else {
            r.top
        }
    }

    /// `qh_newfacet`.
    fn newfacet(&mut self) -> usize {
        let id = self.facet_id;
        self.facet_id += 1;
        self.facets.push(Facet { id, maxoutside: self.distround, simplicial: true, good: true, newfacet: true, f: NULL, previous: NULL, next: NULL, ..Default::default() });
        self.facets.len() - 1
    }

    /// `qh_newvertex`.
    fn newvertex(&mut self, point: usize) -> usize {
        let id = self.vertex_id;
        self.vertex_id += 1;
        self.vertices.push(Vertex { id, point, next: NULL, previous: NULL, ..Default::default() });
        self.vertices.len() - 1
    }

    /// `qh_newridge`.
    fn newridge(&mut self) -> usize {
        self.ridges.push(Ridge { top: NULL, bottom: NULL, ..Default::default() });
        self.ridges.len() - 1
    }

    /// `qh_appendfacet`.
    fn appendfacet(&mut self, facet: usize) {
        let tail = self.facet_tail;
        if tail == self.newfacet_list {
            self.newfacet_list = facet;
            if tail == self.visible_list {
                self.visible_list = facet;
            }
        }
        if tail == self.facet_next {
            self.facet_next = facet;
        }
        let prev = self.facets[tail].previous;
        self.facets[facet].previous = prev;
        self.facets[facet].next = tail;
        if prev != NULL {
            self.facets[prev].next = facet;
        } else {
            self.facet_list = facet;
        }
        self.facets[tail].previous = facet;
        self.num_facets += 1;
    }

    /// `qh_removefacet`.
    fn removefacet(&mut self, facet: usize) {
        let next = self.facets[facet].next;
        let previous = self.facets[facet].previous;
        if facet == self.newfacet_list {
            self.newfacet_list = next;
        }
        if facet == self.facet_next {
            self.facet_next = next;
        }
        if facet == self.visible_list {
            self.visible_list = next;
        }
        if previous != NULL {
            self.facets[previous].next = next;
            self.facets[next].previous = previous;
        } else {
            self.facet_list = next;
            self.facets[next].previous = NULL;
        }
        self.num_facets -= 1;
    }

    /// `qh_prependfacet(facet, &head)`.
    fn prependfacet(&mut self, facet: usize, head: Head) {
        let mut list = match head {
            Head::FacetNext => self.facet_next,
            Head::VisibleList => self.visible_list,
        };
        if list == NULL {
            list = self.facet_tail;
        }
        let prevfacet = self.facets[list].previous;
        self.facets[facet].previous = prevfacet;
        if prevfacet != NULL {
            self.facets[prevfacet].next = facet;
        }
        self.facets[list].previous = facet;
        self.facets[facet].next = list;
        if self.facet_list == list {
            self.facet_list = facet;
        }
        if self.facet_next == list {
            self.facet_next = facet;
        }
        match head {
            Head::FacetNext => self.facet_next = facet,
            Head::VisibleList => self.visible_list = facet,
        }
        self.num_facets += 1;
    }

    /// `qh_appendvertex`.
    fn appendvertex(&mut self, vertex: usize) {
        let tail = self.vertex_tail;
        if tail == self.newvertex_list {
            self.newvertex_list = vertex;
        }
        self.vertices[vertex].newfacet = true;
        let prev = self.vertices[tail].previous;
        self.vertices[vertex].previous = prev;
        self.vertices[vertex].next = tail;
        if prev != NULL {
            self.vertices[prev].next = vertex;
        } else {
            self.vertex_list = vertex;
        }
        self.vertices[tail].previous = vertex;
        self.num_vertices += 1;
    }

    /// `qh_removevertex`.
    fn removevertex(&mut self, vertex: usize) {
        let next = self.vertices[vertex].next;
        let previous = self.vertices[vertex].previous;
        if vertex == self.newvertex_list {
            self.newvertex_list = next;
        }
        if previous != NULL {
            self.vertices[previous].next = next;
            self.vertices[next].previous = previous;
        } else {
            self.vertex_list = next;
            self.vertices[next].previous = NULL;
        }
        self.num_vertices -= 1;
    }

    /// `qh_delfacet`.
    fn delfacet(&mut self, facet: usize) {
        self.removefacet(facet);
        let f = &mut self.facets[facet];
        f.neighbors.clear();
        f.ridges = None;
        f.vertices.clear();
        f.outsideset = None;
        f.coplanarset = None;
    }

    /// `qh_delvertex`.
    fn delvertex(&mut self, vertex: usize) {
        self.removevertex(vertex);
        self.vertices[vertex].neighbors.clear();
    }

    /// `qh_willdelete`.
    fn willdelete(&mut self, facet: usize, replace: usize) {
        self.removefacet(facet);
        self.prependfacet(facet, Head::VisibleList);
        self.num_visible += 1;
        let f = &mut self.facets[facet];
        f.visible = true;
        f.f = replace;
        if let Some(x) = f.ridges.as_mut().and_then(|r| r.first_mut()) {
            *x = NULL;
        }
        if let Some(x) = f.neighbors.first_mut() {
            *x = NULL;
        }
    }

    /// `qh_getreplacement`.
    fn getreplacement(&self, visible: usize) -> usize {
        let mut result = visible;
        while result != NULL && self.facets[result].visible {
            result = self.facets[result].f;
        }
        result
    }

    /// `qh_resetlists`.
    fn resetlists(&mut self, reset_visible: bool) {
        let mut v = self.newvertex_list;
        while v != NULL && self.vertices[v].next != NULL {
            self.vertices[v].newfacet = false;
            v = self.vertices[v].next;
        }
        self.newvertex_list = NULL;
        self.first_newfacet = 0;
        let mut f = self.newfacet_list;
        while f != NULL && self.facets[f].next != NULL {
            self.facets[f].newfacet = false;
            self.facets[f].dupridge = false;
            f = self.facets[f].next;
        }
        self.newfacet_list = NULL;
        if reset_visible {
            let mut f = self.visible_list;
            while f != NULL && self.facets[f].visible {
                self.facets[f].f = NULL;
                self.facets[f].visible = false;
                f = self.facets[f].next;
            }
            self.num_visible = 0;
        }
        self.visible_list = NULL;
        self.newfacets = false;
    }

    /// `qh_deletevisible`.
    fn deletevisible(&mut self) {
        let mut visible = self.visible_list;
        while visible != NULL && self.facets[visible].visible {
            let next = self.facets[visible].next;
            self.delfacet(visible);
            visible = next;
        }
        self.num_visible = 0;
        let dels = self.del_vertices.clone();
        for v in dels {
            self.delvertex(v);
        }
        self.del_vertices.clear();
    }

    /// FORALLfacet_(start): the facets from `start` to the tail.
    fn facets_from(&self, start: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut f = start;
        while f != NULL && self.facets[f].next != NULL {
            out.push(f);
            f = self.facets[f].next;
        }
        out
    }

    // ----------------------------------------------------------------------------------- the build

    /// `qh_qhull` as this subset needs it: the build; the post-merge MuJoCo's options leave a no-op.
    fn qhull(&mut self) -> Result<(), String> {
        self.initbuild()?;
        self.buildhull()
    }

    /// `qh_maxmin`: the extreme points per coordinate, and the scale they set.
    fn maxmin(&mut self) -> Vec<usize> {
        self.max_outside = 0.0;
        self.maxabs_coord = 0.0;
        self.maxwidth = -REALMAX;
        self.maxsumcoord = 0.0;
        self.min_vertex = 0.0;
        self.zeroall_ok = true;
        let mut set = Vec::new();
        for k in 0..DIM {
            let (mut minimum, mut maximum) = (0usize, 0usize);
            for (i, p) in self.points.iter().enumerate() {
                if self.points[maximum][k] < p[k] {
                    maximum = i;
                } else if self.points[minimum][k] > p[k] {
                    minimum = i;
                }
            }
            let (mx, mn) = (self.points[maximum][k], self.points[minimum][k]);
            let maxcoord = if mx < -mn { -mn } else { mx };
            let temp = mx - mn;
            if self.maxwidth < temp {
                self.maxwidth = temp;
            }
            if self.maxabs_coord < maxcoord {
                self.maxabs_coord = maxcoord;
            }
            self.maxsumcoord += maxcoord;
            set.push(minimum);
            set.push(maximum);
            self.near_zero[k] = 80.0 * self.maxsumcoord * REALEPS;
        }
        set
    }

    /// `qh_detroundoff` (with `qh_distround` and `qh_detmaxoutside`) for `C-0`: every tolerance qhull uses.
    fn detroundoff(&mut self) {
        let dim = DIM as f64;
        let mut maxdistsum = self.maxabs_coord * dim.sqrt();
        if maxdistsum > self.maxsumcoord {
            maxdistsum = self.maxsumcoord;
        }
        self.distround = (maxdistsum * dim).mul_add(1.01, self.maxabs_coord) * REALEPS;
        self.mindenom = self.mindenom_1 * self.maxabs_coord;
        self.mindenom_1_2 = (self.mindenom_1 * dim).sqrt();
        self.mindenom_2 = self.mindenom_1_2 * self.maxabs_coord;
        self.anglereound = dim * 1.01 * REALEPS;
        self.premerge_centrum = 2.0f64.mul_add(self.distround, self.premerge_centrum);
        let postmerge_centrum = 2.0f64.mul_add(self.distround, 0.0);
        let maxangle: f64 = 1.0;
        self.one_merge = (self.maxwidth * dim.sqrt()).mul_add((-maxangle).mul_add(maxangle, 1.0).sqrt(), self.distround);
        let maxrho = dim.mul_add(self.premerge_centrum, self.distround);
        if self.one_merge < maxrho {
            self.one_merge = maxrho;
        }
        let maxrho = dim.mul_add(postmerge_centrum, self.distround);
        if self.one_merge < maxrho {
            self.one_merge = maxrho;
        }
        self.near_inside = self.one_merge * QH_RATIONEARINSIDE;
        if self.min_visible > REALMAX / 2.0 {
            self.min_visible = self.premerge_centrum;
        }
        if self.max_coplanar > REALMAX / 2.0 {
            self.max_coplanar = self.min_visible;
        }
        self.min_outside = 2.0 * self.min_visible;
        self.wide_facet = self.min_outside;
        if self.wide_facet < QH_WIDECOPLANAR * self.max_coplanar {
            self.wide_facet = QH_WIDECOPLANAR * self.max_coplanar;
        }
        if self.wide_facet < QH_WIDECOPLANAR * self.min_visible {
            self.wide_facet = QH_WIDECOPLANAR * self.min_visible;
        }
        self.max_vertex = self.distround;
        self.min_vertex = -self.distround;
        self.detmaxoutside();
    }

    /// `qh_detmaxoutside`.
    fn detmaxoutside(&mut self) {
        let a = self.max_outside;
        let b = self.one_merge + self.distround;
        let mut maxoutside = if a < b { b } else { a };
        if maxoutside < self.min_outside {
            maxoutside = self.min_outside;
        }
        self.max_outside_limit = maxoutside;
    }

    /// `qh_USEfindbestnew`.
    fn use_findbestnew(&self) -> bool {
        self.totmerge > 50
    }

    /// `qh_DISToutside`.
    fn dist_outside(&self) -> f64 {
        let a = 2.0 * self.min_outside;
        let b = self.max_outside;
        (if self.use_findbestnew() { 2.0 } else { 1.0 }) * (if a < b { b } else { a })
    }

    /// `qh_SEARCHdist`: `max_outside + 2·DISTround` fused, plus the larger of MINvisible and MAXcoplanar.
    fn search_dist(&self) -> f64 {
        let m = if self.min_visible < self.max_coplanar { self.max_coplanar } else { self.min_visible };
        (self.distround.mul_add(2.0, self.max_outside) + m) * (if self.use_findbestnew() { 2.0 } else { 1.0 })
    }

    /// `qh_initbuild`.
    fn initbuild(&mut self) -> Result<(), String> {
        self.facet_id = 0;
        self.vertex_id = 0;
        self.visit_id = 0;
        self.vertex_visit = 0;
        let maxpoints = self.maxmin();
        self.detroundoff();
        let tail = self.newfacet();
        self.facet_list = tail;
        self.newfacet_list = tail;
        self.facet_tail = tail;
        self.num_facets = 0;
        self.num_vertices = 0;
        self.num_visible = 0;
        let vtail = self.newvertex(NULL);
        self.vertex_list = vtail;
        self.newvertex_list = vtail;
        self.vertex_tail = vtail;
        let vertices = self.initialvertices(&maxpoints)?;
        self.initialhull(&vertices)?;
        self.partitionall(&vertices)?;
        self.resetlists(true);
        self.facet_next = self.facet_list;
        self.furthestnext();
        self.cos_max = self.premerge_cos;
        self.centrum_radius = self.premerge_centrum;
        Ok(())
    }

    /// `qh_initialvertices` for 3-d: `qh_maxsimplex`, then the vertices in reverse.
    fn initialvertices(&mut self, maxpoints: &[usize]) -> Result<Vec<usize>, String> {
        let simplex = self.maxsimplex(maxpoints)?;
        let mut vertices = Vec::new();
        for p in simplex {
            let v = self.newvertex(p);
            vertices.insert(0, v);
        }
        Ok(vertices)
    }

    /// `qh_maxsimplex(dim = 3)`: the extreme pair in x, then the points spanning the largest simplex.
    fn maxsimplex(&self, maxpoints: &[usize]) -> Result<Vec<usize>, String> {
        if self.maxwidth <= 0.0 {
            return Err("qhull input error (qh_maxsimplex): all points are the same".into());
        }
        let mut simplex: Vec<usize> = Vec::new();
        let (mut maxcoord, mut mincoord) = (-REALMAX, REALMAX);
        let (mut maxx, mut minx) = (NULL, NULL);
        if maxpoints.len() >= 2 {
            for &p in maxpoints {
                if maxcoord < self.points[p][0] {
                    maxcoord = self.points[p][0];
                    maxx = p;
                }
                if mincoord > self.points[p][0] {
                    mincoord = self.points[p][0];
                    minx = p;
                }
            }
        }
        let mut maxdet = maxcoord - mincoord;
        set_unique(&mut simplex, minx);
        if simplex.len() < 2 {
            set_unique(&mut simplex, maxx);
        }
        let sizinit = simplex.len();
        if sizinit < 2 {
            return Err("qhull input error (qh_maxsimplex): input has the same x coordinate".into());
        }
        let mut maxnearzero = false;
        for i in sizinit..DIM + 1 {
            let prevdet = maxdet;
            let mut maxpoint = NULL;
            maxdet = -1.0;
            for &p in maxpoints {
                if !set_in(&simplex, p) && p != maxpoint {
                    let (det, nearzero) = self.detsimplex(p, &simplex, i)?;
                    let det = if det < 0.0 { -det } else { det };
                    if det > maxdet {
                        maxdet = det;
                        maxpoint = p;
                        maxnearzero = nearzero;
                    }
                }
            }
            let mut maybe_falsenarrow = false;
            let targetdet = prevdet * self.maxwidth;
            let mindet = 10.0 * QH_RATIOMAXSIMPLEX * targetdet;
            if maxdet > 0.0 {
                let ratio = maxdet / targetdet;
                if ratio < QH_RATIOMAXSIMPLEX {
                    maybe_falsenarrow = true;
                }
            }
            if maxpoint == NULL || maxnearzero || maybe_falsenarrow {
                for p in 0..self.points.len() {
                    if !set_in(maxpoints, p) && !set_in(&simplex, p) {
                        let (det, nearzero) = self.detsimplex(p, &simplex, i)?;
                        let det = if det < 0.0 { -det } else { det };
                        if det > maxdet {
                            maxdet = det;
                            maxpoint = p;
                            maxnearzero = nearzero;
                            if !maxnearzero && maxdet > mindet {
                                break;
                            }
                        }
                    }
                }
            }
            if maxpoint == NULL {
                return Err("qhull precision error (qh_maxsimplex): the input is degenerate (flat)".into());
            }
            simplex.push(maxpoint);
        }
        Ok(simplex)
    }

    /// `qh_createsimplex`.
    fn createsimplex(&mut self, vertices: &[usize]) {
        let mut toporient = true;
        let n = vertices.len();
        for (i, &vertex) in vertices.iter().enumerate() {
            let f = self.newfacet();
            self.facets[f].vertices = set_new_delnthsorted(vertices, n, i, 0);
            if toporient {
                self.facets[f].toporient = true;
            }
            self.appendfacet(f);
            self.facets[f].newfacet = true;
            self.appendvertex(vertex);
            toporient = !toporient;
        }
        let news = self.facets_from(self.newfacet_list);
        for &nf in &news {
            let mut nb = Vec::new();
            for &f in &news {
                if f != nf {
                    nb.push(f);
                }
            }
            nb.truncate(DIM);
            self.facets[nf].neighbors = nb;
        }
    }

    /// `qh_initialhull`.
    fn initialhull(&mut self, vertices: &[usize]) -> Result<(), String> {
        self.createsimplex(vertices);
        self.resetlists(true);
        self.facet_next = self.facet_list;
        self.interior_point = self.getcenter(vertices);
        let firstfacet = self.facet_list;
        self.setfacetplane(firstfacet);
        if self.facets[firstfacet].flipped {
            self.facets[firstfacet].flipped = false;
        }
        let dist = self.distplane(&self.interior_point.clone(), firstfacet);
        if dist > self.distround {
            for f in self.facets_from(self.facet_list) {
                self.facets[f].toporient = !self.facets[f].toporient;
            }
            self.setfacetplane(firstfacet);
        }
        for f in self.facets_from(self.facet_list) {
            if f != firstfacet {
                self.setfacetplane(f);
            }
        }
        for f in self.facets_from(self.facet_list) {
            if self.facets[f].flipped {
                self.facets[f].flipped = false;
            }
            let dist = self.distplane(&self.interior_point.clone(), f);
            if dist > self.distround {
                for g in self.facets_from(self.facet_list) {
                    self.facets[g].flipped = false;
                    self.facets[g].toporient = !self.facets[g].toporient;
                    self.orientoutside(g);
                }
                break;
            }
        }
        let mut minangle = REALMAX;
        for f in self.facets_from(self.facet_list) {
            if !self.checkflipped(f, true) {
                return Err("qhull precision error (qh_initialhull): initial simplex is flat".into());
            }
            for &nb in &self.facets[f].neighbors.clone() {
                if nb == NULL {
                    break;
                }
                let angle = Self::getangle(self.facets[f].normal.as_ref().unwrap(), self.facets[nb].normal.as_ref().unwrap());
                if minangle > angle {
                    minangle = angle;
                }
            }
        }
        if minangle < QH_MAXNARROW {
            self.narrowhull = true;
        }
        // qh_checkpolygon's side effects (its qh_checklists stamps every facet with a new visit_id and
        // advances vertex_visit; then each initial vertex is seen, its visitid counts the facets it is in, and
        // vertex_visit advances by the facet count and one more)
        self.visit_id += 1;
        for f in self.facets_from(self.facet_list) {
            self.facets[f].visitid = self.visit_id;
        }
        self.vertex_visit += 1;
        let mut numfacets = 0u32;
        let mut v = self.vertex_list;
        while v != NULL && self.vertices[v].next != NULL {
            self.vertices[v].seen = false;
            self.vertices[v].visitid = 0;
            v = self.vertices[v].next;
        }
        for f in self.facets_from(self.facet_list) {
            numfacets += 1;
            for &v in &self.facets[f].vertices.clone() {
                self.vertices[v].visitid += 1;
                self.vertices[v].seen = true;
            }
        }
        self.vertex_visit += numfacets + 1;
        // qh_checkconvex(qh_DATAfault): every facet simplicial and ZEROcentrum, so only the error test
        for f in self.facets_from(self.facet_list) {
            let nbs = self.facets[f].neighbors.clone();
            for (i, &nb) in nbs.iter().enumerate() {
                let vertex = self.facets[f].vertices[i];
                let dist = self.distplane(&self.points[self.vertices[vertex].point], nb);
                if dist >= -self.distround {
                    return Err("qhull precision error (qh_checkconvex): the initial simplex is not convex".into());
                }
            }
        }
        Ok(())
    }

    /// `qh_partitionall`.
    fn partitionall(&mut self, vertices: &[usize]) -> Result<(), String> {
        let numpoints = self.points.len();
        let mut pointset: Vec<usize> = (0..numpoints).collect();
        self.num_outside = 0;
        for &v in vertices {
            pointset[self.vertices[v].point] = NULL;
        }
        let distoutside = self.dist_outside();
        for facet in self.facets_from(self.facet_list) {
            let mut outside: Vec<usize> = Vec::new();
            let mut bestpoint = NULL;
            let mut bestdist = -REALMAX;
            let mut point_end = 0usize;
            let point_n = pointset.len();
            for point_i in 0..point_n {
                let point = pointset[point_i];
                if point != NULL {
                    let dist = self.distplane(&self.points[point], facet);
                    if dist < distoutside {
                        pointset[point_end] = point;
                        point_end += 1;
                    } else {
                        self.num_outside += 1;
                        if bestpoint == NULL {
                            bestpoint = point;
                            bestdist = dist;
                        } else if dist > bestdist {
                            outside.push(bestpoint);
                            bestpoint = point;
                            bestdist = dist;
                        } else {
                            outside.push(point);
                        }
                    }
                }
            }
            if bestpoint != NULL {
                outside.push(bestpoint);
                self.facets[facet].furthestdist = bestdist;
                self.facets[facet].outsideset = Some(outside);
            } else {
                self.facets[facet].outsideset = None;
            }
            pointset.truncate(point_end);
        }
        self.findbestnew = true;
        let n = pointset.len();
        for i in 0..n {
            let point = pointset[i];
            if point != NULL {
                let fl = self.facet_list;
                self.partitionpoint(point, fl)?;
            }
        }
        self.findbestnew = false;
        Ok(())
    }

    /// `qh_furthestnext` (option `Q9`): the facet with the furthest outside point goes next.
    fn furthestnext(&mut self) {
        let mut bestfacet = NULL;
        let mut bestdist = -REALMAX;
        for f in self.facets_from(self.facet_list) {
            if self.facets[f].outsideset.is_some() {
                let dist = self.facets[f].furthestdist;
                if dist > bestdist {
                    bestfacet = f;
                    bestdist = dist;
                }
            }
        }
        if bestfacet != NULL {
            self.removefacet(bestfacet);
            self.prependfacet(bestfacet, Head::FacetNext);
        }
    }

    /// `qh_furthestout` — whose `qh_setdel(outsideset, point)` passes the loop variable, NULL by then, so it
    /// only refreshes `furthestdist`.
    fn furthestout(&mut self, facet: usize) {
        let mut bestpoint = NULL;
        let mut bestdist = -REALMAX;
        for &p in self.facets[facet].outsideset.clone().unwrap_or_default().iter() {
            let dist = self.distplane(&self.points[p], facet);
            if dist > bestdist {
                bestpoint = p;
                bestdist = dist;
            }
        }
        if bestpoint != NULL {
            self.facets[facet].furthestdist = bestdist;
        }
        self.facets[facet].notfurthest = false;
    }

    /// `qh_nextfurthest`: the next point to add, and the facet it is above.
    fn nextfurthest(&mut self) -> Option<(usize, usize)> {
        loop {
            let mut facet = self.facet_next;
            if facet == self.facet_tail || facet == NULL {
                return None;
            }
            let size = match &self.facets[facet].outsideset {
                None => {
                    self.facet_next = self.facets[facet].next;
                    continue;
                }
                Some(s) => s.len(),
            };
            if size == 0 {
                self.facets[facet].outsideset = None;
                self.facet_next = self.facets[facet].next;
                continue;
            }
            if self.narrowhull {
                if self.facets[facet].notfurthest {
                    self.furthestout(facet);
                }
                let dist = self.facets[facet].furthestdist;
                if dist < self.min_outside {
                    self.facet_next = self.facets[facet].next;
                    continue;
                }
            }
            if self.pickfurthest {
                self.furthestnext();
                facet = self.facet_next;
            }
            let p = set_dellast(self.facets[facet].outsideset.as_mut().unwrap());
            return if p == NULL { None } else { Some((p, facet)) };
        }
    }

    /// `qh_buildhull`.
    fn buildhull(&mut self) -> Result<(), String> {
        self.facet_next = self.facet_list;
        while let Some((furthest, facet)) = self.nextfurthest() {
            self.num_outside -= 1;
            if self.stopadd > 0 && self.num_vertices - DIM as i64 > self.stopadd - 1 {
                return Ok(());
            }
            self.addpoint(furthest, facet)?;
        }
        Ok(())
    }

    /// `qh_addpoint(furthest, facet, checkdist = False)`.
    fn addpoint(&mut self, furthest: usize, facet: usize) -> Result<(), String> {
        self.detmaxoutside();
        self.findhorizon(furthest, facet)?;
        let apex = self.buildcone(furthest)?;
        self.findbestnew = false;
        self.initmergesets();
        self.premerge(apex)?;
        if self.use_findbestnew() {
            self.findbestnew = true;
        } else {
            for nf in self.facets_from(self.newfacet_list) {
                if !self.facets[nf].simplicial {
                    self.findbestnew = true;
                    break;
                }
            }
        }
        self.partitionvisible()?;
        self.findbestnew = false;
        self.findbest_notsharp = false;
        self.deletevisible();
        self.newfacets = false;
        self.resetlists(true);
        if self.facet_mergeset.is_some() {
            self.all_vertexmerges()?;
            self.freemergesets();
        }
        Ok(())
    }

    /// `qh_findhorizon`: the facets visible from `point`, moved to the end of the list as `visible_list`.
    fn findhorizon(&mut self, point: usize, facet: usize) -> Result<(), String> {
        let p = self.points[point];
        self.removefacet(facet);
        self.appendfacet(facet);
        self.num_visible = 1;
        self.visible_list = facet;
        self.facets[facet].visible = true;
        self.facets[facet].f = NULL;
        self.visit_id += 1;
        let mut numhorizon = 0;
        let mut visible = self.visible_list;
        while visible != NULL && self.facets[visible].visible {
            if self.facets[visible].neighbors.is_empty() {
                return Err("qhull internal error (qh_findhorizon): visible facet has no neighbors".into());
            }
            self.facets[visible].visitid = self.visit_id;
            let mut i = 0;
            loop {
                let neighbor = at(&self.facets[visible].neighbors, i);
                if neighbor == NULL {
                    break;
                }
                i += 1;
                if self.facets[neighbor].visitid == self.visit_id {
                    continue;
                }
                self.facets[neighbor].visitid = self.visit_id;
                let dist = self.distplane(&p, neighbor);
                if dist > self.min_visible {
                    self.removefacet(neighbor);
                    self.appendfacet(neighbor);
                    self.facets[neighbor].visible = true;
                    self.facets[neighbor].f = NULL;
                    self.num_visible += 1;
                } else {
                    if dist >= -self.max_coplanar {
                        self.facets[neighbor].coplanarhorizon = true;
                        if dist > 0.0 {
                            if self.max_outside < dist {
                                self.max_outside = dist;
                            }
                            if self.max_vertex < dist {
                                self.max_vertex = dist;
                            }
                            if self.facets[neighbor].maxoutside < dist {
                                self.facets[neighbor].maxoutside = dist;
                            }
                        } else if self.min_vertex > dist {
                            self.min_vertex = dist;
                        }
                    } else {
                        self.facets[neighbor].coplanarhorizon = false;
                    }
                    numhorizon += 1;
                }
            }
            visible = self.facets[visible].next;
        }
        if numhorizon == 0 {
            return Err("qhull precision error (qh_findhorizon): empty horizon".into());
        }
        Ok(())
    }

    /// `qh_buildcone` without `Q14`/`Qg`: new facets, matched, with planes, vertex neighbours updated.
    fn buildcone(&mut self, furthest: usize) -> Result<usize, String> {
        self.first_newfacet = self.facet_id;
        let apex = self.makenewfacets(furthest)?;
        self.matchnewfacets()?;
        self.makenewplanes();
        self.update_vertexneighbors_cone();
        Ok(apex)
    }

    /// `qh_makenewfacets`.
    fn makenewfacets(&mut self, point: usize) -> Result<usize, String> {
        self.newfacet_list = self.facet_tail;
        self.newvertex_list = self.vertex_tail;
        let apex = self.newvertex(point);
        self.appendvertex(apex);
        self.visit_id += 1;
        let (mut newfacet, mut newfacet2) = (NULL, NULL);
        let mut visible = self.visible_list;
        while visible != NULL && self.facets[visible].visible {
            for nb in self.facets[visible].neighbors.clone() {
                if nb == NULL {
                    break;
                }
                self.facets[nb].seen = false;
            }
            if self.facets[visible].ridges.is_some() {
                self.facets[visible].visitid = self.visit_id;
                newfacet2 = self.makenew_nonsimplicial(visible, apex)?;
            }
            if self.facets[visible].simplicial {
                newfacet = self.makenew_simplicial(visible, apex)?;
            }
            if newfacet2 != NULL {
                newfacet = newfacet2;
            }
            if newfacet != NULL {
                self.facets[visible].f = newfacet;
            }
            if let Some(x) = self.facets[visible].ridges.as_mut().and_then(|r| r.first_mut()) {
                *x = NULL;
            }
            if let Some(x) = self.facets[visible].neighbors.first_mut() {
                *x = NULL;
            }
            visible = self.facets[visible].next;
        }
        self.newfacets = true;
        Ok(apex)
    }

    /// `qh_makenewfacet`.
    fn makenewfacet(&mut self, vertices: Vec<usize>, toporient: bool, horizon: usize) -> usize {
        for &v in &vertices {
            if v == NULL {
                break;
            }
            if !self.vertices[v].newfacet {
                self.removevertex(v);
                self.appendvertex(v);
            }
        }
        let nf = self.newfacet();
        self.facets[nf].vertices = vertices;
        if toporient {
            self.facets[nf].toporient = true;
        }
        if horizon != NULL {
            set_append(&mut self.facets[nf].neighbors, horizon);
        }
        self.appendfacet(nf);
        nf
    }

    /// `qh_facetintersect(facetA, facetB, &skipA, &skipB, prepend = 1)`.
    fn facetintersect(&self, a: usize, b: usize) -> Result<(Vec<usize>, usize, usize), String> {
        let na = &self.facets[a].neighbors;
        let nb = &self.facets[b].neighbors;
        let skip_a = (0..DIM).find(|&i| at(na, i) == b);
        let skip_b = (0..DIM).find(|&i| at(nb, i) == a);
        match (skip_a, skip_b) {
            (Some(sa), Some(sb)) => Ok((set_new_delnthsorted(&self.facets[a].vertices, DIM, sa, 1), sa, sb)),
            _ => Err("qhull internal error (qh_facetintersect): facets are not neighbors".into()),
        }
    }

    /// `qh_makenew_simplicial`.
    fn makenew_simplicial(&mut self, visible: usize, apex: usize) -> Result<usize, String> {
        let mut newfacet = NULL;
        let nbs = self.facets[visible].neighbors.clone();
        for &neighbor in &nbs {
            if neighbor == NULL {
                break;
            }
            if !self.facets[neighbor].seen && !self.facets[neighbor].visible {
                let (mut vertices, horizonskip, _visibleskip) = self.facetintersect(neighbor, visible)?;
                vertices[0] = apex;
                let toporient = if self.facets[neighbor].toporient { horizonskip & 1 == 1 } else { horizonskip & 1 == 0 };
                newfacet = self.makenewfacet(vertices, toporient, neighbor);
                if self.facets[neighbor].coplanarhorizon {
                    self.facets[newfacet].f = newfacet;
                    self.facets[newfacet].mergehorizon = true;
                }
                self.facets[neighbor].neighbors[horizonskip] = newfacet;
            }
        }
        Ok(newfacet)
    }

    /// `qh_makenew_nonsimplicial` (not `NEWtentative`).
    fn makenew_nonsimplicial(&mut self, visible: usize, apex: usize) -> Result<usize, String> {
        let mut newfacet = NULL;
        let ridges = self.facets[visible].ridges.clone().unwrap_or_default();
        for &ridge in &ridges {
            if ridge == NULL {
                break;
            }
            let neighbor = self.otherfacet(ridge, visible);
            if self.facets[neighbor].visible {
                // a ridge between two visible facets is freed by the second of them
            } else {
                let toporient = self.ridges[ridge].top == visible;
                let mut vertices = vec![apex];
                vertices.extend(self.ridges[ridge].vertices.iter().copied());
                newfacet = self.makenewfacet(vertices, toporient, neighbor);
                if self.facets[neighbor].coplanarhorizon {
                    self.facets[newfacet].mergehorizon = true;
                    if !self.facets[neighbor].seen {
                        self.facets[newfacet].f = newfacet;
                        self.facets[neighbor].f = newfacet;
                    } else {
                        let samecycle = self.facets[neighbor].f;
                        self.facets[newfacet].f = self.facets[samecycle].f;
                        self.facets[samecycle].f = newfacet;
                    }
                }
                if self.facets[neighbor].seen {
                    if self.facets[neighbor].simplicial {
                        return Err("qhull internal error (qh_makenew_nonsimplicial): simplicial horizon with duplicate ridges".into());
                    }
                    set_append(&mut self.facets[neighbor].neighbors, newfacet);
                } else {
                    set_replace(&mut self.facets[neighbor].neighbors, visible, newfacet)?;
                }
                if self.facets[neighbor].simplicial {
                    if let Some(r) = self.facets[neighbor].ridges.as_mut() {
                        set_del(r, ridge);
                    }
                } else {
                    self.facets[newfacet].ridges.get_or_insert_with(Vec::new).push(ridge);
                    if toporient {
                        self.ridges[ridge].top = newfacet;
                        self.ridges[ridge].simplicialtop = true;
                    } else {
                        self.ridges[ridge].bottom = newfacet;
                        self.ridges[ridge].simplicialbot = true;
                    }
                }
            }
            self.facets[neighbor].seen = true;
        }
        Ok(newfacet)
    }

    /// `qh_matchnewfacets`: pairs the new facets' ridges through a hash table of their vertex sets.
    fn matchnewfacets(&mut self) -> Result<(), String> {
        let news = self.facets_from(self.newfacet_list);
        for &nf in &news {
            let n = &mut self.facets[nf].neighbors;
            n.resize(DIM, NULL);
            for x in n.iter_mut().skip(1) {
                *x = NULL;
            }
        }
        let numnew = news.len();
        let hashsize = newhashtable_size(numnew * (DIM - 1));
        self.hash_table = vec![NULL; hashsize];
        let mut hashcount: i64 = 0;
        for &nf in &news {
            if !self.facets[nf].simplicial {
                return Err("qhull internal error (qh_matchnewfacets): new facet is not simplicial".into());
            }
            for newskip in 1..DIM {
                self.matchneighbor(nf, newskip, hashsize, &mut hashcount)?;
            }
        }
        if hashcount != 0 {
            return Err("qhull: duplicate ridges (qh_matchdupridge) — not ported".into());
        }
        self.hash_table.clear();
        Ok(())
    }

    /// `qh_gethash` over vertex ids rather than addresses (the table only finds candidates).
    fn gethash(&self, hashsize: usize, set: &[usize], firstindex: usize, skipelem: usize) -> usize {
        let mut hash: u64 = 0;
        for &v in &set[firstindex..DIM] {
            hash = hash.wrapping_add(self.vertices[v].id as u64);
        }
        hash = hash.wrapping_sub(self.vertices[skipelem].id as u64);
        (hash as u32 as usize) % hashsize
    }

    /// `qh_matchneighbor`.
    fn matchneighbor(&mut self, newfacet: usize, newskip: usize, hashsize: usize, hashcount: &mut i64) -> Result<(), String> {
        let hash = self.gethash(hashsize, &self.facets[newfacet].vertices, 1, self.facets[newfacet].vertices[newskip]);
        let mut newfound = false;
        let mut scan = hash;
        loop {
            let facet = self.hash_table[scan];
            if facet == NULL {
                break;
            }
            if facet == newfacet {
                newfound = true;
            } else if let Some((skip, same)) = matchvertices(&self.facets[newfacet].vertices, newskip, &self.facets[facet].vertices) {
                if self.facets[newfacet].vertices[newskip] == self.facets[facet].vertices[skip] {
                    return Err("qhull precision error (qh_matchneighbor): two new facets with the same vertices".into());
                }
                let ismatch = same == (self.facets[newfacet].toporient ^ self.facets[facet].toporient);
                let matchfacet = at(&self.facets[facet].neighbors, skip);
                if ismatch && matchfacet == NULL {
                    self.facets[facet].neighbors[skip] = newfacet;
                    self.facets[newfacet].neighbors[newskip] = facet;
                    *hashcount -= 1;
                    return Ok(());
                }
                return Err("qhull: a ridge with more than two neighbors (dupridge) — not ported".into());
            }
            scan += 1;
            if scan >= hashsize {
                scan = 0;
            }
        }
        if !newfound {
            self.hash_table[scan] = newfacet;
        }
        *hashcount += 1;
        Ok(())
    }

    /// `qh_makenewplanes`.
    fn makenewplanes(&mut self) {
        for nf in self.facets_from(self.newfacet_list) {
            if !self.facets[nf].mergehorizon {
                self.setfacetplane(nf);
            }
        }
    }

    /// `qh_update_vertexneighbors_cone`.
    fn update_vertexneighbors_cone(&mut self) {
        if self.vertexneighbors {
            let mut v = self.newvertex_list;
            while v != NULL && self.vertices[v].next != NULL {
                let mut i = 0;
                loop {
                    let nb = at(&self.vertices[v].neighbors, i);
                    if nb == NULL {
                        break;
                    }
                    if self.facets[nb].visible {
                        // qh_setdelnth(SETindex_) then neighborp--: revisit the slot
                        self.vertices[v].neighbors.swap_remove(i);
                        continue;
                    }
                    i += 1;
                }
                v = self.vertices[v].next;
            }
            for nf in self.facets_from(self.newfacet_list) {
                for v in self.facets[nf].vertices.clone() {
                    if v == NULL {
                        break;
                    }
                    set_append(&mut self.vertices[v].neighbors, nf);
                }
            }
            self.delete_interior_vertices(true);
        } else {
            self.delete_interior_vertices(false);
        }
    }

    /// The last loop of `qh_update_vertexneighbors[_cone]`: a visible facet's old vertex with no other
    /// facet left goes to `del_vertices`.
    fn delete_interior_vertices(&mut self, with_neighbors: bool) {
        let mut visible = self.visible_list;
        while visible != NULL && self.facets[visible].visible {
            for v in self.facets[visible].vertices.clone() {
                if v == NULL {
                    break;
                }
                if !self.vertices[v].newfacet && !self.vertices[v].deleted {
                    if with_neighbors {
                        let keep = self.vertices[v].neighbors.iter().take_while(|&&n| n != NULL).any(|&n| !self.facets[n].visible);
                        if keep {
                            set_del(&mut self.vertices[v].neighbors, visible);
                        } else {
                            self.vertices[v].deleted = true;
                            self.del_vertices.push(v);
                        }
                    } else {
                        self.vertices[v].deleted = true;
                        self.del_vertices.push(v);
                    }
                }
            }
            visible = self.facets[visible].next;
        }
    }

    /// `qh_update_vertexneighbors` (after `qh_triangulate_facet`, and in `qh_all_vertexmerges`).
    fn update_vertexneighbors(&mut self) {
        if self.vertexneighbors {
            let mut v = self.newvertex_list;
            while v != NULL && self.vertices[v].next != NULL {
                let mut count = 0;
                for x in self.vertices[v].neighbors.iter_mut() {
                    if *x == NULL {
                        break;
                    }
                    if self.facets[*x].visible {
                        count += 1;
                        *x = NULL;
                    }
                }
                if count > 0 {
                    self.vertices[v].neighbors.retain(|&x| x != NULL);
                }
                v = self.vertices[v].next;
            }
            for nf in self.facets_from(self.newfacet_list) {
                let fresh = self.first_newfacet != 0 && self.facets[nf].id >= self.first_newfacet;
                for v in self.facets[nf].vertices.clone() {
                    if v == NULL {
                        break;
                    }
                    if fresh {
                        set_append(&mut self.vertices[v].neighbors, nf);
                    } else {
                        set_unique(&mut self.vertices[v].neighbors, nf);
                    }
                }
            }
            self.delete_interior_vertices(true);
        } else {
            self.delete_interior_vertices(false);
        }
    }
}

/// `qh_newhashtable`'s size.
fn newhashtable_size(newsize: usize) -> usize {
    let mut size = ((newsize + 1) * 2) | 1;
    while size.is_multiple_of(3) || size.is_multiple_of(5) {
        size += 2;
    }
    size
}

/// `qh_matchvertices(firstindex = 1, A, skipA, B)`: `Some((skipB, same))` when A without `skipA` is B
/// without one vertex.
fn matchvertices(a: &[usize], skip_a: usize, b: &[usize]) -> Option<(usize, bool)> {
    let mut ia = 1usize;
    let mut ib = 1usize;
    let mut skip_bp: Option<usize> = None;
    loop {
        if ia != skip_a {
            loop {
                let eb = at(b, ib);
                ib += 1;
                if at(a, ia) == eb {
                    break;
                }
                if skip_bp.is_some() {
                    return None;
                }
                skip_bp = Some(ib);
            }
        }
        ia += 1;
        if at(a, ia) == NULL {
            break;
        }
    }
    let sbp = match skip_bp {
        Some(x) => x,
        None => {
            ib += 1;
            ib
        }
    };
    let skip_b = sbp - 1;
    let same = ((skip_a & 1) ^ (skip_b & 1)) == 0;
    Some((skip_b, same))
}

impl Qh {
    // ------------------------------------------------------------------------------- partitioning

    /// `qh_partitionvisible(allpoints = False)`.
    fn partitionvisible(&mut self) -> Result<(), String> {
        let mut visible = self.visible_list;
        while visible != NULL && self.facets[visible].visible {
            let next = self.facets[visible].next;
            if self.facets[visible].outsideset.is_none() && self.facets[visible].coplanarset.is_none() {
                visible = next;
                continue;
            }
            let mut newfacet = self.getreplacement(visible);
            if newfacet == NULL {
                newfacet = self.newfacet_list;
            }
            if self.facets[newfacet].next == NULL {
                return Err("qhull topology error (qh_partitionvisible): all new facets deleted as degenerate".into());
            }
            if let Some(outside) = self.facets[visible].outsideset.clone() {
                self.num_outside -= outside.len() as i64;
                for p in outside {
                    if p == NULL {
                        break;
                    }
                    self.partitionpoint(p, newfacet)?;
                }
            }
            if let Some(coplanar) = self.facets[visible].coplanarset.clone() {
                for p in coplanar {
                    if p == NULL {
                        break;
                    }
                    let fb = self.findbestnew;
                    self.partitioncoplanar(p, newfacet, None, fb)?;
                }
            }
            visible = self.facets[visible].next;
        }
        if !self.del_vertices.is_empty() {
            for v in self.del_vertices.clone() {
                let point = self.vertices[v].point;
                if point != NULL && !self.vertices[v].partitioned {
                    if self.newfacet_list == NULL || self.newfacet_list == self.facet_tail {
                        return Err("qhull internal error (qh_partitionvisible): all new facets deleted or none defined".into());
                    }
                    let nl = self.newfacet_list;
                    self.partitioncoplanar(point, nl, None, true)?;
                    self.vertices[v].partitioned = true;
                }
            }
        }
        Ok(())
    }

    /// `qh_partitionpoint`.
    fn partitionpoint(&mut self, point: usize, facet: usize) -> Result<(), String> {
        let (bestfacet, bestdist, mut isoutside) = if self.findbestnew {
            self.findbestnew_(point, facet, false)?
        } else {
            self.findbest(point, facet, false, true, false)?
        };
        if self.facets[bestfacet].visible {
            return Err("qhull internal error (qh_partitionpoint): partitioned into a visible facet".into());
        }
        if self.narrowhull && bestdist >= -self.near_inside {
            isoutside = true;
        }
        if isoutside {
            let mut isnewoutside = false;
            let empty = match &self.facets[bestfacet].outsideset {
                None => true,
                Some(s) => s.last().copied().unwrap_or(NULL) == NULL,
            };
            if empty {
                set_append_opt(&mut self.facets[bestfacet].outsideset, point);
                if !self.narrowhull || bestdist > self.min_outside {
                    isnewoutside = true;
                }
                self.facets[bestfacet].furthestdist = bestdist;
            } else {
                let previousdist = self.facets[bestfacet].furthestdist;
                if previousdist < bestdist {
                    set_append_opt(&mut self.facets[bestfacet].outsideset, point);
                    self.facets[bestfacet].furthestdist = bestdist;
                    if self.narrowhull && previousdist < self.min_outside && bestdist >= self.min_outside {
                        isnewoutside = true;
                    }
                } else {
                    let s = self.facets[bestfacet].outsideset.as_mut().unwrap();
                    let n = s.len();
                    s.insert(n - 1, point);
                }
            }
            if isnewoutside && self.facet_next != bestfacet {
                if self.facets[bestfacet].newfacet {
                    if self.facets[self.facet_next].newfacet {
                        self.facet_next = self.newfacet_list;
                    }
                } else {
                    self.removefacet(bestfacet);
                    self.appendfacet(bestfacet);
                    if self.newfacet_list != NULL {
                        self.facets[bestfacet].newfacet = true;
                    }
                }
            }
            self.num_outside += 1;
        } else if bestdist >= -self.max_coplanar {
            // KEEPnearinside: always kept
            let fb = self.findbestnew;
            self.partitioncoplanar(point, bestfacet, Some(bestdist), fb)?;
        } else if bestdist >= -self.near_inside {
            let fb = self.findbestnew;
            self.partitioncoplanar(point, bestfacet, Some(bestdist), fb)?;
        }
        Ok(())
    }

    /// `qh_partitioncoplanar`.
    fn partitioncoplanar(&mut self, point: usize, facet: usize, dist: Option<f64>, allnew: bool) -> Result<(), String> {
        let (bestfacet, bestdist) = match dist {
            None => {
                let (bf, bd, _) = if allnew { self.findbestnew_(point, facet, true)? } else { self.findbest(point, facet, true, false, false)? };
                // KEEPnearinside
                if bd < -self.near_inside {
                    self.repart_facetid = 0;
                    return Ok(());
                }
                (bf, bd)
            }
            Some(d) => (facet, d),
        };
        if self.facets[bestfacet].visible {
            return Err("qhull internal error (qh_partitioncoplanar): partitioned into a visible facet".into());
        }
        if bestdist > self.max_outside {
            let mut repartition = false;
            if dist.is_none() && facet != bestfacet {
                let angle = Self::getangle(self.facets[facet].normal.as_ref().unwrap(), self.facets[bestfacet].normal.as_ref().unwrap());
                if angle < 0.0 {
                    repartition = true;
                }
            }
            if !repartition && bestdist > self.max_outside_limit * QH_RATIOCOPLANAROUTSIDE {
                if self.facets[facet].id == self.facets[bestfacet].id {
                    if self.facets[facet].id == self.repart_facetid {
                        return Err("qhull internal error (qh_partitioncoplanar): infinite loop repartitioning a point".into());
                    }
                    self.repart_facetid = self.facets[facet].id;
                }
                repartition = true;
            }
            if repartition {
                let oldfindbest = self.findbestnew;
                self.findbestnew = false;
                self.partitionpoint(point, bestfacet)?;
                self.findbestnew = oldfindbest;
                self.repart_facetid = 0;
                return Ok(());
            }
            self.repart_facetid = 0;
            self.max_outside = bestdist;
        }
        // KEEPnearinside: the coplanar set keeps its furthest point last
        let oldfurthest = self.facets[bestfacet].coplanarset.as_ref().and_then(|s| s.last().copied()).unwrap_or(NULL);
        let mut dist2 = 0.0;
        if oldfurthest != NULL {
            dist2 = self.distplane(&self.points[oldfurthest], bestfacet);
        }
        if oldfurthest == NULL || dist2 < bestdist {
            set_append_opt(&mut self.facets[bestfacet].coplanarset, point);
        } else {
            let s = self.facets[bestfacet].coplanarset.as_mut().unwrap();
            let n = s.len();
            s.insert(n - 1, point);
        }
        Ok(())
    }

    /// `qh_findbest(point, startfacet, bestoutside, isnewfacets, noupper)`: `(facet, dist, isoutside)`.
    fn findbest(&mut self, point: usize, startfacet: usize, bestoutside: bool, isnewfacets: bool, _noupper: bool) -> Result<(usize, f64, bool), String> {
        let p = self.points[point];
        let mut bestdist = -REALMAX / 2.0;
        let mut bestfacet = NULL;
        let mut lastfacet;
        self.visit_id += 1;
        let visitid = self.visit_id;
        let mut testhorizon = true;
        let mut isoutside = true;
        if !self.facets[startfacet].flipped {
            let dist = self.distplane(&p, startfacet);
            if !bestoutside && dist >= self.min_outside {
                return Ok((startfacet, dist, true));
            }
            bestdist = dist;
            bestfacet = startfacet;
        }
        self.facets[startfacet].visitid = visitid;
        let mut facet = startfacet;
        while facet != NULL {
            lastfacet = facet;
            let mut next = NULL;
            let mut i = 0;
            loop {
                let neighbor = at(&self.facets[facet].neighbors, i);
                if neighbor == NULL {
                    break;
                }
                i += 1;
                if !self.facets[neighbor].newfacet && isnewfacets {
                    continue;
                }
                if self.facets[neighbor].visitid == visitid {
                    continue;
                }
                self.facets[neighbor].visitid = visitid;
                if !self.facets[neighbor].flipped {
                    let dist = self.distplane(&p, neighbor);
                    if dist > bestdist {
                        if !bestoutside && dist >= self.min_outside {
                            return Ok((neighbor, dist, true));
                        }
                        bestfacet = neighbor;
                        bestdist = dist;
                        next = neighbor;
                        break;
                    }
                }
            }
            facet = next;
            if facet == NULL {
                let _ = lastfacet;
                if isnewfacets {
                    if bestfacet == NULL {
                        let nl = self.newfacet_list;
                        let (bf, bd, io) = self.findbestnew_(point, nl, bestoutside)?;
                        return Ok((bf, bd, io));
                    } else if !self.findbest_notsharp && bestdist < -self.distround {
                        if self.sharpnewfacets() {
                            let (bf, bd, io) = self.findbestnew_(point, bestfacet, bestoutside)?;
                            self.findbestnew = true;
                            return Ok((bf, bd, io));
                        } else {
                            self.findbest_notsharp = true;
                        }
                    }
                }
                if bestfacet == NULL {
                    return Err("qhull: qh_findbestlower on a facet with no unflipped neighbor — not ported".into());
                }
                if testhorizon {
                    let (bf, bd) = self.findbesthorizon(&p, bestfacet, bestdist);
                    bestfacet = bf;
                    bestdist = bd;
                    testhorizon = false;
                }
                let _ = testhorizon;
                if bestdist < self.min_outside {
                    isoutside = false;
                }
                return Ok((bestfacet, bestdist, isoutside));
            }
        }
        unreachable!()
    }

    /// `qh_findbestnew`: every new facet (starting at `startfacet`), then the horizon.
    fn findbestnew_(&mut self, point: usize, startfacet: usize, bestoutside: bool) -> Result<(usize, f64, bool), String> {
        let p = self.points[point];
        let mut bestdist = -REALMAX / 2.0;
        let mut bestfacet = NULL;
        self.visit_id += 1;
        let visitid = self.visit_id;
        if startfacet == NULL || self.facets[startfacet].next == NULL {
            return Err("qhull precision error (qh_findbestnew): merging has formed and deleted a cone of new facets".into());
        }
        let isdistoutside = !bestoutside;
        let distoutside = if isdistoutside { self.dist_outside() } else { 0.0 };
        let mut start = startfacet;
        for i in 0..2 {
            let mut facet = start;
            while facet != NULL && self.facets[facet].next != NULL {
                if facet == startfacet && i == 1 {
                    break;
                }
                self.facets[facet].visitid = visitid;
                if !self.facets[facet].flipped {
                    let dist = self.distplane(&p, facet);
                    if dist > bestdist {
                        bestfacet = facet;
                        if isdistoutside && dist >= distoutside {
                            return Ok((bestfacet, dist, true));
                        }
                        bestdist = dist;
                    }
                }
                facet = self.facets[facet].next;
            }
            start = self.newfacet_list;
        }
        let from = if bestfacet != NULL { bestfacet } else { startfacet };
        let (bf, bd) = self.findbesthorizon(&p, from, bestdist);
        Ok((bf, bd, bd >= self.min_outside))
    }

    /// `qh_findbesthorizon(ischeckmax = False)`: walks coplanar-ish neighbours for a better facet.
    fn findbesthorizon(&mut self, point: &[f64; 3], startfacet: usize, bestdist_in: f64) -> (usize, f64) {
        let mut bestfacet = startfacet;
        let mut bestdist = bestdist_in;
        self.visit_id += 1;
        let visitid = self.visit_id;
        let searchdist = self.search_dist();
        let mut minsearch = bestdist - searchdist;
        let mut coplanarfacetset_size = 0usize;
        self.facets[startfacet].visitid = visitid;
        let mut facet = startfacet;
        let mut nextfacet = NULL;
        loop {
            let mut i = 0;
            loop {
                let neighbor = at(&self.facets[facet].neighbors, i);
                if neighbor == NULL {
                    break;
                }
                i += 1;
                if self.facets[neighbor].visitid == visitid {
                    continue;
                }
                self.facets[neighbor].visitid = visitid;
                if !self.facets[neighbor].flipped {
                    let dist = self.distplane(point, neighbor);
                    if dist > bestdist {
                        minsearch = dist - searchdist;
                        if dist > bestdist + searchdist {
                            coplanarfacetset_size = 0;
                        }
                        bestfacet = neighbor;
                        bestdist = dist;
                    } else if dist < minsearch {
                        continue;
                    }
                }
                if nextfacet != NULL {
                    let first = coplanarfacetset_size == 0;
                    coplanarfacetset_size += 1;
                    if first {
                        if self.coplanarfacetset.is_empty() {
                            self.coplanarfacetset.push(nextfacet);
                        } else {
                            self.coplanarfacetset[0] = nextfacet;
                        }
                        self.coplanarfacetset.truncate(1);
                    } else {
                        self.coplanarfacetset.push(nextfacet);
                    }
                }
                nextfacet = neighbor;
            }
            facet = nextfacet;
            if facet != NULL {
                nextfacet = NULL;
            } else if coplanarfacetset_size == 0 {
                break;
            } else {
                coplanarfacetset_size -= 1;
                if coplanarfacetset_size == 0 {
                    facet = self.coplanarfacetset[0];
                    self.coplanarfacetset.clear();
                } else {
                    facet = set_dellast(&mut self.coplanarfacetset);
                }
            }
        }
        (bestfacet, bestdist)
    }

    /// `qh_sharpnewfacets`: do the new facets' normals point into more than one orthant?
    fn sharpnewfacets(&self) -> bool {
        let mut quadrant = [false; 3];
        let first = self.newfacet_list;
        for f in self.facets_from(self.newfacet_list) {
            let n = self.facets[f].normal.unwrap();
            if f == first {
                for k in 0..DIM {
                    quadrant[k] = n[k] > 0.0;
                }
            } else {
                for k in 0..DIM {
                    if quadrant[k] != (n[k] > 0.0) {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl Qh {
    // ------------------------------------------------------------------------------------ merging

    fn initmergesets(&mut self) {
        self.degen_mergeset = Some(Vec::new());
        self.vertex_mergeset = Some(Vec::new());
        self.facet_mergeset = Some(Vec::new());
    }

    fn freemergesets(&mut self) {
        self.facet_mergeset = None;
        self.degen_mergeset = None;
        self.vertex_mergeset = None;
    }

    /// `qh_premerge`: merge the cone of new facets until every ridge is clearly convex.
    fn premerge(&mut self, _apex: usize) -> Result<(), String> {
        if self.checkzero() {
            return Ok(());
        }
        self.centrum_radius = self.premerge_centrum;
        self.cos_max = self.premerge_cos;
        let mut othermerge = false;
        // qh_mark_dupridges: no dupridges reach here, so it only clears the flags
        for f in self.facets_from(self.newfacet_list) {
            self.facets[f].mergeridge2 = false;
            self.facets[f].mergeridge = false;
        }
        self.mergecycle_all(&mut othermerge)?;
        // qh_forcedmerges: with no MRGdupridge merges it hands the same merges to a fresh set, in order
        self.flippedmerges(&mut othermerge)?;
        self.postmerging = false;
        self.getmergeset_initial()?;
        self.all_merges(othermerge)
    }

    /// `qh_checkzero(testall = False)`: are the new facets clearly convex to their neighbours and horizon?
    fn checkzero(&mut self) -> bool {
        let news = self.facets_from(self.newfacet_list);
        for &f in &news {
            let horizon = at(&self.facets[f].neighbors, 0);
            if !self.facets[horizon].simplicial || self.facets[f].flipped || self.facets[f].dupridge || self.facets[f].normal.is_none() {
                self.zeroall_ok = false;
                return false;
            }
        }
        for &f in &news {
            self.vertex_visit += 1;
            let mut horizon = NULL;
            let n = self.facets[f].neighbors.len();
            for i in 0..n {
                let neighbor = self.facets[f].neighbors[i];
                if i == 0 {
                    horizon = neighbor;
                    continue;
                }
                let vertex = self.facets[f].vertices[i];
                self.vertices[vertex].visitid = self.vertex_visit;
                let dist = self.distplane(&self.points[self.vertices[vertex].point], neighbor);
                if dist >= -2.0 * self.distround {
                    self.zeroall_ok = false;
                    return false;
                }
            }
            if horizon != NULL {
                for &v in &self.facets[horizon].vertices.clone() {
                    if v == NULL {
                        break;
                    }
                    if self.vertices[v].visitid != self.vertex_visit {
                        let dist = self.distplane(&self.points[self.vertices[v].point], f);
                        if dist >= -2.0 * self.distround {
                            self.zeroall_ok = false;
                            return false;
                        }
                        break;
                    }
                }
            }
        }
        true
    }

    /// `qh_mergecycle_all`: merge each new facet coplanar with its horizon into that horizon facet.
    fn mergecycle_all(&mut self, wasmerge: &mut bool) -> Result<(), String> {
        let mut cycles = 0;
        let mut facet = self.newfacet_list;
        while facet != NULL {
            let nextfacet = self.facets[facet].next;
            if nextfacet == NULL {
                break;
            }
            if self.facets[facet].normal.is_some() {
                facet = nextfacet;
                continue;
            }
            if !self.facets[facet].mergehorizon {
                return Err("qhull internal error (qh_mergecycle_all): new facet without a normal is not mergehorizon".into());
            }
            let horizon = self.facets[facet].neighbors[0];
            if self.facets[facet].f == facet {
                let verts = self.facets[facet].vertices.clone();
                let apex = verts[0];
                for &v in &verts {
                    if v == NULL {
                        break;
                    }
                    if v != apex {
                        self.vertices[v].delridge = true;
                    }
                }
                self.facets[horizon].f = NULL;
                self.mergefacet(facet, horizon, MRG_COPLANARHORIZON, None, true)?;
            } else {
                return Err("qhull: a coplanar-horizon cycle of several facets (qh_mergecycle) — not ported".into());
            }
            cycles += 1;
            facet = nextfacet;
        }
        if cycles > 0 {
            let mut nf = self.newfacet_list;
            while nf != NULL && self.facets[nf].next != NULL {
                if self.facets[nf].coplanarhorizon {
                    self.test_redundant_neighbors(nf)?;
                    self.facets[nf].coplanarhorizon = false;
                }
                nf = self.facets[nf].next;
            }
            self.merge_degenredundant()?;
            *wasmerge = true;
        }
        Ok(())
    }

    /// `qh_flippedmerges`.
    fn flippedmerges(&mut self, wasmerge: &mut bool) -> Result<(), String> {
        for f in self.facets_from(self.newfacet_list) {
            if self.facets[f].flipped && !self.facets[f].visible {
                self.appendmergeset(f, f, MRG_FLIP, 0.0, 1.0)?;
            }
        }
        let othermerges = self.facet_mergeset.take().unwrap_or_default();
        self.facet_mergeset = Some(Vec::new());
        let mut nummerge = 0;
        for &m in &othermerges {
            let merge = self.merges[m];
            let facet1 = merge.facet1;
            if merge.mergetype != MRG_FLIP || self.facets[facet1].visible {
                continue;
            }
            let (neighbor, _dist, mindist, maxdist) = self.findbestneighbor(facet1)?;
            self.mergefacet(facet1, neighbor, merge.mergetype, Some((mindist, maxdist)), false)?;
            nummerge += 1;
        }
        for &m in &othermerges {
            let merge = self.merges[m];
            if !(self.facets[merge.facet1].visible || self.facets[merge.facet2].visible) {
                self.facet_mergeset.as_mut().unwrap().push(m);
            }
        }
        self.merge_degenredundant()?;
        if nummerge > 0 {
            *wasmerge = true;
        }
        Ok(())
    }

    /// `qh_compare_facetmerge` (never 0).
    fn compare_facetmerge(&self, a: usize, b: usize) -> i32 {
        let (a, b) = (&self.merges[a], &self.merges[b]);
        if a.mergetype != b.mergetype {
            if a.mergetype < b.mergetype {
                1
            } else {
                -1
            }
        } else if a.mergetype == MRG_ANGLECOPLANAR {
            if a.angle > b.angle {
                1
            } else {
                -1
            }
        } else if a.distance < b.distance {
            1
        } else {
            -1
        }
    }

    fn sort_facet_mergeset(&mut self) {
        let mut set = self.facet_mergeset.take().unwrap_or_default();
        apple_qsort(&mut set, &mut |a, b| self.compare_facetmerge(a, b));
        self.facet_mergeset = Some(set);
    }

    /// `qh_getmergeset_initial`.
    fn getmergeset_initial(&mut self) -> Result<(), String> {
        self.visit_id += 1;
        let mut facet = self.newfacet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            self.facets[facet].visitid = self.visit_id;
            let mut i = 0;
            loop {
                let neighbor = at(&self.facets[facet].neighbors, i);
                if neighbor == NULL {
                    break;
                }
                i += 1;
                if self.facets[neighbor].visitid != self.visit_id {
                    let simplicial = self.facets[facet].simplicial && self.facets[neighbor].simplicial;
                    if self.test_appendmerge(facet, neighbor, simplicial)? {
                        for r in self.facets[neighbor].ridges.clone().unwrap_or_default() {
                            if r == NULL {
                                break;
                            }
                            if facet == self.otherfacet(r, neighbor) {
                                self.ridges[r].nonconvex = true;
                                break;
                            }
                        }
                    }
                }
            }
            self.facets[facet].tested = true;
            for r in self.facets[facet].ridges.clone().unwrap_or_default() {
                if r == NULL {
                    break;
                }
                self.ridges[r].tested = true;
            }
            facet = self.facets[facet].next;
        }
        self.sort_facet_mergeset();
        Ok(())
    }

    /// `qh_getmergeset`.
    fn getmergeset(&mut self) -> Result<(), String> {
        self.visit_id += 1;
        let mut facet = self.newfacet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            if self.facets[facet].tested {
                facet = self.facets[facet].next;
                continue;
            }
            self.facets[facet].visitid = self.visit_id;
            for nb in self.facets[facet].neighbors.clone() {
                if nb == NULL {
                    break;
                }
                self.facets[nb].seen = false;
            }
            for r in self.facets[facet].ridges.clone().unwrap_or_default() {
                if r == NULL {
                    break;
                }
                if self.ridges[r].tested && !self.ridges[r].nonconvex {
                    continue;
                }
                let neighbor = self.otherfacet(r, facet);
                if self.facets[neighbor].seen {
                    self.ridges[r].tested = true;
                    self.ridges[r].nonconvex = false;
                } else if self.facets[neighbor].visitid != self.visit_id {
                    self.facets[neighbor].seen = true;
                    self.ridges[r].nonconvex = false;
                    let simplicial = self.ridges[r].simplicialbot && self.ridges[r].simplicialtop;
                    if self.test_appendmerge(facet, neighbor, simplicial)? {
                        self.ridges[r].nonconvex = true;
                    }
                    self.ridges[r].tested = true;
                }
            }
            self.facets[facet].tested = true;
            facet = self.facets[facet].next;
        }
        self.sort_facet_mergeset();
        Ok(())
    }

    /// `qh_test_appendmerge` (no angle test: `cos_max` is REALmax).
    fn test_appendmerge(&mut self, facet: usize, neighbor: usize, _simplicial: bool) -> Result<bool, String> {
        self.test_centrum_merge(facet, neighbor, -REALMAX)
    }

    /// `qh_test_centrum_merge`.
    fn test_centrum_merge(&mut self, facet: usize, neighbor: usize, angle: f64) -> Result<bool, String> {
        if self.facets[facet].center.is_none() {
            self.facets[facet].center = Some(self.getcentrum(facet));
        }
        let c = self.facets[facet].center.unwrap();
        let dist = self.distplane(&c, neighbor);
        let (mut isconcave, mut iscoplanar) = (false, false);
        if dist > self.centrum_radius {
            isconcave = true;
        } else if dist >= -self.centrum_radius {
            iscoplanar = true;
        }
        if self.facets[neighbor].center.is_none() {
            self.facets[neighbor].center = Some(self.getcentrum(neighbor));
        }
        let c2 = self.facets[neighbor].center.unwrap();
        let dist2 = self.distplane(&c2, facet);
        if dist2 > self.centrum_radius {
            isconcave = true;
        } else if !iscoplanar && dist2 >= -self.centrum_radius {
            iscoplanar = true;
        }
        if !isconcave && !iscoplanar {
            return Ok(false);
        }
        if isconcave && iscoplanar {
            if dist > dist2 {
                self.appendmergeset(facet, neighbor, MRG_CONCAVECOPLANAR, dist, angle)?;
            } else {
                self.appendmergeset(neighbor, facet, MRG_CONCAVECOPLANAR, dist2, angle)?;
            }
        } else if isconcave {
            let mergedist = if dist < dist2 { dist2 } else { dist };
            self.appendmergeset(facet, neighbor, MRG_CONCAVE, mergedist, angle)?;
        } else {
            let a = if dist < 0.0 { -dist } else { dist };
            let b = if dist2 < 0.0 { -dist2 } else { dist2 };
            let mergedist = if a > b { b } else { a };
            self.appendmergeset(facet, neighbor, MRG_COPLANAR, mergedist, angle)?;
        }
        Ok(true)
    }

    /// `qh_appendmergeset`.
    fn appendmergeset(&mut self, facet: usize, neighbor: usize, mergetype: u8, dist: f64, angle: f64) -> Result<(), String> {
        if (self.facets[facet].redundant && mergetype != MRG_MIRROR) || self.facets[neighbor].redundant {
            return Ok(());
        }
        if self.facets[facet].degenerate && mergetype == MRG_DEGEN {
            return Ok(());
        }
        if self.facets[neighbor].flipped && !self.facets[facet].flipped && mergetype != MRG_DUPRIDGE {
            return Err("qhull internal error (qh_appendmergeset): cannot merge into a flipped facet".into());
        }
        self.merges.push(Merge { angle, distance: dist, facet1: facet, facet2: neighbor, mergetype });
        let m = self.merges.len() - 1;
        if mergetype < MRG_DEGEN {
            self.facet_mergeset.as_mut().unwrap().push(m);
        } else if mergetype == MRG_DEGEN {
            self.facets[facet].degenerate = true;
            let last = self.degen_mergeset.as_ref().unwrap().last().copied().unwrap_or(NULL);
            if last == NULL || self.merges[last].mergetype == MRG_DEGEN {
                self.degen_mergeset.as_mut().unwrap().push(m);
            } else {
                self.degen_mergeset.as_mut().unwrap().insert(0, m);
            }
        } else if mergetype == MRG_REDUNDANT {
            self.facets[facet].redundant = true;
            self.degen_mergeset.as_mut().unwrap().push(m);
        } else {
            // MRGmirror
            if self.facets[facet].redundant || self.facets[neighbor].redundant {
                return Err("qhull internal error (qh_appendmergeset): mirror facet is already redundant".into());
            }
            if self.facets[facet].vertices != self.facets[neighbor].vertices {
                return Err("qhull internal error (qh_appendmergeset): mirrored facets have different vertices".into());
            }
            self.facets[facet].redundant = true;
            self.facets[neighbor].redundant = true;
            self.degen_mergeset.as_mut().unwrap().push(m);
        }
        Ok(())
    }

    /// `qh_all_merges(othermerge, vneighbors = False)`.
    fn all_merges(&mut self, mut othermerge: bool) -> Result<(), String> {
        let mut wasmerge;
        loop {
            wasmerge = false;
            while !self.facet_mergeset.as_ref().unwrap().is_empty() || !self.degen_mergeset.as_ref().unwrap().is_empty() {
                if !self.degen_mergeset.as_ref().unwrap().is_empty() {
                    self.merge_degenredundant()?;
                    wasmerge = true;
                }
                loop {
                    let m = set_dellast(self.facet_mergeset.as_mut().unwrap());
                    if m == NULL {
                        break;
                    }
                    let merge = self.merges[m];
                    let (facet1, facet2, mergetype) = (merge.facet1, merge.facet2, merge.mergetype);
                    if self.facets[facet1].visible || self.facets[facet2].visible {
                        continue;
                    }
                    if (mergetype == MRG_COPLANAR || mergetype == MRG_ANGLECOPLANAR)
                        && ((!self.facets[facet1].tested && self.facets[facet1].newfacet) || (!self.facets[facet2].tested && self.facets[facet2].newfacet))
                    {
                        continue;
                    }
                    if mergetype == MRG_TWISTED {
                        return Err("qhull: a twisted ridge (qh_merge_twisted) — not ported".into());
                    }
                    self.merge_nonconvex(facet1, facet2, mergetype)?;
                    self.merge_degenredundant()?;
                    wasmerge = true;
                }
                self.getmergeset()?;
            }
            if self.vertexneighbors && (wasmerge || othermerge) {
                wasmerge = false;
                othermerge = false;
                if self.reducevertices()? {
                    self.getmergeset()?;
                    continue;
                }
            }
            break;
        }
        if wasmerge || othermerge {
            for f in self.facets_from(self.newfacet_list) {
                self.facets[f].newmerge = false;
            }
        }
        Ok(())
    }

    /// `qh_merge_nonconvex`.
    fn merge_nonconvex(&mut self, facet1: usize, facet2: usize, mergetype: u8) -> Result<(), String> {
        if !(MRG_COPLANAR..=MRG_CONCAVECOPLANAR).contains(&mergetype) {
            return Err("qhull internal error (qh_merge_nonconvex): unexpected mergetype".into());
        }
        let (bestfacet, facet2) = if !self.facets[facet1].newfacet { (facet2, facet1) } else { (facet1, facet2) };
        let (bestneighbor, dist, mindist, maxdist) = self.findbestneighbor(bestfacet)?;
        let (neighbor, dist2, mindist2, maxdist2) = self.findbestneighbor(facet2)?;
        if dist < dist2 {
            self.mergefacet(bestfacet, bestneighbor, mergetype, Some((mindist, maxdist)), false)
        } else {
            self.mergefacet(facet2, neighbor, mergetype, Some((mindist2, maxdist2)), false)
        }
    }

    /// `qh_findbestneighbor`: `(neighbor, dist, mindist, maxdist)`.
    fn findbestneighbor(&mut self, facet: usize) -> Result<(usize, f64, f64, f64), String> {
        let size = self.facets[facet].vertices.len();
        let mut distp = REALMAX;
        let (mut mindistp, mut maxdistp) = (0.0, 0.0);
        let mut bestfacet = NULL;
        let mut testcentrum = false;
        if size > QH_BESTCENTRUM2 * DIM + QH_BESTCENTRUM {
            testcentrum = true;
            if self.facets[facet].center.is_none() {
                self.facets[facet].center = Some(self.getcentrum(facet));
            }
        }
        if size > DIM + QH_BESTNONCONVEX {
            for r in self.facets[facet].ridges.clone().unwrap_or_default() {
                if r == NULL {
                    break;
                }
                if self.ridges[r].nonconvex {
                    let neighbor = self.otherfacet(r, facet);
                    self.findbest_test(testcentrum, facet, neighbor, &mut bestfacet, &mut distp, &mut mindistp, &mut maxdistp);
                }
            }
        }
        if bestfacet == NULL {
            for nb in self.facets[facet].neighbors.clone() {
                if nb == NULL {
                    break;
                }
                self.findbest_test(testcentrum, facet, nb, &mut bestfacet, &mut distp, &mut mindistp, &mut maxdistp);
            }
        }
        if bestfacet == NULL {
            return Err("qhull internal error (qh_findbestneighbor): no neighbors".into());
        }
        if testcentrum {
            let (_, mn, mx) = self.getdistance(facet, bestfacet);
            mindistp = mn;
            maxdistp = mx;
        }
        Ok((bestfacet, distp, mindistp, maxdistp))
    }

    /// `qh_findbest_test`.
    #[allow(clippy::too_many_arguments)]
    fn findbest_test(&mut self, testcentrum: bool, facet: usize, neighbor: usize, bestfacet: &mut usize, distp: &mut f64, mindistp: &mut f64, maxdistp: &mut f64) {
        if self.facets[facet].flipped && self.facets[neighbor].flipped && *bestfacet != NULL && !self.facets[*bestfacet].flipped {
            return;
        }
        let (dist, mindist, maxdist);
        if testcentrum {
            let c = self.facets[facet].center.unwrap();
            let d = self.distplane(&c, neighbor) * DIM as f64;
            if d < 0.0 {
                maxdist = 0.0;
                mindist = d;
                dist = -d;
            } else {
                mindist = 0.0;
                maxdist = d;
                dist = d;
            }
        } else {
            let (d, mn, mx) = self.getdistance(facet, neighbor);
            dist = d;
            mindist = mn;
            maxdist = mx;
        }
        if dist < *distp {
            *bestfacet = neighbor;
            *mindistp = mindist;
            *maxdistp = maxdist;
            *distp = dist;
        }
    }

    /// `qh_getdistance`: how far `facet`'s own vertices are from `neighbor`'s plane — `(dist, min, max)`.
    fn getdistance(&mut self, facet: usize, neighbor: usize) -> (f64, f64, f64) {
        for &v in &self.facets[facet].vertices.clone() {
            self.vertices[v].seen = false;
        }
        for &v in &self.facets[neighbor].vertices.clone() {
            self.vertices[v].seen = true;
        }
        let (mut mind, mut maxd) = (0.0f64, 0.0f64);
        for &v in &self.facets[facet].vertices.clone() {
            if !self.vertices[v].seen {
                let dist = self.distplane(&self.points[self.vertices[v].point], neighbor);
                if dist < mind {
                    mind = dist;
                } else if dist > maxd {
                    maxd = dist;
                }
            }
        }
        let neg = -mind;
        (if maxd > neg { maxd } else { neg }, mind, maxd)
    }

    /// `qh_mergefacet(facet1, facet2, mergetype, mindist/maxdist, mergeapex)`: merge `facet1` into `facet2`.
    fn mergefacet(&mut self, facet1: usize, facet2: usize, mergetype: u8, dists: Option<(f64, f64)>, mergeapex: bool) -> Result<(), String> {
        self.totmerge += 1;
        if let Some((mindist, maxdist)) = dists {
            let mut mintwisted = 100.0 * self.one_merge;
            if mintwisted < self.facets[facet1].maxoutside {
                mintwisted = self.facets[facet1].maxoutside;
            }
            if mintwisted < self.facets[facet2].maxoutside {
                mintwisted = self.facets[facet2].maxoutside;
            }
            if maxdist > mintwisted || -mindist > mintwisted {
                return Err("qhull precision error (qh_mergefacet): wide merge".into());
            }
        }
        if facet1 == facet2 || self.facets[facet1].visible || self.facets[facet2].visible {
            return Err("qhull internal error (qh_mergefacet): same or visible facets".into());
        }
        if self.num_facets - self.num_visible <= DIM as i64 + 1 {
            return Err("qhull topology error: only 4 facets remain; the input is too degenerate".into());
        }
        if !self.vertexneighbors {
            self.vertexneighbors();
        }
        self.makeridges(facet1);
        self.makeridges(facet2);
        if let Some((mindist, maxdist)) = dists {
            if self.max_outside < maxdist {
                self.max_outside = maxdist;
            }
            if self.max_vertex < maxdist {
                self.max_vertex = maxdist;
            }
            if self.facets[facet2].maxoutside < maxdist {
                self.facets[facet2].maxoutside = maxdist;
            }
            if self.min_vertex > mindist {
                self.min_vertex = mindist;
            }
            if !self.facets[facet2].keepcentrum && (maxdist > self.wide_facet || mindist < -self.wide_facet) {
                self.facets[facet2].keepcentrum = true;
            }
        }
        let nummerge = self.facets[facet1].nummerge + self.facets[facet2].nummerge + 1;
        self.facets[facet2].nummerge = nummerge.min(QH_MAXNUMMERGE);
        self.facets[facet2].newmerge = true;
        self.facets[facet2].dupridge = false;
        self.updatetested(facet1, facet2);
        if self.facets[facet1].vertices.len() == DIM {
            self.mergesimplex(facet1, facet2, mergeapex)?;
        } else {
            self.vertex_visit += 1;
            for &v in &self.facets[facet2].vertices.clone() {
                self.vertices[v].visitid = self.vertex_visit;
            }
            self.mergeneighbors(facet1, facet2)?;
            self.mergevertices(facet1, facet2);
            self.mergeridges(facet1, facet2);
            self.mergevertex_neighbors(facet1, facet2)?;
            if !self.facets[facet2].newfacet {
                self.newvertices(facet2);
            }
        }
        self.removefacet(facet2);
        self.appendfacet(facet2);
        self.facets[facet2].newfacet = true;
        self.facets[facet2].tested = false;
        if mergetype != MRG_COPLANARHORIZON {
            self.test_redundant_neighbors(facet2)?;
            self.test_degen_neighbors(facet1)?;
            self.degen_redundant_facet(facet2)?;
        }
        self.willdelete(facet1, facet2);
        Ok(())
    }

    /// `qh_makeridges`: explicit ridges for a simplicial facet about to merge.
    fn makeridges(&mut self, facet: usize) {
        if !self.facets[facet].simplicial {
            return;
        }
        self.facets[facet].simplicial = false;
        let mut mergeridge = false;
        let nbs = self.facets[facet].neighbors.clone();
        for &nb in &nbs {
            if nb == NULL {
                break;
            }
            if nb == MERGERIDGE {
                mergeridge = true;
            } else {
                self.facets[nb].seen = false;
            }
        }
        for r in self.facets[facet].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            let o = self.otherfacet(r, facet);
            self.facets[o].seen = true;
        }
        for (i, &nb) in nbs.iter().enumerate() {
            if nb == MERGERIDGE {
                continue;
            }
            if !self.facets[nb].seen {
                let ridge = self.newridge();
                self.ridges[ridge].vertices = set_new_delnthsorted(&self.facets[facet].vertices, DIM, i, 0);
                let toporient = self.facets[facet].toporient ^ (i & 1 == 1);
                if toporient {
                    self.ridges[ridge].top = facet;
                    self.ridges[ridge].bottom = nb;
                    self.ridges[ridge].simplicialtop = true;
                    self.ridges[ridge].simplicialbot = self.facets[nb].simplicial;
                } else {
                    self.ridges[ridge].top = nb;
                    self.ridges[ridge].bottom = facet;
                    self.ridges[ridge].simplicialtop = self.facets[nb].simplicial;
                    self.ridges[ridge].simplicialbot = true;
                }
                if self.facets[facet].tested && !mergeridge {
                    self.ridges[ridge].tested = true;
                }
                self.facets[facet].ridges.get_or_insert_with(Vec::new).push(ridge);
                self.facets[nb].ridges.get_or_insert_with(Vec::new).push(ridge);
            }
        }
        if mergeridge {
            while set_del(&mut self.facets[facet].neighbors, MERGERIDGE) {}
        }
    }

    /// `qh_updatetested`.
    fn updatetested(&mut self, facet1: usize, facet2: usize) {
        self.facets[facet2].tested = false;
        for r in self.facets[facet1].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            self.ridges[r].tested = false;
        }
        if self.facets[facet2].center.is_none() {
            return;
        }
        let size = self.facets[facet2].vertices.len();
        if !self.facets[facet2].keepcentrum {
            if size > DIM + QH_MAXNEWCENTRUM {
                self.facets[facet2].keepcentrum = true;
            }
        } else if size <= DIM + QH_MAXNEWCENTRUM && (size == DIM || self.postmerging) {
            self.facets[facet2].keepcentrum = false;
        }
        if !self.facets[facet2].keepcentrum {
            self.facets[facet2].center = None;
            for r in self.facets[facet2].ridges.clone().unwrap_or_default() {
                if r == NULL {
                    break;
                }
                self.ridges[r].tested = false;
            }
        }
    }

    /// `qh_mergesimplex`: merge a simplicial `facet1` (three vertices) into `facet2`.
    fn mergesimplex(&mut self, facet1: usize, facet2: usize, mergeapex: bool) -> Result<(), String> {
        let opposite;
        let mut isnew = false;
        if mergeapex {
            opposite = self.facets[facet1].vertices[0];
            if !self.facets[facet2].newfacet {
                self.newvertices(facet2);
            }
            if at(&self.facets[facet2].vertices, 0) != opposite {
                self.facets[facet2].vertices.insert(0, opposite);
                isnew = true;
            }
        } else {
            for &v in &self.facets[facet1].vertices.clone() {
                self.vertices[v].seen = false;
            }
            for r in self.facets[facet1].ridges.clone().unwrap_or_default() {
                if r == NULL {
                    break;
                }
                if self.otherfacet(r, facet1) == facet2 {
                    for &v in &self.ridges[r].vertices.clone() {
                        self.vertices[v].seen = true;
                        self.vertices[v].delridge = true;
                    }
                    break;
                }
            }
            let mut opp = NULL;
            for &v in &self.facets[facet1].vertices {
                if !self.vertices[v].seen {
                    opp = v;
                    break;
                }
            }
            opposite = opp;
            isnew = self.addfacetvertex(facet2, opposite);
            if !self.facets[facet2].newfacet {
                self.newvertices(facet2);
            } else if !self.vertices[opposite].newfacet {
                self.removevertex(opposite);
                self.appendvertex(opposite);
            }
        }
        for v in self.facets[facet1].vertices.clone() {
            if v == opposite && isnew {
                set_replace(&mut self.vertices[v].neighbors, facet1, facet2)?;
            } else {
                set_del(&mut self.vertices[v].neighbors, facet1);
                if at(&self.vertices[v].neighbors, 1) == NULL {
                    self.mergevertex_del(v, facet2);
                }
            }
        }
        self.visit_id += 1;
        for nb in self.facets[facet2].neighbors.clone() {
            if nb == NULL {
                break;
            }
            self.facets[nb].visitid = self.visit_id;
        }
        let mut i = 0;
        loop {
            let ridge = match self.facets[facet1].ridges.as_ref() {
                Some(r) => at(r, i),
                None => NULL,
            };
            if ridge == NULL {
                break;
            }
            let otherfacet = self.otherfacet(ridge, facet1);
            if otherfacet == facet2 {
                self.delridge_merge(ridge);
                set_del(&mut self.facets[facet2].neighbors, facet1);
                continue;
            } else if self.facets[otherfacet].dupridge && !set_in(&self.facets[otherfacet].neighbors, facet1) {
                return Err("qhull internal error (qh_mergesimplex): dupridge neighbor".into());
            } else {
                self.facets[facet2].ridges.get_or_insert_with(Vec::new).push(ridge);
                if self.facets[otherfacet].visitid != self.visit_id {
                    set_append(&mut self.facets[facet2].neighbors, otherfacet);
                    set_replace(&mut self.facets[otherfacet].neighbors, facet1, facet2)?;
                    self.facets[otherfacet].visitid = self.visit_id;
                } else {
                    if self.facets[otherfacet].simplicial {
                        self.makeridges(otherfacet);
                    }
                    if at(&self.facets[otherfacet].neighbors, 0) == facet1 {
                        set_del(&mut self.facets[otherfacet].neighbors, facet2);
                        set_replace(&mut self.facets[otherfacet].neighbors, facet1, facet2)?;
                    } else {
                        set_del(&mut self.facets[otherfacet].neighbors, facet1);
                    }
                }
                if self.ridges[ridge].top == facet1 {
                    self.ridges[ridge].top = facet2;
                    self.ridges[ridge].simplicialtop = false;
                } else {
                    self.ridges[ridge].bottom = facet2;
                    self.ridges[ridge].simplicialbot = false;
                }
            }
            i += 1;
        }
        Ok(())
    }

    /// `qh_addfacetvertex`: insert into the descending-id vertex set; false if already there.
    fn addfacetvertex(&mut self, facet: usize, newvertex: usize) -> bool {
        let nid = self.vertices[newvertex].id;
        let verts = &self.facets[facet].vertices;
        let mut idx = verts.len();
        for (i, &v) in verts.iter().enumerate() {
            let id = self.vertices[v].id;
            if id < nid {
                idx = i;
                break;
            } else if id == nid {
                return false;
            }
        }
        self.facets[facet].vertices.insert(idx, newvertex);
        true
    }

    /// `qh_mergevertex_del`.
    fn mergevertex_del(&mut self, vertex: usize, facet2: usize) {
        set_delsorted(&mut self.facets[facet2].vertices, vertex);
        self.vertices[vertex].deleted = true;
        self.del_vertices.push(vertex);
    }

    /// `qh_delridge_merge`.
    fn delridge_merge(&mut self, ridge: usize) {
        if self.ridges[ridge].nonconvex {
            self.copynonconvex(ridge);
        }
        for &v in &self.ridges[ridge].vertices.clone() {
            self.vertices[v].delridge = true;
        }
        let (top, bottom) = (self.ridges[ridge].top, self.ridges[ridge].bottom);
        if let Some(r) = self.facets[top].ridges.as_mut() {
            set_del(r, ridge);
        }
        if let Some(r) = self.facets[bottom].ridges.as_mut() {
            set_del(r, ridge);
        }
    }

    /// `qh_copynonconvex`.
    fn copynonconvex(&mut self, atridge: usize) {
        let facet = self.ridges[atridge].top;
        let otherfacet = self.ridges[atridge].bottom;
        self.ridges[atridge].nonconvex = false;
        for r in self.facets[facet].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            if (otherfacet == self.ridges[r].top || otherfacet == self.ridges[r].bottom) && r != atridge {
                self.ridges[r].nonconvex = true;
                break;
            }
        }
    }

    /// `qh_mergeneighbors`.
    fn mergeneighbors(&mut self, facet1: usize, facet2: usize) -> Result<(), String> {
        self.visit_id += 1;
        for nb in self.facets[facet2].neighbors.clone() {
            if nb == NULL {
                break;
            }
            self.facets[nb].visitid = self.visit_id;
        }
        for nb in self.facets[facet1].neighbors.clone() {
            if nb == NULL {
                break;
            }
            if self.facets[nb].visitid == self.visit_id {
                if self.facets[nb].simplicial {
                    self.makeridges(nb);
                }
                if at(&self.facets[nb].neighbors, 0) != facet1 {
                    set_del(&mut self.facets[nb].neighbors, facet1);
                } else {
                    set_del(&mut self.facets[nb].neighbors, facet2);
                    set_replace(&mut self.facets[nb].neighbors, facet1, facet2)?;
                }
            } else if nb != facet2 {
                set_append(&mut self.facets[facet2].neighbors, nb);
                set_replace(&mut self.facets[nb].neighbors, facet1, facet2)?;
            }
        }
        set_del(&mut self.facets[facet1].neighbors, facet2);
        set_del(&mut self.facets[facet2].neighbors, facet1);
        Ok(())
    }

    /// `qh_mergevertices`: the union of two descending-id vertex sets, into `facet2`.
    fn mergevertices(&mut self, facet1: usize, facet2: usize) {
        let v1 = self.facets[facet1].vertices.clone();
        let v2 = std::mem::take(&mut self.facets[facet2].vertices);
        let mut merged = Vec::with_capacity(v1.len() + v2.len());
        let mut j = 0;
        let id = |v: usize| self.vertices[v].id;
        for &vertex in &v1 {
            if j >= v2.len() || id(vertex) > id(v2[j]) {
                merged.push(vertex);
            } else {
                while j < v2.len() && id(v2[j]) > id(vertex) {
                    merged.push(v2[j]);
                    j += 1;
                }
                if j >= v2.len() || id(v2[j]) < id(vertex) {
                    merged.push(vertex);
                } else {
                    merged.push(v2[j]);
                    j += 1;
                }
            }
        }
        while j < v2.len() {
            merged.push(v2[j]);
            j += 1;
        }
        self.facets[facet2].vertices = merged;
    }

    /// `qh_mergeridges`.
    fn mergeridges(&mut self, facet1: usize, facet2: usize) {
        let mut i = 0;
        loop {
            let ridge = match self.facets[facet2].ridges.as_ref() {
                Some(r) => at(r, i),
                None => NULL,
            };
            if ridge == NULL {
                break;
            }
            if self.ridges[ridge].top == facet1 || self.ridges[ridge].bottom == facet1 {
                self.delridge_merge(ridge);
                continue;
            }
            i += 1;
        }
        for r in self.facets[facet1].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            if self.ridges[r].top == facet1 {
                self.ridges[r].top = facet2;
                self.ridges[r].simplicialtop = false;
            } else {
                self.ridges[r].bottom = facet2;
                self.ridges[r].simplicialbot = false;
            }
            self.facets[facet2].ridges.get_or_insert_with(Vec::new).push(r);
        }
    }

    /// `qh_mergevertex_neighbors`.
    fn mergevertex_neighbors(&mut self, facet1: usize, facet2: usize) -> Result<(), String> {
        for v in self.facets[facet1].vertices.clone() {
            if self.vertices[v].visitid != self.vertex_visit {
                set_replace(&mut self.vertices[v].neighbors, facet1, facet2)?;
            } else {
                set_del(&mut self.vertices[v].neighbors, facet1);
                if at(&self.vertices[v].neighbors, 1) == NULL {
                    self.mergevertex_del(v, facet2);
                }
            }
        }
        Ok(())
    }

    /// `qh_newvertices`: move a facet's vertices onto the new-vertex list.
    fn newvertices(&mut self, facet: usize) {
        for v in self.facets[facet].vertices.clone() {
            if !self.vertices[v].newfacet {
                self.removevertex(v);
                self.appendvertex(v);
            }
        }
    }

    /// `qh_test_redundant_neighbors`.
    fn test_redundant_neighbors(&mut self, facet: usize) -> Result<(), String> {
        if self.facets[facet].neighbors.len() < DIM {
            return self.appendmergeset(facet, facet, MRG_DEGEN, 0.0, 1.0);
        }
        self.vertex_visit += 1;
        for &v in &self.facets[facet].vertices.clone() {
            self.vertices[v].visitid = self.vertex_visit;
        }
        for nb in self.facets[facet].neighbors.clone() {
            if nb == NULL {
                break;
            }
            if self.facets[nb].visible {
                return Err("qhull internal error (qh_test_redundant_neighbors): visible neighbor".into());
            }
            let n = &self.facets[nb];
            if n.degenerate || n.redundant || n.dupridge {
                continue;
            }
            if self.facets[facet].flipped && !n.flipped {
                continue;
            }
            if n.vertices.iter().all(|&v| self.vertices[v].visitid == self.vertex_visit) {
                self.appendmergeset(nb, facet, MRG_REDUNDANT, 0.0, 1.0)?;
            }
        }
        Ok(())
    }

    /// `qh_test_degen_neighbors`.
    fn test_degen_neighbors(&mut self, facet: usize) -> Result<(), String> {
        for nb in self.facets[facet].neighbors.clone() {
            if nb == NULL {
                break;
            }
            if self.facets[nb].visible {
                return Err("qhull internal error (qh_test_degen_neighbors): visible neighbor".into());
            }
            let n = &self.facets[nb];
            if n.degenerate || n.redundant || n.dupridge {
                continue;
            }
            if n.neighbors.len() < DIM {
                self.appendmergeset(nb, nb, MRG_DEGEN, 0.0, 1.0)?;
            }
        }
        Ok(())
    }

    /// `qh_degen_redundant_facet`.
    fn degen_redundant_facet(&mut self, facet: usize) -> Result<(), String> {
        if self.facets[facet].flipped {
            return Ok(());
        }
        for nb in self.facets[facet].neighbors.clone() {
            if nb == NULL {
                break;
            }
            if self.facets[nb].flipped {
                continue;
            }
            if self.facets[nb].visible {
                return Err("qhull internal error (qh_degen_redundant_facet): visible neighbor".into());
            }
            self.vertex_visit += 1;
            for &v in &self.facets[nb].vertices.clone() {
                self.vertices[v].visitid = self.vertex_visit;
            }
            if self.facets[facet].vertices.iter().all(|&v| self.vertices[v].visitid == self.vertex_visit) {
                return self.appendmergeset(facet, nb, MRG_REDUNDANT, 0.0, 1.0);
            }
        }
        if self.facets[facet].neighbors.len() < DIM {
            self.appendmergeset(facet, facet, MRG_DEGEN, 0.0, 1.0)?;
        }
        Ok(())
    }

    /// `qh_merge_degenredundant`: returns the number of merges.
    fn merge_degenredundant(&mut self) -> Result<usize, String> {
        let mut nummerges = 0;
        let mut mergedfacets: Vec<usize> = Vec::new();
        loop {
            let m = set_dellast(self.degen_mergeset.as_mut().unwrap());
            if m == NULL {
                break;
            }
            let merge = self.merges[m];
            let (facet1, facet2, mergetype) = (merge.facet1, merge.facet2, merge.mergetype);
            if self.facets[facet1].visible {
                continue;
            }
            self.facets[facet1].degenerate = false;
            self.facets[facet1].redundant = false;
            if mergetype == MRG_REDUNDANT {
                let facet3 = self.getreplacement(facet2);
                if facet3 == NULL {
                    return Err("qhull internal error (qh_merge_degenredundant): redundant facet's neighbor was deleted".into());
                }
                set_unique(&mut mergedfacets, facet3);
                if facet1 == facet3 {
                    continue;
                }
                self.mergefacet(facet1, facet3, mergetype, None, false)?;
                nummerges += 1;
            } else {
                let size = self.facets[facet1].neighbors.len();
                if size == 0 {
                    self.willdelete(facet1, NULL);
                    for v in self.facets[facet1].vertices.clone() {
                        set_del(&mut self.vertices[v].neighbors, facet1);
                        if at(&self.vertices[v].neighbors, 0) == NULL {
                            self.vertices[v].deleted = true;
                            self.del_vertices.push(v);
                        }
                    }
                    nummerges += 1;
                } else if size < DIM {
                    let (bestneighbor, _dist, mindist, maxdist) = self.findbestneighbor(facet1)?;
                    self.mergefacet(facet1, bestneighbor, mergetype, Some((mindist, maxdist)), false)?;
                    nummerges += 1;
                }
            }
        }
        Ok(nummerges)
    }

    /// `qh_reducevertices` (3-d: no `qh_redundant_vertex`).
    fn reducevertices(&mut self) -> Result<bool, String> {
        let mut degenredun = false;
        if self.merge_degenredundant()? > 0 {
            degenredun = true;
        }
        'restart: loop {
            let mut nf = self.newfacet_list;
            while nf != NULL && self.facets[nf].next != NULL {
                if self.facets[nf].newmerge && self.remove_extravertices(nf) {
                    self.degen_redundant_facet(nf)?;
                    if self.merge_degenredundant()? > 0 {
                        degenredun = true;
                        continue 'restart;
                    }
                }
                nf = self.facets[nf].next;
            }
            let mut nf = self.newfacet_list;
            while nf != NULL && self.facets[nf].next != NULL {
                if self.facets[nf].newmerge {
                    self.facets[nf].newmerge = false;
                    let mut i = 0;
                    loop {
                        let vertex = at(&self.facets[nf].vertices, i);
                        if vertex == NULL {
                            break;
                        }
                        if self.vertices[vertex].delridge && self.rename_sharedvertex(vertex, nf)? != NULL {
                            if self.merge_degenredundant()? > 0 {
                                degenredun = true;
                                continue 'restart;
                            }
                            continue;
                        }
                        i += 1;
                    }
                }
                nf = self.facets[nf].next;
            }
            break;
        }
        let mut v = self.newvertex_list;
        while v != NULL && self.vertices[v].next != NULL {
            if self.vertices[v].delridge && !self.vertices[v].deleted {
                self.vertices[v].delridge = false;
            }
            v = self.vertices[v].next;
        }
        Ok(degenredun)
    }

    /// `qh_remove_extravertices`: drop vertices no ridge of the facet uses.
    fn remove_extravertices(&mut self, facet: usize) -> bool {
        if self.facets[facet].simplicial {
            return false;
        }
        let mut foundrem = false;
        for &v in &self.facets[facet].vertices.clone() {
            self.vertices[v].seen = false;
        }
        for r in self.facets[facet].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            for &v in &self.ridges[r].vertices.clone() {
                self.vertices[v].seen = true;
            }
        }
        let mut i = 0;
        loop {
            let vertex = at(&self.facets[facet].vertices, i);
            if vertex == NULL {
                break;
            }
            if !self.vertices[vertex].seen {
                foundrem = true;
                set_delsorted(&mut self.facets[facet].vertices, vertex);
                set_del(&mut self.vertices[vertex].neighbors, facet);
                if self.vertices[vertex].neighbors.is_empty() {
                    self.vertices[vertex].deleted = true;
                    self.del_vertices.push(vertex);
                }
                continue;
            }
            i += 1;
        }
        foundrem
    }

    /// `qh_rename_sharedvertex` (3-d): a vertex shared by exactly two facets that merged may be renamed.
    fn rename_sharedvertex(&mut self, vertex: usize, facet: usize) -> Result<usize, String> {
        let nbs = &self.vertices[vertex].neighbors;
        if nbs.len() != 2 {
            return Ok(NULL);
        }
        let mut neighbor_a = nbs[0];
        if neighbor_a == facet {
            neighbor_a = nbs[1];
        }
        self.visit_id += 1;
        self.facets[neighbor_a].visitid = self.visit_id;
        let mut ridges = Vec::new();
        self.vertexridges_facet(vertex, facet, &mut ridges);
        let mut vertices = self.vertexintersect_new(&self.facets[facet].vertices, &self.facets[neighbor_a].vertices);
        set_del(&mut vertices, vertex);
        let newvertex = self.find_newvertex(vertex, &mut vertices, &ridges)?;
        if newvertex != NULL {
            self.renamevertex(vertex, newvertex, &ridges, facet, neighbor_a)?;
        }
        Ok(newvertex)
    }

    /// `qh_vertexridges_facet` (3-d): the facet's ridges containing `vertex` whose other facet is marked.
    fn vertexridges_facet(&mut self, vertex: usize, facet: usize, ridges: &mut Vec<usize>) {
        for r in self.facets[facet].ridges.clone().unwrap_or_default() {
            if r == NULL {
                break;
            }
            let neighbor = self.otherfacet(r, facet);
            if self.facets[neighbor].visitid == self.visit_id {
                let rv = &self.ridges[r].vertices;
                if at(rv, 0) == vertex || at(rv, DIM - 2) == vertex {
                    ridges.push(r);
                }
            }
        }
        self.facets[facet].visitid = self.visit_id - 1;
    }

    /// `qh_vertexridges(vertex, allneighbors = False)`.
    fn vertexridges(&mut self, vertex: usize) -> Vec<usize> {
        let mut ridges = Vec::new();
        self.visit_id += 2;
        let nbs = self.vertices[vertex].neighbors.clone();
        for &nb in &nbs {
            if nb == NULL {
                break;
            }
            self.facets[nb].visitid = self.visit_id;
        }
        for (i, &nb) in nbs.iter().enumerate() {
            if nb == NULL {
                break;
            }
            // `if (*neighborp || allneighbors)`: the LAST neighbour is skipped
            if at(&nbs, i + 1) != NULL {
                self.vertexridges_facet(vertex, nb, &mut ridges);
            }
        }
        ridges
    }

    /// `qh_vertexintersect_new`.
    fn vertexintersect_new(&self, a: &[usize], b: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            if a[i] == b[j] {
                out.push(a[i]);
                i += 1;
                j += 1;
            } else if self.vertices[a[i]].id > self.vertices[b[j]].id {
                i += 1;
            } else {
                j += 1;
            }
        }
        out
    }

    /// `qh_find_newvertex`: the vertex `oldvertex` can become without duplicating a ridge.
    fn find_newvertex(&mut self, oldvertex: usize, vertices: &mut Vec<usize>, ridges: &[usize]) -> Result<usize, String> {
        for &r in ridges {
            for &v in &self.ridges[r].vertices.clone() {
                self.vertices[v].seen = false;
            }
        }
        for &v in vertices.iter() {
            self.vertices[v].visitid = 0;
            self.vertices[v].seen = true;
        }
        for &r in ridges {
            for &v in &self.ridges[r].vertices.clone() {
                if self.vertices[v].seen {
                    self.vertices[v].visitid += 1;
                }
            }
        }
        let mut i = 0;
        while i < vertices.len() {
            if self.vertices[vertices[i]].visitid == 0 {
                vertices.swap_remove(i);
                continue;
            }
            i += 1;
        }
        let maxvisit = ridges.len() as u32;
        if self.vertex_visit < maxvisit {
            self.vertex_visit = maxvisit;
        }
        if vertices.is_empty() {
            return Ok(NULL);
        }
        {
            let verts = &self.vertices;
            apple_qsort(vertices, &mut |a, b| if verts[a].visitid > verts[b].visitid { 1 } else { -1 });
        }
        let hashsize = newhashtable_size(ridges.len());
        self.hash_table = vec![NULL; hashsize];
        for &r in ridges {
            self.hashridge(hashsize, r, oldvertex);
        }
        let mut found = NULL;
        for &vertex in vertices.iter() {
            let newridges = self.vertexridges(vertex);
            let mut dup = false;
            for &r in &newridges {
                if self.hashridge_find(hashsize, r, vertex, oldvertex) {
                    dup = true;
                    break;
                }
            }
            if !dup {
                found = vertex;
                break;
            }
        }
        self.hash_table.clear();
        Ok(found)
    }

    /// `qh_gethash` for a ridge (two vertices), by vertex id.
    fn ridgehash(&self, hashsize: usize, ridge: usize, skipelem: usize) -> usize {
        let mut hash: u64 = 0;
        for &v in &self.ridges[ridge].vertices {
            hash = hash.wrapping_add(self.vertices[v].id as u64);
        }
        hash = hash.wrapping_sub(self.vertices[skipelem].id as u64);
        (hash as u32 as usize) % hashsize
    }

    /// `qh_hashridge`.
    fn hashridge(&mut self, hashsize: usize, ridge: usize, oldvertex: usize) {
        let mut hash = self.ridgehash(hashsize, ridge, oldvertex);
        loop {
            let a = self.hash_table[hash];
            if a == NULL {
                self.hash_table[hash] = ridge;
                break;
            } else if a == ridge {
                break;
            }
            hash += 1;
            if hash == hashsize {
                hash = 0;
            }
        }
    }

    /// `qh_hashridge_find`: is there a ridge equal to `ridge` with `vertex` for `oldvertex`?
    fn hashridge_find(&self, hashsize: usize, ridge: usize, vertex: usize, oldvertex: usize) -> bool {
        let mut hash = self.ridgehash(hashsize, ridge, vertex);
        loop {
            let a = self.hash_table[hash];
            if a == NULL {
                return false;
            }
            if a != ridge && set_equal_except(&self.ridges[ridge].vertices, vertex, &self.ridges[a].vertices, oldvertex) {
                return true;
            }
            hash += 1;
            if hash == hashsize {
                hash = 0;
            }
        }
    }

    /// `qh_renamevertex` with an `oldfacet` (the only form 3-d reaches).
    fn renamevertex(&mut self, oldvertex: usize, newvertex: usize, ridges: &[usize], oldfacet: usize, neighbor_a: usize) -> Result<(), String> {
        for &ridge in ridges {
            if self.renameridgevertex(ridge, oldvertex, newvertex)? {
                let (top, bottom) = (self.ridges[ridge].top, self.ridges[ridge].bottom);
                let topsize = self.facets[top].vertices.len();
                let bottomsize = self.facets[bottom].vertices.len();
                // a ridge of a facet about to be merged away as degenerate is not checked for duplicates
                let degenerate = |f: usize, size: usize| size < DIM || (size == DIM && !self.facets[f].simplicial && set_in(&self.facets[f].vertices, newvertex));
                if !degenerate(top, topsize) && !degenerate(bottom, bottomsize) {
                    self.maybe_duplicateridge(ridge)?;
                }
            }
        }
        if self.vertices[oldvertex].neighbors.len() == 2 {
            for nb in self.vertices[oldvertex].neighbors.clone() {
                if nb == NULL {
                    break;
                }
                set_delsorted(&mut self.facets[nb].vertices, oldvertex);
                self.degen_redundant_facet(nb)?;
            }
            self.vertices[oldvertex].deleted = true;
            self.del_vertices.push(oldvertex);
        } else {
            set_delsorted(&mut self.facets[oldfacet].vertices, oldvertex);
            set_del(&mut self.vertices[oldvertex].neighbors, oldfacet);
            if self.remove_extravertices(neighbor_a) {
                self.degen_redundant_facet(neighbor_a)?;
            }
        }
        self.degen_redundant_facet(oldfacet)
    }

    /// `qh_renameridgevertex`: false when the ridge became a duplicate and was deleted.
    fn renameridgevertex(&mut self, ridge: usize, oldvertex: usize, newvertex: usize) -> Result<bool, String> {
        let oldnth = set_index(&self.ridges[ridge].vertices, oldvertex).ok_or("qhull internal error (qh_renameridgevertex): oldvertex not in ridge")?;
        self.ridges[ridge].vertices.remove(oldnth);
        let nid = self.vertices[newvertex].id;
        let mut nth = 0usize;
        for &v in &self.ridges[ridge].vertices.clone() {
            if v == newvertex {
                if self.ridges[ridge].nonconvex {
                    self.copynonconvex(ridge);
                }
                self.delridge_merge(ridge);
                return Ok(false);
            }
            if self.vertices[v].id < nid {
                break;
            }
            nth += 1;
        }
        self.ridges[ridge].vertices.insert(nth, newvertex);
        self.ridges[ridge].simplicialtop = false;
        self.ridges[ridge].simplicialbot = false;
        if (oldnth as i64 - nth as i64).abs() % 2 == 1 {
            let r = &mut self.ridges[ridge];
            std::mem::swap(&mut r.top, &mut r.bottom);
        }
        Ok(true)
    }

    /// `qh_maybe_duplicateridge` (3-d): a merged neighbour holding another ridge with the same vertices
    /// would need a vertex merge — not ported.
    fn maybe_duplicateridge(&mut self, ridge_a: usize) -> Result<(), String> {
        for neighbor in [self.ridges[ridge_a].top, self.ridges[ridge_a].bottom] {
            if !self.facets[neighbor].simplicial && self.facets[neighbor].nummerge > 0 {
                for r in self.facets[neighbor].ridges.clone().unwrap_or_default() {
                    if r == NULL {
                        break;
                    }
                    if r != ridge_a && at(&self.ridges[r].vertices, 0) == at(&self.ridges[ridge_a].vertices, 0) && at(&self.ridges[r].vertices, DIM - 2) == at(&self.ridges[ridge_a].vertices, DIM - 2) {
                        return Err("qhull: duplicate ridges after a vertex rename (MRGvertices) — not ported".into());
                    }
                }
            }
        }
        Ok(())
    }

    /// `qh_all_vertexmerges`: nothing to do unless a vertex merge was queued, which this port refuses.
    fn all_vertexmerges(&mut self) -> Result<(), String> {
        if self.vertex_mergeset.as_ref().is_some_and(|s| !s.is_empty()) {
            return Err("qhull: pinched-vertex merges (qh_merge_pinchedvertices) — not ported".into());
        }
        Ok(())
    }

    /// `qh_vertexneighbors`: each vertex's facets, in facet-list order.
    fn vertexneighbors(&mut self) {
        if self.vertexneighbors {
            return;
        }
        self.vertex_visit += 1;
        for f in self.facets_from(self.facet_list) {
            if self.facets[f].visible {
                continue;
            }
            for v in self.facets[f].vertices.clone() {
                if v == NULL {
                    break;
                }
                if self.vertices[v].visitid != self.vertex_visit {
                    self.vertices[v].visitid = self.vertex_visit;
                    self.vertices[v].neighbors = Vec::new();
                }
                self.vertices[v].neighbors.push(f);
            }
        }
        self.vertexneighbors = true;
    }
}

impl Qh {
    // ------------------------------------------------------------------------------ triangulation

    /// `qh_triangulate`: split every non-simplicial facet into a fan from its first vertex.
    fn triangulate(&mut self) -> Result<(), String> {
        if self.has_triangulation {
            return Ok(());
        }
        self.visit_id += 1;
        self.initmergesets();
        self.newvertex_list = self.vertex_tail;
        let mut triangulated_facet_list = NULL;
        let mut triangulated_vertex_list = NULL;
        let mut facet = self.facet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            let nextfacet = self.facets[facet].next;
            if !(self.facets[facet].visible || self.facets[facet].simplicial) {
                if triangulated_facet_list == NULL {
                    triangulated_facet_list = facet;
                }
                self.triangulate_facet(facet, &mut triangulated_vertex_list)?;
            }
            facet = nextfacet;
        }
        let mut facet = triangulated_facet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            let nextfacet = self.facets[facet].next;
            if !self.facets[facet].visible {
                if let Some(r) = &self.facets[facet].ridges {
                    if !r.is_empty() {
                        return Err("qhull internal error (qh_triangulate): ridges still defined for a triangulated facet".into());
                    }
                    self.facets[facet].ridges = None;
                }
                if at(&self.facets[facet].vertices, 0) == at(&self.facets[facet].vertices, 1) {
                    self.triangulate_null(facet)?;
                }
            }
            facet = nextfacet;
        }
        self.visible_list = self.facet_tail;
        loop {
            let m = set_dellast(self.degen_mergeset.as_mut().unwrap());
            if m == NULL {
                break;
            }
            let merge = self.merges[m];
            if merge.mergetype == MRG_MIRROR {
                self.triangulate_mirror(merge.facet1, merge.facet2)?;
            }
        }
        self.freemergesets();
        self.newvertex_list = triangulated_vertex_list;
        self.visible_list = NULL;
        self.update_vertexneighbors();
        self.resetlists(false);
        let mut facet = triangulated_facet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            if self.facets[facet].tricoplanar && !self.facets[facet].visible {
                let nbs = self.facets[facet].neighbors.clone();
                let mut orig_neighbor = NULL;
                for (i, &nb) in nbs.iter().enumerate() {
                    let owner = if self.facets[nb].tricoplanar { self.facets[nb].f } else { nb };
                    if i == 0 {
                        orig_neighbor = owner;
                    } else if orig_neighbor == owner {
                        self.facets[facet].degenerate = true;
                        break;
                    }
                }
            }
            facet = self.facets[facet].next;
        }
        let mut owner = NULL;
        let mut visible = NULL;
        let mut facet = triangulated_facet_list;
        while facet != NULL && self.facets[facet].next != NULL {
            let mut nextfacet = self.facets[facet].next;
            if self.facets[facet].visible {
                if self.facets[facet].tricoplanar {
                    self.delfacet(facet);
                    self.num_visible -= 1;
                } else {
                    if visible != NULL && owner == NULL {
                        self.delfacet(visible);
                        self.num_visible -= 1;
                    }
                    visible = facet;
                    owner = NULL;
                }
            } else if self.facets[facet].tricoplanar {
                if self.facets[facet].f != visible || visible == NULL {
                    return Err("qhull internal error (qh_triangulate): tricoplanar facet not owned by its visible facet".into());
                }
                if owner != NULL {
                    self.facets[facet].f = owner;
                } else if !self.facets[facet].degenerate {
                    owner = facet;
                    nextfacet = self.facets[visible].next;
                    self.facets[facet].keepcentrum = true;
                    self.facets[facet].coplanarset = self.facets[visible].coplanarset.take();
                    self.facets[facet].outsideset = self.facets[visible].outsideset.take();
                    self.delfacet(visible);
                    self.num_visible -= 1;
                }
            }
            self.facets[facet].degenerate = false;
            facet = nextfacet;
        }
        if visible != NULL && owner == NULL {
            self.delfacet(visible);
            self.num_visible -= 1;
        }
        self.has_triangulation = true;
        Ok(())
    }

    /// `qh_triangulate_facet`: a fan of tricoplanar facets from `facetA`'s first vertex.
    fn triangulate_facet(&mut self, facet_a: usize, first_vertex: &mut usize) -> Result<(), String> {
        self.first_newfacet = self.facet_id;
        for nb in self.facets[facet_a].neighbors.clone() {
            if nb == NULL {
                break;
            }
            self.facets[nb].seen = false;
            self.facets[nb].coplanarhorizon = false;
        }
        self.visible_list = self.facet_tail;
        self.newfacet_list = self.facet_tail;
        self.facets[facet_a].visitid = self.visit_id;
        let apex = self.facets[facet_a].vertices[0];
        self.makenew_nonsimplicial(facet_a, apex)?;
        self.willdelete(facet_a, NULL);
        let (normal, center, offset, maxoutside, good) = {
            let a = &self.facets[facet_a];
            (a.normal, a.center, a.offset, a.maxoutside, a.good)
        };
        for nf in self.facets_from(self.newfacet_list) {
            let f = &mut self.facets[nf];
            f.tricoplanar = true;
            f.f = facet_a;
            f.degenerate = false;
            f.good = good;
            f.keepcentrum = false;
            f.normal = normal;
            f.center = center;
            f.offset = offset;
            f.maxoutside = maxoutside;
        }
        self.matchnewfacets()?;
        if *first_vertex == NULL {
            *first_vertex = self.newvertex_list;
        }
        self.newvertex_list = NULL;
        self.visible_list = NULL;
        self.update_vertexneighbors();
        self.resetlists(false);
        Ok(())
    }

    /// `qh_triangulate_null`: a fan facet through the apex twice.
    fn triangulate_null(&mut self, facet_a: usize) -> Result<(), String> {
        let neighbor = at(&self.facets[facet_a].neighbors, 0);
        let otherfacet = at(&self.facets[facet_a].neighbors, 1);
        self.triangulate_link(facet_a, neighbor, facet_a, otherfacet)?;
        self.willdelete(facet_a, NULL);
        Ok(())
    }

    /// `qh_hasmerge`.
    fn hasmerge(&self, mergetype: u8, a: usize, b: usize) -> bool {
        self.degen_mergeset.as_ref().is_some_and(|s| {
            s.iter().any(|&m| {
                let x = &self.merges[m];
                x.mergetype == mergetype && ((x.facet1 == a && x.facet2 == b) || (x.facet1 == b && x.facet2 == a))
            })
        })
    }

    /// `qh_triangulate_link`.
    fn triangulate_link(&mut self, oldfacet_a: usize, facet_a: usize, oldfacet_b: usize, facet_b: usize) -> Result<(), String> {
        let mut errmirror = false;
        if set_in(&self.facets[facet_a].neighbors, facet_b) {
            if !set_in(&self.facets[facet_b].neighbors, facet_a) {
                errmirror = true;
            } else if !self.facets[facet_a].redundant || !self.facets[facet_b].redundant || !self.hasmerge(MRG_MIRROR, facet_a, facet_b) {
                self.appendmergeset(facet_a, facet_b, MRG_MIRROR, 0.0, 1.0)?;
            }
        } else if set_in(&self.facets[facet_b].neighbors, facet_a) {
            errmirror = true;
        }
        if errmirror {
            return Err("qhull internal error (qh_triangulate_link): mirror facets are not neighbors of each other".into());
        }
        set_replace(&mut self.facets[facet_b].neighbors, oldfacet_b, facet_a)?;
        set_replace(&mut self.facets[facet_a].neighbors, oldfacet_a, facet_b)
    }

    /// `qh_triangulate_mirror`.
    fn triangulate_mirror(&mut self, facet_a: usize, facet_b: usize) -> Result<(), String> {
        let na = self.facets[facet_a].neighbors.clone();
        for (i, &neighbor) in na.iter().enumerate() {
            let neighbor_b = at(&self.facets[facet_b].neighbors, i);
            let mirrored = neighbor == facet_b && neighbor_b == facet_a;
            if mirrored || (self.facets[neighbor].redundant && self.facets[neighbor_b].redundant && self.hasmerge(MRG_MIRROR, neighbor, neighbor_b)) {
                continue;
            }
            if self.facets[neighbor].visible && self.facets[neighbor_b].visible {
                continue;
            }
            self.triangulate_link(facet_a, neighbor, facet_b, neighbor_b)?;
        }
        self.willdelete(facet_a, NULL);
        self.willdelete(facet_b, NULL);
        Ok(())
    }

    // ------------------------------------------------------------------------------------- output

    /// `mjCMesh::MakeGraph`'s walk over qhull's lists.
    fn graph(&self) -> Vec<i32> {
        let mut verts = Vec::new();
        let mut v = self.vertex_list;
        while v != NULL && self.vertices[v].next != NULL {
            verts.push(v);
            v = self.vertices[v].next;
        }
        let faces = self.facets_from(self.facet_list);
        let numvert = verts.len();
        let numface = faces.len();
        let mut vert_edgeadr = Vec::with_capacity(numvert);
        let mut vert_globalid = Vec::with_capacity(numvert);
        let mut edge_localid: Vec<i32> = Vec::with_capacity(numvert + 3 * numface);
        for &v in &verts {
            let pid = self.vertices[v].point as i32;
            vert_edgeadr.push(edge_localid.len() as i32);
            vert_globalid.push(pid);
            let start = edge_localid.len();
            for &f in &self.vertices[v].neighbors {
                if f == NULL {
                    break;
                }
                for &v1 in &self.facets[f].vertices {
                    let pid1 = self.vertices[v1].point as i32;
                    if pid != pid1 && !edge_localid[start..].contains(&pid1) {
                        edge_localid.push(pid1);
                    }
                }
            }
            edge_localid.push(-1);
        }
        let mut face_globalid = Vec::with_capacity(3 * numface);
        for &f in &faces {
            let fv = &self.facets[f].vertices;
            let mut tri = [0i32; 3];
            let ind: [usize; 3] = if self.facets[f].toporient { [1, 0, 2] } else { [0, 1, 2] };
            for (ii, &v1) in fv.iter().enumerate().take(3) {
                tri[ind[ii]] = self.vertices[v1].point as i32;
            }
            face_globalid.extend_from_slice(&tri);
        }
        let mut hullid = vec![-1i32; self.points.len()];
        for (i, &g) in vert_globalid.iter().enumerate() {
            hullid[g as usize] = i as i32;
        }
        for e in edge_localid.iter_mut() {
            if *e >= 0 {
                *e = hullid[*e as usize];
            }
        }
        let mut out = Vec::with_capacity(2 + 3 * numvert + 6 * numface);
        out.push(numvert as i32);
        out.push(numface as i32);
        out.extend(vert_edgeadr);
        out.extend(vert_globalid);
        out.extend(edge_localid);
        out.extend(face_globalid);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `unitree_z1/z1_GripperMover_col_1` as the file states it (`f32`).
    const Z1_GRIPPER: [[f64; 3]; 26] = [
        [0.07339072972536087, 0.03667756915092468, -0.008999941870570183],
        [0.07339072972536087, 0.03667757287621498, 0.00532058160752058],
        [0.0866953656077385, 0.03347479924559593, 0.0021906362380832434],
        [0.0866953656077385, 0.03347479924559593, -0.008999942801892757],
        [0.07339072972536087, 0.03667756915092468, 0.019641105085611343],
        [0.0866953656077385, 0.03347479924559593, 0.013381215743720531],
        [0.10000000894069672, 0.030272027477622032, 0.007121325936168432],
        [0.10000000149011612, 0.030272027477622032, -0.0009393090149387717],
        [0.10000000894069672, 0.030272027477622032, -0.008999943733215332],
        [0.10000000149011612, 1.1641532182693481e-10, -0.0009393089567311108],
        [0.10000000149011612, 0.0, -0.008999944664537907],
        [0.10000000149011612, 0.0, 0.007121325936168432],
        [0.10000000894069672, -0.030272027477622032, 0.007121325936168432],
        [0.10000000149011612, -0.030272027477622032, -0.0009393090149387717],
        [0.10000000894069672, -0.030272027477622032, -0.008999943733215332],
        [0.0866953581571579, -0.03347479924559593, 0.002190636470913887],
        [0.08669537305831909, -0.03347479924559593, -0.008999942801892757],
        [0.08669537305831909, -0.03347479924559593, 0.013381215743720531],
        [0.07339072972536087, -0.03667756915092468, 0.019641105085611343],
        [0.07339072972536087, -0.03667757287621498, 0.00532058160752058],
        [0.07339072972536087, -0.03667756915092468, -0.008999941870570183],
        [0.0866953656077385, 5.820766091346741e-10, -0.008999942801892757],
        [0.07339072972536087, 0.0, -0.008999941870570183],
        [0.0866953581571579, -1.1641532182693481e-10, 0.013381215743720531],
        [0.07339072972536087, 0.0, 0.019641105085611343],
        [0.07339072972536087, 9.313225746154785e-10, 0.005320582073181868],
    ];
    /// MuJoCo 3.13.0's `mesh_graph` for it.
    const Z1_GRIPPER_GRAPH: [i32; 203] = [15, 26, 0, 4, 10, 18, 26, 31, 36, 43, 50, 56, 63, 69, 76, 83, 88, 5, 17, 20, 1, 0, 18, 19, 4, 12, 8, 14, 6, 10, 16, 3, 3, 7, 11, -1, 5, 7, 11, 6, 8, -1, 13, 6, 12, 4, 3, 5, 7, -1, 9, 11, 0, 7, 14, 4, 2, -1, 14, 3, 12, 2, -1, 1, 7, 6, 2, -1, 13, 10, 1, 5, 2, 8, -1, 0, 11, 3, 1, 5, 2, -1, 1, 11, 10, 6, 9, -1, 3, 11, 14, 12, 10, 8, -1, 8, 6, 13, 12, 9, -1, 9, 3, 0, 7, 1, 8, -1, 9, 10, 2, 4, 13, 14, -1, 2, 6, 10, 12, -1, 3, 4, 9, 12, -1, 8, 1, 6, 5, 1, 4, 5, 4, 6, 1, 5, 6, 18, 17, 4, 4, 17, 6, 17, 18, 19, 1, 3, 0, 8, 3, 1, 20, 16, 19, 16, 14, 19, 17, 12, 6, 14, 12, 19, 12, 17, 19, 10, 8, 14, 10, 20, 0, 20, 1, 0, 18, 20, 19, 1, 20, 4, 20, 18, 4, 8, 12, 14, 12, 8, 6, 10, 16, 20, 16, 10, 14, 10, 3, 8, 3, 10, 0];
    /// `trossen_wxai/link_6_collision` as the file states it (`f32`).
    const WXAI_LINK6: [[f64; 3]; 40] = [
        [0.0004220000118948519, -0.028578000143170357, -0.009573999792337418],
        [0.0004220000118948519, -0.028578000143170357, 0.009417999535799026],
        [0.0004220000118948519, -0.023561999201774597, -0.018573999404907227],
        [0.058421000838279724, -0.08656200021505356, -0.023572999984025955],
        [0.06341800093650818, -0.091559998691082, -0.023572999984025955],
        [0.0004220000118948519, -0.023561999201774597, 0.018417999148368835],
        [0.06341800093650818, -0.091559998691082, 0.023416999727487564],
        [0.058421000838279724, -0.08656200021505356, 0.023416999727487564],
        [0.0004220000118948519, -0.018564000725746155, -0.023572999984025955],
        [0.06341800093650818, -0.08656200021505356, -0.028578000143170357],
        [0.07742200046777725, -0.091559998691082, -0.023572999984025955],
        [0.0004220000118948519, -0.018564000725746155, 0.023416999727487564],
        [0.07742200046777725, -0.091559998691082, 0.023416999727487564],
        [0.06341800093650818, -0.08656200021505356, 0.028421999886631966],
        [0.010424000211060047, -0.023561999201774597, -0.028578000143170357],
        [0.0004220000118948519, -0.009572000242769718, -0.028578000143170357],
        [0.07742200046777725, -0.08656200021505356, -0.028578000143170357],
        [0.010424000211060047, -0.023561999201774597, 0.028421999886631966],
        [0.0004220000118948519, -0.009572000242769718, 0.028421999886631966],
        [0.07742200046777725, -0.08656200021505356, 0.028421999886631966],
        [0.0004220000118948519, 0.009415999986231327, -0.028578000143170357],
        [0.07742200046777725, 0.08640599995851517, -0.028578000143170357],
        [0.0004220000118948519, 0.009415999986231327, 0.028421999886631966],
        [0.07742200046777725, 0.08640599995851517, 0.028421999886631966],
        [0.0004220000118948519, 0.018408000469207764, -0.023572999984025955],
        [0.010424000211060047, 0.023406000807881355, -0.028578000143170357],
        [0.06341800093650818, 0.08640599995851517, -0.028578000143170357],
        [0.07742200046777725, 0.09140399843454361, -0.023572999984025955],
        [0.010424000211060047, 0.023406000807881355, 0.028421999886631966],
        [0.0004220000118948519, 0.018408000469207764, 0.023416999727487564],
        [0.07742200046777725, 0.09140399843454361, 0.023416999727487564],
        [0.06341800093650818, 0.08640599995851517, 0.028421999886631966],
        [0.0004220000118948519, 0.023406000807881355, -0.018573999404907227],
        [0.058421000838279724, 0.08640599995851517, -0.023572999984025955],
        [0.06341800093650818, 0.09140399843454361, -0.023572999984025955],
        [0.058421000838279724, 0.08640599995851517, 0.023416999727487564],
        [0.0004220000118948519, 0.023406000807881355, 0.018417999148368835],
        [0.06341800093650818, 0.09140399843454361, 0.023416999727487564],
        [0.0004220000118948519, 0.028421999886631966, -0.009573999792337418],
        [0.0004220000118948519, 0.028421999886631966, 0.009417999535799026],
    ];
    /// MuJoCo 3.13.0's `mesh_graph` for it.
    const WXAI_LINK6_GRAPH: [i32; 578] = [40, 76, 0, 6, 12, 18, 24, 33, 39, 46, 51, 58, 64, 70, 76, 85, 91, 97, 103, 110, 117, 125, 131, 140, 147, 153, 159, 168, 173, 179, 184, 193, 199, 204, 209, 226, 233, 238, 245, 252, 258, 263, 35, 7, 33, 3, 30, 23, 19, 12, 10, 21, 27, 16, 6, 13, 4, 1, 0, 9, 26, 34, 37, 31, 39, 38, 17, 28, 22, 18, 14, 20, 15, 25, 32, 11, 5, 8, 29, 24, 36, 2, 20, 22, 36, 21, 38, -1, 12, 15, 33, 13, 34, -1, 18, 19, 37, 23, 32, -1, 17, 14, 16, 39, 35, -1, 5, 6, 7, 8, 9, 10, 11, 20, -1, 4, 6, 20, 21, 24, -1, 4, 5, 7, 12, 13, 24, -1, 4, 8, 6, 12, -1, 4, 7, 11, 12, 14, 17, -1, 4, 10, 11, 18, 28, -1, 4, 9, 18, 19, 20, -1, 4, 8, 9, 17, 28, -1, 1, 15, 13, 6, 7, 8, 14, 16, -1, 24, 33, 1, 12, 6, -1, 3, 17, 16, 12, 8, -1, 1, 12, 34, 16, 32, -1, 3, 14, 39, 12, 15, 32, -1, 28, 35, 3, 14, 11, 8, -1, 31, 37, 2, 19, 9, 10, 28, -1, 2, 18, 23, 10, 20, -1, 0, 21, 22, 5, 4, 10, 19, 23, -1, 20, 0, 36, 25, 5, 24, -1, 20, 0, 38, 23, 32, -1, 2, 19, 32, 20, 22, -1, 33, 27, 13, 5, 21, 25, 6, 26, -1, 26, 36, 21, 24, -1, 25, 36, 24, 27, 32, -1, 24, 33, 26, 32, -1, 35, 30, 17, 29, 9, 11, 31, 18, -1, 31, 37, 28, 30, 32, -1, 28, 35, 29, 32, -1, 37, 29, 18, 28, -1, 2, 23, 37, 33, 27, 26, 15, 16, 22, 29, 30, 34, 35, 36, 38, 39, -1, 24, 13, 27, 1, 34, 32, -1, 1, 15, 33, 32, -1, 28, 17, 30, 3, 39, 32, -1, 38, 0, 21, 25, 26, 32, -1, 31, 18, 29, 2, 32, -1, 0, 36, 22, 32, -1, 3, 16, 35, 32, -1, 35, 29, 31, 37, 35, 31, 35, 37, 39, 35, 36, 29, 36, 35, 39, 24, 25, 20, 25, 24, 26, 22, 28, 29, 29, 28, 31, 11, 17, 18, 17, 11, 13, 14, 8, 15, 8, 14, 9, 6, 7, 1, 7, 6, 13, 11, 7, 13, 7, 5, 1, 5, 7, 11, 26, 33, 34, 24, 33, 26, 34, 33, 38, 33, 32, 38, 32, 33, 24, 3, 9, 4, 3, 4, 0, 2, 3, 0, 8, 3, 2, 3, 8, 9, 30, 23, 19, 30, 12, 10, 30, 21, 27, 16, 30, 10, 21, 30, 16, 12, 30, 19, 6, 19, 13, 6, 12, 19, 12, 6, 10, 6, 4, 10, 6, 1, 0, 4, 6, 0, 9, 16, 10, 4, 9, 10, 21, 26, 27, 26, 34, 27, 23, 37, 31, 30, 37, 23, 37, 30, 27, 34, 37, 27, 39, 37, 38, 37, 34, 38, 17, 23, 31, 28, 17, 31, 19, 17, 13, 23, 17, 19, 17, 22, 18, 17, 28, 22, 20, 14, 15, 14, 21, 16, 25, 14, 20, 9, 14, 16, 14, 26, 21, 14, 25, 26, 32, 11, 18, 22, 32, 18, 1, 32, 0, 32, 39, 38, 32, 20, 15, 5, 32, 1, 8, 32, 15, 32, 22, 29, 32, 24, 20, 36, 32, 29, 32, 2, 0, 32, 8, 2, 32, 36, 39, 32, 5, 11];

    /// **Two Menagerie meshes whose hull depends on fused multiply–add**, against MuJoCo 3.13.0's own
    /// `mesh_graph`, integer for integer: built WITHOUT contraction, qhull returns a different graph for
    /// both. One is uncapped (a gripper plate whose coplanar faces qhull merges then re-triangulates), the
    /// other capped by `maxhullvert` 64 — above its 40 vertices, but `Q9` still changes the insertion order.
    #[test]
    fn menagerie_meshes_whose_hull_depends_on_fma_match_mujoco() {
        assert_eq!(mesh_graph(&Z1_GRIPPER, None).unwrap(), Z1_GRIPPER_GRAPH.to_vec());
        assert_eq!(mesh_graph(&WXAI_LINK6, Some(64)).unwrap(), WXAI_LINK6_GRAPH.to_vec());
    }

    /// A unit cube's eight corners: qhull merges each face's two triangles, then `Qt` splits them again.
    #[test]
    fn a_cube_has_eight_vertices_and_twelve_triangles() {
        let pts: Vec<[f64; 3]> = (0..8).map(|i| [(i & 1) as f64, ((i >> 1) & 1) as f64, ((i >> 2) & 1) as f64]).collect();
        let g = mesh_graph(&pts, None).unwrap();
        assert_eq!((g[0], g[1]), (8, 12));
    }

    /// `apple_qsort` against a hand check: ties under a never-zero comparator keep libc's order.
    #[test]
    fn apple_qsort_sorts_and_keeps_libc_tie_order() {
        let keys = [0i32; 9];
        let mut a: Vec<usize> = (0..9).collect();
        apple_qsort(&mut a, &mut |x, y| if keys[x] < keys[y] { 1 } else { -1 });
        // measured on macOS: all-equal keys under `a < b ? 1 : -1` come back as 8 1 2 3 0 5 6 7 4
        assert_eq!(a, vec![8, 1, 2, 3, 0, 5, 6, 7, 4]);
        let keys = [5, 3, 9, 1, 7, 3, 3, 8, 0, 2, 6, 4];
        let mut a: Vec<usize> = (0..keys.len()).collect();
        apple_qsort(&mut a, &mut |x, y| if keys[x] > keys[y] { 1 } else { -1 });
        assert!(a.windows(2).all(|w| keys[w[0]] <= keys[w[1]]));
    }
}
