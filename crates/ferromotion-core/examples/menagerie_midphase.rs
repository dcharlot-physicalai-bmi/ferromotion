//! **MuJoCo's collision driver, decision by decision** — the broadphase and the BVH midphase
//! ([`ferromotion_core::MjcfTree::collide_qpos`]) against MuJoCo 3.13.0's own, on every Menagerie model.
//!
//! ```text
//! python scripts/mujoco_bvh_oracle.py <menagerie root> <bvh oracle txt> [states per model]
//! cargo run --release --example menagerie_midphase -- <menagerie root> <bvh oracle txt> [substring]
//! ```
//!
//! Three things, each bit for bit: the hierarchy the compiler builds over each body's geoms
//! ([`ferromotion_core::MjcfTree::body_bvhs`] against `bvh_aabb`/`bvh_child`/`bvh_nodeid`/`bvh_depth`, with
//! `body_margin` and every `geom_aabb`); at `qpos0` and at random states, the set of nodes the midphase found
//! overlapping (MuJoCo's `bvh_active`); and the contacts that came out — how many, which geoms in which
//! order, and each `dist`.

use ferromotion_core::tree_from_mjcf;
use std::collections::BTreeSet;

/// Python's `float.hex`, to the bit.
fn hex(s: &str) -> f64 {
    let (neg, s) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let s = s.strip_prefix("0x").expect("a hex float");
    let (mant, exp) = s.split_once('p').expect("an exponent");
    let (int, frac) = mant.split_once('.').unwrap_or((mant, ""));
    let mut digits = frac.to_string();
    while digits.len() < 13 {
        digits.push('0');
    }
    let fbits = u64::from_str_radix(&digits, 16).unwrap();
    let exp: i64 = exp.parse().unwrap();
    let bits = match (int, fbits) {
        ("0", 0) => 0,
        ("0", f) => f,
        _ => (((exp + 1023) as u64) << 52) | fbits,
    };
    let v = f64::from_bits(bits);
    if neg { -v } else { v }
}

#[derive(Default)]
struct Body {
    bvhadr: i64,
    margin: f64,
    /// (child0, child1, nodeid, depth, aabb)
    nodes: Vec<(i32, i32, i32, i32, [f64; 6])>,
}

#[derive(Default)]
struct State {
    qpos: Vec<f64>,
    active: Vec<usize>,
    contacts: Vec<(usize, usize, f64)>,
}

#[derive(Default)]
struct Model {
    rel: String,
    bodies: Vec<Body>,
    gaabb: Vec<[f64; 6]>,
    states: Vec<State>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: menagerie_midphase <menagerie root> <bvh oracle txt> [substring]");
        std::process::exit(2);
    }
    let root = std::path::Path::new(&args[1]);
    let filter = args.get(3).cloned();
    let mut models: Vec<Model> = Vec::new();
    for line in std::fs::read_to_string(&args[2]).unwrap().lines() {
        let t: Vec<&str> = line.split('\t').collect();
        let six = |s: &str| -> [f64; 6] { s.split_whitespace().map(hex).collect::<Vec<_>>().try_into().unwrap() };
        match t[0] {
            "model" => models.push(Model { rel: t[1].to_string(), ..Default::default() }),
            "body" => models.last_mut().unwrap().bodies.push(Body { bvhadr: t[2].parse().unwrap(), margin: hex(t[4]), nodes: Vec::new() }),
            "node" => {
                let m = models.last_mut().unwrap();
                let b: usize = t[1].parse().unwrap();
                m.bodies[b].nodes.push((t[3].parse().unwrap(), t[4].parse().unwrap(), t[5].parse().unwrap(), t[6].parse().unwrap(), six(t[7])));
            }
            "gaabb" => models.last_mut().unwrap().gaabb.push(six(t[2])),
            "qpos" => models.last_mut().unwrap().states.push(State { qpos: t[1..].iter().filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect(), ..Default::default() }),
            "active" => models.last_mut().unwrap().states.last_mut().unwrap().active = t.get(1).map_or(Vec::new(), |s| s.split_whitespace().map(|x| x.parse().unwrap()).collect()),
            "contact" => models.last_mut().unwrap().states.last_mut().unwrap().contacts.push((t[1].parse().unwrap(), t[2].parse().unwrap(), hex(t[3]))),
            _ => {}
        }
    }
    let (mut nmodels, mut refused) = (0usize, 0usize);
    let (mut bodies, mut bodies_exact, mut nodes, mut gaabb, mut gaabb_exact) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (mut states, mut active_same, mut contacts_same, mut ncontacts) = (0usize, 0usize, 0usize, 0usize);
    let mut nactive = 0usize;
    let mut notes: Vec<String> = Vec::new();
    // contact differences get their own list: they are the ones that change the physics
    let mut contact_notes: Vec<String> = Vec::new();
    for m in &models {
        if filter.as_ref().is_some_and(|f| !m.rel.contains(f.as_str())) {
            continue;
        }
        let path = root.join(&m.rel);
        let dir = path.parent().unwrap().to_path_buf();
        let Ok(xml) = std::fs::read_to_string(&path) else { continue };
        let t = match tree_from_mjcf(&xml, &|p: &str| std::fs::read(dir.join(p)).ok()) {
            Ok(t) => t,
            Err(e) => {
                refused += 1;
                notes.push(format!("{}: refused: {e}", m.rel));
                continue;
            }
        };
        nmodels += 1;
        // the compiled hierarchies
        let ours = t.body_bvhs();
        for (b, want) in m.bodies.iter().enumerate() {
            bodies += 1;
            nodes += want.nodes.len();
            let Some(&(got, margin)) = ours.get(b) else {
                notes.push(format!("{}: body {b} missing here", m.rel));
                continue;
            };
            let same = margin.to_bits() == want.margin.to_bits()
                && match got {
                    None => want.bvhadr < 0,
                    Some(h) => {
                        h.aabb.len() == want.nodes.len()
                            && want.nodes.iter().enumerate().all(|(k, &(c0, c1, id, depth, a))| h.child[k] == [c0, c1] && h.nodeid[k] == id && h.depth[k] == depth && h.aabb[k].iter().zip(&a).all(|(x, y)| x.to_bits() == y.to_bits()))
                    }
                };
            if same {
                bodies_exact += 1;
            } else if notes.len() < 30 {
                let detail = match got {
                    None => format!("none here, MuJoCo has {} nodes", want.nodes.len()),
                    Some(h) => {
                        let k = (0..h.aabb.len().min(want.nodes.len())).find(|&k| {
                            let (c0, c1, id, depth, a) = want.nodes[k];
                            h.child[k] != [c0, c1] || h.nodeid[k] != id || h.depth[k] != depth || h.aabb[k].iter().zip(&a).any(|(x, y)| x.to_bits() != y.to_bits())
                        });
                        match k {
                            Some(k) => format!("{} vs {} nodes; first difference at node {k}: ours {:?} {:?} {} vs {:?}", h.aabb.len(), want.nodes.len(), h.child[k], h.aabb[k], h.nodeid[k], want.nodes[k]),
                            None => format!("{} vs {} nodes; margin {:e} vs {:e}", h.aabb.len(), want.nodes.len(), margin, want.margin),
                        }
                    }
                };
                notes.push(format!("{}: body {b}: {detail}", m.rel));
            }
        }
        for (g, want) in m.gaabb.iter().enumerate() {
            gaabb += 1;
            if t.geoms.get(g).is_some_and(|x| x.aabb.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits())) {
                gaabb_exact += 1;
            } else if notes.len() < 30 {
                notes.push(format!("{}: geom_aabb {g}: ours {:?} vs {:?}", m.rel, t.geoms.get(g).map(|x| x.aabb), want));
            }
        }
        // what the driver decided at each state
        let adr_of: std::collections::HashMap<usize, (usize, usize)> = m.bodies.iter().enumerate().flat_map(|(b, body)| (0..body.nodes.len()).map(move |n| (body.bvhadr as usize + n, (b, n)))).collect();
        for (s, st) in m.states.iter().enumerate() {
            states += 1;
            ncontacts += st.contacts.len();
            nactive += st.active.len();
            let c = t.collide_qpos(&st.qpos);
            let want_active: BTreeSet<(usize, usize)> = st.active.iter().map(|a| adr_of[a]).collect();
            let got_active: BTreeSet<(usize, usize)> = c.bvh_active.iter().copied().collect();
            if want_active == got_active {
                active_same += 1;
            } else if notes.len() < 30 {
                let only_mj: Vec<_> = want_active.difference(&got_active).take(6).collect();
                let only_ours: Vec<_> = got_active.difference(&want_active).take(6).collect();
                notes.push(format!("{} state {s}: active nodes differ — only MuJoCo {only_mj:?}, only ours {only_ours:?}", m.rel));
            }
            let got: Vec<(usize, usize, f64)> = c.contacts.iter().map(|k| (k.geom[0], k.geom[1], k.record.dist)).collect();
            if got.len() == st.contacts.len() && got.iter().zip(&st.contacts).all(|(a, b)| a.0 == b.0 && a.1 == b.1 && a.2.to_bits() == b.2.to_bits()) {
                contacts_same += 1;
            } else if contact_notes.len() < 30 {
                let k = got.iter().zip(&st.contacts).position(|(a, b)| a.0 != b.0 || a.1 != b.1 || a.2.to_bits() != b.2.to_bits());
                contact_notes.push(format!(
                    "{} state {s}: contacts differ — {} here, {} in MuJoCo; first difference at {k:?}: ours {:?} vs {:?}",
                    m.rel,
                    got.len(),
                    st.contacts.len(),
                    k.and_then(|k| got.get(k)),
                    k.and_then(|k| st.contacts.get(k))
                ));
            }
        }
    }
    println!("collision driver against MuJoCo 3.13.0, BIT FOR BIT: {nmodels} models ({refused} refused)");
    println!("  body hierarchies (bvh_aabb, child, nodeid, depth, body_margin): {bodies_exact} of {bodies} bodies ({nodes} nodes)");
    println!("  geom_aabb: {gaabb_exact} of {gaabb}");
    println!("  states: {states}; the nodes the midphase marked (bvh_active, {nactive} in MuJoCo) identical on {active_same}; contacts identical (count, geoms, order, dist) on {contacts_same} ({ncontacts} MuJoCo contacts)");
    for n in contact_notes.iter().chain(&notes) {
        println!("  {n}");
    }
}
