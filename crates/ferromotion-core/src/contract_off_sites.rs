//! Contract-off pins for the three sites unfused in `f59d562`.
//!
//! Expected bits are the oracle's. `qfrc_spring`, `efc_R`, `efc_D`, and `efc_diagA` are the
//! hex floats in `x86-native.txt` sha256
//! `a3168d763bed63c9181275b6a908348c0c0609fde425be1ece840c75d110b5b7`, written by libmujoco
//! `64156abef7859ac3cedfa4f589c318ff642d1d5d8d0e078efabff21ab02040bc` for Menagerie
//! `4d038b3feae26ec82b46a4d586379114012a8ac7`. Contact position is not a field of that dump.
//! `berkeley_humanoid/scene.xml` state 2 contact 30 `pos.x` was read from that same library
//! with `mj_forward` at the dump's `qpos`/`qvel`/`ctrl`, and is `0x3fe30ae42ddd7fc3`.
//!
//! The checked-in meshes are a tetrahedron. Those arrays do not depend on the visual mesh:
//! the same library produced these bits on the original meshes and on the tetrahedron.
//! The published default is not selected here.

use crate::tree_from_mjcf;
use std::path::{Path, PathBuf};

fn f64s(bits: &[u64]) -> Vec<f64> {
    bits.iter().copied().map(f64::from_bits).collect()
}

fn assert_bits(got: &[f64], want: &[u64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.to_bits(), *w, "{what}[{i}] {g:?}");
    }
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        crate::reset_policy();
    }
}

fn contract_off() -> Reset {
    crate::install_policy(
        crate::ORACLE_X86_CONTRACT_OFF_ID,
        crate::ORACLE_X86_CONTRACT_OFF_SHA256,
        "",
    )
    .unwrap();
    Reset
}

fn load(rel: &str) -> crate::MjcfTree {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/contract-off");
    let path = root.join(rel);
    let xml =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let base = path.parent().unwrap().to_path_buf();
    tree_from_mjcf(&xml, &|p: &str| std::fs::read(resolve(&base, p)).ok())
        .unwrap_or_else(|e| panic!("load {rel}: {e}"))
}

fn resolve(base: &Path, p: &str) -> PathBuf {
    let direct = base.join(p);
    if direct.is_file() {
        return direct;
    }
    base.join(Path::new(p).file_name().unwrap_or_default())
        .to_path_buf()
}

const TETHERIA_QPOS: &[u64] = &[
    4602207767682083636,
    4607415914645390346,
    4602089868675335815,
    4601064718653230893,
    4599061880511341895,
    4604524522287429348,
    4608549516150593662,
    4609637975798798020,
    4606395906572030471,
    4597703223117344230,
    4599626296435071516,
    4605227513452123772,
    4607035516001005626,
    4605419591279292562,
    4608540174996434186,
    4602345875643268104,
];

const TETHERIA_QVEL: &[u64] = &[
    13824374189303430368,
    4601415798370847680,
    4604541153467293288,
    13820234608426214032,
    4603862746389609270,
    4602891119569725232,
    4589502259171320768,
    13806932200619950016,
    13821030667219708728,
    4601932562256601656,
    4604012264964273548,
    4600670994517948996,
    4602782067767337458,
    4603097164435791396,
    13829792067690783332,
    13823024675736909180,
];

const TETHERIA_CTRL: &[u64] = &[
    4592493127423856390,
    4589763257598343134,
    4591139560350473966,
    4591938230710277772,
    4592561350041312144,
    4583569372457320792,
    4591241399507637430,
];

const TETHERIA_QFRC_SPRING: &[u64] = &[
    4567108914343049863,
    13819171403040943709,
    0,
    4567832982974369599,
    13818377479115138888,
    0,
    13771965208088761771,
    13820113947836180617,
    0,
    4570129696964371678,
    13818295099310919969,
    0,
    13594467030924241550,
    13813937119716669110,
    13822883473725928293,
    0,
];

const BERKELEY_QPOS: &[u64] = &[
    4602803107748637922,
    13822505981320689932,
    13797169536238300672,
    4606136903014292384,
    4605776133779214710,
    4574560201851604985,
    4608995236786519405,
    4597861735798061944,
    4589227523845311880,
    13822607794333786724,
    4609295271452409811,
    4586747217810991808,
    4591412586916411184,
    4568107372753941248,
    13826112284964193245,
    13819315862880541320,
    4609175508047570175,
    4594484349551082780,
    13820669486706786828,
];

const ANYMAL0_QPOS: &[u64] = &[
    0,
    0,
    4603759683083215831,
    0,
    0,
    0,
    4607182418800017408,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

const ANYMAL0_QVEL: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const ANYMAL0_CTRL: &[u64] = &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

const ANYMAL0_EFC_R: &[u64] = &[
    4584217575720935433,
    4584217575720935433,
    4584217575720935433,
    4584217575720935433,
    4584217514836025400,
    4584217514836025400,
    4584217514836025400,
    4584217514836025400,
    4584220345751425418,
    4584220345751425418,
    4584220345751425418,
    4584220345751425418,
    4584220405928359560,
    4584220405928359560,
    4584220405928359560,
    4584220405928359560,
];

const ANYMAL0_EFC_D: &[u64] = &[
    4629935501916048364,
    4629935501916048364,
    4629935501916048364,
    4629935501916048364,
    4629935535619844663,
    4629935535619844663,
    4629935535619844663,
    4629935535619844663,
    4629933969029181751,
    4629933969029181751,
    4629933969029181751,
    4629933969029181751,
    4629933935739315534,
    4629933935739315534,
    4629933935739315534,
    4629933935739315534,
];

const ANYMAL2_QPOS: &[u64] = &[
    4592037128677224432,
    4606126458205798266,
    13830184497965481112,
    13822151467192386456,
    13826071199134661272,
    4588240744333139505,
    13827046397362040481,
    13826292994305262007,
    13823548451187058656,
    4619700492505010114,
    13823042407530694668,
    13844389848101720837,
    4599286644511301760,
    13827352413092173209,
    13839937337911520960,
    13844169136496739016,
    13822512435659122962,
    4613886119749705032,
    13834787733189688800,
];

const ANYMAL2_QVEL: &[u64] = &[
    13829372964575150524,
    4600778342306358064,
    4599728061658020364,
    4591672046983041408,
    13826580156718886354,
    4603683571972063244,
    13821826575245176024,
    4607131390214331856,
    4593845318749575536,
    13823090284054886684,
    13824376954625405880,
    13797242467695616512,
    4593241662367750848,
    4599163996543862196,
    4603791634233430392,
    4591525889563687328,
    4604203838229585660,
    4601902981192386416,
];

const ANYMAL2_CTRL: &[u64] = &[
    13840125267196957478,
    13838371448359579488,
    4597446530483993760,
    13842977109405368164,
    13827477530737890568,
    4616524604424112105,
    13843198629093265500,
    4620098356832171869,
    4616757836554492387,
    13836919653023208582,
    4619229164430347929,
    4619019104651907265,
];

const ANYMAL2_EFC_DIAGA: &[u64] = &[
    4576550981167895474,
    4576550981167895474,
    4576550981167895474,
    4576550981167895474,
    4576550942779561454,
    4576550942779561454,
    4576550942779561454,
    4576550942779561454,
    4576552727690238027,
    4576552727690238027,
    4576552727690238027,
    4576552727690238027,
    4576552765632188618,
    4576552765632188618,
    4576552765632188618,
    4576552765632188618,
];

#[test]
fn contract_off_tendon_spring_matches_the_oracle_bits() {
    let _guard = contract_off();
    let tree = load("tetheria_aero_hand_open/left_hand.xml");
    let qpos = f64s(TETHERIA_QPOS);
    let qvel = f64s(TETHERIA_QVEL);
    let ctrl = f64s(TETHERIA_CTRL);
    let rows: std::collections::BTreeMap<&str, Vec<f64>> = tree
        .mujoco_native_forward(&qpos, &qvel, &ctrl, &[])
        .into_iter()
        .collect();
    let spring = rows.get("qfrc_spring").expect("qfrc_spring");
    assert_bits(spring, TETHERIA_QFRC_SPRING, "qfrc_spring");
}

#[test]
fn contract_off_plane_cylinder_contact_matches_the_oracle_bits() {
    let _guard = contract_off();
    let tree = load("berkeley_humanoid/scene.xml");
    let qpos = f64s(BERKELEY_QPOS);
    let contacts = tree.collide_qpos(&qpos).contacts;
    assert!(contacts.len() > 30, "contacts {}", contacts.len());
    assert_eq!(
        contacts[30].record.pos.x.to_bits(),
        0x3fe30ae42ddd7fc3,
        "berkeley_humanoid/scene.xml state 2 contact 30 pos.x"
    );
}

#[test]
fn contract_off_pyramidal_rows_match_the_oracle_bits() {
    let _guard = contract_off();
    let tree = load("anybotics_anymal_c/scene_mjx.xml");
    let rows0: std::collections::BTreeMap<&str, Vec<f64>> = tree
        .mujoco_native_forward(
            &f64s(ANYMAL0_QPOS),
            &f64s(ANYMAL0_QVEL),
            &f64s(ANYMAL0_CTRL),
            &[],
        )
        .into_iter()
        .collect();
    assert_bits(
        rows0.get("efc_R").expect("efc_R"),
        ANYMAL0_EFC_R,
        "efc_R state 0",
    );
    assert_bits(
        rows0.get("efc_D").expect("efc_D"),
        ANYMAL0_EFC_D,
        "efc_D state 0",
    );
    let rows2: std::collections::BTreeMap<&str, Vec<f64>> = tree
        .mujoco_native_forward(
            &f64s(ANYMAL2_QPOS),
            &f64s(ANYMAL2_QVEL),
            &f64s(ANYMAL2_CTRL),
            &[],
        )
        .into_iter()
        .collect();
    assert_bits(
        rows2.get("efc_diagA").expect("efc_diagA"),
        ANYMAL2_EFC_DIAGA,
        "efc_diagA state 2",
    );
}
