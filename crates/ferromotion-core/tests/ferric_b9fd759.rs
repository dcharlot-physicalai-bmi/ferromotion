//! Ferric chunk-v3 codec fixtures at toolchain `b9fd759` and fleet `160b0be`.
//! The contract text stays `bc5f67f0`. These are codec fixtures, not model output.

use ferromotion_core::{
    BindingState, ChunkBinding, ChunkFacts, EMPTY_GOLDEN_DIGEST, EpisodeEnvelopeV2,
    FERRIC_CONTRACT_REVISION, FERRIC_EMPTY_FIXTURE_REVISION, FERRIC_V3_CODEC_FLEET_REVISION,
    FERRIC_V3_CODEC_REVISION, FERRIC_V3_CONTRACT_SHA256, GOLDEN_DIGEST, NsRational, V3Error,
    decode_action_chunk_v3, digest_sha256, predict_action_chunk, steps_per_action,
    verify_chunk_binding,
};

fn file(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/acceptance/contracts/fixtures/ferric-b9fd759")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn json_strings(text: &str, key: &str) -> Vec<String> {
    let pat = format!("\"{key}\": \"");
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(at) = rest.find(&pat) {
        let start = at + pat.len();
        let end = rest[start..].find('"').unwrap();
        out.push(rest[start..start + end].to_string());
        rest = &rest[start + end..];
    }
    out
}

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn envelope() -> EpisodeEnvelopeV2 {
    let mut nonce = [0u8; 32];
    for (i, byte) in nonce.iter_mut().enumerate() {
        *byte = i as u8;
    }
    EpisodeEnvelopeV2::try_new(
        1,
        NsRational::try_new(5_000_000, 3).unwrap(),
        NsRational::try_new(100_000_000, 3).unwrap(),
        0,
        nonce,
    )
    .unwrap()
}

fn bind(bytes: &[u8], index: u64, previous: Option<&BindingState>) -> BindingState {
    let (chunk, digest) = decode_action_chunk_v3(bytes).unwrap();
    assert_eq!(digest_sha256(bytes), hex_of(&digest));
    let env = envelope();
    assert_eq!(env.steps_per_action(), 20);
    assert_eq!(steps_per_action(env.period(), env.dt()).unwrap(), 20);
    let t0 = NsRational::try_new(chunk.t0_num_ns, chunk.t0_den).unwrap();
    let period = NsRational::try_new(chunk.period_num_ns, chunk.period_den).unwrap();
    assert_eq!(period, env.period());
    let binding = ChunkBinding::for_envelope(&env, index, t0, chunk.n_action_steps, digest);
    let facts = ChunkFacts {
        t0,
        period,
        n_action_steps: chunk.n_action_steps,
    };
    verify_chunk_binding(&env, &binding, &facts, previous).unwrap()
}

/// Byte offset of the `u64` seed: magic, version, two strings, two hashes,
/// the stats key, then five hashes.
fn seed_at(bytes: &[u8]) -> usize {
    let mut i = b"FERRIC-SMOLVLA-CHUNK\0".len() + 4;
    for _ in 0..2 {
        let n = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4 + n;
    }
    i += 64 * 2;
    let n = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
    i += 4 + n;
    i + 64 * 5
}

fn hex_of(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn ferric_b9fd759_v3_fixtures_match_recomputed_sha256_and_bind() {
    assert_eq!(
        FERRIC_CONTRACT_REVISION,
        "3ee11c015e3478c7c4aa0409499346f79ab1c8ce"
    );
    assert_eq!(
        FERRIC_EMPTY_FIXTURE_REVISION,
        "c05228929efc3bcb4210b475dd959370fa69893a"
    );
    assert_eq!(
        FERRIC_V3_CONTRACT_SHA256,
        "bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd"
    );
    assert_eq!(
        FERRIC_V3_CODEC_REVISION,
        "b9fd7597c24c6f05b162c572a126749998f5c5a2"
    );
    assert_eq!(
        FERRIC_V3_CODEC_FLEET_REVISION,
        "160b0be18f486cf1697272c6d462d1fa1e140b0e"
    );

    let v2 = include_str!("../../../docs/acceptance/contracts/fixtures/smolvla-chunk-v2.json");
    let v2_empty =
        include_str!("../../../docs/acceptance/contracts/fixtures/smolvla-empty-chunk-v2.json");
    let v2_bytes = from_hex(&json_strings(v2, "canonical_hex")[0]);
    let v2_empty_bytes = from_hex(&json_strings(v2_empty, "canonical_hex")[0]);
    assert_eq!(v2_bytes.len(), 465);
    assert_eq!(v2_empty_bytes.len(), 449);
    assert_eq!(digest_sha256(&v2_bytes), GOLDEN_DIGEST);
    assert_eq!(digest_sha256(&v2_empty_bytes), EMPTY_GOLDEN_DIGEST);

    let protocol = file("smolvla-chunk-v3.json");
    let real = file("smolvla-chunk-v3-real-noise.json");
    let sequence = file("smolvla-chunk-v3-sequence.json");
    let record = file("smolvla-v3-fixture-record.json");
    assert_eq!(protocol.len(), 2524);
    assert_eq!(real.len(), 5412);
    assert_eq!(sequence.len(), 5390);
    assert_eq!(record.len(), 2303);
    assert_eq!(
        digest_sha256(&protocol),
        "a500b2e9dfce00d070c9b0f825d53edb49a19933ab992410e7dd2d23721442bc"
    );
    assert_eq!(
        digest_sha256(&real),
        "44e444846a3aa61987201346fb80bd18bc58a642ccb3822ff9b5b6bb751b6742"
    );
    assert_eq!(
        digest_sha256(&sequence),
        "8dbd31344e8fd9b1a12db9562bfc5787ab420a9cf2ec33079a931736f0fc8f1c"
    );
    assert_eq!(
        digest_sha256(&record),
        "e89abc688a40e2702049e6175f316691bbb646e64aa4f7dff3670c96e7777dd3"
    );

    let protocol_text = String::from_utf8(protocol).unwrap();
    let protocol_bytes = from_hex(&json_strings(&protocol_text, "canonical_hex")[0]);
    assert_eq!(protocol_bytes.len(), 943);
    assert_eq!(
        digest_sha256(&protocol_bytes),
        "4e1722a629708d81221e5c75c1946dcd28af40ab639239bde1ea3893dbb643e5"
    );
    let (chunk, _) = decode_action_chunk_v3(&protocol_bytes).unwrap();
    assert_eq!(chunk.seed, 3);
    assert_eq!(chunk.n_action_steps, 1);
    assert_eq!((chunk.t0_num_ns, chunk.t0_den), (100_000_000, 3));
    bind(&protocol_bytes, 0, None);
    assert_eq!(
        predict_action_chunk().unwrap_err(),
        V3Error::ProposedEntryInvoked
    );

    let real_text = String::from_utf8(real).unwrap();
    let real_bytes = from_hex(&json_strings(&real_text, "canonical_hex")[0]);
    assert_eq!(real_bytes.len(), 2119);
    assert_eq!(
        digest_sha256(&real_bytes),
        "13e82f106898b5a2b1c6fb48f487cf7d6cbe81cb2f8410a71e8f626a2f0c329b"
    );
    let (real_chunk, _) = decode_action_chunk_v3(&real_bytes).unwrap();
    assert_eq!(real_chunk.n_action_steps, 50);
    assert_eq!(real_chunk.seed, 0x5a17);
    assert_eq!(
        real_chunk.noise_sha256,
        "8db2ebc813c83673d6b61b10b2c2a2352018371cbfe9c661f3b405ac6370b61c"
    );
    bind(&real_bytes, 0, None);
    assert_eq!(
        predict_action_chunk().unwrap_err(),
        V3Error::ProposedEntryInvoked
    );

    let sequence_text = String::from_utf8(sequence).unwrap();
    let hexes = json_strings(&sequence_text, "canonical_hex");
    let first = from_hex(&hexes[0]);
    let second = from_hex(&hexes[1]);
    assert_eq!(first.len(), 943);
    assert_eq!(second.len(), 943);
    assert_eq!(
        digest_sha256(&first),
        "7700ea35fbdc35587a111acb1dd605826fb21088da84ec17613af8598799c1c7"
    );
    assert_eq!(
        digest_sha256(&second),
        "516bb02fe060793533d079c60987137f6a659226852a2cd58b893cb39bf943ba"
    );
    let (zero, _) = decode_action_chunk_v3(&first).unwrap();
    let (one, _) = decode_action_chunk_v3(&second).unwrap();
    assert_eq!((zero.t0_num_ns, zero.t0_den), (0, 1));
    assert_eq!((one.t0_num_ns, one.t0_den), (100_000_000, 3));
    let accepted = bind(&first, 0, None);
    bind(&second, 1, Some(&accepted));
}

#[test]
fn v3_decoder_refuses_an_even_seed_and_a_placeholder_hash() {
    let protocol = String::from_utf8(file("smolvla-chunk-v3.json")).unwrap();
    let mut bytes = from_hex(&json_strings(&protocol, "canonical_hex")[0]);
    let at = seed_at(&bytes);
    assert_eq!(&bytes[at..at + 8], &3u64.to_le_bytes());
    bytes[at..at + 8].copy_from_slice(&2u64.to_le_bytes());
    assert_eq!(
        decode_action_chunk_v3(&bytes).unwrap_err(),
        V3Error::EvenSeed
    );

    let mut placeholder = from_hex(&json_strings(&protocol, "canonical_hex")[0]);
    let weight = b"7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb";
    let hash_at = placeholder.windows(64).position(|w| w == weight).unwrap();
    placeholder[hash_at..hash_at + 64].fill(b'1');
    assert_eq!(
        decode_action_chunk_v3(&placeholder).unwrap_err(),
        V3Error::MissingHash
    );
}
