//! Cross-check of ferromorphic `30812e26027cafa50e17c848e5afe57b1f246838`
//! (branch `cursor/scientific-verification-ci-5e0f`) against this receiver.
//!
//! Fixture bytes are copied under
//! `docs/acceptance/contracts/fixtures/ferromorphic-30812e26/` with
//! `PROVENANCE.txt`. The ferric chunk revision pin is not moved.

use ferromotion_core::{
    CodecError, EpisodeEnvelope, FERRIC_CONTRACT_REVISION, FERRIC_EMPTY_FIXTURE_REVISION,
    TimebaseError, authorize_rollout, check_horizon, digest_sha256, parse_chunk, verify_chunk,
};

/// Same bound as `smolvla_receiver::CHUNK_SIZE` and ferromorphic `MAX_ACTION_STEPS`.
const CHUNK_SIZE: u32 = 50;

fn file(name: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/acceptance/contracts/fixtures/ferromorphic-30812e26")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn text(name: &str) -> String {
    String::from_utf8(file(name)).unwrap()
}

fn sha(bytes: &[u8]) -> String {
    digest_sha256(bytes)
}

#[test]
fn ferromorphic_30812e26_fixtures_match_their_sha256_and_our_verdicts() {
    assert_eq!(
        FERRIC_CONTRACT_REVISION,
        "3ee11c015e3478c7c4aa0409499346f79ab1c8ce"
    );
    assert_eq!(
        FERRIC_EMPTY_FIXTURE_REVISION,
        "c05228929efc3bcb4210b475dd959370fa69893a"
    );

    let episode_20 = file("episode-v2-20ms.txt");
    let episode_30 = file("episode-v2-30hz.txt");
    let binding_full = file("chunk-binding-v1-nonempty.txt");
    let binding_empty = file("chunk-binding-v1-empty.txt");
    let chunk_full = file("ferric-v2-nonempty.bin");
    let chunk_empty = file("ferric-v2-empty.bin");
    assert_eq!(
        sha(&episode_20),
        "dd525f577974d4065d04cb546a1d9636d1006bce40049f630d1275a4448510d3"
    );
    assert_eq!(
        sha(&episode_30),
        "4463c0471d8d4bb0451d3170e9a9dcd337662679225639345427f450b380b6ba"
    );
    assert_eq!(
        sha(&binding_full),
        "a16fe7ec9ff4fc20cbc5971e160ad1b724468bac23681474545be39012f3f174"
    );
    assert_eq!(
        sha(&binding_empty),
        "f9c519d86ad155cc57fba372e4e11d3fb62208802b9f013f8b98d14ddf918f60"
    );
    assert_eq!(
        sha(&chunk_full),
        "6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86"
    );
    assert_eq!(
        sha(&chunk_empty),
        "6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e"
    );

    let manifest = text("episode-v2-golden.sha256");
    for (name, digest) in [
        (
            "episode-v2-20ms.txt",
            "dd525f577974d4065d04cb546a1d9636d1006bce40049f630d1275a4448510d3",
        ),
        (
            "episode-v2-30hz.txt",
            "4463c0471d8d4bb0451d3170e9a9dcd337662679225639345427f450b380b6ba",
        ),
        (
            "chunk-binding-v1-nonempty.txt",
            "a16fe7ec9ff4fc20cbc5971e160ad1b724468bac23681474545be39012f3f174",
        ),
        (
            "chunk-binding-v1-empty.txt",
            "f9c519d86ad155cc57fba372e4e11d3fb62208802b9f013f8b98d14ddf918f60",
        ),
    ] {
        assert!(
            manifest.contains(&format!("{digest}  {name}")),
            "{name} missing from the copied manifest"
        );
    }

    // Envelope v2. Our parser is envelope v1 only. The header is not
    // `ferromorphic-episode 1`, so the refusal is `BadTick`, not ferromorphic's
    // v1 `UnknownLine` and not a v2 parse. `dt` as a rational, `period` as a
    // rational, and `episode_nonce` have no field.
    for name in ["episode-v2-20ms.txt", "episode-v2-30hz.txt"] {
        let body = text(name);
        assert_eq!(
            EpisodeEnvelope::from_text(&body),
            Err(TimebaseError::BadTick),
            "{name}"
        );
        assert_eq!(
            parse_chunk(body.as_bytes()),
            Err(CodecError::ChunkShape),
            "{name}"
        );
        assert_eq!(
            verify_chunk(body.as_bytes()),
            Err(CodecError::ChunkShape),
            "{name}"
        );
        assert!(
            !body.contains("n_action_steps"),
            "{name} has no step count for check_horizon"
        );
    }

    // Binding 1 text. There is no binding parser. The envelope parser refuses the
    // header. The chunk parsers refuse the bytes because they are not a Ferric body.
    // Rational `t0` and `period` have no field.
    for (name, n) in [
        ("chunk-binding-v1-nonempty.txt", 2u32),
        ("chunk-binding-v1-empty.txt", 0),
    ] {
        let body = text(name);
        assert_eq!(
            EpisodeEnvelope::from_text(&body),
            Err(TimebaseError::BadTick),
            "{name}"
        );
        assert_eq!(
            parse_chunk(body.as_bytes()),
            Err(CodecError::ChunkShape),
            "{name}"
        );
        assert_eq!(
            verify_chunk(body.as_bytes()),
            Err(CodecError::ChunkShape),
            "{name}"
        );
        assert!(body.contains(&format!("n_action_steps {n}")), "{name}");
        assert_eq!(
            check_horizon(n, CHUNK_SIZE),
            if n == 0 {
                Err(CodecError::Horizon)
            } else {
                Ok(())
            },
            "{name}"
        );
    }

    // The chunk the nonempty binding names. Parses and verifies. Rollout stays
    // refused because the entry is still proposed. That is not a binding-grid error.
    let parsed = parse_chunk(&chunk_full).unwrap();
    assert_eq!(parsed.n_action_steps, 2);
    assert_eq!(verify_chunk(&chunk_full).unwrap(), parsed);
    assert_eq!(authorize_rollout(&parsed), Err(CodecError::EntryProposed));
    assert_eq!(check_horizon(parsed.n_action_steps, CHUNK_SIZE), Ok(()));

    // The chunk the empty binding names. Ferromorphic parses the binding text and
    // `verify_chunk_binding` returns `NoActionSteps`. This receiver refuses the
    // chunk bytes themselves with `Horizon`, so the class name differs and there
    // is no separate "parsed, then NoActionSteps" stage.
    assert_eq!(parse_chunk(&chunk_empty), Err(CodecError::Horizon));
    assert_eq!(verify_chunk(&chunk_empty), Err(CodecError::Horizon));
    assert_eq!(check_horizon(0, CHUNK_SIZE), Err(CodecError::Horizon));
    assert_eq!(check_horizon(51, CHUNK_SIZE), Err(CodecError::Horizon));
    assert_eq!(check_horizon(1, CHUNK_SIZE), Ok(()));
    assert_eq!(check_horizon(50, CHUNK_SIZE), Ok(()));

    // No type here implements RepeatedT0, Late, BeforeOrigin, OffGrid, or the
    // binding verifier's Overflow. `TimebaseError` has Late, OffGrid, and Overflow
    // for the clock, not for a chunk binding. A value above `u64::MAX` cannot be
    // passed to those clock functions.
}
