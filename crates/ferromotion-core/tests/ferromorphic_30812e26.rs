//! Envelope v2 and binding 1 against ferromorphic
//! `30812e26027cafa50e17c848e5afe57b1f246838`, spec
//! `docs/contracts/ferromotion-clock.md` in that commit.
//!
//! Full-state v1 is generated in their test and is not a fixture file yet.
//! It is not pinned here. The ferric chunk pins are not moved.

use ferromotion_core::{
    ChunkBinding, ChunkFacts, CodecError, EpisodeEnvelopeV2, EpisodeError,
    FERRIC_CONTRACT_REVISION, FERRIC_EMPTY_FIXTURE_REVISION, MAX_ACTION_STEPS, NsRational,
    authorize_rollout, check_horizon, digest_sha256, parse_chunk, verify_chunk,
    verify_chunk_binding,
};

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

fn rat(num_ns: u64, den: u64) -> NsRational {
    NsRational::try_new(num_ns, den).unwrap()
}

fn facts(t0: NsRational, period: NsRational, n: u32) -> ChunkFacts {
    ChunkFacts {
        t0,
        period,
        n_action_steps: n,
    }
}

/// `d = t0_num * period_den - origin * period_den * t0_den`, then
/// `period_num * t0_den` divides `d`. Integer only. `None` when a product overflows.
fn grid_k(origin: u64, period: NsRational, t0: NsRational) -> Option<Result<u128, ()>> {
    let lhs = u128::from(t0.num_ns()).checked_mul(u128::from(period.den()))?;
    let base = u128::from(origin)
        .checked_mul(u128::from(period.den()))?
        .checked_mul(u128::from(t0.den()))?;
    if lhs < base {
        return Some(Err(()));
    }
    let step = u128::from(period.num_ns()).checked_mul(u128::from(t0.den()))?;
    let d = lhs - base;
    if step == 0 || d % step != 0 {
        return Some(Err(()));
    }
    Some(Ok(d / step))
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

    let episode_20 = text("episode-v2-20ms.txt");
    let episode_30 = text("episode-v2-30hz.txt");
    let binding_full = text("chunk-binding-v1-nonempty.txt");
    let binding_empty = text("chunk-binding-v1-empty.txt");
    assert_eq!(
        sha(episode_20.as_bytes()),
        "dd525f577974d4065d04cb546a1d9636d1006bce40049f630d1275a4448510d3"
    );
    assert_eq!(
        sha(episode_30.as_bytes()),
        "4463c0471d8d4bb0451d3170e9a9dcd337662679225639345427f450b380b6ba"
    );
    assert_eq!(
        sha(binding_full.as_bytes()),
        "a16fe7ec9ff4fc20cbc5971e160ad1b724468bac23681474545be39012f3f174"
    );
    assert_eq!(
        sha(binding_empty.as_bytes()),
        "f9c519d86ad155cc57fba372e4e11d3fb62208802b9f013f8b98d14ddf918f60"
    );
    assert_eq!(
        sha(&file("ferric-v2-nonempty.bin")),
        "6d668d7e25c89249c890d895ef15b8fdcbb97dab7a1339a0a97605c6e1172d86"
    );
    assert_eq!(
        sha(&file("ferric-v2-empty.bin")),
        "6295818fa11b85b37ad8054c75e0a1935b0c0cbb5d7e419090c2520a17b1744e"
    );

    let e20 = EpisodeEnvelopeV2::from_text(&episode_20).unwrap();
    assert_eq!(e20.to_text(), episode_20);
    assert_eq!(e20.steps_per_action(), 10);
    assert_eq!(e20.dt_bits(), 0x3f60_624d_d2f1_a9fc);
    assert_eq!(
        (e20.dt(), e20.period()),
        (rat(2_000_000, 1), rat(20_000_000, 1))
    );

    let e30 = EpisodeEnvelopeV2::from_text(&episode_30).unwrap();
    assert_eq!(e30.to_text(), episode_30);
    assert_eq!(e30.steps_per_action(), 20);
    assert_eq!(e30.dt_bits(), 0x3f5b_4e81_b4e8_1b4f);
    assert_eq!(
        (e30.dt(), e30.period()),
        (rat(5_000_000, 3), rat(100_000_000, 3))
    );

    let full = ChunkBinding::from_text(&binding_full).unwrap();
    assert_eq!(full.to_text(), binding_full);
    let period = rat(20_000_000, 1);
    let t0 = rat(1_000_000_000, 1);
    assert_eq!(grid_k(0, period, t0), Some(Ok(50)));
    let state = verify_chunk_binding(&e20, &full, &facts(t0, period, 2), None).unwrap();
    assert_eq!((state.epoch(), state.chunk_index(), state.t0()), (1, 0, t0));

    let empty = ChunkBinding::from_text(&binding_empty).unwrap();
    assert_eq!(empty.to_text(), binding_empty);
    assert_eq!(
        verify_chunk_binding(&e20, &empty, &facts(t0, period, 0), None),
        Err(EpisodeError::NoActionSteps)
    );
    assert_eq!(check_horizon(0, MAX_ACTION_STEPS), Err(CodecError::Horizon));

    let chunk = parse_chunk(&file("ferric-v2-nonempty.bin")).unwrap();
    assert_eq!(
        verify_chunk(&file("ferric-v2-nonempty.bin"))
            .unwrap()
            .n_action_steps,
        2
    );
    assert_eq!(authorize_rollout(&chunk), Err(CodecError::EntryProposed));
    assert_eq!(
        parse_chunk(&file("ferric-v2-empty.bin")),
        Err(CodecError::Horizon)
    );
}

#[test]
fn binding_verifier_refuses_each_class_by_integer_arithmetic() {
    let env = EpisodeEnvelopeV2::from_text(&text("episode-v2-30hz.txt")).unwrap();
    let period = env.period();
    let t0 = rat(1_000_000_000, 1);
    // 1_000_000_000 * 3 = 3_000_000_000; period_num * 1 = 100_000_000; k = 30.
    assert_eq!(grid_k(0, period, t0), Some(Ok(30)));
    let at = |index: u64, t0: NsRational, n: u32| {
        ChunkBinding::for_envelope(&env, index, t0, n, [7; 32])
    };
    let first = verify_chunk_binding(&env, &at(0, t0, 50), &facts(t0, period, 50), None).unwrap();

    assert_eq!(
        verify_chunk_binding(&env, &at(0, t0, 0), &facts(t0, period, 0), None),
        Err(EpisodeError::NoActionSteps)
    );
    assert_eq!(check_horizon(0, MAX_ACTION_STEPS), Err(CodecError::Horizon));
    assert_eq!(
        verify_chunk_binding(&env, &at(0, t0, 51), &facts(t0, period, 51), None),
        Err(EpisodeError::TooManyActionSteps { n: 51 })
    );
    assert_eq!(
        check_horizon(51, MAX_ACTION_STEPS),
        Err(CodecError::Horizon)
    );

    // Half a 30 Hz period: 50_000_000/3. Remainder is not zero, so OffGrid.
    let half = rat(50_000_000, 3);
    assert_eq!(grid_k(0, period, half), Some(Err(())));
    assert_eq!(
        verify_chunk_binding(&env, &at(0, half, 5), &facts(half, period, 5), None),
        Err(EpisodeError::OffGrid)
    );

    let shifted = EpisodeEnvelopeV2::try_new(1, env.dt(), period, 5, env.episode_nonce()).unwrap();
    let before = rat(4, 1);
    // origin 5, t0 4/1: lhs = 4 * 3 = 12, base = 5 * 3 * 1 = 15, 12 < 15.
    assert!(4u128 * u128::from(period.den()) < 5u128 * u128::from(period.den()));
    assert_eq!(
        verify_chunk_binding(
            &shifted,
            &ChunkBinding::for_envelope(&shifted, 0, before, 5, [7; 32]),
            &facts(before, period, 5),
            None
        ),
        Err(EpisodeError::BeforeOrigin)
    );

    let next = rat(3_100_000_000, 3);
    assert_eq!(grid_k(0, period, next), Some(Ok(31)));
    let second = verify_chunk_binding(
        &env,
        &at(1, next, 50),
        &facts(next, period, 50),
        Some(&first),
    )
    .unwrap();
    assert_eq!(
        verify_chunk_binding(
            &env,
            &at(5, next, 50),
            &facts(next, period, 50),
            Some(&second)
        ),
        Err(EpisodeError::RepeatedT0)
    );
    let earlier = rat(2_900_000_000, 3);
    assert_eq!(grid_k(0, period, earlier), Some(Ok(29)));
    assert_eq!(
        verify_chunk_binding(
            &env,
            &at(2, earlier, 50),
            &facts(earlier, period, 50),
            Some(&second)
        ),
        Err(EpisodeError::Late)
    );

    let mut nonce = env.episode_nonce();
    nonce[0] ^= 0xff;
    let other = EpisodeEnvelopeV2::try_new(1, env.dt(), period, 0, nonce).unwrap();
    let foreign = verify_chunk_binding(
        &other,
        &ChunkBinding::for_envelope(&other, 0, t0, 5, [7; 32]),
        &facts(t0, period, 5),
        None,
    )
    .unwrap();
    assert_eq!(
        verify_chunk_binding(
            &env,
            &at(9, next, 5),
            &facts(next, period, 5),
            Some(&foreign)
        ),
        Err(EpisodeError::ForeignPrevious)
    );

    let g = text("episode-v2-20ms.txt");
    assert_eq!(
        EpisodeEnvelopeV2::from_text(
            &g.replace("origin_ns 0\n", "origin_ns 18446744073709551616\n")
        ),
        Err(EpisodeError::Overflow)
    );
    assert_eq!(
        EpisodeEnvelopeV2::from_text(&g.replace("epoch 1\n", "epoch -1\n")),
        Err(EpisodeError::BadDecimal)
    );
    assert_eq!(
        EpisodeEnvelopeV2::from_text(&g.replace("epoch 1\n", "epoch 01\n")),
        Err(EpisodeError::BadDecimal)
    );
    assert_eq!(
        EpisodeEnvelopeV2::from_text(&g.replace("epoch 1\n", "epoch +1\n")),
        Err(EpisodeError::BadDecimal)
    );
    let binding = text("chunk-binding-v1-nonempty.txt");
    assert_eq!(
        ChunkBinding::from_text(
            &binding.replace("n_action_steps 2\n", "n_action_steps 4294967296\n")
        ),
        Err(EpisodeError::Overflow)
    );
    // origin * period_den * t0_den does not fit u128: u64::MAX * 3 * u64::MAX.
    let huge_origin =
        EpisodeEnvelopeV2::try_new(1, env.dt(), period, u64::MAX, env.episode_nonce()).unwrap();
    let tiny = rat(1, u64::MAX);
    let product = u128::from(u64::MAX)
        .checked_mul(u128::from(period.den()))
        .and_then(|v| v.checked_mul(u128::from(u64::MAX)));
    assert!(product.is_none());
    assert_eq!(
        verify_chunk_binding(
            &huge_origin,
            &ChunkBinding::for_envelope(&huge_origin, 0, tiny, 5, [7; 32]),
            &facts(tiny, period, 5),
            None
        ),
        Err(EpisodeError::Overflow)
    );
    let _ = second;
}
