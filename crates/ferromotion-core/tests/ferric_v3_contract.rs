//! Ferric chunk-v3 contract text. The codec is not in this tree, so this test
//! pins the file and does not claim a codec verdict.

use ferromotion_core::{
    FERRIC_CONTRACT_REVISION, FERRIC_EMPTY_FIXTURE_REVISION, FERRIC_V3_CONTRACT_BLOB,
    FERRIC_V3_CONTRACT_FLEET_REVISION, FERRIC_V3_CONTRACT_REVISION, FERRIC_V3_CONTRACT_SHA256,
    digest_sha256,
};

fn contract_bytes() -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/acceptance/contracts/fixtures/ferric-9b0a2ffa/smolvla-ferromotion-v3.md");
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn ferric_v3_contract_matches_the_signed_off_bytes_and_the_v2_pins_stay() {
    assert_eq!(
        FERRIC_CONTRACT_REVISION,
        "3ee11c015e3478c7c4aa0409499346f79ab1c8ce"
    );
    assert_eq!(
        FERRIC_EMPTY_FIXTURE_REVISION,
        "c05228929efc3bcb4210b475dd959370fa69893a"
    );
    assert_eq!(
        FERRIC_V3_CONTRACT_REVISION,
        "9b0a2ffab2dcf902e0bec726cf30c437c69a6e84"
    );
    assert_eq!(
        FERRIC_V3_CONTRACT_FLEET_REVISION,
        "13ffe43a30ade6f514697286a2af00f0bbbc6c2a"
    );
    assert_eq!(
        FERRIC_V3_CONTRACT_BLOB,
        "b178ba4dfa051412653a9694211d383e2fa049c6"
    );

    let bytes = contract_bytes();
    assert_eq!(bytes.len(), 21831);
    assert_eq!(digest_sha256(&bytes), FERRIC_V3_CONTRACT_SHA256);
    assert_eq!(
        FERRIC_V3_CONTRACT_SHA256,
        "bc5f67f054b1a84226d0438f959750697fea7a9c9eeca417b134b39a3e5c19fd"
    );

    let text = String::from_utf8(bytes).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[245], "## SO-100 profile clock");
    assert_eq!(
        lines[247],
        "The envelope itself carries any reduced period and any reduced dt. `ferromorphic-episode 2` does not refuse a different pair."
    );
    assert_eq!(
        lines[249],
        "The ferric SO-100 profile is what refuses anything else. For this profile the only legal clock is 30 Hz with 20 substeps. `period_num_ns / period_den` is `100000000/3`. `dt_num_ns / dt_den` is `5000000/3`, which is that period divided by 20. Both fractions are already reduced (`gcd` with 3 is 1). Any other period or dt is refused."
    );
}
