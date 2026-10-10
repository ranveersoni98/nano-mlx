//! `decode(encode(x)) == x` for random inputs.
//!
//! - Random `String`: lossy decode gives back exactly what went in (valid
//!   UTF-8 always survives byte-level BPE).
//! - Random `Vec<u8>`: byte-exact round trip through the bytes path
//!   (bad UTF-8 never merges, never panics).

use proptest::prelude::*;
use tokenizer_core::decode::{decode_lossy, decode_to_bytes};
use tokenizer_core::encode::{encode_bytes, encode_text};
use tokenizer_core::Tokenizer;

fn test_tok() -> Tokenizer {
    // Small but merge-rich vocab: overlapping pairs force multi-step merges.
    Tokenizer::from_merges(
        vec![
            (116, 104), // 256: th
            (256, 101), // 257: the
            (104, 101), // 258: he (higher rank than parts of "the")
            (97, 110),  // 259: an
            (259, 100), // 260: and
            (32, 32),   // 261: double space
        ],
        vec!["<|endoftext|>".to_owned()],
    )
    .expect("test vocab")
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]
    #[test]
    fn round_trip_arbitrary_string(s in "\\PC*") {
        let tok = test_tok();
        let ids = encode_text(&tok, &s, &[]).expect("encode");
        prop_assert_eq!(decode_lossy(&tok, &ids).expect("decode"), s);
    }

    #[test]
    fn round_trip_arbitrary_bytes(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let tok = test_tok();
        let ids = encode_bytes(&tok, &bytes, &[]).expect("encode");
        prop_assert_eq!(decode_to_bytes(&tok, &ids).expect("decode"), bytes);
    }

    #[test]
    fn no_panic_mixed_edge_inputs(
        s in prop::collection::vec(any::<u8>(), 0..1024),
    ) {
        // The bytes path must never panic, whatever the input.
        let tok = test_tok();
        let ids = encode_bytes(&tok, &s, &[]).expect("encode");
        let _ = decode_lossy(&tok, &ids).expect("decode");
    }
}

/// Huge single chunks have to survive. Outside `proptest!` so it runs once.
#[test]
fn huge_single_chunk_round_trip() {
    let tok = test_tok();
    for big in [vec![b'a'; 1_000_000], vec![b' '; 1_000_000]] {
        let ids = encode_bytes(&tok, &big, &[]).expect("encode big");
        assert_eq!(decode_to_bytes(&tok, &ids).expect("decode big"), big);
    }
}
