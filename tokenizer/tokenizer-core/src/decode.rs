//! Decoding: token ids back to bytes to `str`.
//!
//! Every id under `vocab_size` maps to bytes (normal tokens directly,
//! specials through their strings), so decoding always lands somewhere.
//! The only bit that can fail is UTF-8 checking: [`decode_lossy`] swaps in
//! the replacement char (the default), [`decode_strict`] errors.

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::vocab::Tokenizer;

/// Raw bytes for `ids`. Out-of-range ids come back as [`Error::InvalidId`].
pub fn decode_to_bytes(tok: &Tokenizer, ids: &[u32]) -> Result<Vec<u8>> {
    let vocab_size = tok.vocab_size() as u32;
    let mut out = Vec::new();
    for &id in ids {
        if id >= vocab_size {
            return Err(Error::InvalidId(id));
        }
        if let Some(b) = tok.token_bytes(id) {
            out.extend_from_slice(b);
        } else {
            // A special token: index into the specials table.
            let idx = (id as usize) - tok.all_token_bytes().len();
            match tok.special_tokens().get(idx) {
                Some(s) => out.extend_from_slice(s.as_bytes()),
                None => return Err(Error::InvalidId(id)),
            }
        }
    }
    Ok(out)
}

/// Decode with the replacement-character strategy for invalid UTF-8.
/// Never fails on id range (still validates) and never panics.
pub fn decode_lossy(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    Ok(String::from_utf8_lossy(&decode_to_bytes(tok, ids)?).into_owned())
}

/// Decode many id sequences. Large batches spread over rayon; small ones
/// stay sequential. Output order equals input order.
pub fn decode_batch_lossy(tok: &Tokenizer, batches: &[Vec<u32>]) -> Result<Vec<String>> {
    if batches.len() <= 1 || rayon::current_num_threads() <= 1 {
        return batches.iter().map(|ids| decode_lossy(tok, ids)).collect();
    }
    batches
        .par_iter()
        .map(|ids| decode_lossy(tok, ids))
        .collect()
}

/// Decode, raising [`Error::StrictUtf8`] on invalid UTF-8.
pub fn decode_strict(tok: &Tokenizer, ids: &[u32]) -> Result<String> {
    Ok(String::from_utf8(decode_to_bytes(tok, ids)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_id_rejected() {
        let tok = Tokenizer::from_merges(vec![], vec![]).expect("build");
        let err = decode_to_bytes(&tok, &[999]).unwrap_err();
        assert!(matches!(err, Error::InvalidId(999)));
    }

    #[test]
    fn lossy_vs_strict_on_split_emoji() {
        // '🎉' = F0 9F 8E 89. Feed only the first two bytes.
        let tok = Tokenizer::from_merges(vec![], vec![]).expect("build");
        let ids = vec![0xF0u32, 0x9Fu32];
        // An incomplete sequence becomes a single U+FFFD, not one per byte.
        assert_eq!(decode_lossy(&tok, &ids).expect("lossy"), "�");
        assert!(matches!(
            decode_strict(&tok, &ids).unwrap_err(),
            Error::StrictUtf8(_)
        ));
    }

    #[test]
    fn batch_decode_matches_single() {
        let tok = Tokenizer::from_merges(vec![(104, 105)], vec!["<|endoftext|>".to_owned()])
            .expect("build");
        let batches: Vec<Vec<u32>> = vec![
            vec![104, 105],
            vec![256],
            vec![],
            vec![257, 104, 105],
            vec![72, 101, 108, 108, 111],
        ];
        let single: Vec<String> = batches
            .iter()
            .map(|ids| decode_lossy(&tok, ids).expect("single"))
            .collect();
        assert_eq!(decode_batch_lossy(&tok, &batches).expect("batch"), single);
        assert_eq!(
            decode_batch_lossy(&tok, &[]).expect("empty"),
            Vec::<String>::new()
        );
    }
}
