//! Vocabulary: merges, ranks, token bytes, special tokens.
//!
//!
//! ```text
//! 0..=255                single bytes (any input round-trips, no UNK)
//! 256..256+n_merges      merged tokens, in merge (training) order
//! 256+n_merges..         special tokens, in the order given at train time
//! ```
//!
//! Merge **rank** = index in the merge list (lower = learned earlier =
//! applied first at encode time). A pair's merged id is therefore always
//! `256 + rank`, which the encoder relies on instead of a second lookup.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;

use ahash::AHasher;

use crate::error::{Error, Result};
use crate::pretokenize::pattern_text;

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;

/// Portable 64-bit FNV-1a hash.
///
/// Used for the pattern fingerprint and the merge checksum so that the Rust
/// core and the Python reference (`reference_bpe.py`) compute
/// **identical** values without sharing a crypto dependency. (Neither side
/// is allowed a new hashing crate without approval; FNV-1a is ~10 lines in
/// both languages.)
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Byte-level BPE vocabulary + encoder state.
///
/// Cheap to share across threads (`&Tokenizer` is `Send + Sync`): all
/// interior mutability (compiled regex, chunk cache) lives in thread-locals
/// inside [`crate::pretokenize`] and [`crate::encode`], never here.
#[derive(Debug)]
pub struct Tokenizer {
    merges: Vec<(u32, u32)>,
    rank: FastMap<(u32, u32), u32>,
    token_bytes: Vec<Vec<u8>>,
    bytes_to_id: FastMap<Vec<u8>, u32>,
    special_tokens: Vec<String>,
    special_to_id: FastMap<String, u32>,
}

impl Tokenizer {
    /// Build a vocabulary from an ordered merge list and special tokens.
    ///
    /// `merges[i]` creates token id `256 + i` with bytes
    /// `bytes(left) ++ bytes(right)`. Each side must already exist (ids
    /// `< 256 + i`), otherwise [`Error::InvalidId`] is returned.
    pub fn from_merges(merges: Vec<(u32, u32)>, special_tokens: Vec<String>) -> Result<Self> {
        let mut token_bytes: Vec<Vec<u8>> = (0u32..=255).map(|b| vec![b as u8]).collect();
        let mut bytes_to_id: FastMap<Vec<u8>, u32> =
            HashMap::with_capacity_and_hasher(token_bytes.len() + merges.len(), Default::default());
        for (id, bytes) in token_bytes.iter().enumerate() {
            bytes_to_id.insert(bytes.clone(), id as u32);
        }

        let mut rank: FastMap<(u32, u32), u32> =
            HashMap::with_capacity_and_hasher(merges.len(), Default::default());
        for (i, &(left, right)) in merges.iter().enumerate() {
            let new_id = 256 + i as u32;
            let lb = token_bytes
                .get(left as usize)
                .ok_or(Error::InvalidId(left))?;
            let rb = token_bytes
                .get(right as usize)
                .ok_or(Error::InvalidId(right))?;
            let mut joined = Vec::with_capacity(lb.len() + rb.len());
            joined.extend_from_slice(lb);
            joined.extend_from_slice(rb);
            rank.insert((left, right), i as u32);
            bytes_to_id.insert(joined.clone(), new_id);
            token_bytes.push(joined);
        }

        let base = 256 + merges.len() as u32;
        let mut special_to_id: FastMap<String, u32> =
            HashMap::with_capacity_and_hasher(special_tokens.len(), Default::default());
        for (i, s) in special_tokens.iter().enumerate() {
            special_to_id.insert(s.clone(), base + i as u32);
        }

        Ok(Self {
            merges,
            rank,
            token_bytes,
            bytes_to_id,
            special_tokens,
            special_to_id,
        })
    }

    /// Total ids: `256 + n_merges + n_specials`.
    pub fn vocab_size(&self) -> usize {
        self.token_bytes.len() + self.special_tokens.len()
    }

    /// Ordered merge list: `merges[i]` is the pair for token `256 + i`.
    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// Special tokens in id order.
    pub fn special_tokens(&self) -> &[String] {
        &self.special_tokens
    }

    /// Merge rank of a pair (lower = applied first), if the pair merges.
    pub fn pair_rank(&self, left: u32, right: u32) -> Option<u32> {
        self.rank.get(&(left, right)).copied()
    }

    /// Raw bytes of a normal (non-special) token id.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes.get(id as usize).map(Vec::as_slice)
    }

    /// All normal-token bytes (index = id). Specials are appended at decode.
    pub fn all_token_bytes(&self) -> &[Vec<u8>] {
        &self.token_bytes
    }

    /// Id of a special token string, if registered.
    pub fn special_id(&self, s: &str) -> Option<u32> {
        self.special_to_id.get(s).copied()
    }

    /// Fast path for whole-chunk lookup: the id whose bytes equal `chunk`.
    pub fn chunk_id(&self, chunk: &[u8]) -> Option<u32> {
        self.bytes_to_id.get(chunk).copied()
    }

    /// Number of learned merges.
    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    /// Fingerprint of the pre-tokenization pattern this vocab was built with.
    /// [`crate::io`] stores it in the save file and refuses to load vocabs
    /// trained under a different pattern.
    pub fn pattern_hash(&self) -> u64 {
        fnv1a64(pattern_text().as_bytes())
    }

    /// Checksum over the ordered merge list (little-endian `u32` pairs).
    pub fn checksum(&self) -> u64 {
        let mut buf = Vec::with_capacity(self.merges.len() * 8);
        for &(l, r) in &self.merges {
            buf.extend_from_slice(&l.to_le_bytes());
            buf.extend_from_slice(&r.to_le_bytes());
        }
        fnv1a64(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_layout_and_round_trip_bytes() {
        let t =
            Tokenizer::from_merges(vec![(104, 105)], vec!["<|endoftext|>".into()]).expect("build");
        assert_eq!(t.vocab_size(), 256 + 1 + 1);
        // 'h' = 104, 'i' = 105 -> merged id 256 = "hi".
        assert_eq!(t.token_bytes(256), Some(b"hi".as_slice()));
        assert_eq!(t.chunk_id(b"hi"), Some(256));
        assert_eq!(t.special_id("<|endoftext|>"), Some(257));
        assert_eq!(t.pair_rank(104, 105), Some(0));
        assert_eq!(t.pair_rank(105, 104), None);
    }

    #[test]
    fn rejects_forward_reference() {
        // Pair side 900 does not exist yet when merge 0 is defined.
        let err = Tokenizer::from_merges(vec![(900, 901)], vec![]).unwrap_err();
        assert!(matches!(err, Error::InvalidId(_)));
    }
}
