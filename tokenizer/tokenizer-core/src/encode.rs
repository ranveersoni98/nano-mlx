//! Encoding: text -> token ids (the hot path).
//!
//! Pipeline per input: split specials -> regex pre-tokenize -> per-chunk BPE.
//! Merges never cross chunk boundaries (frozen rule; SuperBPE is M9).
//!
//! Per-chunk algorithm (fast path): bytes start as one symbol each in a
//! **doubly linked list**; every adjacent pair with a known merge rank is
//! pushed onto a **min-heap keyed by rank** (lower rank = learned earlier =
//! merged first). Pop the best pair, splice it, push the two new neighbor
//! pairs. Stale heap entries (pairs destroyed by an earlier merge) are
//! detected by re-checking liveness + current rank and skipped - lazy
//! deletion, no heap removals. Complexity is `O(n log n)` merges instead of
//! the naive `O(n²)` re-scan; the naive version is kept as
//! [`encode_chunk_naive`] for criterion A/B comparisons and as an
//! independent correctness oracle in tests.
//!
//! Allocation discipline: the linked-list vectors are reused per call frame,
//! chunk bytes map to `u32` ids without `String` round trips, and repeat
//! chunks hit a **bounded thread-local cache** (`ahash`, cleared when full
//! so memory stays flat). `encode_batch` fans out over rayon; output order
//! always equals input order.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;

use ahash::AHasher;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, split_special, Split};
use crate::vocab::Tokenizer;

/// Upper bound on cached chunks per thread. When full the cache is simply
/// cleared (documented, deterministic, no LRU bookkeeping on the hot path).
pub const MAX_CACHE_ENTRIES: usize = 8192;

thread_local! {
    static CHUNK_CACHE: RefCell<HashMap<Vec<u8>, Vec<u32>, BuildHasherDefault<AHasher>>> =
        RefCell::new(HashMap::default());
}

fn cache_get(chunk: &[u8]) -> Option<Vec<u32>> {
    CHUNK_CACHE
        .try_with(|c| c.borrow().get(chunk).cloned())
        .unwrap_or(None)
}

fn cache_put(chunk: &[u8], ids: &[u32]) {
    let _ = CHUNK_CACHE.try_with(|c| {
        let mut cache = c.borrow_mut();
        if cache.len() >= MAX_CACHE_ENTRIES {
            cache.clear();
        }
        cache.insert(chunk.to_vec(), ids.to_vec());
    });
}

/// Naive BPE over one chunk's bytes: repeatedly merge the lowest-rank pair
/// anywhere in the sequence. `O(n²)` - correctness oracle and Step-A
/// benchmark baseline, never the serving path.
pub fn encode_chunk_naive(chunk: &[u8], tok: &Tokenizer) -> Vec<u32> {
    let mut ids: Vec<u32> = chunk.iter().map(|&b| u32::from(b)).collect();
    if ids.len() < 2 {
        return ids;
    }
    loop {
        let mut best: Option<(u32, usize)> = None; // (rank, position)
        for i in 0..ids.len().saturating_sub(1) {
            if let Some(r) = tok.pair_rank(ids[i], ids[i + 1]) {
                if best.is_none_or(|(br, _)| r < br) {
                    best = Some((r, i));
                }
            }
        }
        match best {
            Some((r, i)) => {
                let merged = 256 + r;
                ids[i] = merged;
                ids.remove(i + 1);
            }
            None => break,
        }
    }
    ids
}

/// Fast BPE over one chunk's bytes: linked list + min-heap on merge rank.
///
/// Invariants: `id[k]`/`next[k]` describe live symbols; `alive[k]` guards
/// stale heap entries; a heap entry `(rank, pos)` is valid iff `alive[pos]`,
/// `next[pos]` is alive, and the pair's current rank still equals `rank`.
/// The merged id for rank `r` is `256 + r` (vocab layout guarantee).
pub fn encode_chunk_fast(chunk: &[u8], tok: &Tokenizer) -> Vec<u32> {
    if chunk.is_empty() {
        return Vec::new();
    }
    // Fast path: the whole chunk is already one vocab token.
    if let Some(id) = tok.chunk_id(chunk) {
        return vec![id];
    }

    let n = chunk.len();
    let mut id: Vec<u32> = chunk.iter().map(|&b| u32::from(b)).collect();
    let mut prev: Vec<Option<usize>> = (0..n).map(|i| i.checked_sub(1)).collect();
    let mut next: Vec<Option<usize>> = (0..n)
        .map(|i| if i + 1 < n { Some(i + 1) } else { None })
        .collect();
    let mut alive = vec![true; n];

    // (rank, position); `BinaryHeap` is a max-heap so both are wrapped in
    // `Reverse` to pop the lowest rank first. The position tie-break is
    // load-bearing: equal-rank pairs (e.g. `(32, 32)` twice in `"   "`)
    // MUST merge leftmost-first, exactly like the naive scan and the
    // Python reference. Popping rightmost-first produces different ids
    // (regression-tested).
    let mut heap: BinaryHeap<(Reverse<u32>, Reverse<usize>)> = BinaryHeap::new();
    let mut cur = Some(0usize);
    while let Some(i) = cur {
        if let Some(j) = next[i] {
            if let Some(r) = tok.pair_rank(id[i], id[j]) {
                heap.push((Reverse(r), Reverse(i)));
            }
        }
        cur = next[i];
    }

    while let Some((Reverse(r), Reverse(i))) = heap.pop() {
        if !alive[i] {
            continue;
        }
        let j = match next[i] {
            Some(j) if alive[j] => j,
            _ => continue,
        };
        // Still the same pair with the same rank? Else stale.
        if tok.pair_rank(id[i], id[j]) != Some(r) {
            continue;
        }
        // Merge j into i.
        id[i] = 256 + r;
        let after = next[j];
        next[i] = after;
        if let Some(k) = after {
            prev[k] = Some(i);
        }
        alive[j] = false;
        // Refresh the two neighbor pairs touching the merged symbol.
        if let Some(p) = prev[i] {
            if let Some(rr) = tok.pair_rank(id[p], id[i]) {
                heap.push((Reverse(rr), Reverse(p)));
            }
        }
        if let Some(k) = next[i] {
            if let Some(rr) = tok.pair_rank(id[i], id[k]) {
                heap.push((Reverse(rr), Reverse(i)));
            }
        }
    }

    // Head is index 0 (merges always reuse the left slot, so 0 never dies).
    let mut out = Vec::new();
    let mut cur = Some(0usize);
    while let Some(i) = cur {
        debug_assert!(alive[i]);
        out.push(id[i]);
        cur = next[i];
    }
    out
}

/// Encode one chunk, consulting the thread-local cache.
fn encode_chunk_cached(chunk: &[u8], tok: &Tokenizer) -> Vec<u32> {
    if let Some(hit) = cache_get(chunk) {
        return hit;
    }
    let ids = encode_chunk_fast(chunk, tok);
    cache_put(chunk, &ids);
    ids
}

/// Encode `text` with `allowed_specials` available as control tokens.
///
/// Default (`allowed_specials` empty): special-looking text is ordinary text.
/// Every id in `allowed_specials` must be registered in the vocab, else
/// [`Error::UnknownSpecialToken`].
pub fn encode_text(tok: &Tokenizer, text: &str, allowed_specials: &[String]) -> Result<Vec<u32>> {
    let allowed_refs: Vec<&str> = allowed_specials.iter().map(String::as_str).collect();
    for s in &allowed_refs {
        if tok.special_id(s).is_none() {
            return Err(Error::UnknownSpecialToken((*s).to_owned()));
        }
    }
    let mut out = Vec::new();
    for part in split_special(text, &allowed_refs) {
        match part {
            Split::Special(s) => {
                // Validated above, so this is always `Some`.
                match tok.special_id(s) {
                    Some(id) => out.push(id),
                    None => return Err(Error::UnknownSpecialToken(s.to_owned())),
                }
            }
            Split::Text(seg) => {
                for chunk in pretokenize(seg)? {
                    out.extend_from_slice(&encode_chunk_cached(chunk.as_bytes(), tok));
                }
            }
        }
    }
    Ok(out)
}

/// Encode arbitrary bytes without panicking.
///
/// Valid-UTF-8 runs go through the normal text pipeline; invalid bytes are
/// emitted as single-byte tokens. Either way every input byte round-trips
/// through [`crate::decode`], so `decode(encode_bytes(x)) == x` for all `x`.
pub fn encode_bytes(
    tok: &Tokenizer,
    bytes: &[u8],
    allowed_specials: &[String],
) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for chunk in bytes.utf8_chunks() {
        let valid = chunk.valid();
        if !valid.is_empty() {
            out.extend(encode_text(tok, valid, allowed_specials)?);
        }
        for &b in chunk.invalid() {
            out.push(u32::from(b));
        }
    }
    Ok(out)
}

/// Encode many strings in parallel. Output order equals input order.
/// The GIL (in the PyO3 wrapper) must be released around this call.
pub fn encode_batch(
    tok: &Tokenizer,
    texts: &[String],
    allowed_specials: &[String],
) -> Result<Vec<Vec<u32>>> {
    texts
        .par_iter()
        .map(|t| encode_text(tok, t, allowed_specials))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_tok() -> Tokenizer {
        // merges: "h"+"i"->256, then 256+"!"->257 ("hi!").
        Tokenizer::from_merges(vec![(104, 105), (256, 33)], vec![]).expect("build")
    }

    #[test]
    fn fast_matches_naive_on_chunks() {
        let tok = mini_tok();
        // A second vocab WITH self-pair merges: equal-rank pairs at
        // adjacent positions are exactly where heap tie-breaks matter.
        // (Regression: the heap once popped rightmost-first here.)
        let self_pair_tok =
            Tokenizer::from_merges(vec![(32, 32), (256, 32), (97, 97), (258, 97)], vec![])
                .expect("build");
        // Targeted overlap cases.
        assert_eq!(
            encode_chunk_fast(b"   ", &self_pair_tok),
            encode_chunk_naive(b"   ", &self_pair_tok)
        );
        assert_eq!(
            encode_chunk_fast(b"aaa", &self_pair_tok),
            encode_chunk_naive(b"aaa", &self_pair_tok)
        );
        assert_eq!(
            encode_chunk_fast(b"aaaa", &self_pair_tok),
            encode_chunk_naive(b"aaaa", &self_pair_tok)
        );
        // Exhaustive over a small alphabet incl. mergeable bytes.
        // Run the sweep against both vocabs; the self-pair-heavy alphabet
        // (`a`/`space`) stresses equal-rank tie-breaks specifically.
        for (tok, alphabet) in [
            (&tok, b"hi!xyz \n".as_slice()),
            (&self_pair_tok, b"a \n".as_slice()),
        ] {
            for len in 0..=6usize {
                // Deterministic pseudo-random sweep (no dev-dep needed here).
                let mut seed: u64 = 0x9e3779b97f4a7c15;
                for _ in 0..200 {
                    let mut chunk = Vec::with_capacity(len);
                    for _ in 0..len {
                        seed = seed
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        chunk.push(alphabet[(seed >> 33) as usize % alphabet.len()]);
                    }
                    assert_eq!(
                        encode_chunk_fast(&chunk, tok),
                        encode_chunk_naive(&chunk, tok),
                        "chunk {chunk:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn linked_list_merge_order_uses_lowest_rank_first() {
        // "ab"+"c": pair (a,b) rank 0 must win over (b,c) rank 1.
        let tok = Tokenizer::from_merges(vec![(97, 98), (98, 99)], vec![]).expect("build");
        assert_eq!(encode_chunk_fast(b"abc", &tok), vec![256, 99]);
        assert_eq!(encode_chunk_naive(b"abc", &tok), vec![256, 99]);
    }

    #[test]
    fn empty_and_single_byte() {
        let tok = mini_tok();
        assert_eq!(encode_chunk_fast(b"", &tok), Vec::<u32>::new());
        assert_eq!(encode_chunk_fast(b"q", &tok), vec![113]);
    }

    #[test]
    fn text_round_trips_through_decode() {
        let tok = mini_tok();
        for s in ["hi! hi!", "hello", "", "xyz xyz"] {
            let ids = encode_text(&tok, s, &[]).expect("encode");
            let back = crate::decode::decode_lossy(&tok, &ids).expect("decode");
            assert_eq!(back, s);
        }
    }

    #[test]
    fn batch_preserves_order() {
        let tok = mini_tok();
        let texts: Vec<String> = (0..64).map(|i| format!("hi! {i}")).collect();
        let batched = encode_batch(&tok, &texts, &[]).expect("batch");
        for (t, ids) in texts.iter().zip(batched.iter()) {
            assert_eq!(ids, &encode_text(&tok, t, &[]).expect("single"));
        }
    }
}
