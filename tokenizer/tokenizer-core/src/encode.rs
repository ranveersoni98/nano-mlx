//! Encoding: text to token ids (the hot path).
//!
//! Each input goes: split specials, regex pre-tokenize, BPE per chunk.
//! Merges never cross chunk edges.
//!
//! Per-chunk trick (fast path): bytes start as one symbol each in a doubly
//! linked list. Every neighboring pair with a known merge rank goes on a
//! min-heap keyed by rank (lower rank means learned earlier means merge
//! first). Pop the best pair, splice it in, push the two new neighbor
//! pairs. Heap entries killed by an earlier merge get noticed by rechecking
//! liveness and rank and skipped. No heap removals. This is `O(n log n)`
//! merges instead of the naive `O(n^2)` rescan. The naive version sticks
//! around as [`encode_chunk_naive`] for tiny chunks (where heap setup costs
//! more than a rescan), for criterion A/B runs, and as a second opinion in
//! tests.
//!
//! Allocs: single bytes return without any lookup, whole-chunk hits skip
//! BPE via a length-gated map probe, tiny chunks use the allocation-light
//! naive scan, and repeat chunks hit a small thread-local cache (`ahash`,
//! cleared when full so memory stays flat). `encode_batch` spreads over
//! rayon, and single inputs larger than [`PAR_TEXT_BYTES`] encode their
//! 64K pre-tokenize pieces in parallel. Parallelism only ever distributes
//! the exact same 64K pieces the sequential path runs (see
//! `pretokenize::split_long_segment`), so parallel output is
//! byte-identical to sequential output on every input, at any thread
//! count. Output order always matches input order.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;

use ahash::AHasher;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, split_long_segment, split_special, Split};
use crate::vocab::Tokenizer;

/// How many chunks we cache per thread. When it's full we just clear it
/// (simple, deterministic, no LRU bookkeeping on the hot path).
pub const MAX_CACHE_ENTRIES: usize = 8192;

/// Only cache chunks in this byte-length window. Shorter ones are cheaper
/// to re-encode than to hash + clone through the cache; longer ones are
/// rarely repeated and would bloat memory (worst case `entries * len`).
pub const MIN_CACHE_CHUNK_LEN: usize = 4;
pub const MAX_CACHE_CHUNK_LEN: usize = 512;

/// Chunks at or below this length use the naive rescan: `O(n^2)` with tiny
/// `n` beats linked-list + heap setup (no `prev`/`next`/`alive`/heap
/// allocs, and the whole thing often stays in one cache line).
pub const NAIVE_CHUNK_LEN: usize = 16;

/// Inputs larger than this encode their 64K pre-tokenize pieces in
/// parallel (single `encode`/`encode_text` calls, not just batch/file
/// paths). Below it, sequential encode avoids rayon dispatch overhead.
/// The pieces are identical either way, so this never changes output.
pub const PAR_TEXT_BYTES: usize = 128 * 1024;

thread_local! {
    static CHUNK_CACHE: RefCell<HashMap<Vec<u8>, Vec<u32>, BuildHasherDefault<AHasher>>> =
        RefCell::new(HashMap::default());
}

fn cache_get(chunk: &[u8]) -> Option<Vec<u32>> {
    if chunk.len() < MIN_CACHE_CHUNK_LEN || chunk.len() > MAX_CACHE_CHUNK_LEN {
        return None;
    }
    CHUNK_CACHE
        .try_with(|c| c.borrow().get(chunk).cloned())
        .unwrap_or(None)
}

fn cache_put(chunk: &[u8], ids: &[u32]) {
    if chunk.len() < MIN_CACHE_CHUNK_LEN || chunk.len() > MAX_CACHE_CHUNK_LEN {
        return;
    }
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
    // Single bytes are their own ids (vocab layout guarantee) - no lookup.
    if chunk.len() == 1 {
        return vec![u32::from(chunk[0])];
    }
    // Fast path: the whole chunk is already one vocab token. Length-gated:
    // chunks longer than any vocab token hash a lot of bytes for a lookup
    // that can never hit.
    if chunk.len() <= tok.max_token_len() {
        if let Some(id) = tok.chunk_id(chunk) {
            return vec![id];
        }
    }
    // Tiny chunks: the naive rescan does fewer allocs (no linked-list vecs,
    // no heap) and usually wins below ~16 bytes.
    if chunk.len() <= NAIVE_CHUNK_LEN {
        return encode_chunk_naive(chunk, tok);
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
    //
    // Pre-sized to `n`: at most one live entry per position is pushed
    // initially, which avoids several heap grows on big chunks.
    let mut heap: BinaryHeap<(Reverse<u32>, Reverse<usize>)> =
        BinaryHeap::with_capacity(n.next_power_of_two().min(1 << 20));
    let mut cur = Some(0usize);
    let mut mergeable = 0usize;
    while let Some(i) = cur {
        if let Some(j) = next[i] {
            if let Some(r) = tok.pair_rank(id[i], id[j]) {
                heap.push((Reverse(r), Reverse(i)));
                mergeable += 1;
            }
        }
        cur = next[i];
    }
    // Nothing can merge: skip the heap loop entirely (common for chunks
    // made of bytes the vocab never combined, e.g. random tails).
    if mergeable == 0 {
        return id;
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

/// Every id in `allowed` must be registered, else
/// [`Error::UnknownSpecialToken`]. Checked once per public call so workers
/// never re-validate.
fn validate_specials(tok: &Tokenizer, allowed_refs: &[&str]) -> Result<()> {
    for s in allowed_refs {
        if tok.special_id(s).is_none() {
            return Err(Error::UnknownSpecialToken((*s).to_owned()));
        }
    }
    Ok(())
}

/// Encode one 64K pre-tokenize piece (no specials inside): regex-split it
/// into chunks and BPE each chunk. The unit of parallel work.
fn encode_piece(tok: &Tokenizer, piece: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for chunk in pretokenize(piece)? {
        out.extend_from_slice(&encode_chunk_cached(chunk.as_bytes(), tok));
    }
    Ok(out)
}

/// Encode `text` with `allowed_specials` available as control tokens.
///
/// Default (`allowed_specials` empty): special-looking text is ordinary text.
/// Every id in `allowed_specials` must be registered in the vocab, else
/// [`Error::UnknownSpecialToken`].
///
/// Large inputs encode their 64K pre-tokenize pieces on the rayon pool;
/// small inputs stay on this thread (no dispatch overhead on the latency
/// path). Either way the pieces - and therefore the output - are identical.
pub fn encode_text(tok: &Tokenizer, text: &str, allowed_specials: &[String]) -> Result<Vec<u32>> {
    let allowed_refs: Vec<&str> = allowed_specials.iter().map(String::as_str).collect();
    validate_specials(tok, &allowed_refs)?;
    encode_text_inner(tok, text, &allowed_refs, true)
}

/// Sequential body shared by [`encode_text`] and the parallel workers.
/// `allow_par` gates one level of intra-text parallelism; workers always
/// pass `false` so parallelism never nests.
fn encode_text_inner(
    tok: &Tokenizer,
    text: &str,
    allowed_refs: &[&str],
    allow_par: bool,
) -> Result<Vec<u32>> {
    // Split specials first (single pass), then chop Text segments into the
    // exact 64K pieces the sequential path runs.
    let parts = split_special(text, allowed_refs);
    // Flatten to (part index, piece) jobs; specials resolve to inline ids.
    // The flat list is in output order, and rayon's indexed collect
    // preserves it, so stitching is a straight concatenation.
    let mut specials: Vec<(usize, u32)> = Vec::new();
    let mut jobs: Vec<(usize, &str)> = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        match part {
            Split::Special(s) => match tok.special_id(s) {
                Some(id) => specials.push((i, id)),
                None => return Err(Error::UnknownSpecialToken((*s).to_owned())),
            },
            Split::Text(seg) => {
                for piece in split_long_segment(seg) {
                    jobs.push((i, piece));
                }
            }
        }
    }
    let par = allow_par && text.len() >= PAR_TEXT_BYTES;
    let encoded = encode_piece_jobs(tok, &jobs, par)?;
    // Stitch: one slot per part; specials are single ids, texts concatenate
    // their pieces in job order (which is output order).
    let mut by_part: Vec<Vec<u32>> = vec![Vec::new(); parts.len()];
    for (i, id) in specials {
        by_part[i].push(id);
    }
    for (i, ids) in encoded {
        by_part[i].extend_from_slice(&ids);
    }
    let mut out = Vec::new();
    for mut ids in by_part {
        out.append(&mut ids);
    }
    Ok(out)
}

/// Encode piece jobs, sequentially or on the rayon pool. Both paths
/// preserve job order in the output, so stitching is concatenation.
/// `allow_par` lets callers suppress nesting (workers pass `false`).
fn encode_piece_jobs<K>(
    tok: &Tokenizer,
    jobs: &[(K, &str)],
    allow_par: bool,
) -> Result<Vec<(K, Vec<u32>)>>
where
    K: Copy + Send + Sync,
{
    if jobs.len() <= 1 || !allow_par || rayon::current_num_threads() <= 1 {
        jobs.iter()
            .map(|&(k, piece)| encode_piece(tok, piece).map(|ids| (k, ids)))
            .collect()
    } else {
        jobs.par_iter()
            .map(|&(k, piece)| encode_piece(tok, piece).map(|ids| (k, ids)))
            .collect()
    }
}

/// Encode arbitrary bytes without panicking.
///
/// Valid-UTF-8 runs go through the normal text pipeline; invalid bytes are
/// emitted as single-byte tokens. Either way every input byte round-trips
/// through [`crate::decode`], so `decode(encode_bytes(x)) == x` for all `x`.
///
/// Large inputs encode their 64K pieces in parallel, exactly as
/// [`encode_text`] does; small inputs stay sequential.
pub fn encode_bytes(
    tok: &Tokenizer,
    bytes: &[u8],
    allowed_specials: &[String],
) -> Result<Vec<u32>> {
    let allowed_refs: Vec<&str> = allowed_specials.iter().map(String::as_str).collect();
    validate_specials(tok, &allowed_refs)?;
    // Flatten the `utf8_chunks` run stream into piece jobs. Each run is
    // (valid text, trailing invalid bytes); valid text splits on specials
    // then into 64K pieces, exactly like the text path, so outputs agree.
    // Keys are (run, part); invalid bytes are in run order already.
    let mut specials: Vec<(usize, usize, u32)> = Vec::new();
    let mut jobs: Vec<((usize, usize), &str)> = Vec::new();
    let mut invalids: Vec<(usize, u8)> = Vec::new();
    let mut n_runs = 0usize;
    for chunk in bytes.utf8_chunks() {
        let r = n_runs;
        n_runs += 1;
        let valid = chunk.valid();
        if !valid.is_empty() {
            for (p, part) in split_special(valid, &allowed_refs).iter().enumerate() {
                match part {
                    Split::Special(s) => match tok.special_id(s) {
                        Some(id) => specials.push((r, p, id)),
                        None => return Err(Error::UnknownSpecialToken((*s).to_owned())),
                    },
                    Split::Text(seg) => {
                        for piece in split_long_segment(seg) {
                            jobs.push(((r, p), piece));
                        }
                    }
                }
            }
        }
        for &b in chunk.invalid() {
            invalids.push((r, b));
        }
    }
    let par = bytes.len() >= PAR_TEXT_BYTES;
    let encoded = encode_piece_jobs(tok, &jobs, par)?;
    // Stitch in (run, part) order: specials and piece outputs merge by part,
    // then each run's invalid bytes follow its valid content (utf8_chunks
    // yields valid-then-invalid within a chunk, runs in order).
    let mut out = Vec::new();
    let mut si = 0usize;
    let mut ei = 0usize;
    let mut ii = 0usize;
    for r in 0..n_runs {
        while si < specials.len() && specials[si].0 == r {
            let sp = specials[si].1;
            while ei < encoded.len() && encoded[ei].0 .0 == r && encoded[ei].0 .1 < sp {
                out.extend_from_slice(&encoded[ei].1);
                ei += 1;
            }
            out.push(specials[si].2);
            si += 1;
        }
        while ei < encoded.len() && encoded[ei].0 .0 == r {
            out.extend_from_slice(&encoded[ei].1);
            ei += 1;
        }
        while ii < invalids.len() && invalids[ii].0 == r {
            out.push(u32::from(invalids[ii].1));
            ii += 1;
        }
    }
    Ok(out)
}

/// Encode many strings in parallel. Output order equals input order.
/// The GIL (in the PyO3 wrapper) must be released around this call.
///
/// Small batches skip rayon dispatch and go sequential; when the batch
/// already fills the pool, each doc encodes sequentially to avoid nested
/// parallelism (docs still run on all threads via the outer split).
pub fn encode_batch(
    tok: &Tokenizer,
    texts: &[String],
    allowed_specials: &[String],
) -> Result<Vec<Vec<u32>>> {
    if texts.len() <= 1 || rayon::current_num_threads() <= 1 {
        return texts
            .iter()
            .map(|t| encode_text(tok, t, allowed_specials))
            .collect();
    }
    // Outer parallelism already saturates the pool: keep per-doc work
    // sequential unless there are fewer docs than threads (huge-doc case,
    // where intra-doc sharding earns its overhead).
    let allow_par = texts.len() < rayon::current_num_threads();
    if !allow_par {
        let allowed_refs: Vec<&str> = allowed_specials.iter().map(String::as_str).collect();
        validate_specials(tok, &allowed_refs)?;
        return texts
            .par_iter()
            .map(|t| encode_text_inner(tok, t, &allowed_refs, false))
            .collect();
    }
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

    fn big_tok() -> Tokenizer {
        Tokenizer::from_merges(
            vec![(104, 105), (256, 33), (32, 32), (97, 110), (258, 100)],
            vec!["<|endoftext|>".to_owned()],
        )
        .expect("build")
    }

    #[test]
    fn large_text_parallel_matches_sequential() {
        let tok = big_tok();
        let allowed = vec!["<|endoftext|>".to_owned()];
        let refs: Vec<&str> = allowed.iter().map(String::as_str).collect();
        // Well over PAR_TEXT_BYTES, with newlines, specials, multibyte text.
        let mut big = String::new();
        for i in 0..4000 {
            big.push_str(&format!(
                "hi! line {i} with words and  spaces  <|endoftext|> café 🎉\n"
            ));
        }
        assert!(big.len() > PAR_TEXT_BYTES);
        let par = encode_text(&tok, &big, &allowed).expect("parallel");
        let seq = encode_text_inner(&tok, &big, &refs, false).expect("sequential");
        assert_eq!(par, seq);
        // Batch-of-one and multi-doc batch agree too.
        let batch = encode_batch(&tok, &[big.clone()], &allowed).expect("batch1");
        assert_eq!(batch.into_iter().next().unwrap(), par);
        let docs = vec![big.clone(), "hi!".to_owned(), big.clone()];
        let batch = encode_batch(&tok, &docs, &allowed).expect("batch3");
        assert_eq!(batch[0], par);
        assert_eq!(
            batch[1],
            encode_text(&tok, "hi!", &allowed).expect("single")
        );
        assert_eq!(batch[2], par);
    }

    #[test]
    fn large_text_without_newlines_matches_sequential() {
        // No `\n` anywhere: sharding finds one piece, must still agree.
        let tok = big_tok();
        let big = "hi! and words ".repeat(PAR_TEXT_BYTES / 8);
        assert!(big.len() > PAR_TEXT_BYTES);
        assert!(!big.contains('\n'));
        let par = encode_text(&tok, &big, &[]).expect("parallel");
        let seq = encode_text_inner(&tok, &big, &[], false).expect("sequential");
        assert_eq!(par, seq);
        let bytes = big.as_bytes();
        assert_eq!(
            encode_bytes(&tok, bytes, &[]).expect("bytes"),
            par,
            "bytes path must agree with text path on valid UTF-8"
        );
    }

    #[test]
    fn large_bytes_with_invalid_splits_match_sequential() {
        // Invalid bytes around shard edges must not change output.
        let tok = big_tok();
        let mut bytes = Vec::new();
        for i in 0..12_000 {
            bytes.extend_from_slice(format!("line {i} hi!\n").as_bytes());
            bytes.extend_from_slice(&[0xFF, 0xFE]);
        }
        assert!(bytes.len() > PAR_TEXT_BYTES);
        let allowed: Vec<String> = vec![];
        let par = encode_bytes(&tok, &bytes, &allowed).expect("parallel");
        // Sequential reference: force single-thread via inner sequential body.
        let mut seq = Vec::new();
        for chunk in bytes.utf8_chunks() {
            if !chunk.valid().is_empty() {
                seq.extend(encode_text_inner(&tok, chunk.valid(), &[], false).expect("seq"));
            }
            for &b in chunk.invalid() {
                seq.push(u32::from(b));
            }
        }
        assert_eq!(par, seq);
    }
}
