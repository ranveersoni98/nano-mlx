//! BPE training: incremental pair counts over words.
//!
//! How it works: count word (pre-tokenized chunk) frequencies once, with
//! rayon, then learn merges without ever recounting the corpus:
//!
//! - each distinct word is a token-id sequence with a frequency weight,
//! - `counts`: pair to total hits (weighted by frequency),
//! - `index`: pair to word spots holding it (append-only, stale refs get
//!   skipped by re-checking the word, never removed),
//! - `heap`: max-heap of `(count, smallest-pair)` with lazy throwing away
//!   (a popped entry only counts if it still matches the live count).
//!
//! After each merge we only rewrite words holding that pair (every
//! non-overlapping hit, left to right, same as the Python side), fixing up
//! counts of killed/created neighbor pairs by the word's frequency.
//! Tie-breaks: highest count, then smallest `(left, right)`. New id is the
//! next free one. Stop at target vocab size (minus specials) or when no
//! pair hits count 2.
//!
//! Deterministic: counts are exact sums (add order doesn't matter), the heap
//! fixes merge order, and rayon only does the order-free frequency adding,
//! so output is the same every run and matches `reference_bpe.train`
//! (checked by the 1MB-plus training test).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, split_long_segment};
use crate::vocab::{FastMap, Tokenizer};

/// One distinct word: how often it shows up and its current tokenization.
struct Word {
    freq: u64,
    tokens: Vec<u32>,
}

/// How many times each pair shows up in `tokens` (pair to hits).
fn pair_multiset(tokens: &[u32]) -> FastMap<(u32, u32), u64> {
    let mut map: FastMap<(u32, u32), u64> = FastMap::default();
    for w in tokens.windows(2) {
        *map.entry((w[0], w[1])).or_insert(0) += 1;
    }
    map
}

/// Rewrite `tokens` merging every non-overlapping hit of `pair`
/// into `new_id`, in place with a two-pointer compact. True if anything
/// changed. Cheaper than building a fresh `Vec`: no second allocation,
/// one pass, and the quick presence scan bails on stale index refs before
/// touching anything.
fn apply_merge(tokens: &mut Vec<u32>, pair: (u32, u32), new_id: u32) -> bool {
    if tokens.len() < 2 {
        return false;
    }
    // Quick skip: don't bother with words that can't hold the pair.
    let mut maybe = false;
    for w in tokens.windows(2) {
        if (w[0], w[1]) == pair {
            maybe = true;
            break;
        }
    }
    if !maybe {
        return false;
    }
    let n = tokens.len();
    let mut read = 0usize;
    let mut write = 0usize;
    while read < n {
        // SAFETY-free indexing: `read < n` and the `+1` is bounds-checked.
        if read + 1 < n && tokens[read] == pair.0 && tokens[read + 1] == pair.1 {
            tokens[write] = new_id;
            write += 1;
            read += 2;
        } else {
            tokens[write] = tokens[read];
            write += 1;
            read += 1;
        }
    }
    tokens.truncate(write);
    true
}

/// Does `tokens` contain `pair` anywhere? O(len) scan, no allocs, no
/// hashing - the stale-index filter for the merge loop.
fn contains_pair(tokens: &[u32], pair: (u32, u32)) -> bool {
    tokens.windows(2).any(|w| (w[0], w[1]) == pair)
}

/// Train a vocab on `texts` (each one document string).
///
/// Inputs have to be valid UTF-8 (TinyStories-style corpora are).
/// Returns a [`Tokenizer`] with `merges` in learned order.
pub fn train(
    texts: &[String],
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> Result<Tokenizer> {
    let target_merges = vocab_size
        .saturating_sub(256)
        .saturating_sub(special_tokens.len());

    // 1. Word frequencies, counted once, in parallel. Words are
    // pre-tokenized chunks (chopping rule included), same as the Python
    // side: every byte lands in exactly one word. The 64K pieces are the
    // identical units the sequential path runs
    // (`split_long_segment` from each text start), so parallel counting
    // produces exactly the same words - only the thread assignment
    // differs. Fold per thread, then reduce: ~10 maps merge, not one per
    // piece.
    let mut pieces: Vec<&str> = Vec::new();
    for text in texts {
        pieces.extend(split_long_segment(text));
    }
    // Fold per thread, then reduce: ~10 maps merge, not one per piece.
    // `Result` accumulators give try-fold semantics (a pretokenize failure
    // - only possible from a corrupted pattern file - short-circuits).
    let freq: FastMap<String, u64> = pieces
        .par_iter()
        .fold(
            || Ok(FastMap::default()),
            |acc: Result<FastMap<String, u64>>, piece| {
                let mut local = acc?;
                for chunk in pretokenize(piece)? {
                    if !chunk.is_empty() {
                        *local.entry(chunk.to_owned()).or_insert(0) += 1;
                    }
                }
                Ok(local)
            },
        )
        .reduce(
            || Ok(FastMap::default()),
            |a: Result<FastMap<String, u64>>, b: Result<FastMap<String, u64>>| {
                let mut acc = a?;
                for (w, c) in b? {
                    *acc.entry(w).or_insert(0) += c;
                }
                Ok(acc)
            },
        )?;

    // 2. Words as byte-id sequences.
    let mut words: Vec<Word> = freq
        .into_iter()
        .map(|(w, c)| Word {
            freq: c,
            tokens: w.bytes().map(u32::from).collect(),
        })
        .collect();

    // 3. Initial pair counts + inverted index + heap.
    let mut counts: FastMap<(u32, u32), u64> = FastMap::default();
    let mut index: FastMap<(u32, u32), Vec<usize>> = FastMap::default();
    for (idx, word) in words.iter().enumerate() {
        for (pair, mult) in pair_multiset(&word.tokens) {
            *counts.entry(pair).or_insert(0) += mult * word.freq;
            index.entry(pair).or_default().push(idx);
        }
    }
    // Max-heap: biggest count first, ties go to the smallest pair.
    // `Reverse(pair)` in a max-heap pops the smallest pair.
    let mut heap: BinaryHeap<(u64, Reverse<(u32, u32)>)> = BinaryHeap::new();
    for (&pair, &count) in counts.iter() {
        heap.push((count, Reverse(pair)));
    }

    // 4. Merge loop with incremental updates.
    let mut merges: Vec<(u32, u32)> = Vec::new();
    while merges.len() < target_merges {
        // Find the live best pair (skip heap entries that went stale).
        let best = loop {
            match heap.pop() {
                Some((count, Reverse(pair))) => {
                    if count >= 2 && counts.get(&pair) == Some(&count) {
                        break Some(pair);
                    }
                }
                None => break None,
            }
        };
        let pair = match best {
            Some(p) => p,
            None => break,
        };
        let new_id = 256 + merges.len() as u32;
        merges.push(pair);
        // Rewrite only words containing the pair.
        let mut holders = index.remove(&pair).unwrap_or_default();
        // The index is append-only, so `holders` accumulates stale and
        // duplicate refs over time. Deduping big lists keeps the merge
        // loop from re-scanning the same word over and over.
        if holders.len() > 64 {
            holders.sort_unstable();
            holders.dedup();
        }
        for idx in holders {
            // Stale-index filter BEFORE any allocation: most late-merge
            // holders no longer contain the pair, and this scan costs
            // O(len) with no hashing. (Previously the `old` multiset
            // HashMap was built first and thrown away for these.)
            if !contains_pair(&words[idx].tokens, pair) {
                continue;
            }
            let word = &mut words[idx];
            let old = pair_multiset(&word.tokens);
            if !apply_merge(&mut word.tokens, pair, new_id) {
                continue; // raced by the check above (cannot happen
                          // single-threaded, kept for safety)
            }
            let new = pair_multiset(&word.tokens);
            // Count down killed pairs, count up new ones. Every pair whose
            // count moves, up OR down, gets a fresh heap entry. (Down
            // matters: a pair can drop *into* first place, and without a
            // fresh entry it would sit there invisible while lower pairs
            // merge. Stale entries die when popped.)
            for (p, c) in &old {
                let delta = c * word.freq;
                let cur = match counts.get_mut(p) {
                    Some(v) => {
                        *v = v
                            .checked_sub(delta)
                            .ok_or(Error::NotImplemented("pair count underflow (training bug)"))?;
                        *v
                    }
                    None => {
                        return Err(Error::NotImplemented(
                            "missing pair count during merge (training bug)",
                        ));
                    }
                };
                heap.push((cur, Reverse(*p)));
            }
            for (p, c) in new {
                let entry = counts.entry(p).or_insert(0);
                *entry += c * word.freq;
                heap.push((*entry, Reverse(p)));
                index.entry(p).or_default().push(idx);
            }
        }
    }

    Tokenizer::from_merges(merges, special_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_texts() -> Vec<String> {
        vec![
            "hello hello world world world".to_owned(),
            "hello world test test test test".to_owned(),
        ]
    }

    #[test]
    fn deterministic_across_runs() {
        let a = train(&tiny_texts(), 300, vec![]).expect("train");
        let b = train(&tiny_texts(), 300, vec![]).expect("train");
        assert_eq!(a.merges(), b.merges());
    }

    #[test]
    fn tie_break_picks_smallest_pair() {
        // "ab" and "ba" tie at 2 each: (97,98) < (98,97), so (97,98) wins.
        // Chunks are single words here (no spaces inside bits, spaces
        // split words, so make one word with a tie).
        let texts = vec!["abba".to_owned()];
        let tok = train(&texts, 260, vec![]).expect("train");
        // Pairs in "abba": (a,b)x1, (b,b)x1, (b,a)x1 - no count>=2, no merges.
        assert!(tok.merges().is_empty());

        // "abab": pairs (a,b)x2, (b,a)x1 -> merge (97,98) first.
        let texts = vec!["abab".to_owned(), "baba".to_owned()];
        let tok = train(&texts, 260, vec![]).expect("train");
        assert_eq!(tok.merges()[0], (97, 98));
    }

    #[test]
    fn stops_at_target_and_on_rare_pairs() {
        let texts = tiny_texts();
        let tok = train(&texts, 258, vec![]).expect("train");
        assert!(tok.num_merges() <= 2);
        // Single uncoverable word: no pair reaches 2.
        let tok = train(&vec!["abcdef".to_owned()], 300, vec![]).expect("train");
        assert!(tok.merges().is_empty());
    }

    #[test]
    fn specials_reserve_id_space() {
        let tok = train(
            &tiny_texts(),
            300,
            vec!["<|endoftext|>".to_owned(), "<|pad|>".to_owned()],
        )
        .expect("train");
        assert_eq!(tok.special_tokens().len(), 2);
        assert!(tok.num_merges() <= 300 - 256 - 2);
        assert!(tok.special_id("<|endoftext|>").is_some());
    }
}
