//! BPE training (M3): incremental pair counts over words.
//!
//! Algorithm (§5.7): count word (pre-tokenized chunk) frequencies once -
//! with rayon - then learn merges without ever recounting the corpus:
//!
//! - each distinct word is a token-id sequence with a frequency weight;
//! - `counts`: pair -> total occurrences (frequency-weighted);
//! - `index`: pair -> word indices containing it (append-only; stale refs
//!   are skipped by re-checking the word, never removed);
//! - `heap`: max-heap of `(count, smallest-pair)` with lazy invalidation
//!   (a popped entry is used only if it still equals the live count).
//!
//! After each merge only words containing the merged pair are rewritten
//! (all non-overlapping occurrences, left to right - exactly like the
//! reference), adjusting counts of destroyed/created neighbor pairs by the
//! word's frequency. Frozen tie-breaks (§5.4): highest count, then smallest
//! `(left, right)`; new id = next unused; stop at the target vocab size
//! (minus specials) or when no pair reaches count 2.
//!
//! Determinism: counts are exact sums (addition order irrelevant), the heap
//! fully determines merge order, and rayon is only used for the
//! order-free frequency reduction - so output is identical run to run and
//! identical to `reference_bpe.train` (proven by the ≥1MB parity test).

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::pretokenize::{pretokenize, split_long_segment};
use crate::vocab::{FastMap, Tokenizer};

/// One distinct word: its frequency and current tokenization.
struct Word {
    freq: u64,
    tokens: Vec<u32>,
}

/// Count pair occurrences in `tokens` (multiset: pair -> multiplicity).
fn pair_multiset(tokens: &[u32]) -> FastMap<(u32, u32), u64> {
    let mut map: FastMap<(u32, u32), u64> = FastMap::default();
    for w in tokens.windows(2) {
        *map.entry((w[0], w[1])).or_insert(0) += 1;
    }
    map
}

/// Rewrite `tokens` merging every non-overlapping occurrence of `pair`
/// into `new_id`. Returns true if anything changed.
fn apply_merge(tokens: &mut Vec<u32>, pair: (u32, u32), new_id: u32) -> bool {
    if tokens.len() < 2 {
        return false;
    }
    // Fast prefilter: skip words that cannot contain the pair.
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
    let mut merged = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        if i + 1 < tokens.len() && (tokens[i], tokens[i + 1]) == pair {
            merged.push(new_id);
            i += 2;
        } else {
            merged.push(tokens[i]);
            i += 1;
        }
    }
    *tokens = merged;
    true
}

/// Train a vocabulary on `texts` (each a document string).
///
/// Training inputs must be valid UTF-8 (TinyStories-class corpora are).
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
    // pre-tokenized chunks (force-split rule included), exactly like the
    // reference: every byte belongs to exactly one word.
    let per_text: Result<Vec<FastMap<String, u64>>> = texts
        .par_iter()
        .map(|text| {
            let mut local: FastMap<String, u64> = FastMap::default();
            for piece in split_long_segment(text) {
                for chunk in pretokenize(piece)? {
                    if !chunk.is_empty() {
                        *local.entry(chunk.to_owned()).or_insert(0) += 1;
                    }
                }
            }
            Ok(local)
        })
        .collect();
    let mut freq: FastMap<String, u64> = FastMap::default();
    for local in per_text? {
        for (w, c) in local {
            *freq.entry(w).or_insert(0) += c;
        }
    }

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
    // Max-heap: highest count first, ties broken by smallest pair
    // (frozen §5.4). `Reverse(pair)` in a max-heap pops the smallest pair.
    let mut heap: BinaryHeap<(u64, Reverse<(u32, u32)>)> = BinaryHeap::new();
    for (&pair, &count) in counts.iter() {
        heap.push((count, Reverse(pair)));
    }

    // 4. Merge loop with incremental updates.
    let mut merges: Vec<(u32, u32)> = Vec::new();
    while merges.len() < target_merges {
        // Find the live best pair (skip stale heap entries).
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
        let holders = index.remove(&pair).unwrap_or_default();
        for idx in holders {
            let word = &mut words[idx];
            let old = pair_multiset(&word.tokens);
            if !apply_merge(&mut word.tokens, pair, new_id) {
                continue; // stale index ref
            }
            let new = pair_multiset(&word.tokens);
            // Decrement destroyed pairs, increment created ones. Every pair
            // whose count changes - up OR down - gets a fresh heap entry.
            // (Down matters: a pair can decrease *into* first place, and
            // without a fresh entry it would sit invisible while lower
            // pairs merge. Stale entries die on pop.)
            for (p, c) in &old {
                let delta = c * word.freq;
                match counts.get_mut(p) {
                    Some(v) => {
                        *v = v
                            .checked_sub(delta)
                            .ok_or(Error::NotImplemented("pair count underflow (training bug)"))?;
                    }
                    None => {
                        return Err(Error::NotImplemented(
                            "missing pair count during merge (training bug)",
                        ));
                    }
                }
                heap.push((counts[p], Reverse(*p)));
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
        // Chunks are single words here (no spaces inside pieces... spaces
        // split words, so craft one word with a tie).
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
