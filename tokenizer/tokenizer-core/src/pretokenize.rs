//! Pre-tokenization: chop text into chunks with the shared regex.
//!
//! The pattern lives in `tokenizer/patterns/o200k_style.txt` and that's the
//! only copy. We bake it in with `include_str!` (trimming the trailing
//! newline, since a literal `\n` here would end up inside the pattern)
//! and compile it once per thread (see below), so `pretokenize` stays cheap.
//!
//! How it works:
//! - Roughly o200k style (GPT-4o / Llama-3 family): case-aware word bits,
//!   contractions, numbers in runs of up to 3 digits, punctuation runs,
//!   whitespace with lookahead. Lookahead needs backtracking, so that's
//!   `fancy-regex` on the Rust side and the third-party `regex` crate on
//!   the Python side.
//! - If Unicode classes (`\p{Lu}` etc.) ever disagree between Rust
//!   `fancy-regex` and Python `regex`, that's a bug in the pattern file,
//!   not something to work around per call. If a check fails on Devanagari
//!   / emoji / combining marks, look here first.
//! - `fancy_regex::Regex` sits in a thread-local instead of inside
//!   [`crate::Tokenizer`] so `&Tokenizer` stays `Send + Sync` for rayon
//!   without depending on how the engine implements its auto traits.

use std::cell::RefCell;

use fancy_regex::Regex;

use crate::error::{Error, Result};

const PATTERN_RAW: &str = include_str!("../../patterns/o200k_style.txt");

/// Longest we'll hand the regex in one go, in chars.
///
/// `fancy-regex` 0.14 has a fixed 1M-entry backtrack stack (MAX_STACK in its
/// VM, not tunable). We measured on 2026-10-10 (release): 524 KB runs of
/// letters/spaces are fine, 1 MB runs blow up. 64K chars leaves us ~8x
/// headroom. Longer bits get chopped at char edges (see below). Never an
/// error, never a panic.
pub const MAX_SEGMENT_CHARS: usize = 65_536;

/// The exact pattern text both implementations must use.
pub fn pattern_text() -> &'static str {
    // Trim only the file's trailing newline; interior whitespace is significant.
    PATTERN_RAW.trim_end_matches(['\r', '\n'])
}

thread_local! {
    static CACHED_RE: RefCell<Option<Regex>> = const { RefCell::new(None) };
}

/// Run `f` with this thread's compiled pattern, compiling on first use.
///
/// The pattern ships with the repo, it's not user input, so if it won't
/// compile the file got corrupted. That comes back as [`Error::Regex`],
/// never a panic.
fn with_regex<T>(f: impl FnOnce(&Regex) -> Result<T>) -> Result<T> {
    CACHED_RE
        .try_with(|cell| {
            let mut slot = cell.borrow_mut();
            if slot.is_none() {
                *slot = Some(Regex::new(pattern_text()).map_err(Error::from)?);
            }
            // `slot` is `Some` here by construction.
            match slot.as_ref() {
                Some(re) => f(re),
                None => Err(Error::ThreadLocal("compiled regex")),
            }
        })
        .map_err(|_| Error::ThreadLocal("compiled regex"))?
}

/// Chop `text` into bits of at most [`MAX_SEGMENT_CHARS`] chars on char
/// edges. Short inputs come back as one piece (fast path, costs nothing).
pub fn split_long_segment(text: &str) -> Vec<&str> {
    if text.chars().count() <= MAX_SEGMENT_CHARS {
        return vec![text];
    }
    let mut pieces = Vec::new();
    let mut start = 0usize;
    let mut count = 0usize;
    for (i, _) in text.char_indices() {
        if count >= MAX_SEGMENT_CHARS {
            pieces.push(&text[start..i]);
            start = i;
            count = 0;
        }
        count += 1;
    }
    pieces.push(&text[start..]);
    pieces
}

/// Split `text` on regex match edges, returning borrowed chunks.
///
/// Longer bits than [`MAX_SEGMENT_CHARS`] get chopped first (regex limit,
/// explained above). All on purpose and covered by tests: merges never
/// cross a chop, the Python side does the exact same chopping so both agree,
/// short inputs come out byte-identical to not chopping.
///
/// Matches don't overlap and cover the gaps too: every byte of `text` lands
/// in exactly one chunk (stuff between matches is kept as its own chunk,
/// so chars the pattern skips still get tokenized).
pub fn pretokenize(text: &str) -> Result<Vec<&str>> {
    with_regex(|re| {
        let mut chunks = Vec::new();
        for piece in split_long_segment(text) {
            pretokenize_piece(re, piece, &mut chunks)?;
        }
        Ok(chunks)
    })
}

/// Run the match loop over one bounded piece, appending into `chunks`.
fn pretokenize_piece<'a>(re: &Regex, text: &'a str, chunks: &mut Vec<&'a str>) -> Result<()> {
    let mut pos = 0usize;
    while pos < text.len() {
        let m = re.find_from_pos(text, pos).map_err(Error::from)?;
        match m {
            Some(mt) => {
                let (s, e) = (mt.start(), mt.end());
                if s > pos {
                    chunks.push(&text[pos..s]);
                }
                if e > s {
                    chunks.push(&text[s..e]);
                    pos = e;
                } else {
                    // Zero-width match: can't sit still or we'd spin.
                    // (Our pattern never makes these, this is just extra safety.)
                    let next = text[pos..]
                        .char_indices()
                        .nth(1)
                        .map(|(i, _)| pos + i)
                        .unwrap_or(text.len());
                    chunks.push(&text[pos..next]);
                    pos = next;
                }
            }
            None => {
                chunks.push(&text[pos..]);
                break;
            }
        }
    }
    Ok(())
}

/// One piece of [`split_special`]: either a registered special token (as the
/// exact matched string) or plain text to pre-tokenize.
#[derive(Debug, PartialEq, Eq)]
pub enum Split<'t> {
    Special(&'t str),
    Text(&'t str),
}

/// Split `text` on exact hits of the `allowed` special-token strings.
///
/// Specials match before the regex runs and never get split or merged.
/// Anything not in `allowed` stays normal text (tiktoken-style default, so
/// user data can't sneak in control tokens). Longest match wins at each spot
/// so overlapping names (like `<|a|>` vs `<|ab|>`) behave.
pub fn split_special<'t>(text: &'t str, allowed: &[&'t str]) -> Vec<Split<'t>> {
    if allowed.is_empty() || text.is_empty() {
        return if text.is_empty() {
            Vec::new()
        } else {
            vec![Split::Text(text)]
        };
    }
    // Longest-first so overlaps resolve deterministically.
    let mut pats: Vec<&'t str> = allowed.to_vec();
    pats.sort_by_key(|s| std::cmp::Reverse(s.len()));

    let mut out = Vec::new();
    let mut pos = 0usize;
    let mut run_start = 0usize;
    while pos < text.len() {
        let mut hit: Option<&'t str> = None;
        for p in &pats {
            if text[pos..].starts_with(p) {
                hit = Some(p);
                break;
            }
        }
        match hit {
            Some(p) => {
                if pos > run_start {
                    out.push(Split::Text(&text[run_start..pos]));
                }
                out.push(Split::Special(p));
                pos += p.len();
                run_start = pos;
            }
            None => {
                // Advance one scalar value (specials are matched on bytes,
                // but stepping by char keeps `pos` on a valid boundary).
                let step = text[pos..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(1);
                pos += step;
            }
        }
    }
    if run_start < text.len() {
        out.push(Split::Text(&text[run_start..]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_loads_and_matches() {
        assert!(pattern_text().contains(r"\p{L}"));
        let chunks = pretokenize("Hello, world! 123").expect("pretokenize");
        assert!(chunks.iter().any(|c| c.contains("Hello")));
        assert!(chunks.iter().any(|c| c.contains("world")));
        // Re-joining the chunks reproduces the input byte-for-byte.
        assert_eq!(chunks.concat(), "Hello, world! 123");
    }

    #[test]
    fn chunks_cover_every_byte() {
        for s in [
            "",
            "a",
            "  \n\n  ",
            "emoji 🎉 test",
            "śźćół",
            "a'b 're 1234",
        ] {
            let chunks = pretokenize(s).expect("pretokenize");
            assert_eq!(chunks.concat(), s, "input {s:?}");
        }
    }

    #[test]
    fn special_split_longest_wins() {
        let parts = split_special("x<|ab|>y", &["<|a|>", "<|ab|>"]);
        assert_eq!(
            parts,
            vec![Split::Text("x"), Split::Special("<|ab|>"), Split::Text("y")]
        );
    }

    #[test]
    fn special_split_defaults_to_text() {
        let parts = split_special("a<|endoftext|>b", &[]);
        assert_eq!(parts, vec![Split::Text("a<|endoftext|>b")]);
    }

    #[test]
    fn force_split_triggers_only_above_threshold() {
        let short = "a".repeat(MAX_SEGMENT_CHARS);
        assert_eq!(split_long_segment(&short).len(), 1);
        let long = "a".repeat(MAX_SEGMENT_CHARS + 1);
        let pieces = split_long_segment(&long);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces.concat(), long);
        // Pieces are char-boundary aligned even for multibyte text.
        let uni = "あ".repeat(MAX_SEGMENT_CHARS + 7);
        let pieces = split_long_segment(&uni);
        assert_eq!(pieces.concat(), uni);
        for p in &pieces {
            assert!(p.chars().count() <= MAX_SEGMENT_CHARS);
        }
    }

    #[test]
    fn huge_single_chunk_never_overflows() {
        // 1 MB of one byte has to encode, not error.
        let big = "a".repeat(1_000_000);
        let chunks = pretokenize(&big).expect("pretokenize 1MB");
        assert_eq!(chunks.concat(), big);
        let spaces = " ".repeat(1_000_000);
        let chunks = pretokenize(&spaces).expect("pretokenize 1MB spaces");
        assert_eq!(chunks.concat(), spaces);
    }
}
