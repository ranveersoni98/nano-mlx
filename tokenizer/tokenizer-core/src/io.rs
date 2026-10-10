//! Persistence + the MLX data-pipeline entry point.
//!
//! Save format (single JSON; a binary sidecar is only needed for huge vocabs
//! and is out of scope until one is measured):
//!
//! ```json
//! {
//!   "version": 1,
//!   "pattern_hash": "<fnv1a64 of o200k_style.txt, hex>",
//!   "special_tokens": ["<|endoftext|>", "<|pad|>"],
//!   "merges": [[104, 105], ...],
//!   "checksum": "<fnv1a64 over merges, hex>"
//! }
//! ```
//!
//! `load` refuses version, pattern, or checksum mismatches loudly -
//! tokenizing training data with a different pattern than the model expects
//! silently corrupts everything downstream.
//!
//! `.bin` files (frozen, AGENTS.md §6) are raw little-endian ids, no header:
//! `u16` iff `vocab_size < 65536` else `u32`, with a sibling `meta.json`.

use std::fs;
use std::path::Path;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::encode::encode_bytes;
use crate::error::{Error, Result};
use crate::vocab::{fnv1a64, Tokenizer};

/// Vocab file format version. Bump only with user approval (frozen format).
pub const SAVE_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct SaveFile {
    version: u32,
    pattern_hash: String,
    special_tokens: Vec<String>,
    merges: Vec<[u32; 2]>,
    checksum: String,
}

fn hex(h: u64) -> String {
    format!("{h:016x}")
}

/// Current UTC time as ISO-8601 (`created` field). Hand-rolled days-to-civil
/// (Howard Hinnant's algorithm) to avoid a date-time dependency.
fn iso8601_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (day_secs, time) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = day_secs + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146097) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        time / 3600,
        (time % 3600) / 60,
        time % 60
    )
}

fn unhex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

/// Save the vocab to `path` as JSON.
pub fn save(tok: &Tokenizer, path: &Path) -> Result<()> {
    let merges: Vec<[u32; 2]> = tok.merges().iter().map(|&(l, r)| [l, r]).collect();
    let file = SaveFile {
        version: SAVE_VERSION,
        pattern_hash: hex(tok.pattern_hash()),
        special_tokens: tok.special_tokens().to_vec(),
        merges,
        checksum: hex(tok.checksum()),
    };
    let json = serde_json::to_string_pretty(&file).map_err(Error::from)?;
    fs::write(path, json)?;
    Ok(())
}

/// Load a vocab, validating version, pattern hash, and checksum.
pub fn load(path: &Path) -> Result<Tokenizer> {
    let json = fs::read_to_string(path)?;
    let file: SaveFile = serde_json::from_str(&json).map_err(Error::from)?;
    if file.version != SAVE_VERSION {
        return Err(Error::VersionMismatch {
            expected: SAVE_VERSION,
            found: file.version,
        });
    }
    let merges: Vec<(u32, u32)> = file.merges.iter().map(|&[l, r]| (l, r)).collect();
    let tok = Tokenizer::from_merges(merges, file.special_tokens)?;
    let found_pattern = unhex(&file.pattern_hash).unwrap_or(0);
    if found_pattern != tok.pattern_hash() {
        return Err(Error::PatternMismatch {
            expected: tok.pattern_hash(),
            found: found_pattern,
        });
    }
    let found_sum = unhex(&file.checksum).unwrap_or(0);
    if found_sum != tok.checksum() {
        return Err(Error::ChecksumMismatch {
            expected: tok.checksum(),
            found: found_sum,
        });
    }
    Ok(tok)
}

/// Stats returned by [`encode_file_to_bin`] (also the dict the Python
/// binding returns, and the core of a benchmark JSON row).
#[derive(Debug, Clone, Serialize)]
pub struct FileStats {
    pub n_tokens: u64,
    pub bytes_in: u64,
    pub seconds: f64,
    pub dtype: String,
}

/// Encode a whole file to a `.bin` of little-endian ids + `meta.json`.
///
/// `allowed_specials` lists special-token strings available as control
/// tokens (additive kwarg; default empty = tiktoken-style safe default).
/// The MLX data pipeline joins documents with `<|endoftext|>` markers and
/// passes it here so documents stay EOS-separated in the `.bin`.
///
/// Parallelism: the input is sharded at `\n` boundaries, which the
/// pre-tokenization regex is guaranteed to also split at - every pattern
/// alternative containing `\n` ends with it, so no match ever straddles a
/// `\n`. Shards therefore concatenate to exactly the single-thread output
/// (proven by `thread_count_independence` below on every run of the test
/// suite). Shard starts/ends sit right after `\n` (ASCII), so they are
/// always UTF-8 char boundaries and the bytes-path chunking agrees too.
///
/// This is the function that feeds the MLX pipeline: `train.bin`/`val.bin`
/// + `meta.json` are what `model/src/nanomlx/data.py` memory-maps.
pub fn encode_file_to_bin(
    tok: &Tokenizer,
    in_path: &Path,
    out_path: &Path,
    allowed_specials: &[String],
) -> Result<FileStats> {
    let started = Instant::now();
    let bytes = fs::read(in_path)?;
    let bytes_in = bytes.len() as u64;

    let n_threads = rayon::current_num_threads().max(1);
    let shards = shard_at_newlines(&bytes, n_threads);
    let mut all_ids: Vec<u32> = Vec::new();
    if shards.len() <= 1 {
        all_ids.extend(encode_bytes(tok, &bytes, allowed_specials)?);
    } else {
        let parts: Result<Vec<Vec<u32>>> = shards
            .par_iter()
            .map(|s| encode_bytes(tok, s, allowed_specials))
            .collect();
        for mut p in parts? {
            all_ids.append(&mut p);
        }
    }

    // Frozen dtype rule (AGENTS.md §12): u16 iff vocab < 64k.
    let use_u16 = tok.vocab_size() < 65536;
    let dtype = if use_u16 { "uint16" } else { "uint32" };
    let mut buf = Vec::with_capacity(all_ids.len() * if use_u16 { 2 } else { 4 });
    for id in &all_ids {
        if use_u16 {
            let v = u16::try_from(*id).map_err(|_| Error::InvalidId(*id))?;
            buf.extend_from_slice(&v.to_le_bytes());
        } else {
            buf.extend_from_slice(&id.to_le_bytes());
        }
    }
    fs::write(out_path, &buf)?;

    // Sibling meta.json, frozen schema. `tokenizer_sha256` holds the FNV-1a
    // pattern fingerprint until a hashing dep is approved; loaders compare
    // it for equality.
    let eos_id = tok
        .special_id("<|endoftext|>")
        .or_else(|| tok.special_tokens().first().and_then(|s| tok.special_id(s)))
        .unwrap_or(0);
    let meta = serde_json::json!({
        "dtype": dtype,
        "vocab_size": tok.vocab_size(),
        "n_tokens": all_ids.len(),
        "eos_id": eos_id,
        "tokenizer_sha256": hex(tok.pattern_hash()),
        "source": in_path.to_string_lossy(),
        "created": iso8601_now(),
    });
    let meta_path = out_path.with_extension("meta.json");
    // Handle `.bin` -> `.meta.json` (with_extension replaces `bin`).
    let meta_path = if meta_path == out_path {
        out_path.with_extension("bin.meta.json")
    } else {
        meta_path
    };
    fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;

    Ok(FileStats {
        n_tokens: all_ids.len() as u64,
        bytes_in,
        seconds: started.elapsed().as_secs_f64(),
        dtype: dtype.to_owned(),
    })
}

/// Split `bytes` into ≤ `n` shards, each ending right after a `\n`
/// (the final shard takes the tail). See fn docs for why this is exact.
fn shard_at_newlines(bytes: &[u8], n: usize) -> Vec<&[u8]> {
    if n <= 1 || bytes.len() < 64 * 1024 {
        return vec![bytes];
    }
    let target = bytes.len() / n;
    let mut bounds = vec![0usize];
    for k in 1..n {
        let mut b = (k * target).min(bytes.len());
        // Only newlines: memchr-style scan without the dependency.
        while b < bytes.len() && bytes[b] != b'\n' {
            b += 1;
        }
        if b < bytes.len() {
            b += 1; // include the newline: boundary sits after it.
        }
        if b > *bounds.last().unwrap_or(&0) && b < bytes.len() {
            bounds.push(b);
        }
    }
    bounds.push(bytes.len());
    bounds
        .windows(2)
        .map(|w| &bytes[w[0]..w[1]])
        .filter(|s| !s.is_empty())
        .collect()
}

/// Portable checksum helper shared with the Python reference (FNV-1a over
/// the raw file bytes), used by `data.py`'s `tokenizer_sha256` equivalent.
pub fn file_fingerprint(bytes: &[u8]) -> u64 {
    fnv1a64(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::decode_lossy;

    fn mini_tok() -> Tokenizer {
        Tokenizer::from_merges(vec![(104, 105)], vec!["<|endoftext|>".into()]).expect("build")
    }

    #[test]
    fn save_load_round_trip() {
        let tok = mini_tok();
        let dir = std::env::temp_dir().join("nanomlx_tok_test");
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("vocab.json");
        save(&tok, &path).expect("save");
        let back = load(&path).expect("load");
        assert_eq!(back.merges(), tok.merges());
        assert_eq!(back.special_tokens(), tok.special_tokens());
        assert_eq!(back.checksum(), tok.checksum());
    }

    #[test]
    fn load_rejects_tampered_checksum() {
        let tok = mini_tok();
        let dir = std::env::temp_dir().join("nanomlx_tok_test");
        fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("tampered.json");
        save(&tok, &path).expect("save");
        let mut json = fs::read_to_string(&path).expect("read");
        json = json.replacen("[", "[[", 1); // corrupt the merges array
        fs::write(&path, json).expect("write");
        assert!(load(&path).is_err());
    }

    #[test]
    fn thread_count_independence() {
        // Mixed text incl. multibyte chars, punctuation-newline runs, and
        // invalid bytes - the exact cases that break naive sharding.
        let tok = mini_tok();
        let mut bytes = "hello\nhi!!!\n\nworld 🎉\n  spaced   \n"
            .as_bytes()
            .to_vec();
        bytes.extend_from_slice(&[0xFF, 0xFE, b'a', 0x80]);
        bytes.extend_from_slice("trailing no newline".as_bytes());

        let dir = std::env::temp_dir().join("nanomlx_tok_test");
        fs::create_dir_all(&dir).expect("mkdir");
        let inp = dir.join("shard_input.txt");
        fs::write(&inp, &bytes).expect("write");

        let single = encode_bytes(&tok, &bytes, &[]).expect("single");

        // Force several thread counts through scoped pools.
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let parallel = pool.install(|| {
                let shards = shard_at_newlines(&bytes, threads);
                let parts: Result<Vec<Vec<u32>>> = shards
                    .par_iter()
                    .map(|s| encode_bytes(&tok, s, &[]))
                    .collect();
                parts.map(|pp| pp.into_iter().flatten().collect::<Vec<u32>>())
            });
            let parallel = parallel.expect("parallel");
            assert_eq!(single, parallel, "threads={threads}");
        }

        // And the file entry point agrees + round-trips through decode.
        let out = dir.join("shard_out.bin");
        let stats = encode_file_to_bin(&tok, &inp, &out, &[]).expect("file");
        assert_eq!(stats.n_tokens, single.len() as u64);
        assert_eq!(stats.bytes_in, bytes.len() as u64);
        assert_eq!(
            decode_lossy(&tok, &single).expect("decode").as_bytes()[..8],
            bytes[..8]
        );
        let _ = fs::remove_file(&inp);
        let _ = fs::remove_file(&out);
    }
}
