//! nanoMLX byte-level BPE tokenizer core (pure Rust, no PyO3).
//!
//! Layout (AGENTS.md §2):
//!
//! ```text
//! pretokenize.rs  regex split + special-token split (single source of truth:
//!                 tokenizer/patterns/o200k_style.txt, compiled per thread)
//! vocab.rs        Tokenizer struct: merges, ranks, token bytes, special tokens
//! encode.rs       per-chunk BPE (linked list + binary heap), chunk cache, rayon batch
//! decode.rs       ids -> bytes -> str (lossy + strict)
//! io.rs           save/load vocab JSON, streaming encode_file_to_bin -> .bin
//! train.rs        BPE training (incremental pair counts, M3)
//! error.rs        typed errors, no unwrap on user-input paths
//! ```

pub mod decode;
pub mod encode;
pub mod error;
pub mod io;
pub mod pretokenize;
pub mod train;
pub mod vocab;

pub use error::{Error, Result};
pub use io::FileStats;
pub use vocab::Tokenizer;
