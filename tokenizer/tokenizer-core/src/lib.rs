//! nanoMLX byte-level BPE tokenizer.
//!
//!
//! ```text
//! pretokenize.rs  regex split + special-token split (the pattern lives in
//!                 tokenizer/patterns/o200k_style.txt, compiled per thread)
//! vocab.rs        Tokenizer struct: merges, ranks, token bytes, special tokens
//! encode.rs       per-chunk BPE (linked list + heap), chunk cache, rayon batch
//! decode.rs       ids back to bytes to str (lossy and strict)
//! io.rs           save/load vocab JSON, streaming encode_file_to_bin to .bin
//! train.rs        BPE training (incremental pair counts)
//! error.rs        typed errors, no unwrap on paths user input can hit
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
