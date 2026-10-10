//! Typed errors for `tokenizer-core`.
//!
//! Library code paths reachable from user input must return [`Result`] with
//! this error type - never `unwrap`/`expect`/`panic`. (`unsafe` is forbidden
//! per repo policy.)

use std::fmt;

/// All failure modes of the tokenizer core.
#[derive(Debug)]
pub enum Error {
    /// Filesystem failure (read/write vocab, input corpus, `.bin` output).
    Io(std::io::Error),
    /// Malformed vocab JSON on save/load (boxed: keep `Result` small).
    Json(Box<serde_json::Error>),
    /// The bundled pre-tokenization regex failed to compile or execute.
    /// This can only happen if `patterns/o200k_style.txt` is corrupted;
    /// it is never a function of user input. (Boxed: `fancy_regex::Error`
    /// is large; see `result_large_err`.)
    Regex(Box<fancy_regex::Error>),
    /// Loaded vocab was trained with a different pre-tokenization pattern.
    PatternMismatch {
        /// FNV-1a hash of the pattern this build was compiled with.
        expected: u64,
        /// Hash recorded in the loaded vocab file.
        found: u64,
    },
    /// Vocab file format version is newer/older than this build understands.
    VersionMismatch { expected: u32, found: u32 },
    /// Merge list was corrupted after saving (checksum over merges differs).
    ChecksumMismatch { expected: u64, found: u64 },
    /// A token id is outside `0..vocab_size` (usually a caller bug or a
    /// truncated `.bin` file).
    InvalidId(u32),
    /// Caller asked to encode with a special token that is not in the vocab.
    /// Fix: add it at train time, or remove it from `allowed_special`.
    UnknownSpecialToken(String),
    /// `decode_strict` hit bytes that are not valid UTF-8. Use the lossy
    /// `decode` (replacement character) if that is acceptable.
    StrictUtf8(std::string::FromUtf8Error),
    /// A training/corpus file was not valid UTF-8 (training inputs must be
    /// valid UTF-8; TinyStories-class corpora are).
    InvalidUtf8(String),
    /// A thread-local (compiled regex / chunk cache) was unavailable,
    /// e.g. during thread teardown.
    ThreadLocal(&'static str),
    /// Called functionality scheduled for a later milestone (e.g. Rust
    /// training is M3).
    NotImplemented(&'static str),
}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Json(e) => write!(f, "vocab JSON error: {e}"),
            Error::Regex(e) => write!(f, "regex error: {e}"),
            Error::PatternMismatch { expected, found } => write!(
                f,
                "pattern mismatch: vocab was trained with pattern hash {found:#x}, \
                 this build uses {expected:#x}. Re-train with the current \
                 tokenizer/patterns/o200k_style.txt"
            ),
            Error::VersionMismatch { expected, found } => write!(
                f,
                "vocab version mismatch: file is v{found}, this build reads v{expected}"
            ),
            Error::ChecksumMismatch { expected, found } => write!(
                f,
                "vocab checksum mismatch: expected {expected:#x}, found {found:#x} \
                 (file is corrupted)"
            ),
            Error::InvalidId(id) => write!(f, "invalid token id: {id}"),
            Error::UnknownSpecialToken(s) => write!(
                f,
                "unknown special token: {s:?} (train with it listed in \
                 special_tokens, or drop it from allowed_special)"
            ),
            Error::StrictUtf8(e) => write!(f, "invalid UTF-8 in decoded bytes: {e}"),
            Error::InvalidUtf8(what) => {
                write!(
                    f,
                    "invalid UTF-8 in {what} (training files must be valid UTF-8)"
                )
            }
            Error::ThreadLocal(what) => {
                write!(f, "thread-local {what} unavailable (thread teardown?)")
            }
            Error::NotImplemented(what) => {
                write!(f, "{what} is scheduled for a later milestone")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            Error::Json(e) => Some(e),
            Error::StrictUtf8(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(Box::new(e))
    }
}

impl From<fancy_regex::Error> for Error {
    fn from(e: fancy_regex::Error) -> Self {
        Error::Regex(Box::new(e))
    }
}

impl From<std::string::FromUtf8Error> for Error {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Error::StrictUtf8(e)
    }
}
