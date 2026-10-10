//! Errors for `tokenizer-core`.
//!
//! Anything user input can reach has to return [`Result`] with this type.
//! No `unwrap` / `expect` / `panic` there. (`unsafe` is off limits in this repo.)

use std::fmt;

/// Everything that can go wrong in the tokenizer.
#[derive(Debug)]
pub enum Error {
    /// Filesystem stuff (reading/writing vocabs, corpora, `.bin` files).
    Io(std::io::Error),
    /// Vocab JSON didn't parse on save/load (boxed to keep `Result` small).
    Json(Box<serde_json::Error>),
    /// The bundled split regex wouldn't compile or run. Only happens if
    /// `patterns/o200k_style.txt` got corrupted, never from user input.
    /// (Boxed since `fancy_regex::Error` is big, see `result_large_err`.)
    Regex(Box<fancy_regex::Error>),
    /// Loaded vocab was trained with a different split pattern.
    PatternMismatch {
        /// FNV-1a of the pattern this build was compiled with.
        expected: u64,
        /// Hash written in the loaded vocab file.
        found: u64,
    },
    /// Vocab file version this build doesn't understand.
    VersionMismatch { expected: u32, found: u32 },
    /// Merge list got mangled after saving (checksum over merges differs).
    ChecksumMismatch { expected: u64, found: u64 },
    /// Token id outside `0..vocab_size` (usually a caller bug or a cut-off
    /// `.bin` file).
    InvalidId(u32),
    /// Asked to encode with a special token that's not in the vocab. Fix is
    /// to add it at train time or drop it from `allowed_special`.
    UnknownSpecialToken(String),
    /// `decode_strict` hit bytes that aren't valid UTF-8. Use the lossy
    /// `decode` (replacement char) if that's okay.
    StrictUtf8(std::string::FromUtf8Error),
    /// A training/corpus file was not valid UTF-8 (training inputs must be
    /// valid UTF-8; TinyStories-class corpora are).
    InvalidUtf8(String),
    /// A thread-local (compiled regex / chunk cache) was unavailable,
    /// e.g. during thread teardown.
    ThreadLocal(&'static str),
    /// Called something we haven't built yet.
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
                write!(f, "{what} isn't built yet")
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
