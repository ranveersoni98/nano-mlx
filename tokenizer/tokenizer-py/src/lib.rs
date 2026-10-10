//! Thin PyO3 wrapper around `tokenizer-core`. No real logic here.
//!
//! Just GIL handling (`py.detach` on calls that can take over ~1 ms) and
//! error mapping. All BPE logic lives in `tokenizer-core` and gets tested
//! with plain `cargo test`.

// pyo3's #[pymethods] macro makes an `Into<PyErr>` conversion on values
// that already are `PyErr`. That `clippy::useless_conversion` hit is macro
// fallout, not a real conversion we wrote.
#![allow(clippy::useless_conversion)]

use std::path::PathBuf;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokenizer_core::{FileStats, Tokenizer};

fn to_py(err: tokenizer_core::Error) -> PyErr {
    use tokenizer_core::Error::*;
    match err {
        InvalidId(_) | UnknownSpecialToken(_) | StrictUtf8(_) | InvalidUtf8(_) => {
            PyValueError::new_err(err.to_string())
        }
        Io(_) => PyIOError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Byte-level BPE tokenizer (Rust inside, Python outside).
///
/// encode_file_to_bin / decode / save / load`.
#[pyclass(module = "nanomlx_tokenizer")]
struct PyTokenizer {
    inner: Tokenizer,
}

const DEFAULT_SPECIALS: &[&str] = &["<|endoftext|>", "<|pad|>"];

#[pymethods]
impl PyTokenizer {
    #[new]
    fn new(merges: Vec<(u32, u32)>, special_tokens: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Tokenizer::from_merges(merges, special_tokens).map_err(to_py)?,
        })
    }

    /// Train a vocab. Each entry of `files_or_texts` is read as a file if
    /// it's there, otherwise treated as literal text (same as the Python side).
    /// Training files have to be valid UTF-8.
    #[staticmethod]
    #[pyo3(signature = (files_or_texts, vocab_size, special_tokens=None))]
    fn train(
        py: Python<'_>,
        files_or_texts: Vec<String>,
        vocab_size: usize,
        special_tokens: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let specials: Vec<String> = special_tokens
            .unwrap_or_else(|| DEFAULT_SPECIALS.iter().map(|s| s.to_string()).collect());
        let inner = py.detach(|| {
            let mut texts = Vec::with_capacity(files_or_texts.len());
            for item in &files_or_texts {
                let path = std::path::Path::new(item);
                if path.is_file() {
                    let bytes = std::fs::read(path)
                        .map_err(tokenizer_core::Error::from)
                        .map_err(to_py)?;
                    let text = String::from_utf8(bytes)
                        .map_err(|_| {
                            tokenizer_core::Error::InvalidUtf8(format!(
                                "training input file {}",
                                path.display()
                            ))
                        })
                        .map_err(to_py)?;
                    texts.push(text);
                } else {
                    texts.push(item.clone());
                }
            }
            tokenizer_core::train::train(&texts, vocab_size, specials).map_err(to_py)
        })?;
        Ok(Self { inner })
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn num_merges(&self) -> usize {
        self.inner.num_merges()
    }

    #[pyo3(signature = (text, allowed_special=None))]
    fn encode(
        &self,
        py: Python<'_>,
        text: &str,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<u32>> {
        let allowed = allowed_special.unwrap_or_default();
        py.detach(|| {
            tokenizer_core::encode::encode_text(&self.inner, text, &allowed).map_err(to_py)
        })
    }

    #[pyo3(signature = (texts, allowed_special=None))]
    fn encode_batch(
        &self,
        py: Python<'_>,
        texts: Vec<String>,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<u32>>> {
        let allowed = allowed_special.unwrap_or_default();
        py.detach(|| {
            tokenizer_core::encode::encode_batch(&self.inner, &texts, &allowed).map_err(to_py)
        })
    }

    #[pyo3(signature = (ids, strict=false))]
    fn decode(&self, py: Python<'_>, ids: Vec<u32>, strict: bool) -> PyResult<String> {
        py.detach(|| {
            if strict {
                tokenizer_core::decode::decode_strict(&self.inner, &ids).map_err(to_py)
            } else {
                tokenizer_core::decode::decode_lossy(&self.inner, &ids).map_err(to_py)
            }
        })
    }

    #[pyo3(signature = (in_path, out_path, allowed_special=None))]
    fn encode_file_to_bin(
        &self,
        py: Python<'_>,
        in_path: PathBuf,
        out_path: PathBuf,
        allowed_special: Option<Vec<String>>,
    ) -> PyResult<Py<PyDict>> {
        let allowed = allowed_special.unwrap_or_default();
        let stats: FileStats = py.detach(|| {
            tokenizer_core::io::encode_file_to_bin(&self.inner, &in_path, &out_path, &allowed)
                .map_err(to_py)
        })?;
        let dict = PyDict::new(py);
        dict.set_item("n_tokens", stats.n_tokens)?;
        dict.set_item("bytes_in", stats.bytes_in)?;
        dict.set_item("seconds", stats.seconds)?;
        dict.set_item("dtype", stats.dtype)?;
        Ok(dict.into())
    }

    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        py.detach(|| tokenizer_core::io::save(&self.inner, &path).map_err(to_py))
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py.detach(|| tokenizer_core::io::load(&path).map_err(to_py))?;
        Ok(Self { inner })
    }

    fn merges(&self) -> Vec<(u32, u32)> {
        self.inner.merges().to_vec()
    }

    fn special_tokens(&self) -> Vec<String> {
        self.inner.special_tokens().to_vec()
    }
}

/// `nanomlx_tokenizer` module root.
#[pymodule]
fn nanomlx_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    // Also export as plain `Tokenizer` so Python reads naturally:
    // `from nanomlx_tokenizer import Tokenizer`.
    m.add("Tokenizer", m.getattr("PyTokenizer")?)?;
    Ok(())
}
