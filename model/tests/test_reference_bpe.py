"""Our Python BPE reference.

Checks round-trip on English, code (Python/Rust/JSON), emoji + ZWJ bits,
CJK, Hindi, Arabic, combining chars, whitespace, tabs/newlines/CRLF,
empty string, single byte, special-token handling, plus training giving
the same thing twice and save/load.
"""

import pytest

from nanomlx.reference_bpe import Tokenizer

TINY_CORPUS = (
    "Hello, world! This is a tiny corpus for deterministic BPE training.\n"
    "The quick brown fox jumps over the lazy dog. 123 numbers 4567 here.\n"
    "def hello(name):\n    return f'hi {name}!'\n"
    'fn main() { println!("Hello, world!"); }\n'
    '{"key": "value", "nums": [1, 2, 3]}\n'
    "Emoji party: 🎉🚀 and family 👨‍👩‍👧‍👦 plus flags 🇺🇸🇮🇳\n"
    "CJK: 你好世界 日本語のテスト 한국어 테스트\n"
    "Hindi: नमस्ते दुनिया, यह एक परीक्षण है।\n"
    "Arabic: مرحبا بالعالم، هذا اختبار للرموز\n"
    "Combining: café naïve résumé Z͑ͫ̓ͪ̂ͫ̽͏̴̙̤̞͉͚̯̞͠aͫlͨgͤoͭ\n"
    "Whitespace:    lots\t\tof\n\n\nnewlines\r\nand CRLF   end\n"
)

PROBES = [
    "Hello, world!",
    "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n",
    "fn main() { let x: Vec<u8> = vec![0xFF, 0xFE]; }\n",
    '{"hello": ["world", 123], "nested": {"a": true}}',
    "No emoji here, just plain ASCII text with numbers 123456.",
    "🎉 single emoji",
    "👨‍👩‍👧‍👦 ZWJ family sequence",
    "🇺🇸🇮🇳 flag pairs",
    "你好，世界！",
    "日本語のテストです。",
    "한국어 테스트입니다.",
    "नमस्ते दुनिया",
    "مرحبا بالعالم",
    "café",
    "Z͑ͫ̓ͪ̂ͫ̽͏̴̙̤̞͉͚̯̞͠aͫlͨgͤoͭ",
    "   leading and trailing   ",
    "\t\ttabs\tand\n\nnewlines\n",
    "line1\r\nline2\rline3\n",
    "",
    "a",
    "x" * 500,
    "a" * 10000,  # huge single chunk
    "<|endoftext|> looks special but must be plain text by default",
    "mix <|pad|> tokens as text",
    "don't can't won't it's (contractions)",
    "prices: $19.99, €25, ₹1,999.50 and 1000000",
]


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return Tokenizer.train([TINY_CORPUS], vocab_size=320)


def test_round_trip_all_probes(tok: Tokenizer):
    for text in PROBES:
        assert tok.decode(tok.encode(text)) == text, f"round-trip failed for {text!r:.60}"


def test_encode_deterministic(tok: Tokenizer):
    for text in PROBES:
        assert tok.encode(text) == tok.encode(text)


def test_training_deterministic():
    a = Tokenizer.train([TINY_CORPUS], vocab_size=320)
    b = Tokenizer.train([TINY_CORPUS], vocab_size=320)
    assert a.merges == b.merges
    assert a.special_tokens == b.special_tokens


def test_vocab_layout(tok: Tokenizer):
    assert tok.vocab_size == 256 + len(tok.merges) + len(tok.special_tokens)
    # Specials sit at the end of id space.
    base = 256 + len(tok.merges)
    assert tok.special_to_id["<|endoftext|>"] == base
    assert tok.special_to_id["<|pad|>"] == base + 1


def test_special_tokens_default_to_plain_text(tok: Tokenizer):
    ids_plain = tok.encode("<|endoftext|>")
    ids_allowed = tok.encode("<|endoftext|>", allowed_special="all")
    assert ids_allowed == [tok.special_to_id["<|endoftext|>"]]
    assert ids_plain != ids_allowed  # plain text gets split into normal pieces
    assert tok.decode(ids_plain) == "<|endoftext|>"


def test_special_tokens_explicit_set(tok: Tokenizer):
    ids = tok.encode("a<|pad|>b", allowed_special={"<|pad|>"})
    assert tok.special_to_id["<|pad|>"] in ids
    assert tok.decode(ids) == "a<|pad|>b"


def test_unknown_special_rejected(tok: Tokenizer):
    with pytest.raises(ValueError):
        tok.encode("hi", allowed_special={"<|nope|>"})


def test_empty_and_single_byte(tok: Tokenizer):
    assert tok.encode("") == []
    assert tok.decode([]) == ""
    assert tok.decode(tok.encode("a")) == "a"


def test_invalid_id_rejected(tok: Tokenizer):
    with pytest.raises(ValueError):
        tok.decode([tok.vocab_size + 5])


def test_decode_strict_option(tok: Tokenizer):
    # A lone high byte isn't valid UTF-8 on its own.
    lone = tok.encode_bytes(b"\xff")
    assert tok.decode(lone) == "�"
    with pytest.raises(UnicodeDecodeError):
        tok.decode(lone, strict=True)


def test_bytes_path_round_trip(tok: Tokenizer):
    blobs = [b"\xff\xfe\x00abc", "héllo 🎉".encode(), b"", bytes(range(256))]
    for blob in blobs:
        ids = tok.encode_bytes(blob)
        assert tok.decode(ids).encode("utf-8", errors="replace") is not None
        # Byte-exact round trip:
        raw = bytearray()
        for i in ids:
            if i < len(tok.token_bytes):
                raw.extend(tok.token_bytes[i])
        assert bytes(raw) == blob


def test_save_load_round_trip(tok: Tokenizer, tmp_path):
    path = tmp_path / "vocab.json"
    tok.save(path)
    back = Tokenizer.load(path)
    assert back.merges == tok.merges
    assert back.special_tokens == tok.special_tokens
    for text in PROBES:
        assert back.encode(text) == tok.encode(text)


def test_load_rejects_tampered_file(tok: Tokenizer, tmp_path):
    import json

    path = tmp_path / "vocab.json"
    tok.save(path)
    payload = json.loads(path.read_text())
    payload["merges"] = payload["merges"][::-1]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        Tokenizer.load(path)


def test_encode_file_to_bin(tok: Tokenizer, tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("hello world\n" * 100, encoding="utf-8")
    out = tmp_path / "out.bin"
    stats = tok.encode_file_to_bin(src, out)
    assert stats["n_tokens"] > 0
    assert stats["bytes_in"] == len(b"hello world\n") * 100
    assert stats["dtype"] == "uint16"
    assert out.exists()
    meta = tmp_path / "out.meta.json"
    assert meta.exists()


def test_huge_single_chunk(tok: Tokenizer):
    # Big runs get chopped up, never error.
    for big in ["a" * 1_000_000, " " * 1_000_000, "abあ " * 250_000]:
        ids = tok.encode(big)
        assert tok.decode(ids) == big


def test_force_split_threshold_constant():
    from nanomlx.reference_bpe import MAX_SEGMENT_CHARS

    assert MAX_SEGMENT_CHARS == 65_536  # has to match Rust pretokenize.rs
