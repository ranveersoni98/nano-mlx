//! Fast (linked-list + heap) vs naive rescan, single vs rayon batch. Run with:
//! `cargo bench --manifest-path tokenizer/Cargo.toml` (release only,
//! debug builds are 10-100x slower so don't quote those numbers).

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use tokenizer_core::encode::{encode_batch, encode_chunk_fast, encode_chunk_naive};
use tokenizer_core::Tokenizer;

fn trained_tok() -> Tokenizer {
    // A real small vocab: train-like merges over English bytes so the bench
    // exercises multi-step merges, not just single bytes.
    let merges: Vec<(u32, u32)> = vec![
        (116, 104), // 256: t h -> "th"
        (256, 101), // 257: th e -> "the"
        (105, 110), // 258: i n -> "in"
        (258, 103), // 259: in g -> "ing"
        (32, 257),  // 260: " the"
        (101, 114), // 261: e r -> "er"
        (97, 110),  // 262: a n -> "an"
        (262, 100), // 263: an d -> "and"
        (111, 110), // 264: o n -> "on"
        (99, 111),  // 265: c o -> "co"
        (265, 109), // 266: co m -> "com"
        (115, 116), // 267: s t -> "st"
        (267, 114), // 268: st r -> "str"
        (268, 105), // 269: str i -> "stri"
        (269, 110), // 270: stri n -> "strin"
        (270, 103), // 271: strin g -> "string"
        (121, 111), // 272: y o -> "yo"
        (272, 117), // 273: yo u -> "you"
        (119, 105), // 274: w i -> "wi"
        (274, 116), // 275: wi t -> "wit"
        (275, 104), // 276: wit h -> "with"
    ];
    Tokenizer::from_merges(merges, vec!["<|endoftext|>".into()]).expect("bench vocab")
}

fn bench_chunks(c: &mut Criterion) {
    let tok = trained_tok();
    let chunk = b"the string with you and the other thing in string theory";
    let mut group = c.benchmark_group("chunk_bpe");
    group.throughput(Throughput::Bytes(chunk.len() as u64));
    group.bench_function("naive_scan", |b| {
        b.iter(|| encode_chunk_naive(black_box(chunk), black_box(&tok)))
    });
    group.bench_function("linked_list_heap", |b| {
        b.iter(|| encode_chunk_fast(black_box(chunk), black_box(&tok)))
    });
    group.finish();
}

fn bench_batch(c: &mut Criterion) {
    let tok = trained_tok();
    let texts: Vec<String> = (0..512)
        .map(|i| format!("the string {i} with you and theory of things"))
        .collect();
    let mut group = c.benchmark_group("encode_batch");
    group.throughput(Throughput::Elements(texts.len() as u64));
    group.bench_function("rayon_parallel", |b| {
        b.iter(|| encode_batch(black_box(&tok), black_box(&texts), black_box(&[])))
    });
    group.finish();
}

criterion_group!(benches, bench_chunks, bench_batch);
criterion_main!(benches);
