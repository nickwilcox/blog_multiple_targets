use blog_multiple_targets::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::default::Default;

const BENCHMARK_SAMPLES: usize = 100_000;

fn prepare_buffer() -> (DeinterleavedBuffer71, InterleavedBuffer71) {
    (
        DeinterleavedBuffer71::empty(BENCHMARK_SAMPLES),
        InterleavedBuffer71::empty(BENCHMARK_SAMPLES),
    )
}

fn criterion_benchmark(c: &mut Criterion) {
    let (src, mut dst) = prepare_buffer();
    c.bench_function("scalar", move |b| {
        b.iter(|| interleave_71_inner(&src, &mut dst))
    });
    let (src, mut dst) = prepare_buffer();
    c.bench_function("avx", move |b| {
        b.iter(|| unsafe { interleave_71_avx(&src, &mut dst) })
    });
    let (src, mut dst) = prepare_buffer();
    c.bench_function("avx2", move |b| {
        b.iter(|| unsafe { interleave_71_avx2(&src, &mut dst) })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
