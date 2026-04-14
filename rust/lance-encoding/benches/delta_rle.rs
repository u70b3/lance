// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark tests for Delta+RLE encoding

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{DataType, Field, Schema};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use lance_encoding::{
    encoder::{default_encoding_strategy, encode_batch, EncodingOptions},
    version::LanceFileVersion,
    compression::MiniBlockDecompressor,
};

/// Generate monotonic increasing i64 data (timestamps)
fn generate_timestamp_data(n: usize) -> Vec<i64> {
    (0..n as i64).map(|i| 1700000000000i64 + i * 1000).collect()
}

/// Generate monotonic increasing u32 data (auto-increment IDs)
fn generate_id_data(n: usize) -> Vec<u32> {
    (1..n as u32 + 1).collect()
}

/// Generate monotonic decreasing u64 data
fn generate_decreasing_data(n: usize) -> Vec<u64> {
    (0..n as u64).rev().collect()
}

/// Generate u16 data with small deltas (sensor readings)
fn generate_sensor_data(n: usize) -> Vec<u16> {
    (0..n as u16).map(|i| 1000 + i % 100).collect()
}

// ========== 9.3 Compression Benchmarks ==========

fn bench_delta_rle_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_rle_compress");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        // Timestamp data (i64)
        let data = generate_timestamp_data(size);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let block = lance_encoding::data::DataBlock::FixedWidth(lance_encoding::data::FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: size as u64,
            block_info: lance_encoding::data::BlockInfo::default(),
        });

        let encoder = lance_encoding::encodings::physical::delta_rle::DeltaRleEncoder::new();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressor::compress(
                    black_box(&encoder),
                    black_box(block.clone()),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

fn bench_delta_rle_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("delta_rle_decompress");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        // Timestamp data (i64)
        let data = generate_timestamp_data(size);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let block = lance_encoding::data::DataBlock::FixedWidth(lance_encoding::data::FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: size as u64,
            block_info: lance_encoding::data::BlockInfo::default(),
        });

        let encoder = lance_encoding::encodings::physical::delta_rle::DeltaRleEncoder::new();
        let (compressed, _) = lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressor::compress(
            &encoder,
            block,
        )
        .unwrap();

        let decompressor = lance_encoding::encodings::physical::delta_rle::DeltaRleDecompressor::new(64, data[0] as i64);

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                MiniBlockDecompressor::decompress(
                    black_box(&decompressor),
                    black_box(compressed.data.clone()),
                    black_box(size as u64),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

// ========== Comparison with RLE ==========

fn bench_delta_rle_vs_rle_compress(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_comparison");

    let sizes = [1000, 10000, 100000];

    for size in sizes {
        // Timestamp data (i64)
        let data = generate_timestamp_data(size);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Delta+RLE
        let delta_rle_encoder = lance_encoding::encodings::physical::delta_rle::DeltaRleEncoder::new();
        let delta_rle_block = lance_encoding::data::DataBlock::FixedWidth(lance_encoding::data::FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::from(bytes.clone()),
            bits_per_value: 64,
            num_values: size as u64,
            block_info: lance_encoding::data::BlockInfo::default(),
        });

        group.bench_function(format!("delta_rle_{}", size), |b| {
            b.iter(|| {
                lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressor::compress(
                    black_box(&delta_rle_encoder),
                    black_box(delta_rle_block.clone()),
                )
                .unwrap()
            });
        });

        // RLE
        let rle_encoder = lance_encoding::encodings::physical::rle::RleEncoder::new();
        let rle_block = lance_encoding::data::DataBlock::FixedWidth(lance_encoding::data::FixedWidthDataBlock {
            data: lance_encoding::buffer::LanceBuffer::from(bytes),
            bits_per_value: 64,
            num_values: size as u64,
            block_info: lance_encoding::data::BlockInfo::default(),
        });

        group.bench_function(format!("rle_{}", size), |b| {
            b.iter(|| {
                lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressor::compress(
                    black_box(&rle_encoder),
                    black_box(rle_block.clone()),
                )
                .unwrap()
            });
        });
    }

    group.finish();
}

// ========== End-to-end Encoding Benchmark ==========

fn bench_delta_rle_end_to_end(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("delta_rle_end_to_end");

    const NUM_ROWS: usize = 100_000;
    const NUM_COLUMNS: usize = 1;

    // Create timestamp data
    let timestamps: Vec<i64> = generate_timestamp_data(NUM_ROWS);
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::Int64Array::from(timestamps));

    // Enable delta-rle encoding via metadata
    let mut metadata = HashMap::new();
    metadata.insert(
        "lance-encoding:delta-rle".to_string(),
        "true".to_string(),
    );
    // Disable BSS to isolate delta-rle performance
    metadata.insert("lance-encoding:bss".to_string(), "off".to_string());

    let fields = vec![Field::new("timestamp", DataType::Int64, false).with_metadata(metadata)];
    let schema = Arc::new(Schema::new(fields));
    let columns: Vec<Arc<dyn arrow_array::Array>> = vec![array];
    let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

    let lance_schema = Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
    // V2_2+ required for delta-rle
    let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);

    group.throughput(criterion::Throughput::Elements(NUM_ROWS as u64));
    group.bench_function("timestamp_encoding", |b| {
        b.iter(|| {
            rt.block_on(encode_batch(
                black_box(&data),
                black_box(lance_schema.clone()),
                black_box(encoding_strategy.as_ref()),
                &EncodingOptions::default(),
            ))
            .unwrap()
        });
    });

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = bench_delta_rle_compress, bench_delta_rle_decompress, bench_delta_rle_vs_rle_compress, bench_delta_rle_end_to_end
);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_delta_rle_compress, bench_delta_rle_decompress, bench_delta_rle_vs_rle_compress, bench_delta_rle_end_to_end
);

criterion_main!(benches);
