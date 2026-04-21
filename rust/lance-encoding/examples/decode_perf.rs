// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Standalone decode performance profiling binary.
//!
//! Run with perf:
//! ```bash
//! perf record -g --call-graph=dwarf -- target/release-with-debug/examples/decode_perf <mode>
//! ```
//!
//! Modes: bitpacking, rle, dict, delta_rle, all

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use lance_core::cache::LanceCache;
use lance_encoding::{
    compression::{DefaultDecompressionStrategy, DecompressionStrategy},
    data::{DataBlock, FixedWidthDataBlock},
    decoder::{decode_batch, DecoderPlugins, FilterExpression},
    encoder::{default_encoding_strategy, encode_batch, EncodingOptions},
    encodings::logical::primitive::miniblock::MiniBlockCompressor,
    statistics::ComputeStat,
    version::LanceFileVersion,
};
use rand::Rng;

const DURATION_SECS: u64 = 30;
const NUM_ROWS_128MB: u64 = 1024 * 1024 * 128 / 4; // 32M for u32/i32

/// Decompress a MiniBlockCompressed by iterating over chunks, matching the real decode path.
/// Does not accumulate into a Vec to avoid memcpy artifacts in profiling.
fn decompress_miniblock(
    compressed: &lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressed,
    encoding: &lance_encoding::format::pb21::CompressiveEncoding,
) -> lance_core::Result<u64> {
    let mut offsets = vec![0usize; compressed.data.len()];
    let decompression_strategy = DefaultDecompressionStrategy::default();
    let decompressor = decompression_strategy
        .create_miniblock_decompressor(encoding, &decompression_strategy)?;
    let mut total_bytes = 0u64;

    for chunk in &compressed.chunks {
        let chunk_values = if chunk.log_num_values > 0 {
            1u64 << chunk.log_num_values
        } else {
            compressed.num_values - total_bytes / 4
        };

        let mut chunk_buffers = Vec::new();
        for (i, &size) in chunk.buffer_sizes.iter().enumerate() {
            if i < compressed.data.len() {
                let buffer_data = compressed.data[i].slice_with_length(offsets[i], size as usize);
                chunk_buffers.push(buffer_data);
                offsets[i] += size as usize;
            }
        }

        let chunk_decompressed = decompressor.decompress(chunk_buffers, chunk_values)?;
        total_bytes += chunk_decompressed.data_size();
    }

    Ok(total_bytes)
}

fn run_bitpacking() {
    println!("=== Bitpacking Decode ===");
    let values: Vec<u32> = (0..NUM_ROWS_128MB).map(|i| (i % 1000) as u32).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut block = FixedWidthDataBlock {
        data: lance_encoding::buffer::LanceBuffer::from(bytes),
        bits_per_value: 32,
        num_values: NUM_ROWS_128MB,
        block_info: lance_encoding::data::BlockInfo::default(),
    };
    block.compute_stat();
    let data_block = DataBlock::FixedWidth(block);

    let compressor = lance_encoding::encodings::physical::bitpacking::InlineBitpacking::new(32);
    let (compressed, encoding) = MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

    // Warm-up + verify
    let bytes_decompressed = decompress_miniblock(&compressed, &encoding).unwrap();
    assert_eq!(bytes_decompressed, NUM_ROWS_128MB * 4);

    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        let _ = decompress_miniblock(&compressed, &encoding).unwrap();
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gibs = (iterations as f64 * NUM_ROWS_128MB as f64 * 4.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!("Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s");
}

fn run_rle() {
    println!("=== RLE Decode ===");
    let half = NUM_ROWS_128MB / 2;
    let values: Vec<i32> = (0..NUM_ROWS_128MB)
        .map(|i| if i < half { 0i32 } else { 1i32 })
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut block = FixedWidthDataBlock {
        data: lance_encoding::buffer::LanceBuffer::from(bytes),
        bits_per_value: 32,
        num_values: NUM_ROWS_128MB,
        block_info: lance_encoding::data::BlockInfo::default(),
    };
    block.compute_stat();
    let data_block = DataBlock::FixedWidth(block);

    let compressor = lance_encoding::encodings::physical::rle::RleEncoder::new();
    let (compressed, encoding) = MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

    // Warm-up + verify
    let bytes_decompressed = decompress_miniblock(&compressed, &encoding).unwrap();
    assert_eq!(bytes_decompressed, NUM_ROWS_128MB * 4);

    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        let _ = decompress_miniblock(&compressed, &encoding).unwrap();
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gibs = (iterations as f64 * NUM_ROWS_128MB as f64 * 4.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!("Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s");
}

fn run_delta_rle() {
    println!("=== Delta+RLE Decode ===");
    let values: Vec<i64> = (0..NUM_ROWS_128MB).map(|i| 1700000000000i64 + i as i64 * 1000).collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let block = FixedWidthDataBlock {
        data: lance_encoding::buffer::LanceBuffer::from(bytes),
        bits_per_value: 64,
        num_values: NUM_ROWS_128MB,
        block_info: lance_encoding::data::BlockInfo::default(),
    };
    let data_block = DataBlock::FixedWidth(block);

    let compressor = lance_encoding::encodings::physical::delta_rle::DeltaRleEncoder::new();
    let (compressed, encoding) = MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

    // Warm-up + verify
    let bytes_decompressed = decompress_miniblock(&compressed, &encoding).unwrap();
    assert_eq!(bytes_decompressed, NUM_ROWS_128MB * 8);

    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        let _ = decompress_miniblock(&compressed, &encoding).unwrap();
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gibs = (iterations as f64 * NUM_ROWS_128MB as f64 * 8.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!("Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s");
}

fn run_dict() {
    println!("=== Dictionary Decode ===");
    let rt = tokio::runtime::Runtime::new().unwrap();
    const NUM_ROWS: usize = 5_000_000;

    // Generate string column with 20 unique values
    let string_data = lance_datagen::gen_batch()
        .anon_col(lance_datagen::array::rand_type(&DataType::Utf8))
        .into_batch_rows(lance_datagen::RowCount::from(20))
        .unwrap();
    let string_array = string_data.column(0);

    // Generate random indices pointing to the 20 values
    let mut rng = rand::rng();
    let integer_arr: Vec<u32> = (0..NUM_ROWS).map(|_| rng.random_range(0..20)).collect();
    let integer_array = UInt32Array::from(integer_arr);
    let mapped_strings = arrow_select::take::take(string_array, &integer_array, None).unwrap();

    let schema = Arc::new(Schema::new(vec![Field::new("string", DataType::Utf8, false)]));
    let data = RecordBatch::try_new(schema, vec![Arc::new(mapped_strings)]).unwrap();

    let lance_schema = Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
    let encoding_strategy = default_encoding_strategy(LanceFileVersion::default());
    let encoded = rt
        .block_on(encode_batch(
            &data,
            lance_schema,
            encoding_strategy.as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    println!("Dict encoding: {:?}", encoded.page_table[0].page_infos[0].encoding);

    // Warm-up
    let _ = rt.block_on(decode_batch(
        &encoded,
        &FilterExpression::no_filter(),
        Arc::<DecoderPlugins>::default(),
        false,
        LanceFileVersion::default(),
        Some(Arc::new(LanceCache::no_cache())),
    )).unwrap();

    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        let batch = rt
            .block_on(decode_batch(
                &encoded,
                &FilterExpression::no_filter(),
                Arc::<DecoderPlugins>::default(),
                false,
                LanceFileVersion::default(),
                Some(Arc::new(LanceCache::no_cache())),
            ))
            .unwrap();
        assert_eq!(data.num_rows(), batch.num_rows());
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_melems = (iterations as f64 * NUM_ROWS as f64) / (elapsed * 1_000_000.0);
    println!("Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_melems:.2} Melem/s");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match mode {
        "bitpacking" => run_bitpacking(),
        "rle" => run_rle(),
        "delta_rle" => run_delta_rle(),
        "dict" => run_dict(),
        "all" => {
            run_bitpacking();
            run_rle();
            run_delta_rle();
            run_dict();
        }
        _ => {
            eprintln!("Unknown mode: {}. Use: bitpacking, rle, delta_rle, dict, all", mode);
            std::process::exit(1);
        }
    }
}
