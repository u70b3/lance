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

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::{RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::BytesMut;
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_encoding::{
    BufferScheduler, EncodingsIo,
    compression::{DecompressionStrategy, DefaultDecompressionStrategy},
    data::{DataBlock, FixedWidthDataBlock},
    decoder::{
        ColumnInfo, DecodeBatchScheduler, DecoderConfig, DecoderPlugins, FilterExpression,
        PageInfo, create_decode_stream, decode_batch,
    },
    encoder::{
        BatchEncoder, EncodedBatch, EncodedPage, EncodingOptions, MIN_PAGE_BUFFER_ALIGNMENT,
        OutOfLineBuffers, default_encoding_strategy, encode_batch,
    },
    encodings::logical::primitive::miniblock::MiniBlockCompressor,
    repdef::RepDefBuilder,
    statistics::ComputeStat,
    version::LanceFileVersion,
};
use rand::Rng;
use tokio::sync::mpsc::unbounded_channel;

const DURATION_SECS: u64 = 30;
const NUM_ROWS_128MB: u64 = 1024 * 1024 * 128 / 4; // 32M for u32/i32
const STREAMING_BATCH_ROWS: u32 = 64 * 1024;
const LAYERED_PROFILE_MAX_PAGE_BYTES: u64 = 256 * 1024;

fn write_page_to_data_buffer(page: EncodedPage, data_buffer: &mut BytesMut) -> PageInfo {
    let buffers = page.data;
    let mut buffer_offsets_and_sizes = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        let buffer_offset = data_buffer.len() as u64;
        data_buffer.extend_from_slice(&buffer);
        let size = data_buffer.len() as u64 - buffer_offset;
        buffer_offsets_and_sizes.push((buffer_offset, size));
    }

    PageInfo {
        buffer_offsets_and_sizes: Arc::from(buffer_offsets_and_sizes.into_boxed_slice()),
        encoding: page.description,
        num_rows: page.num_rows,
        priority: page.row_number,
    }
}

/// Decompress a MiniBlockCompressed by iterating over chunks, matching the real decode path.
/// Does not accumulate into a Vec to avoid memcpy artifacts in profiling.
fn decompress_miniblock(
    compressed: &lance_encoding::encodings::logical::primitive::miniblock::MiniBlockCompressed,
    encoding: &lance_encoding::format::pb21::CompressiveEncoding,
) -> lance_core::Result<u64> {
    let mut offsets = vec![0usize; compressed.data.len()];
    let decompression_strategy = DefaultDecompressionStrategy::default();
    let decompressor =
        decompression_strategy.create_miniblock_decompressor(encoding, &decompression_strategy)?;
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
    let (compressed, encoding) =
        MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

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
    let throughput_gibs =
        (iterations as f64 * NUM_ROWS_128MB as f64 * 4.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!(
        "Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s"
    );
}

fn encode_bitpacking_batch(rt: &tokio::runtime::Runtime) -> EncodedBatch {
    let values = (0..NUM_ROWS_128MB)
        .map(|idx| (idx % 1000) as u32)
        .collect::<Vec<_>>();
    let array = Arc::new(UInt32Array::from(values)) as Arc<dyn arrow_array::Array>;

    let mut metadata = HashMap::new();
    metadata.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );
    metadata.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::UInt32, false).with_metadata(metadata),
    ]));
    let lance_schema =
        Arc::new(lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap());
    let options = EncodingOptions {
        cache_bytes_per_column: LAYERED_PROFILE_MAX_PAGE_BYTES,
        max_page_bytes: LAYERED_PROFILE_MAX_PAGE_BYTES,
        keep_original_array: true,
        buffer_alignment: MIN_PAGE_BUFFER_ALIGNMENT,
        version: LanceFileVersion::V2_2,
    };
    let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);
    let batch_encoder =
        BatchEncoder::try_new(&lance_schema, encoding_strategy.as_ref(), &options).unwrap();
    let top_level_columns = batch_encoder
        .field_id_to_column_index
        .iter()
        .map(|(_, idx)| *idx)
        .collect::<Vec<_>>();
    let mut field_encoder = batch_encoder.field_encoders.into_iter().next().unwrap();

    let mut data_buffer = BytesMut::new();
    let mut external_buffers = OutOfLineBuffers::new(0, MIN_PAGE_BUFFER_ALIGNMENT);
    let chunk_rows = (LAYERED_PROFILE_MAX_PAGE_BYTES as usize / std::mem::size_of::<u32>()).max(1);
    let mut tasks = Vec::new();
    for offset in (0..array.len()).step_by(chunk_rows) {
        let chunk_len = (array.len() - offset).min(chunk_rows);
        let chunk = array.slice(offset, chunk_len);
        tasks.extend(
            field_encoder
                .maybe_encode(
                    chunk,
                    &mut external_buffers,
                    RepDefBuilder::default(),
                    offset as u64,
                    chunk_len as u64,
                )
                .unwrap(),
        );
    }
    tasks.extend(field_encoder.flush(&mut external_buffers).unwrap());
    for buffer in external_buffers.take_buffers() {
        data_buffer.extend_from_slice(&buffer);
    }

    let mut pages_by_column = HashMap::<u32, Vec<PageInfo>>::new();
    for task in tasks {
        let encoded_page = rt.block_on(task).unwrap();
        pages_by_column
            .entry(encoded_page.column_idx)
            .or_default()
            .push(write_page_to_data_buffer(encoded_page, &mut data_buffer));
    }

    let mut final_external_buffers =
        OutOfLineBuffers::new(data_buffer.len() as u64, MIN_PAGE_BUFFER_ALIGNMENT);
    let encoded_columns = rt
        .block_on(field_encoder.finish(&mut final_external_buffers))
        .unwrap();
    for buffer in final_external_buffers.take_buffers() {
        data_buffer.extend_from_slice(&buffer);
    }

    let mut page_table = Vec::new();
    for (column_idx, encoded_column) in encoded_columns.into_iter().enumerate() {
        let mut column_buffers = Vec::new();
        for buffer in encoded_column.column_buffers {
            let buffer_offset = data_buffer.len() as u64;
            data_buffer.extend_from_slice(&buffer);
            let size = data_buffer.len() as u64 - buffer_offset;
            column_buffers.push((buffer_offset, size));
        }
        for page in encoded_column.final_pages {
            pages_by_column
                .entry(page.column_idx)
                .or_default()
                .push(write_page_to_data_buffer(page, &mut data_buffer));
        }
        let column_idx = column_idx as u32;
        let column_pages = std::mem::take(pages_by_column.entry(column_idx).or_default());
        page_table.push(Arc::new(ColumnInfo {
            index: column_idx,
            buffer_offsets_and_sizes: Arc::from(column_buffers.into_boxed_slice()),
            page_infos: Arc::from(column_pages.into_boxed_slice()),
            encoding: encoded_column.encoding,
        }));
    }

    EncodedBatch {
        data: data_buffer.freeze(),
        page_table,
        schema: lance_schema,
        top_level_columns,
        num_rows: NUM_ROWS_128MB,
    }
}

fn make_decode_scheduler(
    rt: &tokio::runtime::Runtime,
    encoded: &EncodedBatch,
    io: Arc<dyn EncodingsIo>,
) -> DecodeBatchScheduler {
    rt.block_on(DecodeBatchScheduler::try_new(
        encoded.schema.as_ref(),
        &encoded.top_level_columns,
        &encoded.page_table,
        &vec![],
        encoded.num_rows,
        Arc::<DecoderPlugins>::default(),
        io,
        Arc::new(LanceCache::no_cache()),
        &FilterExpression::no_filter(),
        &DecoderConfig::default(),
    ))
    .unwrap()
}

fn decode_structural_pages(
    rt: &tokio::runtime::Runtime,
    encoded: &EncodedBatch,
) -> lance_core::Result<(u64, u64, usize)> {
    let io = Arc::new(BufferScheduler::new(encoded.data.clone())) as Arc<dyn EncodingsIo>;
    let mut decode_scheduler = make_decode_scheduler(rt, encoded, io.clone());
    let scheduled = decode_scheduler.schedule_ranges_to_vec(
        &[0..encoded.num_rows],
        &FilterExpression::no_filter(),
        io,
        None,
    )?;

    let mut total_rows = 0u64;
    let mut total_data_size = 0u64;
    let mut total_pages = 0usize;
    for message in scheduled {
        for decoder in message.decoders {
            let unloaded_page = decoder.into_structural();
            let loaded_page = rt.block_on(unloaded_page.0)?;
            let mut decoder = loaded_page.decoder;
            let num_rows = decoder.num_rows();
            let decoded = decoder.drain(num_rows)?.decode()?;
            total_rows += num_rows;
            total_data_size += decoded.data.data_size();
            total_pages += 1;
        }
    }
    Ok((total_rows, total_data_size, total_pages))
}

fn decode_with_batch_size(
    rt: &tokio::runtime::Runtime,
    encoded: &EncodedBatch,
    batch_size: u32,
) -> lance_core::Result<u64> {
    let io = Arc::new(BufferScheduler::new(encoded.data.clone())) as Arc<dyn EncodingsIo>;
    let mut decode_scheduler = make_decode_scheduler(rt, encoded, io.clone());
    let (tx, rx) = unbounded_channel();
    decode_scheduler.schedule_range(0..encoded.num_rows, &FilterExpression::no_filter(), tx, io);

    rt.block_on(async move {
        let mut decode_stream = create_decode_stream(
            &encoded.schema,
            encoded.num_rows,
            batch_size,
            true,
            false,
            true,
            rx,
        )?;
        let mut total_rows = 0u64;
        while let Some(task) = decode_stream.next().await {
            let batch = task.task.await?;
            total_rows += batch.num_rows() as u64;
        }
        Ok(total_rows)
    })
}

fn profile_bytes_layer(name: &str, bytes_per_iteration: u64, mut op: impl FnMut()) {
    let start = Instant::now();
    let mut iterations = 0u64;
    while start.elapsed() < Duration::from_secs(DURATION_SECS) {
        op();
        iterations += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    let throughput_gibs =
        (iterations as f64 * bytes_per_iteration as f64) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!(
        "{name}: iterations={iterations}, elapsed={elapsed:.2}s, throughput={throughput_gibs:.2} GiB/s"
    );
}

fn run_bitpacking_layers() {
    println!("=== Bitpacking Layered Decode ===");
    let rt = tokio::runtime::Runtime::new().unwrap();

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
    let (compressed, encoding) =
        MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

    let encoded = encode_bitpacking_batch(&rt);
    let page_count = encoded
        .page_table
        .first()
        .map(|column| column.page_infos.len())
        .unwrap_or(0);
    let num_bytes = encoded.num_rows * 4;

    println!(
        "rows={}, bytes={}, pages={}, streaming_batch_rows={}",
        encoded.num_rows, num_bytes, page_count, STREAMING_BATCH_ROWS
    );

    let codec_bytes = decompress_miniblock(&compressed, &encoding).unwrap();
    assert_eq!(codec_bytes, num_bytes);

    let (page_rows, page_bytes, decoded_pages) = decode_structural_pages(&rt, &encoded).unwrap();
    assert_eq!(page_rows, encoded.num_rows);
    assert_eq!(page_bytes, num_bytes);
    assert_eq!(decoded_pages, page_count);

    let full_batch_rows = decode_with_batch_size(&rt, &encoded, encoded.num_rows as u32).unwrap();
    assert_eq!(full_batch_rows, encoded.num_rows);

    let streaming_rows = decode_with_batch_size(&rt, &encoded, STREAMING_BATCH_ROWS).unwrap();
    assert_eq!(streaming_rows, encoded.num_rows);

    profile_bytes_layer("codec_only", num_bytes, || {
        let bytes = decompress_miniblock(&compressed, &encoding).unwrap();
        assert_eq!(bytes, num_bytes);
    });
    profile_bytes_layer("page_local_decode", num_bytes, || {
        let (rows, bytes, pages) = decode_structural_pages(&rt, &encoded).unwrap();
        assert_eq!(rows, encoded.num_rows);
        assert_eq!(bytes, num_bytes);
        assert_eq!(pages, page_count);
    });
    profile_bytes_layer("full_batch_decode", num_bytes, || {
        let rows = decode_with_batch_size(&rt, &encoded, encoded.num_rows as u32).unwrap();
        assert_eq!(rows, encoded.num_rows);
    });
    profile_bytes_layer("streaming_batch_decode", num_bytes, || {
        let rows = decode_with_batch_size(&rt, &encoded, STREAMING_BATCH_ROWS).unwrap();
        assert_eq!(rows, encoded.num_rows);
    });
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
    let (compressed, encoding) =
        MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

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
    let throughput_gibs =
        (iterations as f64 * NUM_ROWS_128MB as f64 * 4.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!(
        "Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s"
    );
}

fn run_delta_rle() {
    println!("=== Delta+RLE Decode ===");
    let values: Vec<i64> = (0..NUM_ROWS_128MB)
        .map(|i| 1700000000000i64 + i as i64 * 1000)
        .collect();
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let block = FixedWidthDataBlock {
        data: lance_encoding::buffer::LanceBuffer::from(bytes),
        bits_per_value: 64,
        num_values: NUM_ROWS_128MB,
        block_info: lance_encoding::data::BlockInfo::default(),
    };
    let data_block = DataBlock::FixedWidth(block);

    let compressor = lance_encoding::encodings::physical::delta_rle::DeltaRleEncoder::new();
    let (compressed, encoding) =
        MiniBlockCompressor::compress(&compressor, data_block.clone()).unwrap();

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
    let throughput_gibs =
        (iterations as f64 * NUM_ROWS_128MB as f64 * 8.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    println!(
        "Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_gibs:.2} GiB/s"
    );
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

    let schema = Arc::new(Schema::new(vec![Field::new(
        "string",
        DataType::Utf8,
        false,
    )]));
    let data = RecordBatch::try_new(schema, vec![Arc::new(mapped_strings)]).unwrap();

    let lance_schema =
        Arc::new(lance_core::datatypes::Schema::try_from(data.schema().as_ref()).unwrap());
    let encoding_strategy = default_encoding_strategy(LanceFileVersion::default());
    let encoded = rt
        .block_on(encode_batch(
            &data,
            lance_schema,
            encoding_strategy.as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    println!(
        "Dict encoding: {:?}",
        encoded.page_table[0].page_infos[0].encoding
    );

    // Warm-up
    let _ = rt
        .block_on(decode_batch(
            &encoded,
            &FilterExpression::no_filter(),
            Arc::<DecoderPlugins>::default(),
            false,
            LanceFileVersion::default(),
            Some(Arc::new(LanceCache::no_cache())),
        ))
        .unwrap();

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
    println!(
        "Iterations: {iterations}, Elapsed: {elapsed:.2}s, Throughput: {throughput_melems:.2} Melem/s"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");

    match mode {
        "bitpacking" => run_bitpacking(),
        "bitpacking_layers" => run_bitpacking_layers(),
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
            eprintln!(
                "Unknown mode: {}. Use: bitpacking, bitpacking_layers, rle, delta_rle, dict, all",
                mode
            );
            std::process::exit(1);
        }
    }
}
