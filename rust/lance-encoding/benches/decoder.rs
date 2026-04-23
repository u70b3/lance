// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
use std::{collections::HashMap, sync::Arc};

use arrow_array::{ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use arrow_select::take::take;
use bytes::BytesMut;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_datagen::ArrayGeneratorExt;
use lance_encoding::{
    BufferScheduler, EncodingsIo,
    decoder::{
        ColumnInfo, DecodeBatchScheduler, DecoderConfig, DecoderPlugins, FilterExpression,
        PageInfo, create_decode_stream,
    },
    encoder::{
        BatchEncoder, EncodedBatch, EncodedPage, EncodingOptions, MIN_PAGE_BUFFER_ALIGNMENT,
        OutOfLineBuffers, default_encoding_strategy, encode_batch,
    },
    repdef::RepDefBuilder,
    version::LanceFileVersion,
};
use tokio::sync::mpsc::unbounded_channel;

use rand::Rng;

const PRIMITIVE_TYPES: &[DataType] = &[
    DataType::Date32,
    DataType::Date64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::UInt8,
    DataType::UInt16,
    DataType::UInt32,
    DataType::UInt64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Decimal128(10, 10),
    DataType::Decimal256(10, 10),
    DataType::Timestamp(TimeUnit::Nanosecond, None),
    DataType::Time32(TimeUnit::Second),
    DataType::Time64(TimeUnit::Nanosecond),
    DataType::Duration(TimeUnit::Second),
    // The Interval type is supported by the reader but the writer works with Lance schema
    // at the moment and Lance schema can't parse interval
    // DataType::Interval(IntervalUnit::DayTime),
];

// Some types are supported by the encoder/decoder but Lance
// schema doesn't yet parse them in the context of a fixed size list.
const PRIMITIVE_TYPES_FOR_FSL: &[DataType] = &[DataType::Int8, DataType::Float32];
const LAYERED_BENCH_MAX_PAGE_BYTES: u64 = 256 * 1024;

struct UInt32DecodeCase {
    name: &'static str,
    encoded: Arc<EncodedBatch>,
    num_rows: u64,
    num_bytes: u64,
    page_count: usize,
}

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

fn encode_uint32_case(
    rt: &tokio::runtime::Runtime,
    name: &'static str,
    metadata: HashMap<String, String>,
    array: ArrayRef,
    version: LanceFileVersion,
) -> UInt32DecodeCase {
    let arrow_schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::UInt32, false).with_metadata(metadata),
    ]));
    let num_rows = array.len() as u64;
    let num_bytes = num_rows * std::mem::size_of::<u32>() as u64;
    let lance_schema =
        Arc::new(lance_core::datatypes::Schema::try_from(arrow_schema.as_ref()).unwrap());
    let options = EncodingOptions {
        cache_bytes_per_column: LAYERED_BENCH_MAX_PAGE_BYTES,
        max_page_bytes: LAYERED_BENCH_MAX_PAGE_BYTES,
        keep_original_array: true,
        buffer_alignment: MIN_PAGE_BUFFER_ALIGNMENT,
        version,
    };
    let encoding_strategy = default_encoding_strategy(version);
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
    let chunk_rows = (LAYERED_BENCH_MAX_PAGE_BYTES as usize / std::mem::size_of::<u32>()).max(1);
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

    let encoded = EncodedBatch {
        data: data_buffer.freeze(),
        page_table,
        schema: lance_schema,
        top_level_columns,
        num_rows,
    };
    let page_count = encoded
        .page_table
        .first()
        .map(|column| column.page_infos.len())
        .unwrap_or(0);

    UInt32DecodeCase {
        name,
        encoded: Arc::new(encoded),
        num_rows,
        num_bytes,
        page_count,
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
    version: LanceFileVersion,
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
            version >= LanceFileVersion::V2_1,
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

fn bench_decode(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));
    for data_type in PRIMITIVE_TYPES {
        let func_name = format!("{:?}", data_type).to_lowercase();
        let num_rows = NUM_BYTES / data_type.primitive_width().unwrap() as u64;
        group.bench_function(func_name, |b| {
            let data = lance_datagen::gen_batch()
                .anon_col(lance_datagen::array::rand_type(data_type))
                .into_batch_rows(lance_datagen::RowCount::from(num_rows))
                .unwrap();
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

            b.iter(|| {
                for _ in 0..10 {
                    let batch = rt
                        .block_on(lance_encoding::decoder::decode_batch(
                            &encoded,
                            &FilterExpression::no_filter(),
                            Arc::<DecoderPlugins>::default(),
                            false,
                            LanceFileVersion::default(),
                            Some(Arc::new(LanceCache::no_cache())),
                        ))
                        .unwrap();
                    assert_eq!(data.num_rows(), batch.num_rows());
                }
            })
        });
    }
}

fn bench_decode_fsl(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_fsl");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    for version in [
        LanceFileVersion::V2_0,
        LanceFileVersion::V2_1,
        LanceFileVersion::V2_2,
    ] {
        for data_type in PRIMITIVE_TYPES_FOR_FSL {
            for dimension in [4, 16, 32, 64, 128] {
                let nullable_choices: &[bool] = if version == LanceFileVersion::V2_0 {
                    &[false]
                } else {
                    &[false, true]
                };
                for nullable in nullable_choices {
                    let func_name = format!(
                        "{:?}_{}_v{}_null{}",
                        data_type, dimension, version, nullable
                    )
                    .to_lowercase();
                    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));
                    group.bench_function(func_name, |b| {
                        let num_rows =
                            NUM_BYTES / (dimension * data_type.primitive_width().unwrap() as u64);
                        let mut arraygen =
                            lance_datagen::array::rand_type(&DataType::FixedSizeList(
                                Arc::new(Field::new("item", data_type.clone(), true)),
                                dimension as i32,
                            ));
                        if *nullable {
                            arraygen = arraygen.with_random_nulls(0.5);
                        }
                        let data = lance_datagen::gen_batch()
                            .anon_col(arraygen)
                            .into_batch_rows(lance_datagen::RowCount::from(num_rows))
                            .unwrap();
                        let lance_schema = Arc::new(
                            lance_core::datatypes::Schema::try_from(data.schema().as_ref())
                                .unwrap(),
                        );
                        let encoding_strategy = default_encoding_strategy(version);
                        let encoded = rt
                            .block_on(encode_batch(
                                &data,
                                lance_schema,
                                encoding_strategy.as_ref(),
                                &EncodingOptions::default(),
                            ))
                            .unwrap();
                        b.iter(|| {
                            for _ in 0..10 {
                                let batch = rt
                                    .block_on(lance_encoding::decoder::decode_batch(
                                        &encoded,
                                        &FilterExpression::no_filter(),
                                        Arc::<DecoderPlugins>::default(),
                                        false,
                                        version,
                                        Some(Arc::new(LanceCache::no_cache())),
                                    ))
                                    .unwrap();
                                assert_eq!(data.num_rows(), batch.num_rows());
                            }
                        })
                    });
                }
            }
        }
    }
}

fn bench_decode_str_with_dict_encoding(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");
    const NUM_ROWS: u64 = 100000;

    let data_type = DataType::Utf8;
    // generate string column with 20 rows
    let string_data = lance_datagen::gen_batch()
        .anon_col(lance_datagen::array::rand_type(&DataType::Utf8))
        .into_batch_rows(lance_datagen::RowCount::from(20))
        .unwrap();

    group.throughput(criterion::Throughput::Bytes(
        NUM_ROWS * std::mem::size_of::<u32>() as u64 + string_data.get_array_memory_size() as u64,
    ));

    let func_name = format!("{:?}", data_type).to_lowercase();
    group.bench_function(func_name, |b| {
        let string_array = string_data.column(0);

        // generate random int column with 100000 rows
        let mut rng = rand::rng();
        let integer_arr: Vec<u32> = (0..100_000).map(|_| rng.random_range(0..20)).collect();
        let integer_array = UInt32Array::from(integer_arr);

        let mapped_strings = take(string_array, &integer_array, None).unwrap();

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

        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::default(),
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            }
        })
    });
}

fn bench_decode_packed_struct(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");

    const NUM_ROWS: u64 = 10000;
    let size_bytes =
        ((6 * std::mem::size_of::<i32>() as u64) + std::mem::size_of::<f32>() as u64) * NUM_ROWS;
    group.throughput(criterion::Throughput::Bytes(size_bytes));

    let func_name = "struct";
    group.bench_function(func_name, |b| {
        let fields = vec![
            Arc::new(Field::new("int_field", DataType::Int32, false)),
            Arc::new(Field::new("float_field", DataType::Float32, false)),
            Arc::new(Field::new(
                "fsl_field",
                DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 5),
                false,
            )),
        ]
        .into();

        // generate struct column with 1M rows
        let data = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_type(&DataType::Struct(fields)))
            .into_batch_rows(lance_datagen::RowCount::from(NUM_ROWS))
            .unwrap();

        let schema = data.schema();
        let new_fields: Vec<Arc<Field>> = schema
            .fields()
            .iter()
            .map(|field| {
                if matches!(field.data_type(), &DataType::Struct(_)) {
                    let mut metadata = HashMap::new();
                    metadata.insert("packed".to_string(), "true".to_string());
                    let field =
                        Field::new(field.name(), field.data_type().clone(), field.is_nullable());
                    Arc::new(field.with_metadata(metadata))
                } else {
                    field.clone()
                }
            })
            .collect();

        let new_schema = Schema::new(new_fields);
        let data =
            RecordBatch::try_new(Arc::new(new_schema.clone()), data.columns().to_vec()).unwrap();

        let lance_schema = Arc::new(lance_core::datatypes::Schema::try_from(&new_schema).unwrap());
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::default());
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::default(),
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            }
        })
    });
}

#[cfg(target_os = "linux")]
fn bench_decode_str_with_fixed_size_binary_encoding(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_primitive");

    const NUM_ROWS: u64 = 10000;
    // Randomly generated strings are always 12 characters (at the moment)
    // Plus we need 4 bytes for the offset
    const NUM_BYTES: u64 = NUM_ROWS * 16;
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    let func_name = "fixed-utf8".to_string();
    group.bench_function(func_name, |b| {
        // generate string column with 10k rows
        // Currently the generator generates fixed size strings by default
        // This function will need to be updated once that changes.
        let string_data = lance_datagen::gen_batch()
            .anon_col(lance_datagen::array::rand_type(&DataType::Utf8))
            .into_batch_rows(lance_datagen::RowCount::from(10000))
            .unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new(
            "string",
            DataType::Utf8,
            false,
        )]));

        let data = RecordBatch::try_new(schema, string_data.columns().to_vec()).unwrap();

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
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::default(),
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data.num_rows(), batch.num_rows());
            }
        })
    });
}

fn bench_decode_compressed(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_compressed");

    const NUM_ROWS: usize = 5_000_000;
    const NUM_COLUMNS: usize = 10;

    // Generate compressible string data - high cardinality but compressible
    // (unique values to avoid dictionary encoding, repeated prefix for compression)
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        // Disable dictionary encoding to ensure we hit the compression path
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        // Force miniblock encoding (the path that benefits from compressor caching)
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        // V2_2+ required for general compression
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);

        // Encode once during setup
        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        group.throughput(criterion::Throughput::Elements(
            (NUM_ROWS * NUM_COLUMNS) as u64,
        ));
        group.bench_function(
            format!("{}_strings_{}cols", compression, NUM_COLUMNS),
            |b| {
                b.iter(|| {
                    // Decode 10 times to dominate wall-clock time for perf profiling
                    for _ in 0..10 {
                        let batch = rt
                            .block_on(lance_encoding::decoder::decode_batch(
                                &encoded,
                                &FilterExpression::no_filter(),
                                Arc::<DecoderPlugins>::default(),
                                false,
                                LanceFileVersion::V2_2,
                                Some(Arc::new(LanceCache::no_cache())),
                            ))
                            .unwrap();
                        assert_eq!(data.num_rows(), batch.num_rows());
                    }
                })
            },
        );
    }
}

/// Benchmark parallel decoding with multiple concurrent batch decode tasks.
/// This creates contention on the shared decompressor mutex when multiple
/// batches from the same page are decoded in parallel.
fn bench_decode_compressed_parallel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_compressed_parallel");

    const NUM_ROWS: u64 = 1_000_000;
    const NUM_COLUMNS: usize = 10;
    // Small batch size to create many batches that will contend on the same decompressor
    const BATCH_SIZE: u32 = 100_000;

    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS as usize).map(|i| format!("prefix_that_compresses_well_{}", i)),
    ));

    for compression in ["zstd", "lz4"] {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lance-encoding:compression".to_string(),
            compression.to_string(),
        );
        metadata.insert(
            "lance-encoding:dict-divisor".to_string(),
            "100000".to_string(),
        );
        metadata.insert(
            "lance-encoding:structural-encoding".to_string(),
            "miniblock".to_string(),
        );
        let fields: Vec<Field> = (0..NUM_COLUMNS)
            .map(|i| {
                Field::new(format!("s{}", i), DataType::Utf8, false).with_metadata(metadata.clone())
            })
            .collect();
        let columns: Vec<Arc<dyn arrow_array::Array>> =
            (0..NUM_COLUMNS).map(|_| array.clone()).collect();
        let schema = Arc::new(Schema::new(fields));
        let data = RecordBatch::try_new(schema.clone(), columns).unwrap();

        let lance_schema =
            Arc::new(lance_core::datatypes::Schema::try_from(schema.as_ref()).unwrap());
        let encoding_strategy = default_encoding_strategy(LanceFileVersion::V2_2);

        let encoded = rt
            .block_on(encode_batch(
                &data,
                lance_schema,
                encoding_strategy.as_ref(),
                &EncodingOptions::default(),
            ))
            .unwrap();

        let encoded = Arc::new(encoded);

        // Test with different parallelism levels to see impact of mutex contention
        // parallelism=1 is sequential (no contention), higher values cause contention
        for parallelism in [1, 8] {
            group.throughput(criterion::Throughput::Elements(
                NUM_ROWS * NUM_COLUMNS as u64,
            ));
            group.bench_function(
                format!(
                    "{}_{}cols_parallel_{}",
                    compression, NUM_COLUMNS, parallelism
                ),
                |b| {
                    b.iter(|| {
                        rt.block_on(async {
                            let io_scheduler = Arc::new(lance_encoding::BufferScheduler::new(
                                encoded.data.clone(),
                            ))
                                as Arc<dyn lance_encoding::EncodingsIo>;
                            let cache = Arc::new(LanceCache::no_cache());
                            let filter = FilterExpression::no_filter();

                            let mut decode_scheduler = DecodeBatchScheduler::try_new(
                                encoded.schema.as_ref(),
                                &encoded.top_level_columns,
                                &encoded.page_table,
                                &vec![],
                                encoded.num_rows,
                                Arc::<DecoderPlugins>::default(),
                                io_scheduler.clone(),
                                cache,
                                &filter,
                                &DecoderConfig::default(),
                            )
                            .await
                            .unwrap();

                            let (tx, rx) = unbounded_channel();
                            decode_scheduler.schedule_range(
                                0..encoded.num_rows,
                                &filter,
                                tx,
                                io_scheduler,
                            );

                            let decode_stream = create_decode_stream(
                                &encoded.schema,
                                encoded.num_rows,
                                BATCH_SIZE,
                                true, // is_structural for V2_2
                                false,
                                false,
                                rx,
                            )
                            .unwrap();

                            // Buffer multiple batch decodes in parallel - this causes contention
                            let batches: Vec<_> = decode_stream
                                .map(|task| task.task)
                                .buffered(parallelism)
                                .collect()
                                .await;

                            let total_rows: usize =
                                batches.iter().map(|b| b.as_ref().unwrap().num_rows()).sum();
                            assert_eq!(total_rows, NUM_ROWS as usize);
                        })
                    })
                },
            );
        }
    }
}

fn bench_decode_bitpacking(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_bitpacking");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    const NUM_ROWS: u64 = NUM_BYTES / 4; // UInt32
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    // Generate small-range UInt32 data to trigger bitpacking
    let values: Vec<u32> = (0..NUM_ROWS).map(|i| (i % 1000) as u32).collect();
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::UInt32Array::from(values));

    // Bitpacking path: disable RLE and BSS to isolate bitpacking
    // Do NOT set compression=none, or build_fixed_width_compressor will skip
    // all encodings (including bitpacking) and return ValueEncoder directly.
    let mut metadata_bp = HashMap::new();
    metadata_bp.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );
    metadata_bp.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_bp.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let fields_bp = vec![Field::new("v", DataType::UInt32, false).with_metadata(metadata_bp)];
    let schema_bp = Arc::new(Schema::new(fields_bp));
    let data_bp = RecordBatch::try_new(schema_bp.clone(), vec![array.clone()]).unwrap();

    let lance_schema_bp =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_bp.as_ref()).unwrap());
    let encoded_bp = rt
        .block_on(encode_batch(
            &data_bp,
            lance_schema_bp,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("bitpacking_uint32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_bp,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_bp.num_rows(), batch.num_rows());
            }
        })
    });

    // Flat/Value baseline: same data but force no compression/encoding
    let mut metadata_flat = HashMap::new();
    metadata_flat.insert("lance-encoding:compression".to_string(), "none".to_string());
    metadata_flat.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_flat.insert("lance-encoding:delta-rle".to_string(), "false".to_string());
    metadata_flat.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );

    let fields_flat = vec![Field::new("v", DataType::UInt32, false).with_metadata(metadata_flat)];
    let schema_flat = Arc::new(Schema::new(fields_flat));
    let data_flat = RecordBatch::try_new(schema_flat.clone(), vec![array.clone()]).unwrap();

    let lance_schema_flat =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_flat.as_ref()).unwrap());
    let encoded_flat = rt
        .block_on(encode_batch(
            &data_flat,
            lance_schema_flat,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("flat_uint32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_flat,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_flat.num_rows(), batch.num_rows());
            }
        })
    });

    group.finish();
}

fn bench_decode_bitpacking_layers(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_bitpacking_layers");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    const NUM_ROWS: u64 = NUM_BYTES / 4;
    const STREAMING_BATCH_ROWS: u32 = 64 * 1024;
    let values: Vec<u32> = (0..NUM_ROWS).map(|i| (i % 1000) as u32).collect();
    let array: ArrayRef = Arc::new(UInt32Array::from(values));

    let mut metadata_bp = HashMap::new();
    metadata_bp.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );
    metadata_bp.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_bp.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let mut metadata_flat = HashMap::new();
    metadata_flat.insert("lance-encoding:compression".to_string(), "none".to_string());
    metadata_flat.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_flat.insert("lance-encoding:delta-rle".to_string(), "false".to_string());
    metadata_flat.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );

    let cases = [
        encode_uint32_case(
            &rt,
            "bitpacking_uint32",
            metadata_bp,
            array.clone(),
            LanceFileVersion::V2_2,
        ),
        encode_uint32_case(
            &rt,
            "flat_uint32",
            metadata_flat,
            array,
            LanceFileVersion::V2_2,
        ),
    ];

    for case in &cases {
        group.throughput(Throughput::Bytes(case.num_bytes));
        let case_label = format!("{}_pages{}", case.name, case.page_count);

        group.bench_with_input(
            BenchmarkId::new("page_local", &case_label),
            case,
            |b, case| {
                b.iter(|| {
                    let (decoded_rows, decoded_bytes, decoded_pages) =
                        decode_structural_pages(&rt, case.encoded.as_ref()).unwrap();
                    assert_eq!(decoded_rows, case.num_rows);
                    assert_eq!(decoded_bytes, case.num_bytes);
                    assert_eq!(decoded_pages, case.page_count);
                    black_box((decoded_rows, decoded_bytes, decoded_pages));
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_batch", &case_label),
            case,
            |b, case| {
                b.iter(|| {
                    let decoded_rows = decode_with_batch_size(
                        &rt,
                        case.encoded.as_ref(),
                        LanceFileVersion::V2_2,
                        case.num_rows as u32,
                    )
                    .unwrap();
                    assert_eq!(decoded_rows, case.num_rows);
                    black_box(decoded_rows);
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new(
                "streaming_batch_64k",
                format!("{case_label}_b{STREAMING_BATCH_ROWS}"),
            ),
            case,
            |b, case| {
                b.iter(|| {
                    let decoded_rows = decode_with_batch_size(
                        &rt,
                        case.encoded.as_ref(),
                        LanceFileVersion::V2_2,
                        STREAMING_BATCH_ROWS,
                    )
                    .unwrap();
                    assert_eq!(decoded_rows, case.num_rows);
                    black_box(decoded_rows);
                })
            },
        );
    }

    group.finish();
}

fn bench_decode_bss(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_bss");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    const NUM_ROWS: u64 = NUM_BYTES / 4; // Float32
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    let values: Vec<f32> = (0..NUM_ROWS).map(|i| (i % 10000) as f32).collect();
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::Float32Array::from(values));

    // BSS on (BSS requires general compression to be effective)
    let mut metadata_bss = HashMap::new();
    metadata_bss.insert("lance-encoding:bss".to_string(), "on".to_string());
    metadata_bss.insert("lance-encoding:compression".to_string(), "zstd".to_string());
    metadata_bss.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let fields_bss = vec![Field::new("v", DataType::Float32, false).with_metadata(metadata_bss)];
    let schema_bss = Arc::new(Schema::new(fields_bss));
    let data_bss = RecordBatch::try_new(schema_bss.clone(), vec![array.clone()]).unwrap();

    let lance_schema_bss =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_bss.as_ref()).unwrap());
    let encoded_bss = rt
        .block_on(encode_batch(
            &data_bss,
            lance_schema_bss,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("bss_float32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_bss,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_bss.num_rows(), batch.num_rows());
            }
        })
    });

    // BSS off baseline (same compression, no BSS)
    let mut metadata_no_bss = HashMap::new();
    metadata_no_bss.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_no_bss.insert("lance-encoding:compression".to_string(), "zstd".to_string());
    metadata_no_bss.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let fields_no_bss =
        vec![Field::new("v", DataType::Float32, false).with_metadata(metadata_no_bss)];
    let schema_no_bss = Arc::new(Schema::new(fields_no_bss));
    let data_no_bss = RecordBatch::try_new(schema_no_bss.clone(), vec![array.clone()]).unwrap();

    let lance_schema_no_bss =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_no_bss.as_ref()).unwrap());
    let encoded_no_bss = rt
        .block_on(encode_batch(
            &data_no_bss,
            lance_schema_no_bss,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("no_bss_float32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_no_bss,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_no_bss.num_rows(), batch.num_rows());
            }
        })
    });

    group.finish();
}

fn bench_decode_fsst(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_fsst");
    const NUM_ROWS: usize = 5_000_000; // ~24 bytes per string = ~120MB
    const NUM_BYTES: u64 = NUM_ROWS as u64 * 24;
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    // High-prefix-repeat strings ideal for FSST
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::StringArray::from_iter_values(
        (0..NUM_ROWS).map(|i| format!("category_{}_item_{:08}", i % 10, i)),
    ));

    // FSST path
    let mut metadata_fsst = HashMap::new();
    metadata_fsst.insert("lance-encoding:compression".to_string(), "fsst".to_string());
    metadata_fsst.insert(
        "lance-encoding:dict-divisor".to_string(),
        "100000".to_string(),
    );
    metadata_fsst.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );

    let fields_fsst = vec![Field::new("s", DataType::Utf8, false).with_metadata(metadata_fsst)];
    let schema_fsst = Arc::new(Schema::new(fields_fsst));
    let data_fsst = RecordBatch::try_new(schema_fsst.clone(), vec![array.clone()]).unwrap();

    let lance_schema_fsst =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_fsst.as_ref()).unwrap());
    let encoded_fsst = rt
        .block_on(encode_batch(
            &data_fsst,
            lance_schema_fsst,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("fsst_utf8", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_fsst,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_fsst.num_rows(), batch.num_rows());
            }
        })
    });

    // zstd baseline
    let mut metadata_zstd = HashMap::new();
    metadata_zstd.insert("lance-encoding:compression".to_string(), "zstd".to_string());
    metadata_zstd.insert(
        "lance-encoding:dict-divisor".to_string(),
        "100000".to_string(),
    );
    metadata_zstd.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );

    let fields_zstd = vec![Field::new("s", DataType::Utf8, false).with_metadata(metadata_zstd)];
    let schema_zstd = Arc::new(Schema::new(fields_zstd));
    let data_zstd = RecordBatch::try_new(schema_zstd.clone(), vec![array.clone()]).unwrap();

    let lance_schema_zstd =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_zstd.as_ref()).unwrap());
    let encoded_zstd = rt
        .block_on(encode_batch(
            &data_zstd,
            lance_schema_zstd,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("zstd_utf8", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_zstd,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_zstd.num_rows(), batch.num_rows());
            }
        })
    });

    // flat baseline
    let mut metadata_flat = HashMap::new();
    metadata_flat.insert("lance-encoding:compression".to_string(), "none".to_string());
    metadata_flat.insert(
        "lance-encoding:dict-divisor".to_string(),
        "100000".to_string(),
    );
    metadata_flat.insert(
        "lance-encoding:structural-encoding".to_string(),
        "miniblock".to_string(),
    );

    let fields_flat = vec![Field::new("s", DataType::Utf8, false).with_metadata(metadata_flat)];
    let schema_flat = Arc::new(Schema::new(fields_flat));
    let data_flat = RecordBatch::try_new(schema_flat.clone(), vec![array.clone()]).unwrap();

    let lance_schema_flat =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_flat.as_ref()).unwrap());
    let encoded_flat = rt
        .block_on(encode_batch(
            &data_flat,
            lance_schema_flat,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("flat_utf8", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_flat,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_flat.num_rows(), batch.num_rows());
            }
        })
    });

    group.finish();
}

fn bench_decode_rle(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("decode_rle");
    const NUM_BYTES: u64 = 1024 * 1024 * 128;
    const NUM_ROWS: u64 = NUM_BYTES / 4; // Int32
    group.throughput(criterion::Throughput::Bytes(NUM_BYTES));

    // Only 2 unique values with long runs to make RLE beat bitpacking:
    // bitpacking with bit_width=1 needs ~4MB, but RLE only needs ~10 bytes.
    let half = NUM_ROWS / 2;
    let values: Vec<i32> = (0..NUM_ROWS)
        .map(|i| if i < half { 0i32 } else { 1i32 })
        .collect();
    let array: Arc<dyn arrow_array::Array> = Arc::new(arrow_array::Int32Array::from(values));

    // RLE path: high threshold to encourage RLE, disable other encodings
    // Do NOT set compression=none, or build_fixed_width_compressor will skip
    // all encodings (including RLE) and return ValueEncoder directly.
    let mut metadata_rle = HashMap::new();
    metadata_rle.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.9".to_string(),
    );
    metadata_rle.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_rle.insert("lance-encoding:delta-rle".to_string(), "false".to_string());

    let fields_rle = vec![Field::new("v", DataType::Int32, false).with_metadata(metadata_rle)];
    let schema_rle = Arc::new(Schema::new(fields_rle));
    let data_rle = RecordBatch::try_new(schema_rle.clone(), vec![array.clone()]).unwrap();

    let lance_schema_rle =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_rle.as_ref()).unwrap());
    let encoded_rle = rt
        .block_on(encode_batch(
            &data_rle,
            lance_schema_rle,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("rle_int32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_rle,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_rle.num_rows(), batch.num_rows());
            }
        })
    });

    // Flat baseline: same data but force no encoding
    let mut metadata_flat = HashMap::new();
    metadata_flat.insert("lance-encoding:compression".to_string(), "none".to_string());
    metadata_flat.insert("lance-encoding:bss".to_string(), "off".to_string());
    metadata_flat.insert("lance-encoding:delta-rle".to_string(), "false".to_string());
    metadata_flat.insert(
        "lance-encoding:rle-threshold".to_string(),
        "0.0".to_string(),
    );

    let fields_flat = vec![Field::new("v", DataType::Int32, false).with_metadata(metadata_flat)];
    let schema_flat = Arc::new(Schema::new(fields_flat));
    let data_flat = RecordBatch::try_new(schema_flat.clone(), vec![array.clone()]).unwrap();

    let lance_schema_flat =
        Arc::new(lance_core::datatypes::Schema::try_from(schema_flat.as_ref()).unwrap());
    let encoded_flat = rt
        .block_on(encode_batch(
            &data_flat,
            lance_schema_flat,
            default_encoding_strategy(LanceFileVersion::V2_2).as_ref(),
            &EncodingOptions::default(),
        ))
        .unwrap();

    group.bench_function("flat_int32", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let batch = rt
                    .block_on(lance_encoding::decoder::decode_batch(
                        &encoded_flat,
                        &FilterExpression::no_filter(),
                        Arc::<DecoderPlugins>::default(),
                        false,
                        LanceFileVersion::V2_2,
                        Some(Arc::new(LanceCache::no_cache())),
                    ))
                    .unwrap();
                assert_eq!(data_flat.num_rows(), batch.num_rows());
            }
        })
    });

    group.finish();
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10)
        .with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
    targets = bench_decode, bench_decode_fsl, bench_decode_str_with_dict_encoding, bench_decode_packed_struct,
                bench_decode_str_with_fixed_size_binary_encoding, bench_decode_compressed,
                bench_decode_compressed_parallel, bench_decode_bitpacking, bench_decode_bss,
                bench_decode_bitpacking_layers, bench_decode_fsst, bench_decode_rle);

// Non-linux version does not support pprof.
#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_decode, bench_decode_fsl, bench_decode_str_with_dict_encoding, bench_decode_packed_struct,
                bench_decode_compressed, bench_decode_compressed_parallel, bench_decode_bitpacking,
                bench_decode_bitpacking_layers,
                bench_decode_bss, bench_decode_fsst, bench_decode_rle);
criterion_main!(benches);
