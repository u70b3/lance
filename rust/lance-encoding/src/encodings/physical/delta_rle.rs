// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Delta+RLE Cascading Compression
//!
//! A compression scheme that combines delta encoding with run-length encoding (RLE)
//! for optimal compression of monotonic data such as timestamps, IDs, and sequences.
//!
//! ## Algorithm
//!
//! 1. **Delta Encoding**: Compute first-order differences
//!    - Store first value as-is
//!    - Subsequent values store delta from previous value
//!
//! 2. **RLE Encoding**: Apply run-length encoding to the delta values
//!    - Monotonic data typically has constant or slowly-changing deltas
//!    - RLE compresses repeated delta values efficiently
//!
//! ## Example
//!
//! Input data: `[1000, 1001, 1002, 1003, 1004, 1005]`
//!
//! After delta: `[1000, 1, 1, 1, 1, 1]`
//!
//! After RLE: `[(1000, 1), (1, 5)]`
//!
//! ## Format
//!
//! - **Header**: First value (i64)
//! - **RLE Data**: Values buffer + lengths buffer (reuses existing RLE format)
//!
//! ## Use Cases
//!
//! - Time-series timestamps (typically millisecond intervals)
//! - Auto-increment IDs
//! - Sorted sequences
//! - Sensor readings with regular sampling intervals

use std::io::Write;

use arrow_buffer::ArrowNativeType;
use bytemuck::Pod;
use log::trace;

use crate::buffer::LanceBuffer;
use crate::compression::{BlockCompressor, BlockDecompressor, MiniBlockDecompressor};
use crate::data::{BlockInfo, DataBlock, FixedWidthDataBlock};
use crate::encodings::logical::primitive::miniblock::{
    MiniBlockChunk, MiniBlockCompressed, MiniBlockCompressor,
};
use crate::format::pb21::CompressiveEncoding;
use crate::format::ProtobufUtils21;
// Statistics support can be added later if needed
// use crate::statistics::{GetStat, Stat};

use lance_core::{Error, Result};

/// Minimum number of values to consider Delta+RLE
const MIN_VALUES_FOR_DELTA_RLE: u64 = 10;

/// Minimum monotonicity ratio (0.0 to 1.0) to trigger Delta+RLE
const MIN_MONOTONICITY_RATIO: f64 = 0.8;

/// Minimum estimated compression ratio to use Delta+RLE
const MIN_COMPRESSION_RATIO: f64 = 1.5;

/// Delta+RLE encoder for miniblock format
#[derive(Debug, Default)]
pub struct DeltaRleEncoder;

impl DeltaRleEncoder {
    /// Create a new Delta+RLE encoder
    pub fn new() -> Self {
        Self
    }

    /// Compute delta encoding for fixed-width data
    ///
    /// Returns: (first_value, deltas)
    /// - first_value: The original first value (stored separately)
    /// - deltas: Vec<i64> containing [first_value, v1-v0, v2-v1, ...]
    fn compute_deltas<T>(data: &[T]) -> (i64, Vec<i64>)
    where
        T: ArrowNativeType + Copy,
    {
        if data.is_empty() {
            return (0, Vec::new());
        }

        let first_value = Self::to_i64(data[0]);
        let mut deltas = Vec::with_capacity(data.len());
        deltas.push(first_value);

        for i in 1..data.len() {
            let delta = Self::to_i64(data[i]) - Self::to_i64(data[i - 1]);
            deltas.push(delta);
        }

        (first_value, deltas)
    }

    /// Convert ArrowNativeType to i64
    #[inline]
    fn to_i64<T>(value: T) -> i64
    where
        T: ArrowNativeType,
    {
        value.as_usize() as i64
    }

    /// Encode deltas using RLE
    ///
    /// Returns: (values_buffer, lengths_buffer)
    fn rle_encode_deltas(deltas: &[i64]) -> (Vec<u8>, Vec<u8>) {
        if deltas.is_empty() {
            return (Vec::new(), Vec::new());
        }

        // Estimate capacity: assume ~10:1 compression ratio
        let estimated_runs = deltas.len() / 10;
        let mut values = Vec::with_capacity(estimated_runs * 8);
        let mut lengths = Vec::with_capacity(estimated_runs);

        let mut current_value = deltas[0];
        let mut current_length: u64 = 1;

        for &value in &deltas[1..] {
            if value == current_value {
                current_length += 1;
            } else {
                Self::add_run(&mut values, &mut lengths, current_value, current_length);
                current_value = value;
                current_length = 1;
            }
        }

        // Add final run
        Self::add_run(&mut values, &mut lengths, current_value, current_length);

        (values, lengths)
    }

    /// Add a run to the encoded buffers
    ///
    /// Handles runs longer than 255 by splitting into multiple entries
    fn add_run(values: &mut Vec<u8>, lengths: &mut Vec<u8>, value: i64, length: u64) {
        let value_bytes = value.to_le_bytes();
        let num_full_chunks = (length / 255) as usize;
        let remainder = (length % 255) as u8;

        // Write full 255-length runs
        for _ in 0..num_full_chunks {
            values.extend_from_slice(&value_bytes);
            lengths.push(255);
        }

        // Write remainder (if any)
        if remainder > 0 {
            values.extend_from_slice(&value_bytes);
            lengths.push(remainder);
        }
    }

    /// Encode a chunk of data
    ///
    /// Returns encoded buffers and metadata
    fn encode_chunk<T>(
        &self,
        data: &LanceBuffer,
        num_values: u64,
    ) -> Result<(Vec<LanceBuffer>, Vec<MiniBlockChunk>, i64)>
    where
        T: ArrowNativeType + Pod + Copy,
    {
        if num_values == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }

        let typed_data = data.borrow_to_typed_slice::<T>();
        let typed_slice = typed_data.as_ref();

        // Compute deltas
        let (first_value, deltas) = Self::compute_deltas(typed_slice);

        // RLE encode the deltas
        let (values_buf, lengths_buf) = Self::rle_encode_deltas(&deltas);

        // Create single chunk for simplicity (can be optimized later)
        let chunk = MiniBlockChunk {
            buffer_sizes: vec![values_buf.len() as u32, lengths_buf.len() as u32],
            log_num_values: 0, // Not using power-of-2 optimization
        };

        let buffers = vec![
            LanceBuffer::from(values_buf),
            LanceBuffer::from(lengths_buf),
        ];

        Ok((buffers, vec![chunk], first_value))
    }
}

impl MiniBlockCompressor for DeltaRleEncoder {
    fn compress(&self, data: DataBlock) -> Result<(MiniBlockCompressed, CompressiveEncoding)> {
        match data {
            DataBlock::FixedWidth(fixed) => {
                let num_values = fixed.num_values;
                let bits_per_value = fixed.bits_per_value;

                trace!(
                    "DeltaRleEncoder compressing {} values with {} bits per value",
                    num_values,
                    bits_per_value
                );

                let (buffers, chunks, first_value) = match bits_per_value {
                    8 => self.encode_chunk::<u8>(&fixed.data, num_values)?,
                    16 => self.encode_chunk::<u16>(&fixed.data, num_values)?,
                    32 => self.encode_chunk::<u32>(&fixed.data, num_values)?,
                    64 => self.encode_chunk::<u64>(&fixed.data, num_values)?,
                    _ => {
                        return Err(Error::invalid_input(format!(
                            "DeltaRleEncoder only supports 8, 16, 32, or 64 bit values, got {}",
                            bits_per_value
                        )))
                    }
                };

                let compressed = MiniBlockCompressed {
                    data: buffers,
                    chunks,
                    num_values,
                };

                // Create encoding description
                // Inner RLE encoding uses flat encoding for both values and lengths
                let value_encoding = ProtobufUtils21::flat(64, None); // i64 values
                let length_encoding = ProtobufUtils21::flat(8, None); // u8 lengths

                let encoding = ProtobufUtils21::delta_rle(
                    bits_per_value,
                    first_value,
                    value_encoding,
                    length_encoding,
                );

                trace!(
                    "DeltaRleEncoder compressed {} values, first_value={}",
                    num_values,
                    first_value
                );

                Ok((compressed, encoding))
            }
            _ => Err(Error::invalid_input_source(
                "DeltaRleEncoder only supports FixedWidth data blocks".into(),
            )),
        }
    }
}

impl BlockCompressor for DeltaRleEncoder {
    /// Block format: [8-byte header: first_value][8-byte header: values buffer size]
    ///               [values buffer][run_lengths buffer]
    fn compress(&self, data: DataBlock) -> Result<LanceBuffer> {
        match data {
            DataBlock::FixedWidth(fixed) => {
                let num_values = fixed.num_values;
                let bits_per_value = fixed.bits_per_value;

                let (buffers, _, first_value) = match bits_per_value {
                    8 => self.encode_chunk::<u8>(&fixed.data, num_values)?,
                    16 => self.encode_chunk::<u16>(&fixed.data, num_values)?,
                    32 => self.encode_chunk::<u32>(&fixed.data, num_values)?,
                    64 => self.encode_chunk::<u64>(&fixed.data, num_values)?,
                    _ => {
                        return Err(Error::invalid_input(format!(
                            "DeltaRleEncoder only supports 8, 16, 32, or 64 bit values, got {}",
                            bits_per_value
                        )))
                    }
                };

                let values_size = buffers[0].len() as u64;

                // Build combined buffer: [first_value (8 bytes)][values_size (8 bytes)][values][lengths]
                let mut combined = Vec::with_capacity(16 + buffers[0].len() + buffers[1].len());
                combined.write_all(&first_value.to_le_bytes())?;
                combined.write_all(&values_size.to_le_bytes())?;
                combined.extend_from_slice(&buffers[0]);
                combined.extend_from_slice(&buffers[1]);

                Ok(LanceBuffer::from(combined))
            }
            _ => Err(Error::invalid_input_source(
                "DeltaRleEncoder only supports FixedWidth data blocks".into(),
            )),
        }
    }
}

/// Delta+RLE decompressor
#[derive(Debug)]
pub struct DeltaRleDecompressor {
    bits_per_value: u64,
    first_value: i64,
}

impl DeltaRleDecompressor {
    /// Create a new decompressor
    pub fn new(bits_per_value: u64, first_value: i64) -> Self {
        Self {
            bits_per_value,
            first_value,
        }
    }

    /// Create from protobuf description
    pub fn from_description(desc: &crate::format::pb21::DeltaRle) -> Self {
        Self {
            bits_per_value: desc.uncompressed_bits_per_value,
            first_value: desc.first_value,
        }
    }

    /// Decode RLE data to get deltas
    fn rle_decode_deltas(values_buf: &[u8], lengths_buf: &[u8]) -> Vec<i64> {
        let mut deltas = Vec::new();

        let num_runs = values_buf.len() / 8;
        for (i, &length) in lengths_buf.iter().enumerate().take(num_runs) {
            let value_offset = i * 8;
            let value = i64::from_le_bytes([
                values_buf[value_offset],
                values_buf[value_offset + 1],
                values_buf[value_offset + 2],
                values_buf[value_offset + 3],
                values_buf[value_offset + 4],
                values_buf[value_offset + 5],
                values_buf[value_offset + 6],
                values_buf[value_offset + 7],
            ]);

            deltas.extend(std::iter::repeat_n(value, length as usize));
        }

        deltas
    }

    /// Reconstruct original data from deltas
    fn reconstruct_from_deltas<T>(first_value: i64, deltas: &[i64], num_values: u64) -> Vec<T>
    where
        T: ArrowNativeType,
    {
        let mut result = Vec::with_capacity(num_values as usize);
        let mut current = first_value;

        for (i, &delta) in deltas.iter().enumerate() {
            if i == 0 {
                // First value is stored directly
                current = delta;
            } else {
                current += delta;
            }
            result.push(T::from_usize(current as usize).expect("value fits in target type"));
        }

        result
    }

    /// Decode data for a specific type
    fn decode_data<T>(&self, buffers: &[LanceBuffer], num_values: u64) -> Result<LanceBuffer>
    where
        T: ArrowNativeType + Pod,
    {
        if num_values == 0 {
            return Ok(LanceBuffer::empty());
        }

        if buffers.len() != 2 {
            return Err(Error::invalid_input_source(
                format!(
                    "DeltaRleDecompressor expects exactly 2 buffers, got {}",
                    buffers.len()
                )
                .into(),
            ));
        }

        let values_buf = buffers[0].as_ref();
        let lengths_buf = buffers[1].as_ref();

        // RLE decode to get deltas
        let deltas = Self::rle_decode_deltas(values_buf, lengths_buf);

        if deltas.len() < num_values as usize {
            return Err(Error::invalid_input_source(
                format!(
                    "DeltaRleDecompressor expected {} values but got {}",
                    num_values,
                    deltas.len()
                )
                .into(),
            ));
        }

        // Reconstruct original data
        let reconstructed = Self::reconstruct_from_deltas::<T>(
            self.first_value,
            &deltas[..num_values as usize],
            num_values,
        );

        Ok(LanceBuffer::reinterpret_vec(reconstructed))
    }
}

impl MiniBlockDecompressor for DeltaRleDecompressor {
    fn decompress(&self, data: Vec<LanceBuffer>, num_values: u64) -> Result<DataBlock> {
        let decoded_data = match self.bits_per_value {
            8 => self.decode_data::<u8>(&data, num_values)?,
            16 => self.decode_data::<u16>(&data, num_values)?,
            32 => self.decode_data::<u32>(&data, num_values)?,
            64 => self.decode_data::<u64>(&data, num_values)?,
            _ => {
                return Err(Error::invalid_input(format!(
                    "DeltaRleDecompressor only supports 8, 16, 32, or 64 bit values, got {}",
                    self.bits_per_value
                )))
            }
        };

        Ok(DataBlock::FixedWidth(FixedWidthDataBlock {
            bits_per_value: self.bits_per_value,
            data: decoded_data,
            num_values,
            block_info: BlockInfo::default(),
        }))
    }
}

impl BlockDecompressor for DeltaRleDecompressor {
    fn decompress(&self, data: LanceBuffer, num_values: u64) -> Result<DataBlock> {
        let data_slice = data.as_ref();

        if data_slice.len() < 16 {
            return Err(Error::invalid_input_source(
                "DeltaRle block data too short (minimum 16 bytes for headers)".into(),
            ));
        }

        // Parse headers
        let first_value = i64::from_le_bytes([
            data_slice[0], data_slice[1], data_slice[2], data_slice[3],
            data_slice[4], data_slice[5], data_slice[6], data_slice[7],
        ]);

        let values_size = u64::from_le_bytes([
            data_slice[8], data_slice[9], data_slice[10], data_slice[11],
            data_slice[12], data_slice[13], data_slice[14], data_slice[15],
        ]) as usize;

        if data_slice.len() < 16 + values_size {
            return Err(Error::invalid_input_source(
                "DeltaRle block data too short for values buffer".into(),
            ));
        }

        let values_buf = &data_slice[16..16 + values_size];
        let lengths_buf = &data_slice[16 + values_size..];

        // Create buffers vector for unified decoding
        let buffers = vec![
            LanceBuffer::from(values_buf.to_vec()),
            LanceBuffer::from(lengths_buf.to_vec()),
        ];

        // Use miniblock decompressor
        let decompressor = Self::new(self.bits_per_value, first_value);
        MiniBlockDecompressor::decompress(&decompressor, buffers, num_values)
    }
}

/// Check if data is suitable for Delta+RLE encoding
///
/// Returns Some(estimated_ratio) if suitable, None otherwise
pub fn should_use_delta_rle(data: &FixedWidthDataBlock) -> Option<f64> {
    if data.num_values < MIN_VALUES_FOR_DELTA_RLE {
        return None;
    }

    // Only support standard bit widths
    if !matches!(data.bits_per_value, 8 | 16 | 32 | 64) {
        return None;
    }

    // Check monotonicity
    let monotonicity = compute_monotonicity(data);
    if monotonicity < MIN_MONOTONICITY_RATIO {
        trace!(
            "Data not monotonic enough for Delta+RLE: {} < {}",
            monotonicity,
            MIN_MONOTONICITY_RATIO
        );
        return None;
    }

    // Estimate compression ratio
    let estimated_ratio = estimate_delta_rle_ratio(data, monotonicity);
    if estimated_ratio < MIN_COMPRESSION_RATIO {
        trace!(
            "Estimated compression ratio too low for Delta+RLE: {} < {}",
            estimated_ratio,
            MIN_COMPRESSION_RATIO
        );
        return None;
    }

    Some(estimated_ratio)
}

/// Compute monotonicity ratio (0.0 to 1.0)
///
/// Returns the proportion of adjacent pairs that are monotonic (all increasing or all decreasing)
fn compute_monotonicity(data: &FixedWidthDataBlock) -> f64 {
    match data.bits_per_value {
        8 => compute_monotonicity_typed::<u8>(data),
        16 => compute_monotonicity_typed::<u16>(data),
        32 => compute_monotonicity_typed::<u32>(data),
        64 => compute_monotonicity_typed::<u64>(data),
        _ => 0.0,
    }
}

fn compute_monotonicity_typed<T>(data: &FixedWidthDataBlock) -> f64
where
    T: ArrowNativeType + Pod + Copy,
{
    let typed_data = data.data.borrow_to_typed_slice::<T>();
    let slice = typed_data.as_ref();

    if slice.len() < 2 {
        return 0.0;
    }

    let mut increasing = 0usize;
    let mut decreasing = 0usize;

    for i in 1..slice.len() {
        let prev_i64 = value_to_i64(slice[i - 1]);
        let curr_i64 = value_to_i64(slice[i]);

        if curr_i64 > prev_i64 {
            increasing += 1;
        } else if curr_i64 < prev_i64 {
            decreasing += 1;
        }
        // Equal values don't count toward either
    }

    let total = slice.len() - 1;
    let monotonic_count = increasing.max(decreasing);

    monotonic_count as f64 / total as f64
}

/// Convert any ArrowNativeType to i64
#[inline]
fn value_to_i64<T: ArrowNativeType>(value: T) -> i64 {
    // Use as_usize() for unsigned types, which may truncate for u64 values > i64::MAX
    // but this is acceptable for monotonicity detection
    value.as_usize() as i64
}

/// Estimate the compression ratio for Delta+RLE
///
/// This is a heuristic based on monotonicity and data characteristics
fn estimate_delta_rle_ratio(data: &FixedWidthDataBlock, monotonicity: f64) -> f64 {
    // Base ratio from monotonicity
    let base_ratio = 1.0 + monotonicity * 2.0;

    // Factor based on number of values (more values = better amortization of overhead)
    let size_factor = (data.num_values as f64 / 1000.0).min(2.0);

    // Combine factors
    let estimated = base_ratio * (1.0 + size_factor * 0.5);

    estimated.min(20.0) // Cap at 20x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_deltas() {
        let data = vec![1000i64, 1001, 1002, 1003, 1004];
        let (first, deltas) = DeltaRleEncoder::compute_deltas(&data);

        assert_eq!(first, 1000);
        assert_eq!(deltas, vec![1000, 1, 1, 1, 1]);
    }

    #[test]
    fn test_rle_encode_deltas() {
        let deltas = vec![1000i64, 1, 1, 1, 1];
        let (values, lengths) = DeltaRleEncoder::rle_encode_deltas(&deltas);

        // Should produce 2 runs: (1000, 1) and (1, 4)
        assert_eq!(values.len(), 16); // 2 i64 values = 16 bytes
        assert_eq!(lengths.len(), 2);
        assert_eq!(lengths[0], 1);
        assert_eq!(lengths[1], 4);
    }

    #[test]
    fn test_rle_decode_deltas() {
        // Create encoded data for [1000, 1, 1, 1, 1]
        let values_buf = vec![
            0xE8, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1000 in little-endian
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1 in little-endian
        ];
        let lengths_buf = vec![1, 4];

        let deltas = DeltaRleDecompressor::rle_decode_deltas(&values_buf, &lengths_buf);
        assert_eq!(deltas, vec![1000, 1, 1, 1, 1]);
    }

    #[test]
    fn test_reconstruct_from_deltas() {
        let deltas = vec![1000i64, 1, 1, 1, 1];
        let first_value = 1000;

        let reconstructed: Vec<i64> =
            DeltaRleDecompressor::reconstruct_from_deltas(first_value, &deltas, 5);

        assert_eq!(reconstructed, vec![1000, 1001, 1002, 1003, 1004]);
    }

    #[test]
    fn test_compute_monotonicity() {
        // Fully increasing
        let data = create_data_block(&[1u64, 2, 3, 4, 5]);
        assert_eq!(compute_monotonicity(&data), 1.0);

        // Fully decreasing
        let data = create_data_block(&[5u64, 4, 3, 2, 1]);
        assert_eq!(compute_monotonicity(&data), 1.0);

        // Mixed
        let data = create_data_block(&[1u64, 3, 2, 4, 3]);
        assert!(compute_monotonicity(&data) < 1.0);
    }

    #[test]
    fn test_should_use_delta_rle() {
        // Monotonic data with enough values
        let data = create_data_block(&(0..100u64).collect::<Vec<_>>());
        assert!(should_use_delta_rle(&data).is_some());

        // Too few values
        let data = create_data_block(&[1u64, 2, 3]);
        assert!(should_use_delta_rle(&data).is_none());

        // Not monotonic (oscillating pattern)
        let oscillating: Vec<u64> = (0..100).map(|i| if i % 2 == 0 { i + 2 } else { i }).collect();
        let data = create_data_block(&oscillating);
        assert!(should_use_delta_rle(&data).is_none());
    }

    fn create_data_block(values: &[u64]) -> FixedWidthDataBlock {
        let bytes = bytemuck::cast_slice(values);
        FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 64,
            num_values: values.len() as u64,
            block_info: BlockInfo::default(),
        }
    }

    // ========== 9.1 Integration Tests (Round-trip) ==========

    #[test]
    fn test_delta_rle_roundtrip_i64_monotonic_increasing() {
        // Test monotonic increasing i64 data (like timestamps)
        let data: Vec<i64> = (0..10000i64).map(|i| 1700000000000i64 + i * 1000).collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 64,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress
        let (compressed, _encoding) = MiniBlockCompressor::compress(&encoder, block).unwrap();
        assert_eq!(compressed.num_values, data.len() as u64);

        // Decompress
        let decompressor = DeltaRleDecompressor::new(64, data[0]);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[i64] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_delta_rle_roundtrip_u32_monotonic_increasing() {
        // Test monotonic increasing u32 data (like auto-increment IDs)
        let data: Vec<u32> = (1..10000u32).collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 32,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress
        let (compressed, _encoding) = MiniBlockCompressor::compress(&encoder, block).unwrap();

        // Decompress
        let decompressor = DeltaRleDecompressor::new(32, data[0] as i64);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[u32] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_delta_rle_roundtrip_u64_monotonic_decreasing() {
        // Test monotonic decreasing u64 data
        let data: Vec<u64> = (0..5000u64).rev().collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 64,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress
        let (compressed, _encoding) = MiniBlockCompressor::compress(&encoder, block).unwrap();

        // Decompress
        let decompressor = DeltaRleDecompressor::new(64, data[0] as i64);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[u64] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_delta_rle_roundtrip_u16_with_small_deltas() {
        // Test u16 with small delta values (like sensor readings at regular intervals)
        let data: Vec<u16> = (0..1000u16).map(|i| 1000 + i % 100).collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 16,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress
        let (compressed, _encoding) = MiniBlockCompressor::compress(&encoder, block).unwrap();

        // Decompress
        let decompressor = DeltaRleDecompressor::new(16, data[0] as i64);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[u16] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    #[test]
    fn test_delta_rle_roundtrip_u8_minimal_values() {
        // Test u8 with small dataset (boundary case: MIN_VALUES_FOR_DELTA_RLE = 10)
        let data: Vec<u8> = (10..20u8).collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 8,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress
        let (compressed, _encoding) = MiniBlockCompressor::compress(&encoder, block).unwrap();

        // Decompress
        let decompressor = DeltaRleDecompressor::new(8, data[0] as i64);
        let decompressed = MiniBlockDecompressor::decompress(
            &decompressor,
            compressed.data,
            compressed.num_values,
        )
        .unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[u8] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }
    }

    // ========== 9.2 Compression Ratio Tests ==========

    #[test]
    fn test_delta_rle_compression_ratio_vs_rle() {
        use crate::encodings::physical::rle::RleEncoder;

        // Test timestamp data (i64) with varying sizes
        for count in [1000, 10000] {
            let name = format!("timestamp_i64_{}", count);
            let data: Vec<i64> = (0..count).map(|i| 1700000000000i64 + i * 1000).collect();
            let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

            let encoder = DeltaRleEncoder::new();
            let block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(bytes.clone()),
                bits_per_value: 64,
                num_values: count as u64,
                block_info: BlockInfo::default(),
            });

            // Delta+RLE compression size
            let (delta_rle_compressed, _) = MiniBlockCompressor::compress(&encoder, block).unwrap();
            let delta_rle_size = delta_rle_compressed
                .data
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>();

            // RLE compression size
            let rle_encoder = RleEncoder::new();
            let rle_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(bytes.clone()),
                bits_per_value: 64,
                num_values: count as u64,
                block_info: BlockInfo::default(),
            });
            let (rle_compressed, _) = MiniBlockCompressor::compress(&rle_encoder, rle_block).unwrap();
            let rle_size = rle_compressed
                .data
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>();

            let ratio = delta_rle_size as f64 / rle_size as f64;
            println!(
                "{}: Delta+RLE={} bytes, RLE={} bytes, ratio={:.2}x (Delta+RLE smaller is better)",
                name, delta_rle_size, rle_size, ratio
            );

            // For monotonic data, Delta+RLE should be significantly better than RLE
            assert!(
                ratio < 0.5,
                "{}: Delta+RLE ({}) should be at least 2x better than RLE ({}), got ratio {:.2}",
                name, delta_rle_size, rle_size, ratio
            );
        }

        // Test u32 auto-increment IDs
        for count in [1000, 10000] {
            let name = format!("ids_u32_{}", count);
            let data: Vec<u32> = (1..count as u32 + 1).collect();
            let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

            let encoder = DeltaRleEncoder::new();
            let block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(bytes.clone()),
                bits_per_value: 32,
                num_values: count as u64,
                block_info: BlockInfo::default(),
            });

            // Delta+RLE compression size
            let (delta_rle_compressed, _) = MiniBlockCompressor::compress(&encoder, block).unwrap();
            let delta_rle_size = delta_rle_compressed
                .data
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>();

            // RLE compression size
            let rle_encoder = RleEncoder::new();
            let rle_block = DataBlock::FixedWidth(FixedWidthDataBlock {
                data: LanceBuffer::from(bytes),
                bits_per_value: 32,
                num_values: count as u64,
                block_info: BlockInfo::default(),
            });
            let (rle_compressed, _) = MiniBlockCompressor::compress(&rle_encoder, rle_block).unwrap();
            let rle_size = rle_compressed
                .data
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>();

            let ratio = delta_rle_size as f64 / rle_size as f64;
            println!(
                "{}: Delta+RLE={} bytes, RLE={} bytes, ratio={:.2}x (Delta+RLE smaller is better)",
                name, delta_rle_size, rle_size, ratio
            );

            // For monotonic data, Delta+RLE should be significantly better than RLE
            assert!(
                ratio < 0.5,
                "{}: Delta+RLE ({}) should be at least 2x better than RLE ({}), got ratio {:.2}",
                name, delta_rle_size, rle_size, ratio
            );
        }
    }

    #[test]
    fn test_delta_rle_compression_size_standalone() {
        // Test that Delta+RLE achieves good compression on monotonic data
        let encoder = DeltaRleEncoder::new();

        // Timestamps: 10000 values, delta = 1000
        let data: Vec<i64> = (0..10000i64).map(|i| 1700000000000i64 + i * 1000).collect();
        let bytes = bytemuck::cast_slice(&data);

        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 64,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Original size: 10000 * 8 = 80000 bytes
        let original_size = bytes.len() as f64;

        let (compressed, _) = MiniBlockCompressor::compress(&encoder, block).unwrap();
        let compressed_size = compressed
            .data
            .iter()
            .map(|b| b.len() as f64)
            .sum::<f64>();

        let compression_ratio = original_size / compressed_size;
        println!(
            "Original: {:.0} bytes, Compressed: {:.0} bytes, Ratio: {:.1}x",
            original_size, compressed_size, compression_ratio
        );

        // Expect at least 2x compression for this pattern
        // The deltas are all 1 (small), so RLE should compress them very well
        assert!(
            compression_ratio >= 2.0,
            "Expected at least 2x compression, got {:.1}x",
            compression_ratio
        );
    }

    // ========== Block Format Round-trip Tests ==========

    #[test]
    fn test_delta_rle_block_format_roundtrip() {
        // Test the BlockCompressor/BlockDecompressor round-trip
        let data: Vec<u64> = (0..5000u64).map(|i| 1000000 + i).collect();
        let encoder = DeltaRleEncoder::new();

        let bytes = bytemuck::cast_slice(&data);
        let block = DataBlock::FixedWidth(FixedWidthDataBlock {
            data: LanceBuffer::from(bytes.to_vec()),
            bits_per_value: 64,
            num_values: data.len() as u64,
            block_info: BlockInfo::default(),
        });

        // Compress using block format
        let compressed = BlockCompressor::compress(&encoder, block).unwrap();
        let compressed_size = compressed.len() as u64;

        // Decompress
        let decompressor = DeltaRleDecompressor::new(64, data[0] as i64);
        let decompressed = BlockDecompressor::decompress(&decompressor, compressed, data.len() as u64).unwrap();

        // Verify round-trip correctness
        match decompressed {
            DataBlock::FixedWidth(ref result) => {
                let result_bytes = result.data.as_ref();
                let result_slice: &[u64] = bytemuck::cast_slice(result_bytes);
                assert_eq!(result_slice, &data);
            }
            _ => panic!("Expected FixedWidth block"),
        }

        println!(
            "Block format: {} values, compressed to {} bytes ({:.1}x ratio)",
            data.len(),
            compressed_size,
            (data.len() * 8) as f64 / compressed_size as f64
        );
    }
}
