use crate::DType;
use anyhow::{Result, bail};
use half::{bf16, f16};

/// Converts `LittleEndian` byte representations of f32s into a vector of f32s
pub fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Converts `LittleEndian` byte representations of f16s into a vector of f32s
pub fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Converts `LittleEndian` byte representations of bf16s into a vector of f32s
pub fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

pub fn dtype_to_f32(bytes: &[u8], dtype: &DType) -> Result<Vec<f32>> {
    match dtype {
        DType::F32 => Ok(f32_bytes_to_f32(bytes)),
        DType::F16 => Ok(f16_bytes_to_f32(bytes)),
        DType::BF16 => Ok(bf16_bytes_to_f32(bytes)),
        other => {
            bail!("unrecognised dtype {other:?}")
        }
    }
}

/// Pad the given `data`, to ensure that the number of values is a multiple of `block_size`.
///
/// The data is padded with `0.0f32`
pub fn pad_to_block_size(data: &[f32], block_size: usize) -> Vec<f32> {
    let remainder = data.len() % block_size;
    if remainder == 0 {
        return data.to_vec();
    }

    let pad_size = block_size - remainder;
    let mut padded = Vec::with_capacity(data.len() + pad_size);
    padded.extend_from_slice(data);
    padded.extend(std::iter::repeat_n(0.0f32, pad_size));
    padded
}

#[cfg(test)]
mod tests {
    mod dtype_conversions {
        use super::super::*;
        #[test]
        fn test_f32_bytes_to_f32() {
            let f32_bytes: [u8; 12] = [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x3F,
            ];
            let f32_converted = f32_bytes_to_f32(&f32_bytes);

            assert_eq!(f32_converted, vec![0f32, 1f32, 0.5f32])
        }

        #[test]
        fn test_f16_bytes_to_f32() {
            let f16_bytes: [u8; 6] = [0x00, 0x00, 0x00, 0x3C, 0x00, 0x38];
            let f32_converted = f16_bytes_to_f32(&f16_bytes);

            assert_eq!(f32_converted, vec![0f32, 1f32, 0.5f32])
        }

        #[test]
        fn test_bf16_bytes_to_f32() {
            let bf16_bytes: [u8; 6] = [0x00, 0x00, 0x80, 0xBF, 0x00, 0x3F];
            let f32_converted = bf16_bytes_to_f32(&bf16_bytes);

            assert_eq!(f32_converted, vec![0f32, -1f32, 0.5f32])
        }
    }

    mod block_size_padding {
        use super::super::*;

        #[test]
        fn pad_to_exact_block_size() {
            let data = vec![12.93f32; 8];
            let block_size = 10;
            let padded_data = pad_to_block_size(&data, block_size);

            assert_eq!(padded_data.len(), block_size);
            assert_eq!(padded_data[8..], [0.0f32; 2]);
        }

        #[test]
        fn pad_to_block_size_multiple() {
            let data = vec![12.93f32; 12];
            let block_size = 10;
            let padded_data = pad_to_block_size(&data, block_size);

            assert_eq!(padded_data.len() % block_size, 0);
            assert_eq!(padded_data[12..], [0.0f32; 8]);
        }

        #[test]
        fn pad_to_block_size_empty_data() {
            let data = vec![];
            let block_size = 10;
            let padded_data = pad_to_block_size(&data, block_size);

            assert_eq!(padded_data.len(), 0);
        }
    }
}
