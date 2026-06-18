use crate::DType;
use crate::gguf::tensor_names::hf_to_ggf_name;
use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use safetensors::SafeTensors;
use std::fmt::{Display, Formatter};
use std::fs::File;
use std::path::Path;

/// Block size of the Q8_0 quantization strategy
const Q8_0_BLOCK_SIZE: usize = 32;

/// Layer name suffixes that should not be quantized
const GGUF_KEEP_F32_SUFFIXES: [&str; 4] = [
    ".norm.weight",
    "_norm.weight",
    ".norm_1.weight",
    ".norm_2.weight",
];

/// Exact Layer names that should not be quantized
const GGUF_KEEP_F32_EXACT: [&str; 2] = ["token_embd.weight", "output_norm.weight"];

/// Denotes the GGML Type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    Q8_0 = 8,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
        }
    }
}

/// Tensor Data Post Quantization.
///
/// When a model is quantized, it creates a list of `QuantizedTensor`s for each layer in the model,
/// even if a layer was not quantized.
pub struct QuantizedTensor {
    /// Name of the tensor taken from the safetensor
    pub name: String,
    /// Standard GGUF tensor name
    pub gguf_name: String,
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// The quantized tensor in bytes
    pub data: Vec<u8>,
    /// Number of elements in the vector
    pub num_elements: usize,
    /// GGML type of the vector
    pub ggml_type: GGMLType,
}

/// Returns `true` if a tensor should be quantized to `Q8_0`, `false` if it must be kept in f32.
///
/// Two categories of tensors are always kept in f32:
///     - **1-D Tensors** - bias vectors, norms, and embedding scales; quantizing these produces
/// garbage output because Q8_0 block size assumes 2-D weight matrices
///     - **Named Exceptions** - tensors in `GGUF_KEEP_F32_EXACT` (exact match) or whose name
/// ends with a suffix in `GGUF_KEEP_F32_SUFFIXES` (e.g. `"norm.weight"`) — llama.cpp requires
/// these in F32 to preserve numerical stability during inference
fn should_quantize(tensor_name: &str, shape: &[usize]) -> bool {
    if shape.len() == 1 {
        return false;
    }

    if GGUF_KEEP_F32_EXACT.contains(&tensor_name) {
        return false;
    }

    if GGUF_KEEP_F32_SUFFIXES
        .iter()
        .any(|&n| tensor_name.ends_with(n))
    {
        return false;
    }

    true
}

/// Quantizes a slice of f32 values into Q8_0 blocks.
///
/// Each block contains 32 elements packed as:
/// `[scale: f16 LE, 2 bytes][32 × i8, 32 bytes] = 34 bytes`
fn quantize_q8_0_blocks(elements: &[f32], block_size: usize) -> Vec<u8> {
    assert_eq!(
        elements.len() % block_size,
        0,
        "Tensor is element count {} is not a multiple of block size {}",
        elements.len(),
        block_size
    );

    let num_blocks = elements.len() / block_size;
    let bytes_per_block = block_size + 2;
    let mut output = vec![0u8; num_blocks * bytes_per_block];

    output
        .par_chunks_mut(bytes_per_block)
        .zip(elements.par_chunks(block_size))
        .for_each(|(out_block, in_block)| {
            let amax = in_block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            let scale = if amax == 0.0 { 0.0f32 } else { amax / 127.0f32 };
            let scale_f16 = f16::from_f32(scale);

            out_block[0..2].copy_from_slice(&scale_f16.to_le_bytes());

            let inv_scale = if scale == 0.0 { 0.0f32 } else { 1.0f32 / scale };

            for (i, &val) in in_block.iter().enumerate() {
                let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;

                out_block[2 + i] = q as u8;
            }
        });

    output
}

/// Converts `LittleEndian` byte representations of f32s into a vector of f32s
fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Converts `LittleEndian` byte representations of f16s into a vector of f32s
fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Converts `LittleEndian` byte representations of bf16s into a vector of f32s
fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

/// Returns a vector of Quantized tensors loaded from the given `safetensors` file.
///
/// The safetensor file is loaded as a memory mapped file.
///
/// ```no_run
/// use std::path::{Path, PathBuf};
/// use erissery_core::quantization::quantize_safetensors_q8_0_from_file;
/// let path = PathBuf::from("model.safetensors");
/// let quantized_tensors = quantize_safetensors_q8_0_from_file(&path);
/// ```
pub fn quantize_safetensors_q8_0_from_file(path: &Path) -> Result<Vec<QuantizedTensor>> {
    // todo: Need to modularize this file read operation. I'm using this again in inspect
    let file =
        File::open(path).with_context(|| format!("Failed to open file: {}", path.display()))?;

    let mmap = unsafe { Mmap::map(&file) }
        .with_context(|| format!("Failed to memory map the file {}", path.display()))?;

    let tensors = SafeTensors::deserialize(&mmap).with_context(|| {
        format!(
            "Failed to parse safetensors header — is this a valid .safetensors file? {}",
            path.display()
        )
    })?;

    quantize_safetensors_q8_0(&tensors)
}

/// Pad the given `data`, to ensure that the number of values is a multiple of `block_size`.
///
/// The data is padded with `0.0f32`
fn pad_to_block_size(data: &[f32], block_size: usize) -> Vec<f32> {
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

/// Quantize and return the tensors.
///
/// Currently performs `Q8_0` quantization with support for more on the way.
/// Converts all the data to f32 before quantization. Does not quantize tensors with
/// suffixed defined in `GGUF_KEEP_F32_SUFFIXES` or names defined in `GGUF_KEEP_F32_EXACT`.
pub fn quantize_safetensors_q8_0(tensors: &SafeTensors) -> Result<Vec<QuantizedTensor>> {
    println!("Quantizing {} tensors to Q8_0", tensors.tensors().len());

    let mut results: Vec<QuantizedTensor> = Vec::with_capacity(tensors.tensors().len());

    for (name, view) in tensors.tensors() {
        let gguf_name = hf_to_ggf_name(name.as_str())
            .ok_or_else(|| anyhow!("unmapped tensor name: {}", name))?;
        let dtype = DType::from(view.dtype());
        let raw_bytes = view.data();
        let shape = view.shape().to_vec();
        let num_elements: usize = shape.iter().product();

        let f32_data: Vec<f32> = match dtype {
            DType::F32 => f32_bytes_to_f32(raw_bytes),
            DType::F16 => f16_bytes_to_f32(raw_bytes),
            DType::BF16 => bf16_bytes_to_f32(raw_bytes),
            other => {
                eprintln!("\t SKIP {name}: dtype {other} cannot be quantized");
                continue;
            }
        };

        if !should_quantize(name.as_str(), view.shape()) {
            let bytes: Vec<u8> = f32_data.iter().flat_map(|&x| x.to_le_bytes()).collect();

            println!("\t{name:<100} {:<5} {num_elements:>10}", GGMLType::F32);

            results.push(QuantizedTensor {
                name,
                gguf_name,
                shape,
                data: bytes,
                num_elements: f32_data.len(),
                ggml_type: GGMLType::F32,
            });
        } else {
            if f32_data.len() != num_elements {
                bail!(
                    "Tensor '{}': expected {} elements from shape {:?}, \
                 but decoded {} f32 values. Dtype was {}.",
                    name,
                    num_elements,
                    shape,
                    f32_data.len(),
                    dtype
                )
            }

            let padded_data = pad_to_block_size(&f32_data, Q8_0_BLOCK_SIZE);

            print!("\t{name:<100} {:<5} {num_elements:>10}", GGMLType::Q8_0);
            if f32_data.len() != padded_data.len() {
                print!(" (padded {} > {})", f32_data.len(), padded_data.len());
            }
            println!();

            let quantized_bytes = quantize_q8_0_blocks(&padded_data, Q8_0_BLOCK_SIZE);

            results.push(QuantizedTensor {
                name,
                gguf_name,
                shape,
                data: quantized_bytes,
                num_elements,
                ggml_type: GGMLType::Q8_0,
            })
        }
    }

    println!(
        "Done! Quantized {}/{} tensors.",
        results.len(),
        tensors.len()
    );
    Ok(results)
}

#[cfg(test)]
mod tests {
    use half::f16;

    pub fn dequantize_q8_0(packed: &[u8], block_size: usize, num_elements: usize) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::with_capacity(num_elements);

        for block in packed.chunks(block_size + 2) {
            let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();

            for &byte in &block[2..] {
                let quant = byte as i8;
                out.push(quant as f32 * scale);
                if out.len() == num_elements {
                    // Ignore padding bytes
                    return out;
                }
            }
        }
        out
    }

    pub fn mse(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "MSE: slice lengths must match");
        let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum / a.len() as f32
    }

    mod should_quantize {
        use super::super::*;
        use rand::random;

        #[test]
        fn should_not_quantize_1d_tensor() {
            let tensor_shape: [usize; 1] = [random::<u32>() as usize];

            assert!(!should_quantize("test_tensor", &tensor_shape))
        }

        #[test]
        fn should_not_quantize_norms() {
            let tensor_shape: [usize; 3] = [random::<u32>() as usize; 3];
            for suffix in GGUF_KEEP_F32_SUFFIXES {
                assert!(!should_quantize(
                    format!("test_tensor{}", suffix).as_str(),
                    &tensor_shape
                ))
            }
        }

        #[test]
        fn should_not_quantize_essentials() {
            let tensor_shape: [usize; 3] = [random::<u32>() as usize; 3];
            for suffix in GGUF_KEEP_F32_EXACT {
                assert!(!should_quantize(suffix, &tensor_shape))
            }
        }
    }

    mod quantize_q8_0_blocks {
        use super::super::*;
        use super::*;
        use rand::random;

        #[test]
        #[should_panic]
        fn fail_non_multiple_element_count() {
            let block_size = 10;
            let elements: Vec<f32> = (0..block_size + 3).map(|_| random::<f32>()).collect();

            quantize_q8_0_blocks(&elements, block_size);
        }

        #[test]
        fn quantized_should_dequantize() {
            let block_size = 10;
            let elements: Vec<f32> = (0..block_size * 3).map(|_| random::<f32>()).collect();
            let result = quantize_q8_0_blocks(&elements, block_size);
            let dequantized_elements = dequantize_q8_0(&result, block_size, elements.len());

            assert_eq!(elements.len(), dequantized_elements.len());
        }

        #[test]
        fn low_dequantization_precision() {
            let block_size = 10;
            let elements: Vec<f32> = (0..block_size * 3).map(|_| random::<f32>()).collect();
            let result = quantize_q8_0_blocks(&elements, block_size);
            let dequantized_elements = dequantize_q8_0(&result, block_size, elements.len());
            let error = mse(&elements, &dequantized_elements);

            assert!(-1f32 < error && error < 1f32);
        }
    }

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

    mod safetensors_q8_0_quantization {
        use super::super::*;
        use safetensors::Dtype;
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;

        fn make_f32_safetensors(name: &str, shape: &[usize]) -> (Vec<f32>, Vec<u8>) {
            let num_elements: usize = shape.iter().product();

            // Synthetic Data
            let data: Vec<f32> = (0..num_elements)
                .map(|i| (i as f32 / num_elements as f32) * 2.0 - 1.0)
                .collect();

            let raw: &[u8] = bytemuck::cast_slice(&data);

            let mut tensors: HashMap<String, TensorView> = HashMap::new();
            tensors.insert(
                name.to_string(),
                TensorView::new(Dtype::F32, shape.to_vec(), raw).unwrap(),
            );

            let bytes = safetensors::serialize(&tensors, None).unwrap();

            (data, bytes)
        }

        #[test]
        fn should_not_quantize_dim() {
            let name = "model.layers.0.self_attn.q_proj.weight";
            let shape = vec![2usize];
            let (data, tensor_bytes) = make_f32_safetensors(name, &shape);

            let tensor = SafeTensors::deserialize(&tensor_bytes).unwrap();
            let quantized_tensors = quantize_safetensors_q8_0(&tensor).unwrap();

            assert_eq!(quantized_tensors.len(), 1);
            let qtensor = quantized_tensors.first().unwrap();
            assert_eq!(qtensor.shape, shape);
            assert_eq!(qtensor.name, name);
            assert_eq!(qtensor.data.len(), data.len() * 4);
            assert_eq!(qtensor.ggml_type, GGMLType::F32);
            // todo: Add checks for qtensor.data's values
        }

        #[test]
        fn should_not_quantize_name() {
            let name = "model.norm.weight";
            let shape = vec![2usize, 3usize];
            let (data, tensor_bytes) = make_f32_safetensors(name, &shape);

            let tensor = SafeTensors::deserialize(&tensor_bytes).unwrap();
            let quantized_tensors = quantize_safetensors_q8_0(&tensor).unwrap();

            assert_eq!(quantized_tensors.len(), 1);
            let qtensor = quantized_tensors.first().unwrap();
            assert_eq!(qtensor.shape, shape);
            assert_eq!(qtensor.name, name);
            assert_eq!(qtensor.data.len(), data.len() * 4);
            assert_eq!(qtensor.ggml_type, GGMLType::F32);
            // todo: Add checks for qtensor.data's values
        }

        #[test]
        fn quantize_tensor_block_no_padding() {
            let name = "model.layers.0.self_attn.q_proj.weight";
            let shape = vec![2usize, 16usize];
            let (data, tensor_bytes) = make_f32_safetensors(name, &shape);

            let tensor = SafeTensors::deserialize(&tensor_bytes).unwrap();
            let quantized_tensors = quantize_safetensors_q8_0(&tensor).unwrap();

            assert_eq!(quantized_tensors.len(), 1);
            let qtensor = quantized_tensors.first().unwrap();
            assert_eq!(qtensor.shape, shape);
            assert_eq!(qtensor.name, name);
            assert_eq!(qtensor.data.len(), data.len() + 2);
            assert_eq!(qtensor.ggml_type, GGMLType::Q8_0);
            // todo: Add checks for qtensor.data's values
        }

        #[test]
        fn quantize_tensor_block_padding() {
            let name = "model.layers.0.self_attn.q_proj.weight";
            let shape = vec![3usize, 16usize];
            let (_, tensor_bytes) = make_f32_safetensors(name, &shape);

            let tensor = SafeTensors::deserialize(&tensor_bytes).unwrap();
            let quantized_tensors = quantize_safetensors_q8_0(&tensor).unwrap();

            assert_eq!(quantized_tensors.len(), 1);
            let qtensor = quantized_tensors.first().unwrap();
            assert_eq!(qtensor.shape, shape);
            assert_eq!(qtensor.name, name);
            assert_eq!(qtensor.data.len(), 68);
            assert_eq!(qtensor.ggml_type, GGMLType::Q8_0);
            // todo: Add checks for qtensor.data's values
        }
    }
}
