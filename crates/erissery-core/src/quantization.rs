use std::fmt::{Display, Formatter};
use crate::DType;
use crate::gguf::tensor_names::hf_to_ggf_name;
use anyhow::{Context, Result, anyhow, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

const Q8_0_BLOCK_SIZE: usize = 32;

const GGUF_KEEP_F32_SUFFIXES: [&str; 4] = [
    ".norm.weight",
    "_norm.weight",
    ".norm_1.weight",
    ".norm_2.weight",
];

const GGUF_KEEP_F32_EXACT: [&str; 2] = [
    "token_embd.weight",
    "output_norm.weight",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    Q8_0 = 8,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::Q8_0 => write!(f, "Q8_0")
        }
    }
}

pub struct QuantizedTensor {
    pub name: String,
    pub gguf_name: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub num_elements: usize,
    pub ggml_type: GGMLType,
}

fn should_quantize(tensor_name: &str, shape: &[usize]) -> bool {

    if shape.len() == 1 {
        return false
    }

    if GGUF_KEEP_F32_EXACT.iter().any(|&n| tensor_name == n) {
        return false
    }

    if GGUF_KEEP_F32_SUFFIXES.iter().any(|&n| tensor_name.ends_with(n)) {
        return false
    }

    true

}

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

fn f32_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn f16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f16::from_bits(bits).to_f32()
        })
        .collect()
}

fn bf16_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            bf16::from_bits(bits).to_f32()
        })
        .collect()
}

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
            let bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();

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
                ggml_type: GGMLType::Q8_0
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
