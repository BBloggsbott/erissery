use crate::DType;
use crate::ggml::{GGMLType, QuantizedTensor};
use crate::gguf::tensor_names::hf_to_gguf_name;
use crate::quant::constants::BASE_QUANTIZATION_BLOCK_SIZE;
use crate::quant::utils::{dtype_to_f32, pad_to_block_size};
use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

/// Number of elements in one Q4_K superblock
pub const K_QUANT_BLOCK_N_ELEMENTS: usize = 256;

/// Number of sub blocks in each superblock
pub const K_SCALE_COUNT: usize = K_QUANT_BLOCK_N_ELEMENTS / BASE_QUANTIZATION_BLOCK_SIZE;

/// Number of bytes needed to store `K_SCALE_COUNT` scales + `K_SCALE_COUNT` mins, 6 bits each
pub const K_SCALE_SIZE: usize = (2 * K_SCALE_COUNT * 6) / 8;

/// Number of bytes for the 4bit data
pub const K_DATA_SIZE: usize = K_QUANT_BLOCK_N_ELEMENTS / 2;

/// Total Bytes per superblock - Superblock scale (2 bytes) + Superblock min (2 bytes) + Scale Size + Data Size
pub const Q4_K_BLOCK_SIZE: usize = 2 + 2 + K_SCALE_SIZE + K_DATA_SIZE;

fn quantize_q4k_blocks(data: &[f32]) -> Vec<u8> {
    todo!()
}

pub fn quantize_safetensors_q4k_from_file(path: &Path) -> Result<Vec<QuantizedTensor>> {
    // todo: Need to modularize this file read operation. I'm using this again in inspect and
    //  other quantization operations
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

    quantize_safetensors_q4k(&tensors)
}

fn quantize_safetensors_q4k(tensors: &SafeTensors) -> Result<Vec<QuantizedTensor>> {
    let mut results: Vec<QuantizedTensor> = Vec::new();

    for (name, view) in tensors.tensors() {
        let gguf_name = hf_to_gguf_name(name.as_str())
            .ok_or_else(|| anyhow!("unmapped tensor name: {}", name))?;
        let dtype = DType::from(view.dtype());
        let raw_bytes = view.data();
        let shape = view.shape().to_vec();
        let num_elements: usize = shape.iter().product();

        if let DType::I32 | DType::I64 = &dtype {
            let ggml_type = GGMLType::try_from(dtype)?;
            results.push(QuantizedTensor {
                name,
                gguf_name,
                shape,
                data: view.data().to_vec(),
                num_elements,
                ggml_type,
            });
            continue;
        }

        let f32_data: Vec<f32> = dtype_to_f32(raw_bytes, &dtype)?;

        // todo: Verify and Include should quantize check

        let padded_data = pad_to_block_size(&f32_data, K_QUANT_BLOCK_N_ELEMENTS);
        let quantized_data = quantize_q4k_blocks(&padded_data);

        results.push(QuantizedTensor {
            name,
            gguf_name,
            shape,
            data: quantized_data,
            num_elements,
            ggml_type: GGMLType::Q4K,
        });
    }

    Ok(results)
}
