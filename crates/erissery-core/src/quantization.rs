use crate::DType;
use anyhow::{Context, Result, bail};
use half::{bf16, f16};
use memmap2::Mmap;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use safetensors::SafeTensors;
use std::fs::File;
use std::path::Path;

const Q8_0_BLOCK_SIZE: usize = 32;

struct QuantizedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
    pub num_elements: usize,
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
    let num_chunks = block_size + 4;
    let mut output = vec![0u8; num_blocks * num_chunks];

    output
        .par_chunks_mut(num_chunks)
        .zip(elements.par_chunks(block_size))
        .for_each(|(out_block, in_block)| {
            let amax = in_block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            let scale = if amax == 0.0 { 0.0f32 } else { amax / 127.0f32 };

            out_block[0..4].copy_from_slice(&scale.to_le_bytes());

            let inv_scale = if scale == 0.0 { 0.0f32 } else { 1.0f32 / scale };

            for (i, &val) in in_block.iter().enumerate() {
                let q = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;

                out_block[4 + 1] = q as u8;
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

        print!("\n {name:<100} {num_elements:>10}");
        if f32_data.len() != padded_data.len() {
            print!(" (padded {} > {})", f32_data.len(), padded_data.len());
        }
        println!();

        let quantized_bytes = quantize_q8_0_blocks(&padded_data, Q8_0_BLOCK_SIZE);

        results.push(QuantizedTensor {
            name,
            shape,
            data: quantized_bytes,
            num_elements,
        })
    }

    println!(
        "Done! Quantized {}/{} tensors.",
        results.len(),
        tensors.len()
    );
    Ok(results)
}
