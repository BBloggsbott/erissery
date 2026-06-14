use crate::gguf::kv::GGUFValue;
use crate::gguf::types::{
    GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32, GGUF_TYPE_INT32, GGUF_TYPE_STRING, GGUF_TYPE_UINT32,
    GGUF_TYPE_UINT64,
};
use crate::quantization::QuantizedTensor;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

const GGML_TYPE_Q8_0: u32 = 8;

const GGUF_LE_MAGIC_NUMBER: u32 = 0x46554747;
const GGUF_VERSION: u32 = 8;

fn write_gguf_str<W: Write>(w: &mut W, s: &str) -> Result<()> {
    w.write_u64::<LittleEndian>(s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn write_kv<W: Write>(w: &mut W, key: &str, value: &GGUFValue) -> Result<()> {
    write_gguf_str(w, key)?;

    match value {
        GGUFValue::U32(val) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_UINT32)?;
            w.write_u32::<LittleEndian>(*val)?;
        }
        GGUFValue::U64(val) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_UINT64)?;
            w.write_u64::<LittleEndian>(*val)?;
        }
        GGUFValue::F32(val) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_FLOAT32)?;
            w.write_f32::<LittleEndian>(*val)?;
        }
        GGUFValue::String(s) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_STRING)?;
            write_gguf_str(w, s)?;
        }
        GGUFValue::ArrayString(items) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_ARRAY)?;
            w.write_u32::<LittleEndian>(GGUF_TYPE_STRING)?;
            w.write_u64::<LittleEndian>(items.len() as u64)?;
            for item in items {
                write_gguf_str(w, item)?;
            }
        }
        GGUFValue::ArrayI32(items) => {
            w.write_u32::<LittleEndian>(GGUF_TYPE_ARRAY)?;
            w.write_u32::<LittleEndian>(GGUF_TYPE_INT32)?;
            w.write_u64::<LittleEndian>(items.len() as u64)?;

            for &item in items {
                w.write_i32::<LittleEndian>(item)?;
            }
        }
    }

    Ok(())
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

fn align_write_pos<W: Write + Seek>(w: &mut W, alignment: u64) -> Result<()> {
    let pos = w.stream_position()?;
    let aligned_pos = align_up(pos, alignment);
    let padding = (aligned_pos - pos) as usize;
    if padding > 0 {
        w.write_all(&vec![0u8; padding])?;
    }

    Ok(())
}

fn compute_tensor_data_offsets(tensors: &Vec<QuantizedTensor>, alignment: u64) -> Vec<u64> {
    let mut offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut running_offset: u64 = 0;

    for tensor in tensors {
        offsets.push(running_offset);
        running_offset += tensor.data.len() as u64;

        running_offset = align_up(running_offset, alignment);
    }
    offsets
}

fn write_tensor_descriptors<W: Write>(
    w: &mut W,
    tensors: &[QuantizedTensor],
    offsets: &[u64],
) -> Result<()> {
    for (tensor, &offset) in tensors.iter().zip(offsets) {
        // name
        write_gguf_str(w, tensor.name.as_str())?;

        // n_dimensions
        w.write_u32::<LittleEndian>(tensor.shape.len() as u32)?;
        // dimensions
        for &dim in &tensor.shape {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }

        // type
        w.write_u32::<LittleEndian>(GGML_TYPE_Q8_0)?;

        // offset
        w.write_u64::<LittleEndian>(offset)?;
    }

    Ok(())
}

fn write_tensor_data<W: Write + Seek>(
    w: &mut W,
    tensors: &[QuantizedTensor],
    offsets: &[u64],
    alignment: u64,
) -> Result<()> {
    let data_section_start = w.stream_position()?;

    for (i, tensor) in tensors.iter().enumerate() {
        w.write_all(&tensor.data)?;

        let written_so_far = w.stream_position()? - data_section_start;

        let expected_pos = if i + 1 < offsets.len() {
            offsets[i + 1]
        } else {
            align_up(written_so_far, alignment)
        };

        let padding = (expected_pos - written_so_far) as usize;
        if padding > 0 {
            w.write_all(&vec![0u8; padding])?;
        }
    }

    Ok(())
}

pub struct GGUFWriter {
    kvs: Vec<(String, GGUFValue)>,
    tensors: Vec<QuantizedTensor>,
    alignment: u64,
}

impl GGUFWriter {
    pub fn new(kvs: Vec<(String, GGUFValue)>, tensors: Vec<QuantizedTensor>) -> Self {
        Self {
            kvs,
            tensors,
            alignment: 32,
        }
    }

    // Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#file-structure
    pub fn write(&self, path: &Path) -> Result<()> {
        let file = File::create(path)
            .with_context(|| format!("creating output file {}", path.display()))?;

        let mut w = BufWriter::new(file);

        w.write_u32::<LittleEndian>(GGUF_LE_MAGIC_NUMBER)?;
        w.write_u32::<LittleEndian>(GGUF_VERSION)?;
        w.write_u64::<LittleEndian>(self.tensors.len() as u64)?;
        w.write_u64::<LittleEndian>(self.kvs.len() as u64)?;

        for (key, val) in &self.kvs {
            write_kv(&mut w, key, val)?;
        }

        // Computing offsets for tensor data
        let offsets = compute_tensor_data_offsets(&self.tensors, self.alignment);

        // Writing the tensor_info section
        write_tensor_descriptors(&mut w, &self.tensors, &offsets)?;

        // Padding before tensor data section
        align_write_pos(&mut w, self.alignment)?;

        // Tensor Data Section
        write_tensor_data(&mut w, &self.tensors, &offsets, self.alignment)?;

        w.flush()?;

        Ok(())
    }
}
