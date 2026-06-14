pub mod gguf;
pub mod hf_config;
pub mod model_dir;
pub mod quantization;
pub mod tokenizer;

use anyhow::{Context, Result};
use memmap2::Mmap;
use safetensors::{SafeTensors, tensor};
use std::fmt::Display;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    Other(String),
}

impl From<&str> for DType {
    fn from(value: &str) -> Self {
        match value {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            other => DType::Other(other.to_string()),
        }
    }
}

impl From<tensor::Dtype> for DType {
    fn from(value: tensor::Dtype) -> Self {
        match value {
            tensor::Dtype::F32 => DType::F32,
            tensor::Dtype::F16 => DType::F16,
            tensor::Dtype::BF16 => DType::BF16,
            tensor::Dtype::I32 => DType::I32,
            tensor::Dtype::I64 => DType::I64,
            other => DType::Other(other.to_string()),
        }
    }
}

impl Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::Other(s) => s.as_str(),
        };
        write!(f, "{}", str)
    }
}

impl DType {
    pub fn needs_conversion(&self) -> bool {
        matches!(self, DType::BF16)
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub byte_size: usize,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub fn inspect_tensors_from_file(path: &Path) -> Result<Vec<TensorInfo>> {
    // todo: Need to modularize this file read operation. I'm using this again in quantization
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

    inspect_tensors(&tensors)
}

pub fn inspect_tensors(tensors: &SafeTensors) -> Result<Vec<TensorInfo>> {
    let mut infos: Vec<TensorInfo> = Vec::new();

    for (name, view) in tensors.tensors() {
        let dtype = DType::from(view.dtype());

        infos.push(TensorInfo {
            name,
            shape: view.shape().to_vec(),
            dtype,
            byte_size: view.data().len(),
        })
    }

    Ok(infos)
}
