use crate::DType;
use anyhow::anyhow;
use std::fmt::{Display, Formatter};

/// Denotes the GGML Type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GGMLType {
    F32 = 0,
    I32 = 26,
    I64 = 27,
    Q8_0 = 8,
    Q4K = 12,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q4K => write!(f, "Q4_K"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::I64 => write!(f, "I64"),
        }
    }
}

impl TryFrom<DType> for GGMLType {
    type Error = anyhow::Error;

    fn try_from(value: DType) -> Result<Self, Self::Error> {
        match value {
            DType::F32 => Ok(GGMLType::F32),
            DType::I32 => Ok(GGMLType::I32),
            DType::I64 => Ok(GGMLType::I64),
            other => Err(anyhow!("Unsupported dtype: {other:?}")),
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
