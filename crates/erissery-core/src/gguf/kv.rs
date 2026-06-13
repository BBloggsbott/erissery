use crate::hf_config::HFConfig;
use std::fmt::{Display, Formatter};

pub enum GGUFValue {
    U32(u32),
    U64(u64),
    F32(f32),
    String(String),
}

impl Display for GGUFValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFValue::U32(v) => write!(f, "{}", v),
            GGUFValue::U64(v) => write!(f, "{}", v),
            GGUFValue::F32(v) => write!(f, "{}", v),
            GGUFValue::String(s) => write!(f, "{}", s),
        }
    }
}

//Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs
pub fn architecture_kvs(config: &HFConfig) -> Vec<(String, GGUFValue)> {
    let arch = "llama";

    let kvs: Vec<(String, GGUFValue)> = vec![
        (
            "general.architecture".to_string(),
            GGUFValue::String(arch.to_string()),
        ),
        (
            "general.name".to_string(),
            GGUFValue::String(config.model_name.to_string()),
        ),
        (
            format!("{arch}.context_length"),
            GGUFValue::U32(config.max_position_embeddings),
        ),
        (
            format!("{arch}.embedding_length"),
            GGUFValue::U32(config.hidden_size),
        ),
        (
            format!("{arch}.block_count"),
            GGUFValue::U32(config.num_hidden_layers),
        ),
        (
            format!("{arch}.feed_forward_length"),
            GGUFValue::U32(config.intermediate_size),
        ),
        (
            format!("{arch}.attention.head_count"),
            GGUFValue::U32(config.num_attention_heads),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            GGUFValue::U32(config.num_key_value_heads()),
        ),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GGUFValue::F32(config.rms_norm_eps),
        ),
        (
            format!("{arch}.rope.dimension_count"),
            GGUFValue::U32(config.hidden_size / config.num_attention_heads),
        ),
        (
            format!("{arch}.rope.freq_base"),
            GGUFValue::F32(config.rope_theta()),
        ),
    ];

    kvs
}
