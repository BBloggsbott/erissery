use crate::hf_config::HFConfig;
use crate::tokenizer::TokenizerInfo;
use std::fmt::{Display, Formatter};

const GGUF_QUANTIZATION_VERSION: u32 = 2;

pub enum GGUFValue {
    U32(u32),
    U64(u64),
    F32(f32),
    String(String),
    ArrayString(Vec<String>),
    ArrayI32(Vec<i32>),
}

impl Display for GGUFValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            GGUFValue::U32(v) => write!(f, "{}", v),
            GGUFValue::U64(v) => write!(f, "{}", v),
            GGUFValue::F32(v) => write!(f, "{}", v),
            GGUFValue::String(s) => {
                if s.len() > 30 {
                    write!(f, "{:.10}... [{} chars]", s, s.len())
                } else {
                    write!(f, "{}", s)
                }
            }
            GGUFValue::ArrayString(v) => write!(f, "[{} strings]", v.len()),
            GGUFValue::ArrayI32(v) => write!(f, "[{} i32s]", v.len()),
        }
    }
}

//Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs
pub fn architecture_kvs(config: &HFConfig) -> Vec<(String, GGUFValue)> {
    let arch = match config.model_type.as_str() {
        "qwen2" => "qwen2",
        "llama" | "mistral" => "llama",
        other => other,
    };

    let kvs: Vec<(String, GGUFValue)> = vec![
        (
            "general.architecture".to_string(),
            GGUFValue::String(arch.to_string()),
        ),
        (
            "general.quantization_version".to_string(),
            GGUFValue::U32(GGUF_QUANTIZATION_VERSION),
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
        (
            format!("{arch}.vocab_size"),
            GGUFValue::U32(config.vocab_size),
        ),
    ];

    kvs
}

pub fn tokenizer_kvs(tokenizer_info: &TokenizerInfo) -> Vec<(String, GGUFValue)> {
    let mut kvs = vec![
        (
            "tokenizer.ggml.model".to_string(),
            GGUFValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GGUFValue::ArrayString(tokenizer_info.tokens.clone()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            GGUFValue::ArrayI32(tokenizer_info.token_types.clone()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            GGUFValue::ArrayString(tokenizer_info.merges.clone()),
        ),
    ];

    if let Some(id) = tokenizer_info.bos_token_id {
        kvs.push((
            "tokenizer.ggml.bos_token_id".to_string(),
            GGUFValue::U32(id),
        ));
    }
    if let Some(id) = tokenizer_info.eos_token_id {
        kvs.push((
            "tokenizer.ggml.eos_token_id".to_string(),
            GGUFValue::U32(id),
        ));
    }
    if let Some(id) = tokenizer_info.unk_token_id {
        kvs.push((
            "tokenizer.ggml.unknown_token_id".to_string(),
            GGUFValue::U32(id),
        ));
    }
    if let Some(id) = tokenizer_info.pad_token_id {
        kvs.push((
            "tokenizer.ggml.padding_token_id".to_string(),
            GGUFValue::U32(id),
        ));
    }
    if let Some(template) = &tokenizer_info.chat_template {
        kvs.push((
            "tokenizer.chat_template".to_string(),
            GGUFValue::String(template.clone()),
        ));
    }

    kvs
}
