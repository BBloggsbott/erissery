use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct HFConfig {
    #[serde(rename = "_name_or_path", default)]
    pub model_name: String,

    pub model_type: String,

    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_attention_heads: u32,
    pub num_hidden_layers: u32,
    num_key_value_heads: Option<u32>,
    pub vocab_size: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_eps: f32,
    rope_theta: Option<f32>,

    #[serde(flatten)]
    pub extras: HashMap<String, Value>,
}

impl HFConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;

        let mut config: HFConfig = serde_json::from_str(&text)
            .with_context(|| format!("Parsing {} as HF config.json", path.display()))?;

        match config.model_type.as_str() {
            "llama" | "mistral" | "qwen2" => {}
            other => bail!(
                "model_type '{}' is not yet supported (only llama/mistral/qwen2 architecture metadata is mapped)",
                other
            ),
        }

        if config.model_name.is_empty() {
            let model_name = path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            config.model_name = model_name.to_string();
        }

        Ok(config)
    }

    pub fn num_key_value_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or_else(|| {
            println!("rope_theta not found in HFConfig. Defaulting to 10000.0");
            10000.0
        })
    }
}
