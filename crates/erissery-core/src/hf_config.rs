use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Holds the information from the huggingface model's config file
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
    /// Loads the Huggingface model's config from a file.
    ///
    /// Currently only supports `model_type` "llama", "mistral" and "qwen2".
    /// If config does not have the model name, it defaults to the model directory's name.
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

    /// Returns the value of the number of key value heads if it is set.
    /// If the value is not set, defaults to the number of attention heads.
    pub fn num_key_value_heads(&self) -> u32 {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Returns the value of `rope_theta` if it was set in the config.
    /// If not, it defaults to `10000.0`.
    pub fn rope_theta(&self) -> f32 {
        self.rope_theta.unwrap_or_else(|| {
            println!("rope_theta not found in HFConfig. Defaulting to 10000.0");
            10000.0
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn make_hf_config(num_key_value_heads: Option<u32>, rope_theta: Option<f32>) -> HFConfig {
        HFConfig {
            model_name: "test_model".to_string(),
            model_type: "test_model_type".to_string(),
            hidden_size: 12,
            intermediate_size: 11,
            num_attention_heads: 89,
            num_hidden_layers: 91,
            num_key_value_heads,
            vocab_size: 70,
            max_position_embeddings: 33,
            rms_norm_eps: 1.24f32,
            rope_theta,
            extras: HashMap::new(),
        }
    }

    // todo: Test the load function

    #[test]
    fn test_num_key_value_heads_with_value() {
        let hf_config = make_hf_config(Some(99), Some(1f32));

        assert_eq!(hf_config.num_key_value_heads(), 99);
    }

    #[test]
    fn test_num_key_value_heads_without_value() {
        let hf_config = make_hf_config(None, Some(1f32));

        assert_eq!(
            hf_config.num_key_value_heads(),
            hf_config.num_attention_heads
        );
    }

    #[test]
    fn test_rope_theta_with_value() {
        let hf_config = make_hf_config(Some(99), Some(1f32));

        assert_eq!(hf_config.rope_theta(), 1f32);
    }

    #[test]
    fn test_rope_theta_without_value() {
        let hf_config = make_hf_config(None, None);

        assert_eq!(hf_config.rope_theta(), 10000.0);
    }
}
