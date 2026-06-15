use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

pub struct ModelDir {
    pub safetensors_path: PathBuf, // todo - Handle sharded models
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub tokenizer_config_path: PathBuf,
}

impl ModelDir {
    pub fn resolve(dir: &Path) -> Result<Self> {
        println!("Resolving huggingface model directory");

        if !dir.is_dir() {
            bail!("Provided path `{}` is not a directory", dir.display());
        }

        let safetensors_path = dir.join("model.safetensors");
        if !safetensors_path.exists() {
            if dir.join("model.safetensors.index.json").exists() {
                bail!(
                    "{} appears to be sharded (model.safetensors.index.json was found). Sharded models are not yet supported",
                    dir.display()
                );
            }

            bail!("{} does not contain `model.safetensors`", dir.display());
        }

        let config_path = dir.join("config.json");
        if !config_path.exists() {
            bail!("'{}' does not contain config.json", dir.display());
        }

        let tokenizer_path = dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            bail!("'{}' does not contain tokenizers.json", dir.display())
        }

        let tokenizer_config_path = dir.join("tokenizer_config.json");

        Ok(Self {
            safetensors_path,
            config_path,
            tokenizer_path,
            tokenizer_config_path,
        })
    }
}
