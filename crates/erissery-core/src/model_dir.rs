use anyhow::{Result, bail};
use std::path::{Path, PathBuf};

/// Files in the Huggingface model directory that are needed
/// to quantize the model.
pub struct ModelDir {
    /// Safetensors file that contains the moels weights and tensor metadata
    /// Currently supports only unsharded models.
    pub safetensors_path: PathBuf, // todo - Handle sharded models
    /// Path to the `config.json` file in the model directory
    pub config_path: PathBuf,
    /// Path to the `tokenizer.json` file in the model directory
    pub tokenizer_path: PathBuf,
    /// Path to the `tokenizer_config.json` file in the model directory
    pub tokenizer_config_path: PathBuf,
}

impl ModelDir {
    /// Resolve the paths to the files inside the Huggingface model directory that are needed
    /// to quantize the model.
    ///
    /// Resolves the paths for the `.safetensors` file (supports unsharded models only
    /// and fails for sharded models), `config.json`, `tokenizer.json`, `tokenizer_config.json`.
    ///
    /// ```no_run
    /// use std::path::{Path, PathBuf};
    /// use erissery_core::model_dir::ModelDir;
    ///
    /// let path = PathBuf::from("/hf-model");
    /// let model_dir = ModelDir::resolve(&path).unwrap();
    ///
    /// assert_eq!(model_dir.safetensors_path, path.join("model.safetensors"));
    /// assert_eq!(model_dir.config_path, path.join("config.json"));
    /// assert_eq!(model_dir.tokenizer_path, path.join("tokenizer.json"));
    /// assert_eq!(model_dir.tokenizer_config_path, path.join("tokenizer_config.json"));
    /// ```
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
