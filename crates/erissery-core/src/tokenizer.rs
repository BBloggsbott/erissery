use anyhow::{Result, bail};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MergeEntry {
    Pair(String, String),
    Combined(String),
}

impl MergeEntry {
    fn to_gguf_string(&self) -> String {
        match self {
            MergeEntry::Pair(a, b) => format!("{a} {b}"),
            MergeEntry::Combined(s) => s.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

#[derive(Debug, Deserialize)]
struct TokenizerModel {
    #[serde(rename = "type")]
    model_type: String,

    vocab: HashMap<String, u32>,

    #[serde(default)]
    merges: Vec<MergeEntry>,
}

#[derive(Debug, Deserialize)]
struct TokenizerJson {
    model: TokenizerModel,

    #[serde(default)]
    added_tokens: Vec<AddedToken>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum TokenRef {
    Plain(String),
    Detailed { content: String },
}

impl TokenRef {
    fn content(&self) -> &str {
        match self {
            TokenRef::Plain(s) => s,
            TokenRef::Detailed { content } => content,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct TokenizerConfigJson {
    #[serde(default)]
    chat_template: Option<String>,

    #[serde(default)]
    bos_token: Option<TokenRef>,

    #[serde(default)]
    eos_token: Option<TokenRef>,

    #[serde(default)]
    pad_token: Option<TokenRef>,

    #[serde(default)]
    unk_token: Option<TokenRef>,
}

pub struct TokenizerInfo {
    pub tokens: Vec<String>,
    pub token_types: Vec<i32>,
    pub merges: Vec<String>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub unk_token_id: Option<u32>,
    pub pad_token_id: Option<u32>,
    pub chat_template: Option<String>,
}

// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#ggml
const GGUF_TOKEN_TYPE_NORMAL: i32 = 1;
const GGUF_TOKEN_TYPE_CONTROL: i32 = 3;

fn lookup_id(vocab: &HashMap<String, u32>, token_ref: &Option<TokenRef>, added_tokens: &Vec<AddedToken>) -> Option<u32> {
    let content = token_ref.as_ref()?.content();

    // Check base vocab first, then added_tokens
    vocab.get(content).copied().or_else(|| {
        added_tokens.iter().find(|t| t.content == content).map(|t| t.id)
    })
}

pub fn load_tokenizer(
    tokenizer_path: &Path,
    tokenizer_config_path: &Path,
    vocab_size: u32,
) -> Result<TokenizerInfo> {
    let text = fs::read_to_string(tokenizer_path)?;
    let tokenizer: TokenizerJson = serde_json::from_str(text.as_str())?;

    if tokenizer.model.model_type != "BPE" {
        bail!(
            "tokenizer.json model.type is '{}' — only BPE tokenizers are currently supported",
            tokenizer.model.model_type
        );
    }

    let max_id = tokenizer
        .model
        .vocab
        .values()
        .copied()
        .max()
        .unwrap_or(0u32);
    let mut tokens = vec![String::new(); (max_id + 1) as usize];
    let mut token_types = vec![GGUF_TOKEN_TYPE_NORMAL; (max_id + 1) as usize];

    for (token_str, &id) in &tokenizer.model.vocab {
        tokens[id as usize] = token_str.clone()
    }


    for added in &tokenizer.added_tokens {
        let idx = added.id as usize;
        if idx >= tokens.len() {
            tokens.resize(idx + 1, String::new());
            token_types.resize(idx + 1, GGUF_TOKEN_TYPE_NORMAL);
        }
        tokens[idx] = added.content.clone();
        if added.special {
            token_types[idx] = GGUF_TOKEN_TYPE_CONTROL;
        }
    }

    let vocab_size = vocab_size as usize;
    if tokens.len() < vocab_size {
        tokens.resize(vocab_size, String::new());
        token_types.resize(vocab_size, GGUF_TOKEN_TYPE_NORMAL);
    }

    let merges: Vec<String> = tokenizer
        .model
        .merges
        .iter()
        .map(MergeEntry::to_gguf_string)
        .collect();

    let tokenizer_config = if tokenizer_config_path.exists() {
        let tokenizer_config_text = fs::read_to_string(tokenizer_config_path)?;
        serde_json::from_str(&tokenizer_config_text)?
    } else {
        TokenizerConfigJson::default()
    };

    Ok(TokenizerInfo {
        tokens,
        token_types,
        merges,
        bos_token_id: lookup_id(&tokenizer.model.vocab, &tokenizer_config.bos_token, &tokenizer.added_tokens),
        eos_token_id: lookup_id(&tokenizer.model.vocab, &tokenizer_config.eos_token, &tokenizer.added_tokens),
        unk_token_id: lookup_id(&tokenizer.model.vocab, &tokenizer_config.unk_token, &tokenizer.added_tokens),
        pad_token_id: lookup_id(&tokenizer.model.vocab, &tokenizer_config.pad_token, &tokenizer.added_tokens),
        chat_template: tokenizer_config.chat_template,
    })
}
