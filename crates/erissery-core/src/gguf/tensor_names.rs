// Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md#standardized-tensor-names
pub fn hf_to_ggf_name(hf_name: &str) -> Option<String> {
    let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

    if name == "embed_tokens.weight" {
        return Some("token_embd.weight".to_string());
    }
    if name == "norm.weight" {
        return Some("output_norm.weight".to_string());
    }
    if hf_name == "lm_head.weight" {
        return Some("output.weight".to_string());
    }

    let rest = name.strip_prefix("layers.")?;
    let dot = rest.find(".")?;
    let layer_num = &rest[..dot];
    let suffix = &rest[dot + 1..];

    // println!("Mapping name for {} {}", layer_num, suffix);

    let gguf_suffix = match suffix {
        // Attention
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "self_attn.o_proj.bias" => "attn_output.bias",
        // Multi layer perceptrons
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        // Norms
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        // Unknown — caller decides what to do (log a warning, skip, etc.)
        other => {
            eprintln!("warn: unmapped tensor suffix: {}", other);
            return None;
        }
    };

    Some(format!("blk.{}.{}", layer_num, gguf_suffix))
}
