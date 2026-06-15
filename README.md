# Erissery

A command-line toolkit for quantizing large language models (LLMs). Converts HuggingFace models in SafeTensor format to [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) format for use with GGML-based runtimes such as llama.cpp.

> **Early Development:** This project is under active development. Expect breaking changes, missing features, and rough edges.

## Current Status

Only Q8\_0 quantization is supported right now — Q4\_K\_M and other strategies are in progress. The tool has only been tested with Qwen2.5 0.5B Instruct; testing and support for more models is ongoing. Sharded models (weights split across multiple `.safetensors` files) are not supported yet. Normalization layers and embeddings are always kept in F32; all other multi-dimensional tensors are quantized.

## Building

Clone the repository and build with Cargo:

```sh
git clone https://github.com/vishalramesh/erissery.git
cd erissery
cargo build --release
```

The compiled binary is placed at `target/release/erissery` (or `target\release\erissery.exe` on Windows).

To run directly without installing:

```sh
cargo run --release -- [OPTIONS]
```

## Usage

```
erissery --input <MODEL_DIR> --output <OUTPUT_FILE> [OPTIONS]
```

### Required arguments

| Argument | Description |
|---|---|
| `-i`, `--input <PATH>` | Path to the HuggingFace model directory |
| `-o`, `--output <PATH>` | Path for the output `.gguf` file |

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--quant <TYPE>` | `Q8_0` | Quantization type (`Q8_0`) |
| `--threads <N>` | `0` (all cores) | Number of CPU threads to use |
| `--overwrite` | false | Overwrite the output file if it already exists |
| `--inspect` | — | Print tensor names, shapes, and dtypes without quantizing |

### Examples

Quantize a model with Q8\_0 using all available CPU cores:

```sh
erissery -i ./qwen2.5-0.5b-instruct -o ./qwen2.5-0.5b-instruct-q8_0.gguf
```

Quantize using 8 threads:

```sh
erissery -i ./qwen2.5-0.5b-instruct -o ./qwen2.5-0.5b-instruct-q8_0.gguf --threads 8
```

Inspect a model's tensors without quantizing:

```sh
erissery -i ./qwen2.5-0.5b-instruct -o /dev/null --inspect
```

## Input Requirements

The model directory must contain:

- `model.safetensors` — model weights (single file, not sharded)
- `config.json` — HuggingFace model configuration
- `tokenizer.json` — tokenizer vocabulary and merge rules
- `tokenizer_config.json` — tokenizer metadata

Supported input dtypes: F32, F16, BF16. I32 and I64 tensors are passed through as-is.

Supported model architectures: `llama`, `mistral`, `qwen2`.
