use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use erissery_core::DType::BF16;
use erissery_core::gguf::kv::{architecture_kvs, tokenizer_kvs};
use erissery_core::gguf::writer::GGUFWriter;
use erissery_core::hf_config::HFConfig;
use erissery_core::inspect_tensors_from_file;
use erissery_core::model_dir::ModelDir;
use erissery_core::quantization::quantize_safetensors_q8_0_from_file;
use erissery_core::tokenizer::{TokenizerInfo, load_tokenizer};
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "erissery",
    about = "Quantize Huggingface Safetensor models into GGUF format",
    version = "0.1.0"
)]
struct Cli {
    /// Path to the input model directory
    #[arg(short, long)]
    input: PathBuf,

    /// Path for the output .gguf file
    #[arg(short, long)]
    output: PathBuf,

    /// Overwrite the output file if it already exists
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Quantization type to apply
    #[arg(short, long, value_enum, default_value_t = QuantType::Q8_0)]
    quant: QuantType,

    /// Print tensor names, shapes, and dtypes without quantizing
    #[arg(long, default_value_t = false)]
    inspect: bool,

    /// Number of threads for CPU quantization (0 = use all logical cores)
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq)]
enum QuantType {
    /// 8-bit block quantization — fastest, ~2x size reduction, near-lossless
    #[value(name = "Q8_0")]
    Q8_0,

    /// 4-bit K-quant with mixed precision — best quality/size tradeoff at 4bpw
    #[value(name = "Q4_K_M")]
    Q4KM,
}

fn print_cli_info(num_threads: usize, input: &Path, output: &Path, quant_type: QuantType) {
    println!("erissery-cli v0.1.0");
    println!("Threads : {num_threads}");
    println!("Input   : {}", input.display());
    println!("Output  : {}", output.display());
    println!("Quant   : {:?}", quant_type);
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .context("Failed to initialize Rayon Threadpool")?;

    print_cli_info(
        rayon::current_num_threads(),
        &cli.input,
        &cli.output,
        cli.quant,
    );

    let model_dir = ModelDir::resolve(&cli.input)?;
    let hf_config = HFConfig::load(&model_dir.config_path)?;
    let tokenizer_info = load_tokenizer(
        &model_dir.tokenizer_path,
        &model_dir.tokenizer_config_path,
        hf_config.vocab_size,
    )?;

    if cli.inspect {
        inspect(&model_dir.safetensors_path, &hf_config, &tokenizer_info)
    } else {
        quantize(
            &model_dir.safetensors_path,
            &cli.output,
            cli.overwrite,
            cli.quant,
            &hf_config,
            &tokenizer_info,
        )
    }
}

fn inspect(input: &Path, config: &HFConfig, tokenizer_info: &TokenizerInfo) -> Result<()> {
    println!("Inspecting {}", input.display());

    let mut kvs = architecture_kvs(config);
    kvs.extend(tokenizer_kvs(tokenizer_info));

    println!("{:<100} {:<20}", "Key", "GGUF Value");
    println!("{}", "-".repeat(121));

    for (key, value) in kvs {
        println!("{:<100} {:>20}", key, value);
    }

    println!("{}", "-".repeat(121));

    println!();

    let tensors = inspect_tensors_from_file(input)?;

    println!(
        "{:<100} {:>10}  {:<10}  Shape",
        "Tensor Name", "Elements", "DType"
    );
    println!("{}", "-".repeat(140));

    let mut has_bf16 = false;

    for tensor in &tensors {
        let shape_str = format!(
            "[{}]",
            tensor
                .shape
                .iter()
                .map(|m| m.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );

        println!(
            "{:<100} {:>10}  {:<10}  {}",
            tensor.name,
            tensor.num_elements(),
            tensor.dtype,
            shape_str
        );

        if tensor.dtype == BF16 {
            has_bf16 = true;
        }
    }

    println!("{}", "─".repeat(140));
    println!("Total tensors: {}", tensors.len());

    if has_bf16 {
        println!("!!! BF16 tensors detected. They must be converted to F32 before quantization !!!")
    }

    Ok(())
}

fn quantize(
    input: &Path,
    output: &Path,
    overwrite: bool,
    quant_type: QuantType,
    config: &HFConfig,
    tokenizer_info: &TokenizerInfo,
) -> Result<()> {
    if output.exists() && !overwrite {
        bail!(
            "{} already exists. Aborting to avoid overwriting",
            output.display()
        );
    }

    match quant_type {
        QuantType::Q8_0 => {
            let mut kvs = architecture_kvs(config);
            kvs.extend(tokenizer_kvs(tokenizer_info));
            let quantized = quantize_safetensors_q8_0_from_file(input)?;

            GGUFWriter::new(kvs, quantized).write(output)?;

            println!("Quantized model written to {}", output.display());
        }
        QuantType::Q4KM => {}
    }

    Ok(())
}
