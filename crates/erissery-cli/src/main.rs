use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::{Path, PathBuf};
use erissery_core::DType::BF16;
use erissery_core::inspect_tensors_from_file;

#[derive(Parser, Debug)]
#[command(
    name = "erissery",
    about = "Quantize Huggingface Safetensor models into GGUF format",
    version = "0.1.0"
)]
struct Cli {

    /// Path to the input .safetensors file (or directory for sharded models)
    #[arg(short, long)]
    input: PathBuf,

    /// Path for the output .gguf file
    #[arg(short, long)]
    output: PathBuf,

    /// Quantization type to apply
    #[arg(short, long, value_enum, default_value_t = QuantType::Q8_0)]
    quant: QuantType,

    /// Print tensor names, shapes, and dtypes without quantizing
    #[arg(long, default_value_t = false)]
    inspect: bool,

    /// Number of threads for CPU quantization (0 = use all logical cores)
    #[arg(long, default_value_t = 0)]
    threads: usize

}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq)]
enum QuantType {

    /// 8-bit block quantization — fastest, ~2x size reduction, near-lossless
    #[value(name="Q8_0")]
    Q8_0,

    /// 4-bit K-quant with mixed precision — best quality/size tradeoff at 4bpw
    #[value(name = "Q4_K_M")]
    Q4KM
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


    print_cli_info(rayon::current_num_threads(), &cli.input, &cli.output, cli.quant);

    if cli.inspect {
        inspect(&cli.input)
    } else {
        quantize(&cli.input, &cli.output, cli.quant)
    }

}


fn inspect(input: &PathBuf) -> Result<()> {
    println!("Inspecting {}", input.display());
    let tensors = inspect_tensors_from_file(&input)?;

    println!(
        "{:<100} {:>10}  {:<10}  {}",
        "Tensor Name", "Elements", "DType", "Shape"
    );
    println!("{}", "─".repeat(140));

    let mut has_bf16 = false;

    for tensor in &tensors {
        let shape_str = format!(
            "[{}]",
            tensor.shape.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(", ")
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

fn quantize(input: &PathBuf, output: &PathBuf, quant_type: QuantType) -> Result<()> {
    println!("\n[quantize mode — not yet implemented]");
    println!("Would quantize: {} → {}", input.display(), output.display());
    println!("Quant type: {quant_type:?}");
    Ok(())
}