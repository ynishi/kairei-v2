//! CLI for Kairei-v2 AgentCulture Framework

use clap::Parser;
use kairei_cli::{ModelType, commands, error::CliError};

#[derive(Parser)]
#[command(name = "kairei")]
#[command(about = "CLI for Kairei-v2 AgentCulture Framework", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand)]
enum LoraCommands {
    /// Add a LoRA model to the registry
    Add {
        /// Source (file path or URL)
        source: String,

        /// LoRA name (defaults to filename)
        #[arg(long, short = 'n')]
        name: Option<String>,

        /// Base model this LoRA is for
        #[arg(long, short = 'b', default_value = "stories15M")]
        base_model: Option<String>,

        /// Description
        #[arg(long, short = 'd')]
        description: Option<String>,
    },

    /// List all registered LoRA models
    List,

    /// Show information about a LoRA model
    Show {
        /// LoRA name
        name: String,
    },

    /// Remove a LoRA model from the registry
    Remove {
        /// LoRA name
        name: String,

        /// Keep the file (only remove from registry)
        #[arg(long)]
        keep_file: bool,
    },

    /// Convert PEFT format to candle-lora format
    Convert {
        /// PEFT directory path (containing adapter_config.json and adapter_model.safetensors)
        peft_dir: String,

        /// Output safetensors file path
        #[arg(long, short = 'o')]
        output: String,

        /// Prefix for the converted tensors (default: "lora_llama")
        #[arg(long, short = 'p', default_value = "lora_llama")]
        prefix: String,
    },
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Initialize a new Kairei project
    Init,
    /// Run the Kairei agent
    Run,
    /// Chat with the Kairei agent
    Chat {
        /// Optional initial message
        #[arg(long, short)]
        message: Option<String>,
        /// Run once and exit (non-interactive mode)
        #[arg(long)]
        once: bool,
        /// Use candle backend
        #[arg(long)]
        candle: bool,
        /// LoRA models to apply (can be specified multiple times)
        #[arg(long, action = clap::ArgAction::Append)]
        lora: Vec<String>,
        /// Base model to use
        #[arg(long, short = 'b')]
        base_model: Option<String>,
        /// Model type to use
        #[arg(long, value_enum, default_value = "llama2c")]
        model_type: ModelType,
        /// Path to tokenizer file (defaults to auto-detect based on model)
        #[arg(long, short = 't')]
        tokenizer: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(long, default_value = "100")]
        max_tokens: usize,
    },
    /// Setup models and dependencies
    Setup {
        /// List available models
        #[arg(long)]
        list: bool,
        /// Model name to download (e.g., stories15M, stories42M, stories110M)
        #[arg(long, short)]
        model: Option<String>,
        /// Force re-download even if file exists
        #[arg(long, short)]
        force: bool,
        /// Custom name for the model (used with HuggingFace repo ID)
        #[arg(value_name = "NAME")]
        name: Option<String>,
        /// HuggingFace repository ID (e.g., meta-llama/Llama-3.2-1B)
        #[arg(value_name = "REPO_ID")]
        repo_id: Option<String>,
    },
    /// LoRA culture management
    Lora {
        #[command(subcommand)]
        command: Option<LoraCommands>,
    },
    /// Train a LoRA model
    Train {
        /// Path to training data JSON file (required)
        #[arg(long, short = 'd')]
        train_data: String,

        /// Base model name (registered name or HuggingFace ID)
        #[arg(long, short = 'm')]
        model: String,

        /// Number of training epochs
        #[arg(long, short = 'e', default_value = "5")]
        epochs: u32,

        /// LoRA rank (recommended: 8-16 for better learning)
        #[arg(long, default_value = "8")]
        lora_r: u32,

        /// LoRA alpha (typically 2x rank)
        #[arg(long, default_value = "16")]
        lora_alpha: u32,

        /// Output directory for trained model
        #[arg(long, short = 'o', default_value = "./lora_output")]
        output_dir: String,

        /// Batch size per device
        #[arg(long, default_value = "1")]
        batch_size: u32,

        /// Learning rate
        #[arg(long, default_value = "5e-4")]
        learning_rate: f32,

        /// Run test after training
        #[arg(long)]
        test: bool,
    },
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Init) => {
            commands::run_init().await?;
        }
        Some(Commands::Run) => {
            commands::run_run().await?;
        }
        Some(Commands::Chat {
            message,
            once,
            candle,
            lora,
            base_model,
            model_type,
            tokenizer,
            max_tokens,
        }) => {
            commands::run_chat(
                message.clone(),
                *once,
                *candle,
                lora.clone(),
                base_model.clone(),
                *model_type,
                tokenizer.clone(),
                *max_tokens,
            )
            .await?;
        }
        Some(Commands::Setup {
            list,
            model,
            force,
            name,
            repo_id,
        }) => {
            commands::run_setup(*list, model.clone(), *force, name.clone(), repo_id.clone())
                .await?;
        }
        Some(Commands::Lora { command }) => match command {
            Some(LoraCommands::Add {
                source,
                name,
                base_model,
                description,
            }) => {
                commands::lora_add(
                    source,
                    name.clone(),
                    base_model.clone(),
                    description.clone(),
                )
                .await?;
            }
            Some(LoraCommands::List) => {
                commands::lora_list().await?;
            }
            Some(LoraCommands::Show { name }) => {
                commands::lora_show(name).await?;
            }
            Some(LoraCommands::Remove { name, keep_file }) => {
                commands::lora_remove(name, *keep_file).await?;
            }
            Some(LoraCommands::Convert {
                peft_dir,
                output,
                prefix,
            }) => {
                commands::convert_peft_to_candle_lora(
                    peft_dir.clone(),
                    output.clone(),
                    Some(prefix.clone()),
                )
                .await?;
            }
            None => {
                // Show help when no subcommand is provided
                println!("LoRA model management commands\n");
                println!("Use --help for more information");
            }
        },
        Some(Commands::Train {
            train_data,
            epochs,
            lora_r,
            lora_alpha,
            output_dir,
            model,
            batch_size,
            learning_rate,
            test,
        }) => {
            commands::run_train(
                train_data.clone(),
                *epochs,
                *lora_r,
                *lora_alpha,
                output_dir.clone(),
                model.clone(),
                *batch_size,
                *learning_rate,
                *test,
            )
            .await?;
        }
        None => {
            println!("Kairei-v2 AgentCulture Framework");
            println!("Use --help for more information");
        }
    }

    Ok(())
}
