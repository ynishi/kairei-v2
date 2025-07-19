//! CLI for Kairei-v2 AgentCulture Framework

use clap::Parser;
use kairei_cli::{commands, error::CliError};

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
    },
    /// LoRA culture management
    Lora {
        #[command(subcommand)]
        command: Option<LoraCommands>,
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
        }) => {
            commands::run_chat(
                message.clone(),
                *once,
                *candle,
                lora.clone(),
                base_model.clone(),
            )
            .await?;
        }
        Some(Commands::Setup { list, model, force }) => {
            commands::run_setup(*list, model.clone(), *force).await?;
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
            None => {
                // Show help when no subcommand is provided
                println!("LoRA model management commands\n");
                println!("Use --help for more information");
            }
        },
        None => {
            println!("Kairei-v2 AgentCulture Framework");
            println!("Use --help for more information");
        }
    }

    Ok(())
}
