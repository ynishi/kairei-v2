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

#[derive(clap::ValueEnum, Clone)]
enum SetupComponent {
    Lora,
}

#[derive(clap::Subcommand)]
enum LoraCommands {
    /// Create a new LoRA culture
    New {
        /// Culture name
        culture_name: String,
    },
    /// List all LoRA cultures
    List,
    /// Train a LoRA culture
    Train {
        /// Culture name
        culture_name: String,
    },
    /// Apply a LoRA culture
    Apply {
        /// Culture name
        culture_name: String,
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
        /// Setup component type
        component: Option<SetupComponent>,
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
        }) => {
            commands::run_chat(message.clone(), *once, *candle).await?;
        }
        Some(Commands::Setup {
            list,
            model,
            force,
            component,
        }) => {
            if let Some(comp) = component {
                match comp {
                    SetupComponent::Lora => commands::setup_lora().await?,
                }
            } else {
                commands::run_setup(*list, model.clone(), *force).await?;
            }
        }
        Some(Commands::Lora { command }) => match command {
            Some(LoraCommands::New { culture_name }) => {
                commands::lora_new(culture_name).await?;
            }
            Some(LoraCommands::List) => {
                commands::lora_list().await?;
            }
            Some(LoraCommands::Train { culture_name }) => {
                commands::lora_train(culture_name).await?;
            }
            Some(LoraCommands::Apply { culture_name }) => {
                println!("Applying LoRA culture: {}", culture_name);
                // TODO: Implement apply command
            }
            None => {
                // Show help when no subcommand is provided
                println!("LoRA culture management commands\n");
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
