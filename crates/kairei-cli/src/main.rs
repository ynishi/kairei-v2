//! CLI for Kairei-v2 AgentCulture Framework

mod error;

use clap::Parser;
use error::CliError;
use rustyline::DefaultEditor;

#[derive(Parser)]
#[command(name = "kairei")]
#[command(about = "CLI for Kairei-v2 AgentCulture Framework", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
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
    },
}

#[tokio::main]
async fn main() -> Result<(), CliError> {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Init) => {
            println!("Initializing Kairei project...");
        }
        Some(Commands::Run) => {
            println!("Running Kairei agent...");
        }
        Some(Commands::Chat { message, once }) => {
            run_chat(message.clone(), *once).await?;
        }
        None => {
            println!("Kairei-v2 AgentCulture Framework");
            println!("Use --help for more information");
        }
    }

    Ok(())
}

async fn run_chat(initial_message: Option<String>, once: bool) -> Result<(), CliError> {
    // Only show header in interactive mode
    if !once {
        println!("🤖 Kairei Chat - Type 'exit' to quit");
        println!("================================");
    }

    // Initialize the kernel
    let kernel = kairei_core::KaireiKernel::new();

    // If initial message provided, process it
    if let Some(msg) = initial_message {
        if !once {
            println!("You: {}", msg);
        }
        let request = kairei_core::Request::simple(msg);
        let response = kernel
            .process_request(request)
            .await
            .map_err(|e| CliError::Core(e.to_string()))?;

        // In once mode, just print the response without "Bot:" prefix
        if once {
            println!("{}", response.message);
            return Ok(());
        } else {
            println!("Bot: {}", response.message);
        }
    } else if once {
        // In once mode, we need a message
        return Err(CliError::InvalidInput(
            "Message is required in once mode".to_string(),
        ));
    }

    // Interactive loop (only if not in once mode)
    let mut rl = DefaultEditor::new()?;

    loop {
        let readline = rl.readline("🤖 > ");
        match readline {
            Ok(line) => {
                let input = line.trim();

                if input == "exit" || input == "quit" {
                    println!("Goodbye! 👋");
                    break;
                }

                if input.is_empty() {
                    continue;
                }

                // Add to history
                rl.add_history_entry(&line).ok();

                // Process through core
                let request = kairei_core::Request::simple(input);
                let response = kernel
                    .process_request(request)
                    .await
                    .map_err(|e| CliError::Core(e.to_string()))?;
                println!("Bot: {}", response.message);
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("\nCtrl-C received. Use 'exit' to quit.");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("\nGoodbye! 👋");
                break;
            }
            Err(err) => {
                return Err(err.into());
            }
        }
    }

    Ok(())
}
