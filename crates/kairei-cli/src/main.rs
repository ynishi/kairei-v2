//! CLI for Kairei-v2 AgentCulture Framework

mod error;

use clap::Parser;
use error::CliError;

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
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Some(Commands::Init) => {
            println!("Initializing Kairei project...");
        }
        Some(Commands::Run) => {
            println!("Running Kairei agent...");
        }
        None => {
            println!("Kairei-v2 AgentCulture Framework");
        }
    }
}
