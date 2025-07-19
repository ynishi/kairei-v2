//! Chat command handler

use crate::error::CliError;
use kairei::prelude::*;
use rustyline::DefaultEditor;

pub async fn run_chat(
    initial_message: Option<String>,
    once: bool,
    use_candle: bool,
    lora_models: Vec<String>,
    base_model: Option<String>,
) -> Result<(), CliError> {
    // Only show header in interactive mode
    if !once {
        println!("🤖 Kairei Chat - Type 'exit' to quit");
        println!("================================");
    }

    // Show LoRA configuration if any
    if !lora_models.is_empty() {
        println!("📚 Loading LoRA models: {:?}", lora_models);
        if let Some(ref model) = base_model {
            println!("🧠 Base model: {}", model);
        }
    }

    // Build the application
    let app = if use_candle {
        println!("🔥 Initializing Candle backend (LLaMA2-C)...");
        let processor =
            kairei::Llama2CProcessor::new_tiny().map_err(|e| CliError::Core(e.to_string()))?;
        println!("✅ LLaMA2-C backend ready!");

        KaireiApp::builder("kairei-chat")
            .llm_mode()
            .processor(processor)
            .build()
            .map_err(CliError::Core)?
    } else {
        // Use echo processor as fallback
        KaireiApp::builder("kairei-chat")
            .llm_mode()
            .processor(kairei::app::processors::EchoProcessor)
            .build()
            .map_err(CliError::Core)?
    };

    // If initial message provided, process it
    if let Some(msg) = initial_message {
        if !once {
            println!("You: {}", msg);
        }
        let request = Request::simple(msg);
        let response = app
            .process(request)
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

                // Process through app
                let request = Request::simple(input);
                let response = app
                    .process(request)
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
