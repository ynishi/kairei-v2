//! CLI command handlers

pub mod chat;
pub mod init;
pub mod lora;
pub mod run;
pub mod setup;

pub use chat::run_chat;
pub use init::run_init;
pub use lora::{lora_list, lora_new, setup_lora, lora_train};
pub use run::run_run;
pub use setup::run_setup;
