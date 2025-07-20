//! CLI command handlers

pub mod chat;
pub mod init;
pub mod lora;
pub mod run;
pub mod setup;
pub mod train;

pub use chat::run_chat;
pub use init::run_init;
pub use lora::{
    convert_peft_to_candle_lora, lora_add, lora_list, lora_remove, lora_show, setup_lora,
    update_lora_training_info,
};
pub use run::run_run;
pub use setup::run_setup;
pub use train::run_train;
