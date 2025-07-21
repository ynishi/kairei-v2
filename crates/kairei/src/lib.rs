//! Main crate for Kairei-v2 AgentCulture Framework
//!
//! This is the main application framework layer that provides high-level
//! abstractions for building LLM applications and agents.

pub mod app;
pub mod base_model;
pub mod config;
pub mod error;
pub mod lora;
pub mod storage;

// Re-export core types
pub use kairei_core::{
    CoreError, ProcessingContext, Processor, ProcessorExt, ProcessorMetadata, Request, RequestId,
    RequestMetadata, Response, ResponseMetadata, Result,
};

// Re-export app types
pub use app::{AppMetadata, AppMode, KaireiApp, KaireiAppBuilder};

// Re-export base model types
pub use base_model::{BaseModel, BaseModelId, BaseModelMetadata};

// Re-export error types
pub use error::{KaireiError, Result as KaireiResult};

// Feature-gated re-exports
#[cfg(feature = "candle")]
pub use kairei_candle::{
    Llama2CProcessor, Llama2CProcessorBuilder, Llama2Config, Llama2LoraProcessor, Llama3Config,
    Llama3LoraProcessor,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::app::{AppMode, KaireiApp};
    pub use kairei_core::{Processor, Request, Response};
}
