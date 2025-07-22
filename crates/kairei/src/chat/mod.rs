//! Chat functionality for Kairei
//!
//! This module provides chat session management and message processing
//! for both CLI and API usage.

pub mod model;
pub mod service;

pub use model::{ChatConfig, ChatMessage, ChatSession, GenerationParams, MessageRole, ModelType};
pub use service::ChatService;
