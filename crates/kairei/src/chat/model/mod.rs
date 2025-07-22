//! Model definitions for chat functionality

mod chat_session;
mod message;
mod model_config;

pub use chat_session::ChatSession;
pub use message::{ChatMessage, MessageRole};
pub use model_config::{ChatConfig, GenerationParams, ModelType};
