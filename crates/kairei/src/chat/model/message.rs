//! Message types for chat

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

impl ChatMessage {
    pub fn new_user(content: String) -> Self {
        Self {
            role: MessageRole::User,
            content,
            timestamp: Utc::now(),
        }
    }

    pub fn new_assistant(content: String) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
            timestamp: Utc::now(),
        }
    }
}
