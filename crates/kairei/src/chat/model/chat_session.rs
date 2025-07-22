//! Chat session management

use super::{ChatConfig, ChatMessage};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub config: ChatConfig,
}

impl ChatSession {
    pub fn new(config: ChatConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            messages: Vec::new(),
            config,
        }
    }

    pub fn add_message(&mut self, message: ChatMessage) {
        self.messages.push(message);
    }

    pub fn get_context(&self) -> String {
        self.messages
            .iter()
            .map(|msg| {
                format!(
                    "{}: {}",
                    match msg.role {
                        super::MessageRole::User => "User",
                        super::MessageRole::Assistant => "Assistant",
                    },
                    msg.content
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}
