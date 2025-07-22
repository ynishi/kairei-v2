//! Chat service implementation

use crate::chat::model::{ChatConfig, ChatMessage, ChatSession};
use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct ChatService {
    sessions: Arc<RwLock<HashMap<String, ChatSession>>>,
    // TODO: Add processor for actual LLM interaction
}

impl Default for ChatService {
    fn default() -> Self {
        Self::new()
    }
}

impl ChatService {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_session(&self, config: ChatConfig) -> Result<ChatSession> {
        let session = ChatSession::new(config);
        let session_id = session.id.clone();

        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session.clone());

        Ok(session)
    }

    pub async fn send_message(
        &self,
        session_id: &str,
        message: ChatMessage,
    ) -> Result<ChatMessage> {
        let mut sessions = self.sessions.write().await;
        let session = sessions.get_mut(session_id).ok_or_else(|| {
            crate::error::KaireiError::Build(format!("Session {} not found", session_id))
        })?;

        // Add user message
        session.add_message(message);

        // TODO: Process with actual LLM
        // For now, just echo back
        let response = ChatMessage::new_assistant(format!(
            "Echo: {}",
            session.messages.last().unwrap().content
        ));

        session.add_message(response.clone());

        Ok(response)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<ChatSession> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned().ok_or_else(|| {
            crate::error::KaireiError::Build(format!("Session {} not found", session_id))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::model::{GenerationParams, ModelType};

    #[tokio::test]
    async fn test_create_session() {
        let service = ChatService::new();
        let config = ChatConfig {
            model_type: ModelType::Llama2,
            base_model: None,
            lora_models: vec![],
            parameters: GenerationParams::default(),
        };

        let session = service.create_session(config).await.unwrap();
        assert!(!session.id.is_empty());
        assert_eq!(session.messages.len(), 0);
    }

    #[tokio::test]
    async fn test_send_message() {
        let service = ChatService::new();
        let config = ChatConfig {
            model_type: ModelType::Llama2,
            base_model: None,
            lora_models: vec![],
            parameters: GenerationParams::default(),
        };

        let session = service.create_session(config).await.unwrap();
        let message = ChatMessage::new_user("Hello!".to_string());

        let response = service.send_message(&session.id, message).await.unwrap();
        assert_eq!(response.content, "Echo: Hello!");

        // Check session was updated
        let updated_session = service.get_session(&session.id).await.unwrap();
        assert_eq!(updated_session.messages.len(), 2);
    }
}
