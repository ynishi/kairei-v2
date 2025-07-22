//! Model configuration types

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    Llama2,
    Llama3,
    Llama2c,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatConfig {
    pub model_type: ModelType,
    pub base_model: Option<String>,
    pub lora_models: Vec<String>,
    pub parameters: GenerationParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.9,
            top_k: None,
        }
    }
}
