//! Candle-based Processor implementation
//!
//! This module provides Processor implementations backed by Candle models.
//! Future custom processors like CustomSharedMoEProcessor will be built on top of this.

use async_trait::async_trait;
use kairei_core::{Processor, ProcessorMetadata, Request, Response, Result as CoreResult};

use crate::{CandleError, CandleModel, ModelConfig};

/// A Processor implementation backed by a Candle model
pub struct CandleProcessor {
    model: CandleModel,
    name: String,
}

impl CandleProcessor {
    /// Create a new CandleProcessor with the given configuration
    pub async fn new(name: impl Into<String>, config: ModelConfig) -> Result<Self, CandleError> {
        let mut model = CandleModel::new(config);
        model.load().await?;

        Ok(Self {
            model,
            name: name.into(),
        })
    }

    /// Create from an existing loaded model
    pub fn from_model(name: impl Into<String>, model: CandleModel) -> Self {
        Self {
            model,
            name: name.into(),
        }
    }
}

#[async_trait]
impl Processor for CandleProcessor {
    async fn process(&self, request: Request) -> CoreResult<Response> {
        // Generate response using the Candle model
        let generated = self
            .model
            .generate(&request.message)
            .await
            .map_err(|e| kairei_core::CoreError::Backend(e.to_string()))?;

        // Build response
        Ok(Response::simple(request.id, generated))
    }

    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: Some(self.name.clone()),
            description: Some("Candle-based language model processor".to_string()),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
            capabilities: vec![
                "text-generation".to_string(),
                "async-processing".to_string(),
            ],
        }
    }
}

/// Builder for creating custom Candle processors
pub struct CandleProcessorBuilder {
    name: String,
    config: ModelConfig,
}

impl CandleProcessorBuilder {
    /// Create a new builder with default configuration
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            config: ModelConfig::default(),
        }
    }

    /// Set the model path
    pub fn model_path(mut self, path: impl Into<String>) -> Self {
        self.config.model_path = Some(path.into());
        self
    }

    /// Set the temperature
    pub fn temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp;
        self
    }

    /// Set the max tokens
    pub fn max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = tokens;
        self
    }

    /// Build the processor
    pub async fn build(self) -> Result<CandleProcessor, CandleError> {
        CandleProcessor::new(self.name, self.config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_candle_processor_builder() {
        let processor = CandleProcessorBuilder::new("test-model")
            .temperature(0.8)
            .max_tokens(1024)
            .build()
            .await
            .unwrap();

        let metadata = processor.metadata();
        assert_eq!(metadata.name, Some("test-model".to_string()));
        assert!(
            metadata
                .capabilities
                .contains(&"text-generation".to_string())
        );
    }

    #[tokio::test]
    async fn test_candle_processor_process() {
        let processor = CandleProcessor::new("test", ModelConfig::default())
            .await
            .unwrap();

        let request = Request::simple("Hello, Candle!");
        let response = processor.process(request).await.unwrap();

        assert!(response.message.contains("Candle says"));
    }
}
