//! Processor trait and implementations for kairei-core
//!
//! This module defines the core processing abstraction that will be the heart of KAIREI-v2.
//! Designed to be simple yet extensible, supporting future separation into Curator and Decider patterns.

use crate::{Request, Response, Result};
use async_trait::async_trait;
use std::sync::Arc;

/// Core trait for processing requests in KAIREI-v2
///
/// This trait defines the fundamental abstraction for all processing logic.
/// It's designed to be:
/// - Simple: Just one method to implement
/// - Composable: Can be chained and combined
/// - Extensible: Ready for future Curator/Decider patterns
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process a request and generate a response
    async fn process(&self, request: Request) -> Result<Response>;

    /// Optional method to get processor metadata
    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata::default()
    }
}

/// Metadata about a processor
#[derive(Debug, Clone, Default)]
pub struct ProcessorMetadata {
    /// Human-readable name of the processor
    pub name: Option<String>,
    /// Description of what this processor does
    pub description: Option<String>,
    /// Version information
    pub version: Option<String>,
    /// Additional capabilities or features
    pub capabilities: Vec<String>,
}

/// A boxed processor for dynamic dispatch
pub type BoxedProcessor = Box<dyn Processor>;

/// Arc-wrapped processor for thread-safe sharing
pub type SharedProcessor = Arc<dyn Processor>;

/// Extension trait for processor composition
#[async_trait]
pub trait ProcessorExt: Processor {
    /// Chain this processor with another
    fn chain<P>(self, next: P) -> ChainedProcessor
    where
        Self: Sized + 'static,
        P: Processor + 'static,
    {
        ChainedProcessor::new(Box::new(self), Box::new(next))
    }

    /// Convert to a shared processor
    fn shared(self) -> SharedProcessor
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

// Implement ProcessorExt for all types that implement Processor
impl<T: Processor> ProcessorExt for T {}

/// A processor that chains two processors together
pub struct ChainedProcessor {
    first: BoxedProcessor,
    second: BoxedProcessor,
}

impl ChainedProcessor {
    /// Create a new chained processor
    pub fn new(first: BoxedProcessor, second: BoxedProcessor) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl Processor for ChainedProcessor {
    async fn process(&self, request: Request) -> Result<Response> {
        // Process with first processor
        let intermediate = self.first.process(request).await?;

        // Convert response back to request for second processor
        // This is a simple implementation - could be enhanced with context passing
        let next_request = Request {
            id: intermediate.request_id.clone(),
            message: intermediate.message,
            metadata: None, // Could preserve/transform metadata here
        };

        self.second.process(next_request).await
    }

    fn metadata(&self) -> ProcessorMetadata {
        ProcessorMetadata {
            name: Some("ChainedProcessor".to_string()),
            description: Some("Chains two processors together".to_string()),
            ..Default::default()
        }
    }
}

/// Context that can be passed through processor chains
/// This will be important for future Curator/Decider patterns
#[derive(Debug, Clone, Default)]
pub struct ProcessingContext {
    /// Arbitrary context data
    pub data: std::collections::HashMap<String, serde_json::Value>,
}

impl ProcessingContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a value into the context
    pub fn insert(&mut self, key: impl Into<String>, value: impl serde::Serialize) -> Result<()> {
        let value = serde_json::to_value(value).map_err(|e| {
            crate::CoreError::Processing(format!("Failed to serialize context value: {}", e))
        })?;
        self.data.insert(key.into(), value);
        Ok(())
    }

    /// Get a value from the context
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.data.get(key) {
            Some(value) => {
                let parsed = serde_json::from_value(value.clone()).map_err(|e| {
                    crate::CoreError::Processing(format!(
                        "Failed to deserialize context value: {}",
                        e
                    ))
                })?;
                Ok(Some(parsed))
            }
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple echo processor for testing
    struct EchoProcessor;

    #[async_trait]
    impl Processor for EchoProcessor {
        async fn process(&self, request: Request) -> Result<Response> {
            Ok(Response::simple(request.id.clone(), request.message))
        }
    }

    #[tokio::test]
    async fn test_basic_processor() {
        let processor = EchoProcessor;
        let request = Request::simple("Hello, world!");
        let response = processor.process(request).await.unwrap();
        assert_eq!(response.message, "Hello, world!");
    }

    #[tokio::test]
    async fn test_processor_chaining() {
        let echo = EchoProcessor;
        let uppercase = UppercaseProcessor;
        let chained = echo.chain(uppercase);

        let request = Request::simple("hello");
        let response = chained.process(request).await.unwrap();
        assert_eq!(response.message, "HELLO");
    }

    struct UppercaseProcessor;

    #[async_trait]
    impl Processor for UppercaseProcessor {
        async fn process(&self, request: Request) -> Result<Response> {
            Ok(Response::simple(
                request.id.clone(),
                request.message.to_uppercase(),
            ))
        }
    }
}
