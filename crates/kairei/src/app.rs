//! Application layer for Kairei-v2
//!
//! This module provides the high-level application framework for building
//! LLM applications and agents with Kairei.

use kairei_core::{Processor, Request, Response, Result as CoreResult};
use std::sync::Arc;

/// The main Kairei application
pub struct KaireiApp {
    /// The processor chain for handling requests
    processor: Arc<dyn Processor>,
    /// Application metadata
    metadata: AppMetadata,
}

/// Metadata about the Kairei application
#[derive(Debug, Clone)]
pub struct AppMetadata {
    pub name: String,
    pub version: String,
    pub mode: AppMode,
}

/// Application mode
#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    /// Simple LLM mode
    LLM,
    /// Agent mode with tools and memory
    Agent,
    /// Custom mode
    Custom(String),
}

impl KaireiApp {
    /// Create a new app with a processor
    pub fn new(processor: impl Processor + 'static, metadata: AppMetadata) -> Self {
        Self {
            processor: Arc::new(processor),
            metadata,
        }
    }

    /// Process a request
    pub async fn process(&self, request: Request) -> CoreResult<Response> {
        self.processor.process(request).await
    }

    /// Get app metadata
    pub fn metadata(&self) -> &AppMetadata {
        &self.metadata
    }

    /// Create a new builder
    pub fn builder(name: impl Into<String>) -> KaireiAppBuilder {
        KaireiAppBuilder::new(name)
    }
}

/// Builder for creating Kairei applications
pub struct KaireiAppBuilder {
    name: String,
    version: String,
    mode: AppMode,
    processor: Option<Arc<dyn Processor>>,
}

impl KaireiAppBuilder {
    /// Create a new builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            mode: AppMode::LLM,
            processor: None,
        }
    }

    /// Set the version
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set the mode to LLM
    pub fn llm_mode(mut self) -> Self {
        self.mode = AppMode::LLM;
        self
    }

    /// Set the mode to Agent
    pub fn agent_mode(mut self) -> Self {
        self.mode = AppMode::Agent;
        self
    }

    /// Set a custom mode
    pub fn custom_mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = AppMode::Custom(mode.into());
        self
    }

    /// Set the processor
    pub fn processor(mut self, processor: impl Processor + 'static) -> Self {
        self.processor = Some(Arc::new(processor));
        self
    }

    /// Build the application
    pub fn build(self) -> Result<KaireiApp, String> {
        let processor = self.processor.ok_or("No processor configured")?;

        let metadata = AppMetadata {
            name: self.name,
            version: self.version,
            mode: self.mode,
        };

        Ok(KaireiApp {
            processor,
            metadata,
        })
    }
}

/// Default processor implementations
pub mod processors {
    use super::*;
    use async_trait::async_trait;

    /// Simple echo processor for testing
    pub struct EchoProcessor;

    #[async_trait]
    impl Processor for EchoProcessor {
        async fn process(&self, request: Request) -> CoreResult<Response> {
            Ok(Response::simple(
                request.id.clone(),
                format!("Echo: {}", request.message),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use processors::EchoProcessor;

    #[tokio::test]
    async fn test_app_builder() {
        let app = KaireiApp::builder("test-app")
            .version("1.0.0")
            .llm_mode()
            .processor(EchoProcessor)
            .build()
            .unwrap();

        assert_eq!(app.metadata().name, "test-app");
        assert_eq!(app.metadata().mode, AppMode::LLM);

        let request = Request::simple("Hello");
        let response = app.process(request).await.unwrap();
        assert!(response.message.starts_with("Echo:"));
    }
}
