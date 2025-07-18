//! Low-level kernel implementation

use crate::{Request, Response, Result};

#[cfg(feature = "candle")]
use kairei_candle::{CandleModel, ModelConfig};

/// Core kernel structure for Kairei-v2
pub struct KaireiKernel {
    pub version: String,
    #[cfg(feature = "candle")]
    candle_model: Option<CandleModel>,
}

impl KaireiKernel {
    /// Create a new Kairei kernel instance
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            #[cfg(feature = "candle")]
            candle_model: None,
        }
    }

    /// Initialize with candle backend
    #[cfg(feature = "candle")]
    pub async fn init_candle(&mut self, config: ModelConfig) -> Result<()> {
        let mut model = CandleModel::new(config);
        model
            .load()
            .await
            .map_err(|e| crate::error::CoreError::Backend(e.to_string()))?;
        self.candle_model = Some(model);
        Ok(())
    }

    /// Process a request and generate a response
    pub async fn process_request(&self, request: Request) -> Result<Response> {
        #[cfg(feature = "candle")]
        {
            // If candle model is loaded, use it
            if let Some(model) = &self.candle_model {
                let response_message = model
                    .generate(&request.message)
                    .await
                    .map_err(|e| crate::error::CoreError::Backend(e.to_string()))?;
                return Ok(Response::simple(request.id, response_message));
            }
        }

        // Fallback to simple echo
        let response_message = format!("Hello! You said: {}", request.message);
        Ok(Response::simple(request.id, response_message))
    }
}

impl Default for KaireiKernel {
    fn default() -> Self {
        Self::new()
    }
}
