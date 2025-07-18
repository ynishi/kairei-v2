//! Low-level kernel implementation

use crate::{Request, Response, Result};

/// Core kernel structure for Kairei-v2
pub struct KaireiKernel {
    pub version: String,
}

impl KaireiKernel {
    /// Create a new Kairei kernel instance
    pub fn new() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Process a request and generate a response
    pub async fn process_request(&self, request: Request) -> Result<Response> {
        // For now, just echo the message with a greeting
        let response_message = format!("Hello! You said: {}", request.message);

        Ok(Response::simple(request.id, response_message))
    }
}

impl Default for KaireiKernel {
    fn default() -> Self {
        Self::new()
    }
}
