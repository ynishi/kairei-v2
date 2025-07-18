//! Request and Response types for kairei-core

use serde::{Deserialize, Serialize};
use std::fmt;

/// NewType pattern for Request ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(String);

impl RequestId {
    /// Create a new RequestId
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    /// Create from existing string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get the inner string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for RequestId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Request from user to the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Request {
    /// Unique request ID
    pub id: RequestId,
    /// User's input message
    pub message: String,
    /// Optional metadata
    pub metadata: Option<RequestMetadata>,
}

/// Metadata for request context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestMetadata {
    /// Extension data for processors
    pub ext: serde_json::Value,
}

/// Response from system to user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// Corresponding request ID
    pub request_id: RequestId,
    /// Response message
    pub message: String,
    /// Processing metadata
    pub metadata: Option<ResponseMetadata>,
}

/// Metadata for response details
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponseMetadata {
    /// Extension data for processors
    pub ext: serde_json::Value,
}

impl Request {
    /// Create a simple request with just a message
    pub fn simple(message: impl Into<String>) -> Self {
        Self {
            id: RequestId::new(),
            message: message.into(),
            metadata: None,
        }
    }
}

impl Response {
    /// Create a simple response
    pub fn simple(request_id: RequestId, message: impl Into<String>) -> Self {
        Self {
            request_id,
            message: message.into(),
            metadata: None,
        }
    }
}
