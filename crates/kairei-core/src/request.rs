//! Request and Response types for kairei-core

use crate::types::{RequestId, SessionId};
use serde::{Deserialize, Serialize};

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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetadata {
    /// User ID if available
    pub user_id: Option<String>,
    /// Session ID for conversation tracking
    pub session_id: Option<SessionId>,
    /// Additional context
    pub context: Option<serde_json::Value>,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseMetadata {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Model used for generation
    pub model: Option<String>,
    /// Additional info
    pub info: Option<serde_json::Value>,
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
