//! Common types for kairei-core

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

/// NewType pattern for Session ID
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new SessionId
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

impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}