//! Low-level kernel implementation

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
}

impl Default for KaireiKernel {
    fn default() -> Self {
        Self::new()
    }
}
