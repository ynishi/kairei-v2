//! Tuner trait and implementations for kairei-core
//!
//! This module defines the core tuning abstraction for fine-tuning models.
//! Supports various tuning methods like LoRA, QLoRA, and full fine-tuning.

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Core trait for model tuning in KAIREI-v2
///
/// This trait defines the fundamental abstraction for all tuning/fine-tuning logic.
/// It's designed to be:
/// - Flexible: Supports various tuning methods (LoRA, QLoRA, etc.)
/// - Composable: Can be combined with Processors
/// - Culture-aware: Ready for Culture/Agent level abstractions
#[async_trait]
pub trait Tuner: Send + Sync {
    /// Apply tuning to modify model behavior
    async fn apply(&self, context: TuningContext) -> Result<TuningResult>;

    /// Save the tuned parameters
    async fn save(&self, path: &str) -> Result<()>;

    /// Load tuned parameters
    async fn load(&self, path: &str) -> Result<()>;

    /// Get tuner metadata
    fn metadata(&self) -> TunerMetadata {
        TunerMetadata::default()
    }
}

/// Context for tuning operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningContext {
    /// The input data for tuning
    pub input: String,
    /// Expected output (for supervised tuning)
    pub expected_output: Option<String>,
    /// Additional parameters for tuning
    pub parameters: TuningParameters,
}

/// Parameters for tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningParameters {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training steps
    pub steps: Option<usize>,
    /// Batch size
    pub batch_size: Option<usize>,
    /// Additional method-specific parameters
    pub extra: std::collections::HashMap<String, serde_json::Value>,
}

impl Default for TuningParameters {
    fn default() -> Self {
        Self {
            learning_rate: 5e-4,
            steps: None,
            batch_size: None,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Result of a tuning operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    /// Whether tuning was successful
    pub success: bool,
    /// Loss value if available
    pub loss: Option<f32>,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f32>,
    /// Any messages or logs
    pub messages: Vec<String>,
}

/// Metadata about a tuner
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TunerMetadata {
    /// Human-readable name of the tuner
    pub name: Option<String>,
    /// Type of tuning method (e.g., "LoRA", "QLoRA", "FullFineTune")
    pub tuning_type: Option<String>,
    /// Description of what this tuner does
    pub description: Option<String>,
    /// Version information
    pub version: Option<String>,
    /// Supported model architectures
    pub supported_models: Vec<String>,
    /// Additional capabilities or features
    pub capabilities: Vec<String>,
}

/// A boxed tuner for dynamic dispatch
pub type BoxedTuner = Box<dyn Tuner>;

/// Arc-wrapped tuner for thread-safe sharing
pub type SharedTuner = Arc<dyn Tuner>;

/// LoRA-specific tuner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRATunerConfig {
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha parameter
    pub alpha: f32,
    /// LoRA dropout
    pub dropout: Option<f32>,
    /// Target modules (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
}

impl Default for LoRATunerConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: None,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        }
    }
}

/// Extension trait for tuner composition
#[async_trait]
pub trait TunerExt: Tuner {
    /// Combine this tuner with another (e.g., ensemble tuning)
    fn combine<T>(self, other: T) -> CombinedTuner
    where
        Self: Sized + 'static,
        T: Tuner + 'static,
    {
        CombinedTuner::new(Box::new(self), Box::new(other))
    }

    /// Convert to a shared tuner
    fn shared(self) -> SharedTuner
    where
        Self: Sized + 'static,
    {
        Arc::new(self)
    }
}

// Implement TunerExt for all types that implement Tuner
impl<T: Tuner> TunerExt for T {}

/// A tuner that combines multiple tuning methods
pub struct CombinedTuner {
    first: BoxedTuner,
    second: BoxedTuner,
}

impl CombinedTuner {
    /// Create a new combined tuner
    pub fn new(first: BoxedTuner, second: BoxedTuner) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl Tuner for CombinedTuner {
    async fn apply(&self, context: TuningContext) -> Result<TuningResult> {
        // Apply first tuner
        let result1 = self.first.apply(context.clone()).await?;

        // Apply second tuner
        let result2 = self.second.apply(context).await?;

        // Combine results
        Ok(TuningResult {
            success: result1.success && result2.success,
            loss: match (result1.loss, result2.loss) {
                (Some(l1), Some(l2)) => Some((l1 + l2) / 2.0),
                (Some(l), None) | (None, Some(l)) => Some(l),
                _ => None,
            },
            metrics: {
                let mut metrics = result1.metrics;
                metrics.extend(result2.metrics);
                metrics
            },
            messages: {
                let mut messages = result1.messages;
                messages.extend(result2.messages);
                messages
            },
        })
    }

    async fn save(&self, path: &str) -> Result<()> {
        // Save both tuners
        self.first.save(&format!("{}_first", path)).await?;
        self.second.save(&format!("{}_second", path)).await?;
        Ok(())
    }

    async fn load(&self, path: &str) -> Result<()> {
        // Load both tuners
        self.first.load(&format!("{}_first", path)).await?;
        self.second.load(&format!("{}_second", path)).await?;
        Ok(())
    }

    fn metadata(&self) -> TunerMetadata {
        TunerMetadata {
            name: Some("CombinedTuner".to_string()),
            tuning_type: Some("Combined".to_string()),
            description: Some("Combines multiple tuning methods".to_string()),
            ..Default::default()
        }
    }
}

// Culture and Agent abstractions will be added in the future when needed

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock LoRA tuner for testing
    struct MockLoRATuner {
        config: LoRATunerConfig,
    }

    #[async_trait]
    impl Tuner for MockLoRATuner {
        async fn apply(&self, _context: TuningContext) -> Result<TuningResult> {
            Ok(TuningResult {
                success: true,
                loss: Some(0.5),
                metrics: vec![("accuracy".to_string(), 0.95)].into_iter().collect(),
                messages: vec!["Tuning applied successfully".to_string()],
            })
        }

        async fn save(&self, _path: &str) -> Result<()> {
            Ok(())
        }

        async fn load(&self, _path: &str) -> Result<()> {
            Ok(())
        }

        fn metadata(&self) -> TunerMetadata {
            TunerMetadata {
                name: Some("MockLoRATuner".to_string()),
                tuning_type: Some("LoRA".to_string()),
                description: Some("Mock LoRA tuner for testing".to_string()),
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn test_basic_tuner() {
        let tuner = MockLoRATuner {
            config: LoRATunerConfig::default(),
        };

        let context = TuningContext {
            input: "Test input".to_string(),
            expected_output: Some("Expected output".to_string()),
            parameters: TuningParameters::default(),
        };

        let result = tuner.apply(context).await.unwrap();
        assert!(result.success);
        assert_eq!(result.loss, Some(0.5));
    }
}
