use axum::{Json, Router, extract::State, http::StatusCode, routing::post};
use kairei::trainer::{TrainParams, TrainerService, TrainingResult};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::AppState;

/// Training request DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrainRequest {
    /// Training data path
    pub train_data: String,

    /// Number of epochs
    pub epochs: u32,

    /// LoRA rank (r)
    pub lora_r: u32,

    /// LoRA alpha
    pub lora_alpha: u32,

    /// Output directory (optional, will be generated if not provided)
    pub output_dir: Option<String>,

    /// Model name or HuggingFace ID
    pub model: String,

    /// Batch size
    pub batch_size: u32,

    /// Learning rate
    pub learning_rate: f32,

    /// Run test after training
    pub test: bool,
}

impl From<TrainRequest> for TrainParams {
    fn from(req: TrainRequest) -> Self {
        TrainParams {
            train_data: req.train_data,
            epochs: req.epochs,
            lora_r: req.lora_r,
            lora_alpha: req.lora_alpha,
            output_dir: req.output_dir,
            model: req.model,
            batch_size: req.batch_size,
            learning_rate: req.learning_rate,
            test: req.test,
        }
    }
}

/// Training response DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrainResponse {
    /// Output directory where the model was saved
    pub output_dir: String,

    /// Status of the training
    pub status: String,

    /// Final loss value (if available)
    pub final_loss: Option<f32>,

    /// Total training duration
    pub duration: Option<String>,

    /// Any error message
    pub error: Option<String>,

    /// Model checkpoints created
    pub checkpoints: Vec<String>,
}

impl From<TrainingResult> for TrainResponse {
    fn from(result: TrainingResult) -> Self {
        TrainResponse {
            output_dir: result.output_dir,
            status: format!("{:?}", result.status),
            final_loss: result.final_loss,
            duration: result.duration,
            error: result.error,
            checkpoints: result.checkpoints,
        }
    }
}

/// Start LoRA training
#[utoipa::path(
    post,
    path = "/train/lora",
    request_body = TrainRequest,
    responses(
        (status = 200, description = "Training started successfully", body = TrainResponse),
        (status = 400, description = "Invalid training parameters"),
        (status = 500, description = "Internal server error")
    )
)]
async fn train_lora(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> Result<(StatusCode, Json<TrainResponse>), (StatusCode, Json<serde_json::Value>)> {
    let trainer = TrainerService::new(state.storage);

    match trainer.train_lora(req.into()).await {
        Ok(result) => {
            let response: TrainResponse = result.into();
            Ok((StatusCode::OK, Json(response)))
        }
        Err(e) => {
            let error_response = serde_json::json!({
                "error": e.to_string()
            });
            Err((StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)))
        }
    }
}

/// Trainer routes
pub fn routes() -> Router<AppState> {
    Router::new().route("/lora", post(train_lora))
}
