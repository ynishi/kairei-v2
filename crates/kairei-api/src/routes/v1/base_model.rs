use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use kairei::base_model::{BaseModel, BaseModelId, BaseModelMetadata};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::AppState;

/// Model status in the system
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    /// Model is available for download
    Available,
    /// Model is currently being downloaded
    Downloading,
    /// Model has been downloaded and is ready to use
    Downloaded,
    /// Model download failed
    Failed,
}

/// Model metadata DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq)]
pub struct BaseModelMetadataDto {
    pub repo_id: String,
    pub name: String,
    pub description: Option<String>,
    pub downloaded_at: Option<String>,
    pub parameters: Option<String>,
    pub architecture: Option<String>,
    pub quantization: Option<String>,
}

impl From<BaseModelMetadata> for BaseModelMetadataDto {
    fn from(metadata: BaseModelMetadata) -> Self {
        Self {
            repo_id: metadata.repo_id,
            name: metadata.name,
            description: metadata.description,
            downloaded_at: metadata.downloaded_at,
            parameters: metadata.parameters,
            architecture: metadata.architecture,
            quantization: metadata.quantization,
        }
    }
}

impl From<BaseModelMetadataDto> for BaseModelMetadata {
    fn from(dto: BaseModelMetadataDto) -> Self {
        BaseModelMetadata {
            repo_id: dto.repo_id,
            name: dto.name,
            description: dto.description,
            downloaded_at: dto.downloaded_at,
            parameters: dto.parameters,
            architecture: dto.architecture,
            quantization: dto.quantization,
        }
    }
}

/// API representation of a base model
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelDto {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub repo_id: Option<String>,
    pub filename: Option<String>,
    pub size_mb: Option<u64>,
    pub status: ModelStatus,
    pub metadata: Option<BaseModelMetadataDto>,
}

impl From<BaseModel> for ModelDto {
    fn from(model: BaseModel) -> Self {
        Self {
            id: model.id.to_string(),
            name: model.name,
            description: model.description,
            repo_id: model.repo_id,
            filename: model.filename,
            size_mb: model.size_mb,
            status: ModelStatus::Available, // Default status
            metadata: model.metadata.map(BaseModelMetadataDto::from),
        }
    }
}

/// Request to create a new base model
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateModelRequest {
    pub name: String,
    pub description: Option<String>,
    pub repo_id: Option<String>,
    pub filename: Option<String>,
    pub size_mb: Option<u64>,
    pub metadata: Option<BaseModelMetadataDto>,
}

/// Request to update a base model
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateModelRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub metadata: Option<BaseModelMetadataDto>,
}

/// Response for listing models
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListModelsResponse {
    pub models: Vec<ModelDto>,
    pub total: usize,
}

/// List all base models
#[utoipa::path(
    get,
    path = "/api/v1/models",
    responses(
        (status = 200, description = "List of models", body = ListModelsResponse)
    ),
    tag = "models"
)]
pub async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    match state.base_model_service.list_models().await {
        Ok(models) => {
            let model_list: Vec<ModelDto> = models.into_iter().map(ModelDto::from).collect();

            let total = model_list.len();

            Ok(Json(ListModelsResponse {
                models: model_list,
                total,
            }))
        }
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to list models")),
    }
}

/// Get a specific model by ID
#[utoipa::path(
    get,
    path = "/api/v1/models/{id}",
    params(
        ("id" = String, Path, description = "Model ID")
    ),
    responses(
        (status = 200, description = "Model details", body = ModelDto),
        (status = 404, description = "Model not found")
    ),
    tag = "models"
)]
pub async fn get_model(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let model_id = BaseModelId::from_string(id);

    match state.base_model_service.get_model(&model_id).await {
        Ok(model) => Ok(Json(ModelDto::from(model))),
        Err(_) => Err((StatusCode::NOT_FOUND, "Model not found")),
    }
}

/// Create a new base model
#[utoipa::path(
    post,
    path = "/api/v1/models",
    request_body = CreateModelRequest,
    responses(
        (status = 201, description = "Model created", body = ModelDto),
        (status = 400, description = "Invalid request")
    ),
    tag = "models"
)]
pub async fn create_model(
    State(state): State<AppState>,
    Json(request): Json<CreateModelRequest>,
) -> impl IntoResponse {
    match state
        .base_model_service
        .register_model(
            request.name,
            request.description,
            request.repo_id,
            request.filename,
            request.size_mb,
            request.metadata.map(BaseModelMetadata::from),
        )
        .await
    {
        Ok(model) => Ok((StatusCode::CREATED, Json(ModelDto::from(model)))),
        Err(e) => match e {
            kairei::base_model::BaseModelError::AlreadyExists(_) => {
                Err((StatusCode::CONFLICT, "Model already exists"))
            }
            kairei::base_model::BaseModelError::InvalidData(_msg) => {
                Err((StatusCode::BAD_REQUEST, "Invalid model data"))
            }
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to create model")),
        },
    }
}

/// Update a base model
#[utoipa::path(
    put,
    path = "/api/v1/models/{id}",
    params(
        ("id" = String, Path, description = "Model ID")
    ),
    request_body = UpdateModelRequest,
    responses(
        (status = 200, description = "Model updated", body = ModelDto),
        (status = 404, description = "Model not found")
    ),
    tag = "models"
)]
pub async fn update_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateModelRequest>,
) -> impl IntoResponse {
    let model_id = BaseModelId::from_string(id);

    match state
        .base_model_service
        .update_model(
            &model_id,
            request.name,
            request.description,
            request.metadata.map(BaseModelMetadata::from),
        )
        .await
    {
        Ok(model) => Ok(Json(ModelDto::from(model))),
        Err(e) => match e {
            kairei::base_model::BaseModelError::NotFound(_) => {
                Err((StatusCode::NOT_FOUND, "Model not found"))
            }
            kairei::base_model::BaseModelError::AlreadyExists(_) => {
                Err((StatusCode::CONFLICT, "Model name already exists"))
            }
            kairei::base_model::BaseModelError::InvalidData(_msg) => {
                Err((StatusCode::BAD_REQUEST, "Invalid model data"))
            }
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to update model")),
        },
    }
}

/// Delete a base model
#[utoipa::path(
    delete,
    path = "/api/v1/models/{id}",
    params(
        ("id" = String, Path, description = "Model ID")
    ),
    responses(
        (status = 204, description = "Model deleted"),
        (status = 404, description = "Model not found")
    ),
    tag = "models"
)]
pub async fn delete_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let model_id = BaseModelId::from_string(id);

    match state.base_model_service.delete_model(&model_id).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(kairei::base_model::BaseModelError::NotFound(_)) => {
            Err((StatusCode::NOT_FOUND, "Model not found"))
        }
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to delete model")),
    }
}

/// Download a model from HuggingFace
#[utoipa::path(
    post,
    path = "/api/v1/models/{id}/download",
    params(
        ("id" = String, Path, description = "Model ID")
    ),
    responses(
        (status = 200, description = "Model downloaded successfully", body = ModelDto),
        (status = 404, description = "Model not found"),
        (status = 409, description = "Model already downloaded")
    ),
    tag = "models"
)]
pub async fn download_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let model_id = BaseModelId::from_string(id);

    // Check if model exists first
    let model = match state.base_model_service.get_model(&model_id).await {
        Ok(m) => m,
        Err(kairei::base_model::BaseModelError::NotFound(_)) => {
            return Err((StatusCode::NOT_FOUND, "Model not found"));
        }
        Err(_) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to get model"));
        }
    };

    // Check if already downloaded
    match state
        .base_model_service
        .is_model_downloaded(&model_id)
        .await
    {
        Ok(true) => {
            return Ok(Json(ModelDto::from(model)));
        }
        Ok(false) => {
            // Proceed with download
        }
        Err(_) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to check download status",
            ));
        }
    }

    // Download the model
    match state
        .base_model_service
        .download_model(&model_id, false)
        .await
    {
        Ok(()) => {
            // Get updated model
            match state.base_model_service.get_model(&model_id).await {
                Ok(updated_model) => Ok(Json(ModelDto::from(updated_model))),
                Err(_) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to get updated model",
                )),
            }
        }
        Err(e) => match e {
            kairei::base_model::BaseModelError::DownloadError(_msg) => {
                Err((StatusCode::BAD_GATEWAY, "Download failed"))
            }
            _ => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to download model",
            )),
        },
    }
}

/// Routes for base model endpoints
pub fn routes() -> axum::Router<AppState> {
    use axum::routing::get;

    axum::Router::new()
        .route("/", get(list_models).post(create_model))
        .route(
            "/{id}",
            get(get_model).put(update_model).delete(delete_model),
        )
        .route("/{id}/download", axum::routing::post(download_model))
}
