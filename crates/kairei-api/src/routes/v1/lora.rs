use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use kairei::base_model::BaseModelId;
use kairei::lora::{Lora, LoraId, LoraMetadata, TrainingInfo};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::AppState;

/// Training info DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrainingInfoDto {
    pub training_data: String,
    pub training_data_hash: Option<String>,
    pub epochs: u32,
    pub batch_size: u32,
    pub learning_rate: f32,
    pub final_loss: Option<f32>,
    pub training_duration: Option<String>,
}

impl From<TrainingInfo> for TrainingInfoDto {
    fn from(info: TrainingInfo) -> Self {
        Self {
            training_data: info.training_data,
            training_data_hash: info.training_data_hash,
            epochs: info.epochs,
            batch_size: info.batch_size,
            learning_rate: info.learning_rate,
            final_loss: info.final_loss,
            training_duration: info.training_duration,
        }
    }
}

impl From<TrainingInfoDto> for TrainingInfo {
    fn from(dto: TrainingInfoDto) -> Self {
        TrainingInfo {
            training_data: dto.training_data,
            training_data_hash: dto.training_data_hash,
            epochs: dto.epochs,
            batch_size: dto.batch_size,
            learning_rate: dto.learning_rate,
            final_loss: dto.final_loss,
            training_duration: dto.training_duration,
        }
    }
}

/// LoRA metadata DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LoraMetadataDto {
    pub rank: Option<usize>,
    pub alpha: Option<f64>,
    pub training_info: Option<TrainingInfoDto>,
    pub parent_lora_id: Option<String>,
    pub version: Option<String>,
    pub training_framework: Option<String>,
}

impl From<LoraMetadata> for LoraMetadataDto {
    fn from(metadata: LoraMetadata) -> Self {
        Self {
            rank: metadata.rank,
            alpha: metadata.alpha,
            training_info: metadata.training_info.map(TrainingInfoDto::from),
            parent_lora_id: metadata.parent_lora_id.map(|id| id.to_string()),
            version: metadata.version,
            training_framework: metadata.training_framework,
        }
    }
}

impl From<LoraMetadataDto> for LoraMetadata {
    fn from(dto: LoraMetadataDto) -> Self {
        LoraMetadata {
            rank: dto.rank,
            alpha: dto.alpha,
            training_info: dto.training_info.map(TrainingInfo::from),
            parent_lora_id: dto.parent_lora_id.map(LoraId::from_string),
            version: dto.version,
            training_framework: dto.training_framework,
        }
    }
}

/// API representation of a LoRA model
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct LoraDto {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub base_model_id: Option<String>,
    pub created_at: String,
    pub status: String,
    pub file_path: Option<String>,
    pub size_bytes: Option<u64>,
    pub metadata: LoraMetadataDto,
}

impl From<Lora> for LoraDto {
    fn from(lora: Lora) -> Self {
        Self {
            id: lora.id.to_string(),
            name: lora.name,
            description: lora.description,
            base_model_id: lora.base_model_id.map(|id| id.to_string()),
            created_at: lora.created_at,
            status: lora.status.to_string(),
            file_path: lora.file_path,
            size_bytes: lora.size_bytes,
            metadata: LoraMetadataDto::from(lora.metadata),
        }
    }
}

/// Request to create a new LoRA
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateLoraRequest {
    pub name: String,
    pub description: Option<String>,
    pub base_model_id: Option<String>,
    pub metadata: Option<LoraMetadataDto>,
}

/// Request to update a LoRA
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateLoraRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub status: Option<String>,
}

/// Response for listing LoRAs
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListLorasResponse {
    pub loras: Vec<LoraDto>,
    pub total: usize,
}

/// List all LoRAs
#[utoipa::path(
    get,
    path = "/api/v1/loras",
    responses(
        (status = 200, description = "List of LoRAs", body = ListLorasResponse)
    ),
    tag = "loras"
)]
pub async fn list_loras(State(state): State<AppState>) -> impl IntoResponse {
    match state.lora_service.list().await {
        Ok(loras) => {
            let lora_list: Vec<LoraDto> = loras.into_iter().map(LoraDto::from).collect();
            let total = lora_list.len();

            Ok(Json(ListLorasResponse {
                loras: lora_list,
                total,
            }))
        }
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to list LoRAs")),
    }
}

/// Create a new LoRA
#[utoipa::path(
    post,
    path = "/api/v1/loras",
    request_body = CreateLoraRequest,
    responses(
        (status = 201, description = "LoRA created", body = LoraDto),
        (status = 400, description = "Invalid request")
    ),
    tag = "loras"
)]
pub async fn create_lora(
    State(state): State<AppState>,
    Json(request): Json<CreateLoraRequest>,
) -> impl IntoResponse {
    let metadata = request.metadata.map(LoraMetadata::from).unwrap_or_default();
    let base_model_id = request.base_model_id.map(BaseModelId::from_string);

    match state
        .lora_service
        .create(request.name, request.description, base_model_id, metadata)
        .await
    {
        Ok(lora) => Ok((StatusCode::CREATED, Json(LoraDto::from(lora)))),
        Err(e) => match e {
            kairei::lora::LoraError::AlreadyExists(_) => {
                Err((StatusCode::CONFLICT, "LoRA already exists"))
            }
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to create LoRA")),
        },
    }
}

/// Get a specific LoRA by ID
#[utoipa::path(
    get,
    path = "/api/v1/loras/{id}",
    params(
        ("id" = String, Path, description = "LoRA ID")
    ),
    responses(
        (status = 200, description = "LoRA details", body = LoraDto),
        (status = 404, description = "LoRA not found")
    ),
    tag = "loras"
)]
pub async fn get_lora(State(state): State<AppState>, Path(id): Path<String>) -> impl IntoResponse {
    let lora_id = LoraId::from_string(id);

    match state.lora_service.get(&lora_id).await {
        Ok(Some(lora)) => Ok(Json(LoraDto::from(lora))),
        Ok(None) => Err((StatusCode::NOT_FOUND, "LoRA not found")),
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to get LoRA")),
    }
}

/// Update a LoRA
#[utoipa::path(
    put,
    path = "/api/v1/loras/{id}",
    params(
        ("id" = String, Path, description = "LoRA ID")
    ),
    request_body = UpdateLoraRequest,
    responses(
        (status = 200, description = "LoRA updated", body = LoraDto),
        (status = 404, description = "LoRA not found")
    ),
    tag = "loras"
)]
pub async fn update_lora(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateLoraRequest>,
) -> impl IntoResponse {
    let lora_id = LoraId::from_string(id);

    // Get existing LoRA
    let mut lora = match state.lora_service.get(&lora_id).await {
        Ok(Some(lora)) => lora,
        Ok(None) => return Err((StatusCode::NOT_FOUND, "LoRA not found")),
        Err(_) => return Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to get LoRA")),
    };

    // Update fields if provided
    if let Some(name) = request.name {
        lora.name = name;
    }
    if let Some(description) = request.description {
        lora.description = Some(description);
    }
    if let Some(status_str) = request.status {
        match status_str.parse::<kairei::lora::LoraStatus>() {
            Ok(status) => lora.status = status,
            Err(_) => return Err((StatusCode::BAD_REQUEST, "Invalid status")),
        }
    }

    // Update the LoRA
    match state.lora_service.update(lora).await {
        Ok(updated_lora) => Ok(Json(LoraDto::from(updated_lora))),
        Err(e) => match e {
            kairei::lora::LoraError::NotFound(_) => Err((StatusCode::NOT_FOUND, "LoRA not found")),
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to update LoRA")),
        },
    }
}

/// Delete a LoRA
#[utoipa::path(
    delete,
    path = "/api/v1/loras/{id}",
    params(
        ("id" = String, Path, description = "LoRA ID")
    ),
    responses(
        (status = 204, description = "LoRA deleted"),
        (status = 404, description = "LoRA not found")
    ),
    tag = "loras"
)]
pub async fn delete_lora(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let lora_id = LoraId::from_string(id);

    match state.lora_service.delete(&lora_id).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(e) => match e {
            kairei::lora::LoraError::NotFound(_) => Err((StatusCode::NOT_FOUND, "LoRA not found")),
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to delete LoRA")),
        },
    }
}

/// Get a LoRA by name
#[utoipa::path(
    get,
    path = "/api/v1/loras/by-name/{name}",
    params(
        ("name" = String, Path, description = "LoRA name")
    ),
    responses(
        (status = 200, description = "LoRA details", body = LoraDto),
        (status = 404, description = "LoRA not found")
    ),
    tag = "loras"
)]
pub async fn get_lora_by_name(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    match state.lora_service.get_by_name(&name).await {
        Ok(Some(lora)) => Ok(Json(LoraDto::from(lora))),
        Ok(None) => Err((StatusCode::NOT_FOUND, "LoRA not found")),
        Err(_) => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to get LoRA")),
    }
}

/// Upload file to LoRA
#[utoipa::path(
    post,
    path = "/api/v1/loras/{id}/upload",
    params(
        ("id" = String, Path, description = "LoRA ID")
    ),
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "File uploaded", body = LoraDto),
        (status = 404, description = "LoRA not found"),
        (status = 413, description = "File too large")
    ),
    tag = "loras"
)]
pub async fn upload_lora_file(
    State(state): State<AppState>,
    Path(id): Path<String>,
    mut multipart: axum::extract::Multipart,
) -> impl IntoResponse {
    let lora_id = LoraId::from_string(id);

    // Get the file from multipart
    let mut file_name = String::new();
    let mut file_content = Vec::new();

    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            file_name = field
                .file_name()
                .unwrap_or("adapter.safetensors")
                .to_string();
            file_content = match field.bytes().await {
                Ok(bytes) => bytes.to_vec(),
                Err(_) => {
                    return Err((StatusCode::BAD_REQUEST, "Failed to read file"));
                }
            };
            break;
        }
    }

    if file_content.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "No file uploaded"));
    }

    // Upload the file
    match state
        .lora_service
        .upload_file(&lora_id, &file_name, &file_content)
        .await
    {
        Ok(lora) => Ok(Json(LoraDto::from(lora))),
        Err(e) => match e {
            kairei::lora::LoraError::NotFound(_) => Err((StatusCode::NOT_FOUND, "LoRA not found")),
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to upload file")),
        },
    }
}

/// Routes for LoRA endpoints
pub fn routes() -> axum::Router<AppState> {
    use axum::routing::{get, post};

    axum::Router::new()
        .route("/", get(list_loras).post(create_lora))
        .route("/by-name/{name}", get(get_lora_by_name))
        .route("/{id}", get(get_lora).put(update_lora).delete(delete_lora))
        .route("/{id}/upload", post(upload_lora_file))
}
