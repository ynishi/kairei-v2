use axum::{
    Json,
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use kairei::training_data::{
    DataFormat, DataType, TrainingData, TrainingDataId, TrainingDataMetadata,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::AppState;

/// Training data metadata DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrainingDataMetadataDto {
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub sample_count: Option<usize>,
    pub statistics: Option<serde_json::Value>,
    pub properties: Option<serde_json::Value>,
}

impl From<TrainingDataMetadata> for TrainingDataMetadataDto {
    fn from(metadata: TrainingDataMetadata) -> Self {
        Self {
            source: metadata.source,
            tags: metadata.tags,
            sample_count: metadata.sample_count,
            statistics: metadata.statistics,
            properties: metadata.properties,
        }
    }
}

impl From<TrainingDataMetadataDto> for TrainingDataMetadata {
    fn from(dto: TrainingDataMetadataDto) -> Self {
        TrainingDataMetadata {
            source: dto.source,
            tags: dto.tags,
            sample_count: dto.sample_count,
            statistics: dto.statistics,
            properties: dto.properties,
        }
    }
}

/// Data format DTO
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct DataFormatDto {
    pub mime_type: Option<String>,
    pub extension: Option<String>,
    pub encoding: Option<String>,
    pub compression: Option<String>,
}

impl From<DataFormat> for DataFormatDto {
    fn from(format: DataFormat) -> Self {
        Self {
            mime_type: format.mime_type,
            extension: format.extension,
            encoding: format.encoding,
            compression: format.compression,
        }
    }
}

impl From<DataFormatDto> for DataFormat {
    fn from(dto: DataFormatDto) -> Self {
        DataFormat {
            mime_type: dto.mime_type,
            extension: dto.extension,
            encoding: dto.encoding,
            compression: dto.compression,
        }
    }
}

/// API representation of training data
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct TrainingDataDto {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub data_type: String,
    pub format: DataFormatDto,
    pub file_path: Option<String>,
    pub size_bytes: Option<u64>,
    pub hash: Option<String>,
    pub lora_ids: Vec<String>,
    pub metadata: TrainingDataMetadataDto,
    pub created_at: String,
    pub updated_at: String,
    pub archived: bool,
}

impl From<TrainingData> for TrainingDataDto {
    fn from(data: TrainingData) -> Self {
        Self {
            id: data.id.to_string(),
            name: data.name,
            description: data.description,
            data_type: data.data_type.to_string(),
            format: DataFormatDto::from(data.format),
            file_path: data.file_path,
            size_bytes: data.size_bytes,
            hash: data.hash,
            lora_ids: data.lora_ids,
            metadata: TrainingDataMetadataDto::from(data.metadata),
            created_at: data.created_at,
            updated_at: data.updated_at,
            archived: data.archived,
        }
    }
}

/// Request to create new training data
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct CreateTrainingDataRequest {
    pub name: String,
    pub description: Option<String>,
    pub data_type: String,
    pub format: Option<DataFormatDto>,
    pub metadata: Option<TrainingDataMetadataDto>,
}

/// Request to update training data
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct UpdateTrainingDataRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub archived: Option<bool>,
}

/// Response for listing training data
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ListTrainingDataResponse {
    pub training_data: Vec<TrainingDataDto>,
    pub total: usize,
}

/// List all training data
#[utoipa::path(
    get,
    path = "/api/v1/training-data",
    responses(
        (status = 200, description = "List of training data", body = ListTrainingDataResponse)
    ),
    tag = "training-data"
)]
pub async fn list_training_data(State(state): State<AppState>) -> impl IntoResponse {
    match state.training_data_service.list().await {
        Ok(data_list) => {
            let dto_list: Vec<TrainingDataDto> =
                data_list.into_iter().map(TrainingDataDto::from).collect();
            let total = dto_list.len();

            Ok(Json(ListTrainingDataResponse {
                training_data: dto_list,
                total,
            }))
        }
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to list training data",
        )),
    }
}

/// Create new training data
#[utoipa::path(
    post,
    path = "/api/v1/training-data",
    request_body = CreateTrainingDataRequest,
    responses(
        (status = 201, description = "Training data created", body = TrainingDataDto),
        (status = 400, description = "Invalid request")
    ),
    tag = "training-data"
)]
pub async fn create_training_data(
    State(state): State<AppState>,
    Json(request): Json<CreateTrainingDataRequest>,
) -> impl IntoResponse {
    // Parse data type
    let data_type = match request.data_type.as_str() {
        "text" => DataType::Text,
        "json" => DataType::Json,
        "csv" => DataType::Csv,
        "parquet" => DataType::Parquet,
        "image" => DataType::Image,
        "audio" => DataType::Audio,
        "video" => DataType::Video,
        "binary" => DataType::Binary,
        custom => DataType::Custom(custom.to_string()),
    };

    let format = request.format.map(DataFormat::from).unwrap_or_default();
    let metadata = request
        .metadata
        .map(TrainingDataMetadata::from)
        .unwrap_or_default();

    match state
        .training_data_service
        .create(
            request.name,
            request.description,
            data_type,
            format,
            metadata,
        )
        .await
    {
        Ok(data) => Ok((StatusCode::CREATED, Json(TrainingDataDto::from(data)))),
        Err(e) => match e {
            kairei::training_data::TrainingDataError::AlreadyExists(_) => {
                Err((StatusCode::CONFLICT, "Training data already exists"))
            }
            _ => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to create training data",
            )),
        },
    }
}

/// Get specific training data by ID
#[utoipa::path(
    get,
    path = "/api/v1/training-data/{id}",
    params(
        ("id" = String, Path, description = "Training data ID")
    ),
    responses(
        (status = 200, description = "Training data details", body = TrainingDataDto),
        (status = 404, description = "Training data not found")
    ),
    tag = "training-data"
)]
pub async fn get_training_data(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let data_id = TrainingDataId::from_string(id);

    match state.training_data_service.get(&data_id).await {
        Ok(Some(data)) => Ok(Json(TrainingDataDto::from(data))),
        Ok(None) => Err((StatusCode::NOT_FOUND, "Training data not found")),
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to get training data",
        )),
    }
}

/// Update training data
#[utoipa::path(
    put,
    path = "/api/v1/training-data/{id}",
    params(
        ("id" = String, Path, description = "Training data ID")
    ),
    request_body = UpdateTrainingDataRequest,
    responses(
        (status = 200, description = "Training data updated", body = TrainingDataDto),
        (status = 404, description = "Training data not found")
    ),
    tag = "training-data"
)]
pub async fn update_training_data(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(request): Json<UpdateTrainingDataRequest>,
) -> impl IntoResponse {
    let data_id = TrainingDataId::from_string(id);

    // Get existing data
    let mut data = match state.training_data_service.get(&data_id).await {
        Ok(Some(data)) => data,
        Ok(None) => return Err((StatusCode::NOT_FOUND, "Training data not found")),
        Err(_) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to get training data",
            ));
        }
    };

    // Update fields if provided
    if let Some(name) = request.name {
        data.name = name;
    }
    if let Some(description) = request.description {
        data.description = Some(description);
    }
    if let Some(tags) = request.tags {
        data.metadata.tags = tags;
    }
    if let Some(archived) = request.archived {
        data.archived = archived;
    }

    // Update the data
    match state.training_data_service.update(data).await {
        Ok(updated_data) => Ok(Json(TrainingDataDto::from(updated_data))),
        Err(e) => match e {
            kairei::training_data::TrainingDataError::NotFound(_) => {
                Err((StatusCode::NOT_FOUND, "Training data not found"))
            }
            _ => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to update training data",
            )),
        },
    }
}

/// Delete training data
#[utoipa::path(
    delete,
    path = "/api/v1/training-data/{id}",
    params(
        ("id" = String, Path, description = "Training data ID")
    ),
    responses(
        (status = 204, description = "Training data deleted"),
        (status = 404, description = "Training data not found")
    ),
    tag = "training-data"
)]
pub async fn delete_training_data(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let data_id = TrainingDataId::from_string(id);

    match state.training_data_service.delete(&data_id).await {
        Ok(()) => Ok(StatusCode::NO_CONTENT),
        Err(e) => match e {
            kairei::training_data::TrainingDataError::NotFound(_) => {
                Err((StatusCode::NOT_FOUND, "Training data not found"))
            }
            _ => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to delete training data",
            )),
        },
    }
}

/// Upload file to training data
#[utoipa::path(
    post,
    path = "/api/v1/training-data/{id}/upload",
    params(
        ("id" = String, Path, description = "Training data ID")
    ),
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "File uploaded", body = TrainingDataDto),
        (status = 404, description = "Training data not found"),
        (status = 413, description = "File too large")
    ),
    tag = "training-data"
)]
pub async fn upload_training_file(
    State(state): State<AppState>,
    Path(id): Path<String>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let data_id = TrainingDataId::from_string(id);

    // Get the file from multipart
    let mut file_name = String::new();
    let mut file_content = Vec::new();

    while let Some(field) = multipart.next_field().await.unwrap_or(None) {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            file_name = field.file_name().unwrap_or("data").to_string();
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
        .training_data_service
        .upload_file(&data_id, &file_name, &file_content)
        .await
    {
        Ok(data) => Ok(Json(TrainingDataDto::from(data))),
        Err(e) => match e {
            kairei::training_data::TrainingDataError::NotFound(_) => {
                Err((StatusCode::NOT_FOUND, "Training data not found"))
            }
            kairei::training_data::TrainingDataError::FileTooLarge(_, _) => {
                Err((StatusCode::PAYLOAD_TOO_LARGE, "File too large"))
            }
            _ => Err((StatusCode::INTERNAL_SERVER_ERROR, "Failed to upload file")),
        },
    }
}

/// Get training data by LoRA ID
#[utoipa::path(
    get,
    path = "/api/v1/training-data/by-lora/{lora_id}",
    params(
        ("lora_id" = String, Path, description = "LoRA ID")
    ),
    responses(
        (status = 200, description = "Training data list", body = ListTrainingDataResponse)
    ),
    tag = "training-data"
)]
pub async fn get_training_data_by_lora(
    State(state): State<AppState>,
    Path(lora_id): Path<String>,
) -> impl IntoResponse {
    match state.training_data_service.list_by_lora(&lora_id).await {
        Ok(data_list) => {
            let dto_list: Vec<TrainingDataDto> =
                data_list.into_iter().map(TrainingDataDto::from).collect();
            let total = dto_list.len();

            Ok(Json(ListTrainingDataResponse {
                training_data: dto_list,
                total,
            }))
        }
        Err(_) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to list training data",
        )),
    }
}

/// Routes for training data endpoints
pub fn routes() -> axum::Router<AppState> {
    use axum::routing::{get, post};

    axum::Router::new()
        .route("/", get(list_training_data).post(create_training_data))
        .route("/by-lora/{lora_id}", get(get_training_data_by_lora))
        .route(
            "/{id}",
            get(get_training_data)
                .put(update_training_data)
                .delete(delete_training_data),
        )
        .route("/{id}/upload", post(upload_training_file))
}
