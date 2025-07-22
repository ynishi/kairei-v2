use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use kairei::base_model::{HuggingFaceDownloader, InMemoryBaseModelRepository};
use kairei::lora::InMemoryLoraRepository;
use kairei::storage::LocalStorage;
use kairei::training_data::InMemoryTrainingDataRepository;
use kairei_api::{AppState, build_app, config::AuthConfig};
use serde_json::{Value, json};
use std::sync::Arc;
use tempfile::TempDir;
use tower::util::ServiceExt;

/// Create a test application with in-memory repository and temp storage
fn create_test_app() -> (axum::Router, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let base_model_repository = Arc::new(InMemoryBaseModelRepository::new());
    let lora_repository = Arc::new(InMemoryLoraRepository::new());
    let training_data_repository = Arc::new(InMemoryTrainingDataRepository::new());
    let storage = Arc::new(LocalStorage::new(temp_dir.path().to_str().unwrap()));
    let downloader = Arc::new(HuggingFaceDownloader::new(None));
    let auth_config = AuthConfig::default(); // Auth disabled by default

    let state = AppState::new(
        base_model_repository,
        lora_repository,
        training_data_repository,
        storage,
        downloader,
        auth_config,
    );

    (build_app(state), temp_dir)
}

/// Helper function to make JSON requests
async fn json_request(
    app: &mut axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let request_builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");

    let request = if let Some(body) = body {
        request_builder
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap()
    } else {
        request_builder.body(Body::empty()).unwrap()
    };

    let response = app.oneshot(request).await.unwrap();
    let status = response.status();

    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body_value: Value = if body_bytes.is_empty() {
        json!(null)
    } else {
        match serde_json::from_slice(&body_bytes) {
            Ok(json) => json,
            Err(_) => {
                // If JSON parsing fails, return the raw text as a string
                let text = String::from_utf8_lossy(&body_bytes);
                json!(text.to_string())
            }
        }
    };

    (status, body_value)
}

#[tokio::test]
async fn test_train_lora_endpoint() {
    let (mut app, _temp_dir) = create_test_app();

    let train_request = json!({
        "train_data": "test_data.jsonl",
        "epochs": 3,
        "lora_r": 8,
        "lora_alpha": 16,
        "output_dir": null,
        "model": "test-model",
        "batch_size": 2,
        "learning_rate": 0.0001,
        "test": false
    });

    let (status, body) =
        json_request(&mut app, "POST", "/api/v1/train/lora", Some(train_request)).await;

    // Debug print
    println!("Train LoRA response: {:?}", body);

    // Since we don't have the actual training script in test environment,
    // we expect an error response
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"].as_str().is_some());
    assert!(
        body["error"]
            .as_str()
            .unwrap()
            .contains("Training script not found")
    );
}

#[tokio::test]
async fn test_train_lora_with_all_params() {
    let (mut app, _temp_dir) = create_test_app();

    let train_request = json!({
        "train_data": "training/data/culture.jsonl",
        "epochs": 10,
        "lora_r": 16,
        "lora_alpha": 32,
        "output_dir": "training/output/my_model",
        "model": "meta-llama/Llama-2-7b-hf",
        "batch_size": 4,
        "learning_rate": 0.0003,
        "test": true
    });

    let (status, body) =
        json_request(&mut app, "POST", "/api/v1/train/lora", Some(train_request)).await;

    // We still expect an error due to missing script, but the request should be processed
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    assert!(body["error"].is_string());
}

#[tokio::test]
async fn test_train_lora_invalid_request() {
    let (mut app, _temp_dir) = create_test_app();

    // Missing required fields
    let invalid_request = json!({
        "train_data": "test_data.jsonl",
        "epochs": 3
        // Missing other required fields
    });

    let (status, _body) = json_request(
        &mut app,
        "POST",
        "/api/v1/train/lora",
        Some(invalid_request),
    )
    .await;

    // Should return 422 Unprocessable Entity for invalid JSON
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_train_lora_empty_body() {
    let (mut app, _temp_dir) = create_test_app();

    let (status, _body) = json_request(&mut app, "POST", "/api/v1/train/lora", None).await;

    // Should return 400 Bad Request for missing body
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_train_request_serialization() {
    use kairei_api::routes::v1::trainer::TrainRequest;

    let train_request = TrainRequest {
        train_data: "data.jsonl".to_string(),
        epochs: 5,
        lora_r: 8,
        lora_alpha: 16,
        output_dir: Some("output".to_string()),
        model: "model".to_string(),
        batch_size: 2,
        learning_rate: 0.001,
        test: true,
    };

    // Test serialization
    let json = serde_json::to_value(&train_request).unwrap();
    assert_eq!(json["train_data"], "data.jsonl");
    assert_eq!(json["epochs"], 5);
    assert_eq!(json["lora_r"], 8);
    assert_eq!(json["lora_alpha"], 16);
    assert_eq!(json["output_dir"], "output");
    assert_eq!(json["model"], "model");
    assert_eq!(json["batch_size"], 2);
    assert!((json["learning_rate"].as_f64().unwrap() - 0.001).abs() < 0.00001);
    assert_eq!(json["test"], true);

    // Test deserialization
    let deserialized: TrainRequest = serde_json::from_value(json).unwrap();
    assert_eq!(deserialized.train_data, "data.jsonl");
    assert_eq!(deserialized.epochs, 5);
}

#[tokio::test]
async fn test_train_response_serialization() {
    use kairei_api::routes::v1::trainer::TrainResponse;

    let train_response = TrainResponse {
        output_dir: "output/model".to_string(),
        status: "Completed".to_string(),
        final_loss: Some(0.123),
        duration: Some("1h 30m".to_string()),
        error: None,
        checkpoints: vec!["checkpoint-100".to_string(), "checkpoint-200".to_string()],
    };

    // Test serialization
    let json = serde_json::to_value(&train_response).unwrap();
    assert_eq!(json["output_dir"], "output/model");
    assert_eq!(json["status"], "Completed");
    assert!((json["final_loss"].as_f64().unwrap() - 0.123).abs() < 0.00001);
    assert_eq!(json["duration"], "1h 30m");
    assert!(json["error"].is_null());
    assert_eq!(json["checkpoints"].as_array().unwrap().len(), 2);
}
