use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use kairei::base_model::{HuggingFaceDownloader, InMemoryBaseModelRepository};
use kairei::lora::InMemoryLoraRepository;
use kairei::storage::LocalStorage;
use kairei_api::{AppState, build_app, config::AuthConfig};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::util::ServiceExt;

/// Create a test application with in-memory repository
fn create_test_app() -> axum::Router {
    let repository = Arc::new(InMemoryBaseModelRepository::new());
    let lora_repository = Arc::new(InMemoryLoraRepository::new());
    let storage = Arc::new(LocalStorage::new("test_models"));
    let downloader = Arc::new(HuggingFaceDownloader::new(None));
    let auth_config = AuthConfig::default(); // Auth disabled by default
    let state = AppState::new(
        repository,
        lora_repository,
        storage,
        downloader,
        auth_config,
    );
    build_app(state)
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

    let body: Value = if body_bytes.is_empty() {
        json!(null)
    } else {
        // Try to parse as JSON, if it fails, treat as string
        serde_json::from_slice(&body_bytes)
            .unwrap_or_else(|_| json!(String::from_utf8_lossy(&body_bytes).to_string()))
    };

    (status, body)
}

#[tokio::test]
async fn test_list_models_empty() {
    let mut app = create_test_app();

    let (status, body) = json_request(&mut app, "GET", "/api/v1/models", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["models"].as_array().unwrap().len(), 0);
    assert_eq!(body["total"], 0);
}

#[tokio::test]
async fn test_create_model_success() {
    let mut app = create_test_app();

    let create_request = json!({
        "name": "test-model",
        "description": "Test model for unit tests",
        "repo_id": "test/repo",
        "filename": "model.bin",
        "size_mb": 1000,
        "metadata": {
            "repo_id": "test/repo",
            "name": "Test Model",
            "description": "A test model",
            "parameters": "1B",
            "architecture": "test",
            "quantization": "fp16"
        }
    });

    let (status, body) =
        json_request(&mut app, "POST", "/api/v1/models", Some(create_request)).await;

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["name"], "test-model");
    assert_eq!(body["status"], "available");
    assert!(body["id"].is_string());
}

#[tokio::test]
async fn test_create_duplicate_model_fails() {
    let mut app = create_test_app();

    let create_request = json!({
        "name": "duplicate-test",
        "description": "First model",
        "repo_id": "test/repo1",
        "filename": "model1.bin",
        "size_mb": 1000
    });

    // Create first model
    let (status, _) = json_request(
        &mut app,
        "POST",
        "/api/v1/models",
        Some(create_request.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // Try to create duplicate
    let duplicate_request = json!({
        "name": "duplicate-test",
        "description": "Second model",
        "repo_id": "test/repo2",
        "filename": "model2.bin",
        "size_mb": 2000
    });

    let (status, _) =
        json_request(&mut app, "POST", "/api/v1/models", Some(duplicate_request)).await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_create_model_empty_name_fails() {
    let mut app = create_test_app();

    let invalid_request = json!({
        "name": "",
        "description": "Invalid model",
        "repo_id": "test/repo",
        "filename": "model.bin",
        "size_mb": 1000
    });

    let (status, _) = json_request(&mut app, "POST", "/api/v1/models", Some(invalid_request)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_get_model_by_id() {
    let mut app = create_test_app();

    // Create a model first
    let create_request = json!({
        "name": "get-test",
        "description": "Model for GET test",
        "repo_id": "test/repo",
        "filename": "model.bin",
        "size_mb": 1000
    });

    let (_, created_body) =
        json_request(&mut app, "POST", "/api/v1/models", Some(create_request)).await;
    let model_id = created_body["id"].as_str().unwrap();

    // Get the model
    let (status, body) = json_request(
        &mut app,
        "GET",
        &format!("/api/v1/models/{}", model_id),
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], model_id);
    assert_eq!(body["name"], "get-test");
}

#[tokio::test]
async fn test_get_nonexistent_model() {
    let mut app = create_test_app();

    let (status, _) = json_request(&mut app, "GET", "/api/v1/models/nonexistent-id", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_model() {
    let mut app = create_test_app();

    // Create a model first
    let create_request = json!({
        "name": "update-test",
        "description": "Original description",
        "repo_id": "test/repo",
        "filename": "model.bin",
        "size_mb": 1000
    });

    let (_, created_body) =
        json_request(&mut app, "POST", "/api/v1/models", Some(create_request)).await;
    let model_id = created_body["id"].as_str().unwrap();

    // Update the model
    let update_request = json!({
        "description": "Updated description",
        "metadata": {
            "repo_id": "test/repo",
            "name": "Updated Model",
            "description": "Now with metadata",
            "downloaded_at": "2024-01-01T00:00:00Z",
            "parameters": "1B",
            "architecture": "test",
            "quantization": "fp32"
        }
    });

    let (status, body) = json_request(
        &mut app,
        "PUT",
        &format!("/api/v1/models/{}", model_id),
        Some(update_request),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["description"], "Updated description");
    assert_eq!(body["metadata"]["downloaded_at"], "2024-01-01T00:00:00Z");
}

#[tokio::test]
async fn test_delete_model() {
    let mut app = create_test_app();

    // Create a model first
    let create_request = json!({
        "name": "delete-test",
        "description": "Model to be deleted",
        "repo_id": "test/repo",
        "filename": "model.bin",
        "size_mb": 1000
    });

    let (_, created_body) =
        json_request(&mut app, "POST", "/api/v1/models", Some(create_request)).await;
    let model_id = created_body["id"].as_str().unwrap();

    // Delete the model
    let (status, _) = json_request(
        &mut app,
        "DELETE",
        &format!("/api/v1/models/{}", model_id),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    // Verify it's deleted
    let (status, _) = json_request(
        &mut app,
        "GET",
        &format!("/api/v1/models/{}", model_id),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_nonexistent_model() {
    let mut app = create_test_app();

    let (status, _) = json_request(&mut app, "DELETE", "/api/v1/models/nonexistent-id", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_list_models_after_operations() {
    let mut app = create_test_app();

    // Create multiple models
    for i in 1..=3 {
        let create_request = json!({
            "name": format!("model-{}", i),
            "description": format!("Test model {}", i),
            "repo_id": format!("test/repo{}", i),
            "filename": "model.bin",
            "size_mb": 1000 * i
        });

        let (status, _) =
            json_request(&mut app, "POST", "/api/v1/models", Some(create_request)).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    // List all models
    let (status, body) = json_request(&mut app, "GET", "/api/v1/models", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 3);

    let models = body["models"].as_array().unwrap();
    assert_eq!(models.len(), 3);

    // Verify model names
    let names: Vec<String> = models
        .iter()
        .map(|m| m["name"].as_str().unwrap().to_string())
        .collect();
    assert!(names.contains(&"model-1".to_string()));
    assert!(names.contains(&"model-2".to_string()));
    assert!(names.contains(&"model-3".to_string()));
}
