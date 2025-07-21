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
    let base_model_repository = Arc::new(InMemoryBaseModelRepository::new());
    let lora_repository = Arc::new(InMemoryLoraRepository::new());
    let storage = Arc::new(LocalStorage::new("test_loras"));
    let downloader = Arc::new(HuggingFaceDownloader::new(None));
    let auth_config = AuthConfig::default(); // Auth disabled by default
    let state = AppState::new(
        base_model_repository,
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
async fn test_list_loras_empty() {
    let mut app = create_test_app();

    let (status, body) = json_request(&mut app, "GET", "/api/v1/loras", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["loras"], json!([]));
    assert_eq!(body["total"], 0);
}

#[tokio::test]
async fn test_create_lora() {
    let mut app = create_test_app();

    let create_request = json!({
        "name": "test-lora",
        "description": "Test LoRA model",
        "metadata": {
            "rank": 16,
            "alpha": 32.0
        }
    });

    let (status, body) =
        json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;

    // Debug print
    println!("Create LoRA response: {:?}", body);

    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(body["name"], "test-lora");
    assert_eq!(body["description"], "Test LoRA model");
    assert_eq!(body["metadata"]["rank"], 16);
    assert_eq!(body["metadata"]["alpha"], 32.0);
    assert!(!body["id"].as_str().unwrap().is_empty());
}

#[tokio::test]
async fn test_create_lora_duplicate_name() {
    let mut app = create_test_app();

    let create_request = json!({
        "name": "duplicate-lora",
        "description": "First LoRA"
    });

    // Create first LoRA
    let (status, _) = json_request(
        &mut app,
        "POST",
        "/api/v1/loras",
        Some(create_request.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);

    // Try to create duplicate
    let (status, _) = json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_list_loras_with_data() {
    let mut app = create_test_app();

    // Create some LoRAs
    let lora1 = json!({
        "name": "lora-1",
        "description": "First LoRA"
    });
    let lora2 = json!({
        "name": "lora-2",
        "description": "Second LoRA"
    });

    json_request(&mut app, "POST", "/api/v1/loras", Some(lora1)).await;
    json_request(&mut app, "POST", "/api/v1/loras", Some(lora2)).await;

    // List LoRAs
    let (status, body) = json_request(&mut app, "GET", "/api/v1/loras", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["total"], 2);
    assert_eq!(body["loras"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_get_lora() {
    let mut app = create_test_app();

    // Create a LoRA first
    let create_request = json!({
        "name": "test-get-lora",
        "description": "Test LoRA for GET endpoint"
    });

    let (status, created) =
        json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;
    assert_eq!(status, StatusCode::CREATED);
    let lora_id = created["id"].as_str().unwrap();

    // Get the LoRA by ID
    let (status, body) =
        json_request(&mut app, "GET", &format!("/api/v1/loras/{}", lora_id), None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], lora_id);
    assert_eq!(body["name"], "test-get-lora");
    assert_eq!(body["description"], "Test LoRA for GET endpoint");
}

#[tokio::test]
async fn test_get_lora_not_found() {
    let mut app = create_test_app();

    let (status, _) = json_request(&mut app, "GET", "/api/v1/loras/nonexistent-id", None).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_update_lora() {
    let mut app = create_test_app();

    // Create a LoRA first
    let create_request = json!({
        "name": "test-update-lora",
        "description": "Original description"
    });

    let (status, created) =
        json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;
    assert_eq!(status, StatusCode::CREATED);
    let lora_id = created["id"].as_str().unwrap();

    // Update the LoRA
    let update_request = json!({
        "name": "updated-lora-name",
        "description": "Updated description"
    });

    let (status, body) = json_request(
        &mut app,
        "PUT",
        &format!("/api/v1/loras/{}", lora_id),
        Some(update_request),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], "updated-lora-name");
    assert_eq!(body["description"], "Updated description");
}

#[tokio::test]
async fn test_delete_lora() {
    let mut app = create_test_app();

    // Create a LoRA first
    let create_request = json!({
        "name": "test-delete-lora",
        "description": "LoRA to be deleted"
    });

    let (status, created) =
        json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;
    assert_eq!(status, StatusCode::CREATED);
    let lora_id = created["id"].as_str().unwrap();

    // Delete the LoRA
    let (status, _) = json_request(
        &mut app,
        "DELETE",
        &format!("/api/v1/loras/{}", lora_id),
        None,
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);

    // Verify it's deleted
    let (status, _) =
        json_request(&mut app, "GET", &format!("/api/v1/loras/{}", lora_id), None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_lora_by_name() {
    let mut app = create_test_app();

    // Create a LoRA first
    let create_request = json!({
        "name": "unique-lora-name",
        "description": "LoRA with unique name"
    });

    let (status, _) = json_request(&mut app, "POST", "/api/v1/loras", Some(create_request)).await;
    assert_eq!(status, StatusCode::CREATED);

    // Get by name
    let (status, body) = json_request(
        &mut app,
        "GET",
        "/api/v1/loras/by-name/unique-lora-name",
        None,
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["name"], "unique-lora-name");
    assert_eq!(body["description"], "LoRA with unique name");
}
