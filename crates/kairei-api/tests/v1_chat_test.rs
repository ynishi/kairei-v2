//! Tests for Chat API endpoints

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use kairei::base_model::{HuggingFaceDownloader, InMemoryBaseModelRepository};
use kairei::chat::{ChatConfig, GenerationParams, ModelType};
use kairei::lora::InMemoryLoraRepository;
use kairei::storage::LocalStorage;
use kairei::training_data::InMemoryTrainingDataRepository;
use kairei_api::routes::v1::chat::{
    CreateSessionRequest, CreateSessionResponse,
};
use kairei_api::{AppState, build_app, config::AuthConfig};
use std::sync::Arc;
use tower::util::ServiceExt;

/// Create a test application with in-memory repository
fn create_test_app() -> axum::Router {
    let base_model_repository = Arc::new(InMemoryBaseModelRepository::new());
    let lora_repository = Arc::new(InMemoryLoraRepository::new());
    let training_data_repository = Arc::new(InMemoryTrainingDataRepository::new());
    let storage = Arc::new(LocalStorage::new("test_chat"));
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
    build_app(state)
}

#[tokio::test]
async fn test_create_session() {
    let app = create_test_app();

    let config = ChatConfig {
        model_type: ModelType::Llama2,
        base_model: None,
        lora_models: vec![],
        parameters: GenerationParams::default(),
    };

    let request = Request::builder()
        .uri("/api/v1/chat/sessions")
        .method("POST")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_string(&CreateSessionRequest { config }).unwrap(),
        ))
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::CREATED);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let session_response: CreateSessionResponse = serde_json::from_slice(&body).unwrap();
    assert!(!session_response.session.id.is_empty());
    assert_eq!(session_response.session.messages.len(), 0);
}

/*
#[tokio::test]
async fn test_send_message() -> TestResult<()> {
    let app = app().await?;

    // First create a session
    let config = ChatConfig {
        model_type: ModelType::Llama2,
        base_model: None,
        lora_models: vec![],
        parameters: GenerationParams::default(),
    };

    let create_response = app
        .post("/api/v1/chat/sessions")
        .json(&CreateSessionRequest { config })
        .await;

    let session_response: CreateSessionResponse = create_response.json();
    let session_id = session_response.session.id;

    // Send a message
    let message_response = app
        .post(format!("/api/v1/chat/sessions/{}/messages", session_id))
        .json(&SendMessageRequest { content: "Hello!".to_string() })
        .await;

    message_response.assert_status_ok();
    let body: SendMessageResponse = message_response.json();
    assert_eq!(body.message.content, "Echo: Hello!");

    Ok(())
}

#[tokio::test]
async fn test_get_session() -> TestResult<()> {
    let app = app().await?;

    // First create a session
    let config = ChatConfig {
        model_type: ModelType::Llama2,
        base_model: None,
        lora_models: vec![],
        parameters: GenerationParams::default(),
    };

    let create_response = app
        .post("/api/v1/chat/sessions")
        .json(&CreateSessionRequest { config })
        .await;

    let session_response: CreateSessionResponse = create_response.json();
    let session_id = session_response.session.id;

    // Send a message to have some history
    app.post(format!("/api/v1/chat/sessions/{}/messages", session_id))
        .json(&SendMessageRequest { content: "Test message".to_string() })
        .await;

    // Get session details
    let get_response = app
        .get(format!("/api/v1/chat/sessions/{}", session_id))
        .await;

    get_response.assert_status_ok();
    let body: serde_json::Value = get_response.json();
    assert_eq!(body["session"]["messages"].as_array().unwrap().len(), 2); // User + Assistant

    Ok(())
}

#[tokio::test]
async fn test_session_not_found() -> TestResult<()> {
    let app = app().await?;

    let response = app
        .get("/api/v1/chat/sessions/non-existent-id")
        .await;

    response.assert_status(StatusCode::NOT_FOUND);

    Ok(())
}
*/
