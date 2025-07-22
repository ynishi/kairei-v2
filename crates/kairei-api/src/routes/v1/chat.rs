//! Chat API endpoints

use crate::AppState;
use axum::{
    Router,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
};
use kairei::chat::{ChatConfig, ChatMessage, ChatSession};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub config: ChatConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session: ChatSession,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendMessageRequest {
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SendMessageResponse {
    pub message: ChatMessage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetSessionResponse {
    pub session: ChatSession,
}

/// Create a new chat session
async fn create_session(
    State(state): State<AppState>,
    Json(req): Json<CreateSessionRequest>,
) -> impl IntoResponse {
    match state.chat_service.create_session(req.config).await {
        Ok(session) => Ok((StatusCode::CREATED, Json(CreateSessionResponse { session }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create session: {}", e),
        )),
    }
}

/// Send a message to a session
async fn send_message(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> impl IntoResponse {
    let user_message = ChatMessage::new_user(req.content);

    match state
        .chat_service
        .send_message(&session_id, user_message)
        .await
    {
        Ok(response_message) => Ok((
            StatusCode::OK,
            Json(SendMessageResponse {
                message: response_message,
            }),
        )),
        Err(e) => {
            if e.to_string().contains("not found") {
                Err((StatusCode::NOT_FOUND, format!("Session not found: {}", e)))
            } else {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to send message: {}", e),
                ))
            }
        }
    }
}

/// Get session details
async fn get_session(
    State(state): State<AppState>,
    Path(session_id): Path<String>,
) -> impl IntoResponse {
    match state.chat_service.get_session(&session_id).await {
        Ok(session) => Ok(Json(GetSessionResponse { session })),
        Err(e) => Err((StatusCode::NOT_FOUND, format!("Session not found: {}", e))),
    }
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/sessions", axum::routing::post(create_session))
        .route("/sessions/{session_id}", axum::routing::get(get_session))
        .route(
            "/sessions/{session_id}/messages",
            axum::routing::post(send_message),
        )
}
