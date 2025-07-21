use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

use crate::AppState;

/// JWT Claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,           // Subject (user ID)
    pub exp: Option<u64>,      // Expiration time
    pub iat: Option<u64>,      // Issued at
    pub aud: Option<String>,   // Audience
    pub iss: Option<String>,   // Issuer
    pub email: Option<String>, // User email
}

/// Authentication middleware
pub async fn auth_middleware(
    State(_state): State<AppState>,
    mut request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract Authorization header
    let auth_header = request
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok());

    if let Some(auth_header) = auth_header {
        // Check if it's a Bearer token
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            // TODO: Implement JWT validation
            // For now, we'll just log that we received a token
            tracing::debug!("Received JWT token: {}...", &token[..20.min(token.len())]);

            // TODO: Steps to implement:
            // 1. Validate JWT signature using Auth0 JWKS
            // 2. Check expiration
            // 3. Verify audience and issuer
            // 4. Extract claims and add to request extensions

            // Placeholder: Create dummy claims for now
            let claims = Claims {
                sub: "placeholder-user".to_string(),
                exp: None,
                iat: None,
                aud: None,
                iss: None,
                email: Some("user@example.com".to_string()),
            };

            // Add claims to request extensions so handlers can access them
            request.extensions_mut().insert(claims);
        }
    }

    // For now, allow all requests through
    // TODO: Make this configurable based on route protection requirements
    Ok(next.run(request).await)
}

/// Helper function to require authentication
/// This can be used in handlers to ensure authentication
pub async fn require_auth(
    claims: Option<axum::extract::Extension<Claims>>,
) -> Result<Claims, impl IntoResponse> {
    match claims {
        Some(axum::extract::Extension(claims)) => Ok(claims),
        None => Err((StatusCode::UNAUTHORIZED, "Authentication required")),
    }
}
