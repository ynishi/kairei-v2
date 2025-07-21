use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{Row, postgres::PgPool};
use uuid::Uuid;

use super::{Lora, LoraError, LoraId, LoraMetadata, LoraRepository, LoraStatus, Result};
use crate::base_model::BaseModelId;

/// PostgreSQL implementation of LoraRepository
pub struct PostgresLoraRepository {
    pool: PgPool,
}

impl PostgresLoraRepository {
    /// Create a new PostgresLoraRepository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl LoraRepository for PostgresLoraRepository {
    async fn create(&self, lora: Lora) -> Result<Lora> {
        let metadata_json = serde_json::to_value(&lora.metadata).map_err(|e| {
            LoraError::SerializationError(format!("Failed to serialize metadata: {}", e))
        })?;

        let id = Uuid::parse_str(lora.id.as_str())
            .map_err(|e| LoraError::InvalidMetadata(format!("Invalid UUID: {}", e)))?;

        let base_model_id = lora
            .base_model_id
            .as_ref()
            .map(|id| {
                Uuid::parse_str(id.as_str()).map_err(|e| {
                    LoraError::InvalidMetadata(format!("Invalid base model UUID: {}", e))
                })
            })
            .transpose()?;

        let _result = sqlx::query(
            r#"
            INSERT INTO loras (
                id, name, description, base_model_id, created_at, updated_at,
                status, archived, file_path, size_bytes, metadata
            ) VALUES (
                $1, $2, $3, $4, $5::timestamptz, $6::timestamptz, $7, $8, $9, $10, $11
            )
            "#,
        )
        .bind(id)
        .bind(&lora.name)
        .bind(&lora.description)
        .bind(base_model_id)
        .bind(&lora.created_at)
        .bind(&lora.updated_at)
        .bind(lora.status.to_string())
        .bind(lora.archived)
        .bind(&lora.file_path)
        .bind(lora.size_bytes.map(|s| s as i64))
        .bind(metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(db_err) if db_err.is_unique_violation() => {
                LoraError::AlreadyExists(lora.name.clone())
            }
            _ => LoraError::DatabaseError(e.to_string()),
        })?;

        Ok(lora)
    }

    async fn get(&self, id: &LoraId) -> Result<Option<Lora>> {
        let uuid = Uuid::parse_str(id.as_str())
            .map_err(|e| LoraError::InvalidMetadata(format!("Invalid UUID: {}", e)))?;

        let row = sqlx::query(
            r#"
            SELECT id, name, description, base_model_id, created_at, updated_at,
                   status, archived, file_path, size_bytes, metadata
            FROM loras
            WHERE id = $1
            "#,
        )
        .bind(uuid)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LoraError::DatabaseError(e.to_string()))?;

        match row {
            Some(row) => {
                let id_val: Uuid = row.get("id");
                let base_model_id_val: Option<Uuid> = row.get("base_model_id");
                let metadata_json: serde_json::Value = row.get("metadata");
                let metadata: LoraMetadata =
                    serde_json::from_value(metadata_json).map_err(|e| {
                        LoraError::InvalidMetadata(format!("Failed to deserialize metadata: {}", e))
                    })?;

                let status_str: String = row.get("status");
                let status = status_str.parse::<LoraStatus>().map_err(|_| {
                    LoraError::InvalidMetadata(format!("Invalid status: {}", status_str))
                })?;

                let created_at: DateTime<Utc> = row.get("created_at");
                let updated_at: DateTime<Utc> = row.get("updated_at");

                Ok(Some(Lora {
                    id: LoraId::from_string(id_val.to_string()),
                    name: row.get("name"),
                    description: row.get("description"),
                    base_model_id: base_model_id_val
                        .map(|id| BaseModelId::from_string(id.to_string())),
                    created_at: created_at.to_rfc3339(),
                    updated_at: updated_at.to_rfc3339(),
                    status,
                    archived: row.get("archived"),
                    file_path: row.get("file_path"),
                    size_bytes: row.get::<Option<i64>, _>("size_bytes").map(|s| s as u64),
                    metadata,
                }))
            }
            None => Ok(None),
        }
    }

    async fn get_by_name(&self, name: &str) -> Result<Option<Lora>> {
        let row = sqlx::query(
            r#"
            SELECT id, name, description, base_model_id, created_at, updated_at,
                   status, archived, file_path, size_bytes, metadata
            FROM loras
            WHERE name = $1
            "#,
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| LoraError::DatabaseError(e.to_string()))?;

        match row {
            Some(row) => {
                let id_val: Uuid = row.get("id");
                let base_model_id_val: Option<Uuid> = row.get("base_model_id");
                let metadata_json: serde_json::Value = row.get("metadata");
                let metadata: LoraMetadata =
                    serde_json::from_value(metadata_json).map_err(|e| {
                        LoraError::InvalidMetadata(format!("Failed to deserialize metadata: {}", e))
                    })?;

                let status_str: String = row.get("status");
                let status = status_str.parse::<LoraStatus>().map_err(|_| {
                    LoraError::InvalidMetadata(format!("Invalid status: {}", status_str))
                })?;

                let created_at: DateTime<Utc> = row.get("created_at");
                let updated_at: DateTime<Utc> = row.get("updated_at");

                Ok(Some(Lora {
                    id: LoraId::from_string(id_val.to_string()),
                    name: row.get("name"),
                    description: row.get("description"),
                    base_model_id: base_model_id_val
                        .map(|id| BaseModelId::from_string(id.to_string())),
                    created_at: created_at.to_rfc3339(),
                    updated_at: updated_at.to_rfc3339(),
                    status,
                    archived: row.get("archived"),
                    file_path: row.get("file_path"),
                    size_bytes: row.get::<Option<i64>, _>("size_bytes").map(|s| s as u64),
                    metadata,
                }))
            }
            None => Ok(None),
        }
    }

    async fn list(&self) -> Result<Vec<Lora>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, description, base_model_id, created_at, updated_at,
                   status, archived, file_path, size_bytes, metadata
            FROM loras
            WHERE NOT archived
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| LoraError::DatabaseError(e.to_string()))?;

        let mut loras = Vec::new();
        for row in rows {
            let id_val: Uuid = row.get("id");
            let base_model_id_val: Option<Uuid> = row.get("base_model_id");
            let metadata_json: serde_json::Value = row.get("metadata");
            let metadata: LoraMetadata = serde_json::from_value(metadata_json).map_err(|e| {
                LoraError::InvalidMetadata(format!("Failed to deserialize metadata: {}", e))
            })?;

            let status_str: String = row.get("status");
            let status = status_str.parse::<LoraStatus>().map_err(|_| {
                LoraError::InvalidMetadata(format!("Invalid status: {}", status_str))
            })?;

            let created_at: DateTime<Utc> = row.get("created_at");
            let updated_at: DateTime<Utc> = row.get("updated_at");

            loras.push(Lora {
                id: LoraId::from_string(id_val.to_string()),
                name: row.get("name"),
                description: row.get("description"),
                base_model_id: base_model_id_val.map(|id| BaseModelId::from_string(id.to_string())),
                created_at: created_at.to_rfc3339(),
                updated_at: updated_at.to_rfc3339(),
                status,
                archived: row.get("archived"),
                file_path: row.get("file_path"),
                size_bytes: row.get::<Option<i64>, _>("size_bytes").map(|s| s as u64),
                metadata,
            });
        }

        Ok(loras)
    }

    async fn update(&self, lora: Lora) -> Result<Lora> {
        let metadata_json = serde_json::to_value(&lora.metadata).map_err(|e| {
            LoraError::SerializationError(format!("Failed to serialize metadata: {}", e))
        })?;

        let id = Uuid::parse_str(lora.id.as_str())
            .map_err(|e| LoraError::InvalidMetadata(format!("Invalid UUID: {}", e)))?;

        let base_model_id = lora
            .base_model_id
            .as_ref()
            .map(|id| {
                Uuid::parse_str(id.as_str()).map_err(|e| {
                    LoraError::InvalidMetadata(format!("Invalid base model UUID: {}", e))
                })
            })
            .transpose()?;

        let result = sqlx::query(
            r#"
            UPDATE loras
            SET name = $2, description = $3, base_model_id = $4, updated_at = $5::timestamptz,
                status = $6, archived = $7, file_path = $8, size_bytes = $9, metadata = $10
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&lora.name)
        .bind(&lora.description)
        .bind(base_model_id)
        .bind(&lora.updated_at)
        .bind(lora.status.to_string())
        .bind(lora.archived)
        .bind(&lora.file_path)
        .bind(lora.size_bytes.map(|s| s as i64))
        .bind(metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| LoraError::DatabaseError(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(LoraError::NotFound(lora.id.to_string()));
        }

        Ok(lora)
    }

    async fn delete(&self, id: &LoraId) -> Result<()> {
        let uuid = Uuid::parse_str(id.as_str())
            .map_err(|e| LoraError::InvalidMetadata(format!("Invalid UUID: {}", e)))?;

        let result = sqlx::query("DELETE FROM loras WHERE id = $1")
            .bind(uuid)
            .execute(&self.pool)
            .await
            .map_err(|e| LoraError::DatabaseError(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(LoraError::NotFound(id.to_string()));
        }

        Ok(())
    }
}
