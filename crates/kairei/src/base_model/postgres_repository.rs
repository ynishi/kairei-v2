use async_trait::async_trait;
use sqlx::{Row, postgres::PgPool};
use uuid::Uuid;

use super::{BaseModel, BaseModelError, BaseModelId, BaseModelRepository, BaseModelResult};

/// PostgreSQL implementation of BaseModelRepository
pub struct PostgresBaseModelRepository {
    pool: PgPool,
}

impl PostgresBaseModelRepository {
    /// Create a new PostgresBaseModelRepository
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl BaseModelRepository for PostgresBaseModelRepository {
    async fn create(&self, model: BaseModel) -> BaseModelResult<BaseModel> {
        let metadata_json = model
            .metadata
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to serialize metadata: {}", e))
            })?;

        let id = Uuid::parse_str(model.id.as_str())
            .map_err(|e| BaseModelError::InvalidData(format!("Invalid UUID: {}", e)))?;

        let _result = sqlx::query(
            r#"
            INSERT INTO base_models (
                id, name, description, repo_id, filename, size_mb, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
            "#,
        )
        .bind(id)
        .bind(&model.name)
        .bind(&model.description)
        .bind(&model.repo_id)
        .bind(&model.filename)
        .bind(model.size_mb.map(|s| s as i64))
        .bind(metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| match e {
            sqlx::Error::Database(db_err) if db_err.is_unique_violation() => {
                BaseModelError::AlreadyExists(model.name.clone())
            }
            _ => BaseModelError::DatabaseError(e.to_string()),
        })?;

        Ok(model)
    }

    async fn get(&self, id: &BaseModelId) -> BaseModelResult<BaseModel> {
        let uuid = Uuid::parse_str(id.as_str())
            .map_err(|e| BaseModelError::InvalidData(format!("Invalid UUID: {}", e)))?;

        let row = sqlx::query(
            r#"
            SELECT id, name, description, repo_id, filename, size_mb, metadata
            FROM base_models
            WHERE id = $1
            "#,
        )
        .bind(uuid)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?
        .ok_or_else(|| BaseModelError::NotFound(id.to_string()))?;

        let id_val: Uuid = row.get("id");
        let metadata_json: Option<serde_json::Value> = row.get("metadata");
        let metadata = metadata_json
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to deserialize metadata: {}", e))
            })?;

        Ok(BaseModel {
            id: BaseModelId::from_string(id_val.to_string()),
            name: row.get("name"),
            description: row.get("description"),
            repo_id: row.get("repo_id"),
            filename: row.get("filename"),
            size_mb: row.get::<Option<i64>, _>("size_mb").map(|s| s as u64),
            metadata,
        })
    }

    async fn get_by_name(&self, name: &str) -> BaseModelResult<BaseModel> {
        let row = sqlx::query(
            r#"
            SELECT id, name, description, repo_id, filename, size_mb, metadata
            FROM base_models
            WHERE name = $1
            "#,
        )
        .bind(name)
        .fetch_optional(&self.pool)
        .await
        .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?
        .ok_or_else(|| BaseModelError::NotFound(name.to_string()))?;

        let id_val: Uuid = row.get("id");
        let metadata_json: Option<serde_json::Value> = row.get("metadata");
        let metadata = metadata_json
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to deserialize metadata: {}", e))
            })?;

        Ok(BaseModel {
            id: BaseModelId::from_string(id_val.to_string()),
            name: row.get("name"),
            description: row.get("description"),
            repo_id: row.get("repo_id"),
            filename: row.get("filename"),
            size_mb: row.get::<Option<i64>, _>("size_mb").map(|s| s as u64),
            metadata,
        })
    }

    async fn update(&self, model: BaseModel) -> BaseModelResult<BaseModel> {
        let metadata_json = model
            .metadata
            .as_ref()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|e| {
                BaseModelError::InvalidData(format!("Failed to serialize metadata: {}", e))
            })?;

        let id = Uuid::parse_str(model.id.as_str())
            .map_err(|e| BaseModelError::InvalidData(format!("Invalid UUID: {}", e)))?;

        let result = sqlx::query(
            r#"
            UPDATE base_models 
            SET name = $2, description = $3, repo_id = $4, filename = $5, 
                size_mb = $6, metadata = $7, updated_at = CURRENT_TIMESTAMP
            WHERE id = $1
            "#,
        )
        .bind(id)
        .bind(&model.name)
        .bind(&model.description)
        .bind(&model.repo_id)
        .bind(&model.filename)
        .bind(model.size_mb.map(|s| s as i64))
        .bind(metadata_json)
        .execute(&self.pool)
        .await
        .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(BaseModelError::NotFound(model.id.to_string()));
        }

        Ok(model)
    }

    async fn delete(&self, id: &BaseModelId) -> BaseModelResult<()> {
        let uuid = Uuid::parse_str(id.as_str())
            .map_err(|e| BaseModelError::InvalidData(format!("Invalid UUID: {}", e)))?;

        let result = sqlx::query("DELETE FROM base_models WHERE id = $1")
            .bind(uuid)
            .execute(&self.pool)
            .await
            .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?;

        if result.rows_affected() == 0 {
            return Err(BaseModelError::NotFound(id.to_string()));
        }

        Ok(())
    }

    async fn list(&self) -> BaseModelResult<Vec<BaseModel>> {
        let rows = sqlx::query(
            r#"
            SELECT id, name, description, repo_id, filename, size_mb, metadata
            FROM base_models
            ORDER BY created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?;

        let mut models = Vec::new();
        for row in rows {
            let id_val: Uuid = row.get("id");
            let metadata_json: Option<serde_json::Value> = row.get("metadata");
            let metadata = metadata_json
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| {
                    BaseModelError::InvalidData(format!("Failed to deserialize metadata: {}", e))
                })?;

            models.push(BaseModel {
                id: BaseModelId::from_string(id_val.to_string()),
                name: row.get("name"),
                description: row.get("description"),
                repo_id: row.get("repo_id"),
                filename: row.get("filename"),
                size_mb: row.get::<Option<i64>, _>("size_mb").map(|s| s as u64),
                metadata,
            });
        }

        Ok(models)
    }

    async fn exists_by_name(&self, name: &str) -> BaseModelResult<bool> {
        let result = sqlx::query_scalar::<_, bool>(
            "SELECT EXISTS(SELECT 1 FROM base_models WHERE name = $1)",
        )
        .bind(name)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?;

        Ok(result)
    }

    async fn exists(&self, id: &BaseModelId) -> BaseModelResult<bool> {
        let uuid = Uuid::parse_str(id.as_str())
            .map_err(|e| BaseModelError::InvalidData(format!("Invalid UUID: {}", e)))?;

        let result =
            sqlx::query_scalar::<_, bool>("SELECT EXISTS(SELECT 1 FROM base_models WHERE id = $1)")
                .bind(uuid)
                .fetch_one(&self.pool)
                .await
                .map_err(|e| BaseModelError::DatabaseError(e.to_string()))?;

        Ok(result)
    }
}
