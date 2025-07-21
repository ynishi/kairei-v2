use async_trait::async_trait;

use super::{BaseModel, BaseModelId, BaseModelResult};

#[async_trait]
pub trait BaseModelRepository: Send + Sync {
    /// Get all base models
    async fn list(&self) -> BaseModelResult<Vec<BaseModel>>;

    /// Get a base model by ID
    async fn get(&self, id: &BaseModelId) -> BaseModelResult<BaseModel>;

    /// Get a base model by name
    async fn get_by_name(&self, name: &str) -> BaseModelResult<BaseModel>;

    /// Create a new base model
    async fn create(&self, model: BaseModel) -> BaseModelResult<BaseModel>;

    /// Update an existing base model
    async fn update(&self, model: BaseModel) -> BaseModelResult<BaseModel>;

    /// Delete a base model
    async fn delete(&self, id: &BaseModelId) -> BaseModelResult<()>;

    /// Check if a model exists
    async fn exists(&self, id: &BaseModelId) -> BaseModelResult<bool>;

    /// Check if a model exists by name
    async fn exists_by_name(&self, name: &str) -> BaseModelResult<bool>;
}
