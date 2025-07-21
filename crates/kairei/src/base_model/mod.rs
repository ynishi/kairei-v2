pub mod error;
pub mod in_memory_repository;
pub mod model;
pub mod repository;
pub mod service;

pub use error::{BaseModelError, BaseModelResult};
pub use in_memory_repository::InMemoryBaseModelRepository;
pub use model::{BaseModel, BaseModelId, BaseModelMetadata};
pub use repository::BaseModelRepository;
pub use service::BaseModelService;
