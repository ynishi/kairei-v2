pub mod error;
pub mod local;
pub mod traits;

pub use error::{StorageError, StorageResult};
pub use local::LocalStorage;
pub use traits::{Storage, StorageConfig};
