use thiserror::Error;

#[derive(Error, Debug)]
pub enum BaseModelError {
    #[error("Model not found: {0}")]
    NotFound(String),

    #[error("Model already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid model data: {0}")]
    InvalidData(String),

    #[error("Repository error: {0}")]
    RepositoryError(String),

    #[error("Download error: {0}")]
    DownloadError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type BaseModelResult<T> = Result<T, BaseModelError>;
