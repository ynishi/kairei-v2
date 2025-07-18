//! # Kairei-v2 Core Engine
//!
//! Internal engine implementation for the Kairei-v2 AgentCulture Framework.

pub mod error;
pub mod kernel;
pub mod request;
pub mod types;

pub use error::{CoreError, Result};
pub use kernel::*;
pub use request::{Request, RequestMetadata, Response, ResponseMetadata};
pub use types::{RequestId, SessionId};

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
