//! # Kairei-v2 Core Engine
//!
//! Internal engine implementation for the Kairei-v2 AgentCulture Framework.

pub mod kernel;

pub use kernel::*;

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
