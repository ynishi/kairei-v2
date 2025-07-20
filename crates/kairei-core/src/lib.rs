//! # Kairei-v2 Core Engine
//!
//! Internal engine implementation for the Kairei-v2 AgentCulture Framework.

pub mod error;
pub mod processor;
pub mod request;
pub mod tuner;

pub use error::{CoreError, Result};
pub use processor::{ProcessingContext, Processor, ProcessorExt, ProcessorMetadata};
pub use request::{Request, RequestId, RequestMetadata, Response, ResponseMetadata};
pub use tuner::{
    BoxedTuner, LoRATunerConfig, SharedTuner, Tuner, TunerExt, TunerMetadata, TuningContext,
    TuningParameters, TuningResult,
};

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
