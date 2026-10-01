//! Inference request validation and processing logic.

use serde::{Deserialize, Serialize};

/// Errors encountered during inference request payload validation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RequestValidationError {
    /// The prompt string is empty.
    #[error("prompt cannot be empty")]
    EmptyPrompt,
    /// `max_tokens` is 0 or exceeds the maximum configured sequence length.
    #[error("max_tokens out of range: {max_tokens} (allowed: 1..={max_seq_len})")]
    MaxTokensOutOfRange {
        /// Provided token generation limit.
        max_tokens: usize,
        /// Maximum allowed sequence length for the engine.
        max_seq_len: usize,
    },
    /// Sampling temperature is negative, exceeds 2.0, or is NaN.
    #[error("temperature must be between 0.0 and 2.0, got {0}")]
    InvalidTemperature(f32),
    /// Nucleus sampling `top_p` is not within `(0.0, 1.0]` or is NaN.
    #[error("top_p must be between 0.0 (exclusive) and 1.0 (inclusive), got {0}")]
    InvalidTopP(f32),
}

/// High-level client inference request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    /// Unique identifier for the client request.
    pub id: String,
    /// Text prompt to generate completion for.
    pub prompt: String,
    /// Maximum new tokens to produce.
    pub max_tokens: usize,
    /// Sampling temperature (higher means more random).
    pub temperature: f32,
    /// Nucleus sampling cutoff threshold.
    pub top_p: f32,
}

impl InferenceRequest {
    /// Validates the request fields against engine constraints.
    ///
    /// # Errors
    /// Returns [`RequestValidationError`] if the prompt is empty, `max_tokens` is out of bounds,
    /// or sampling parameters (`temperature`, `top_p`) are out of their allowed ranges.
    pub fn validate(&self, max_seq_len: usize) -> Result<(), RequestValidationError> {
        if self.prompt.is_empty() {
            return Err(RequestValidationError::EmptyPrompt);
        }
        if self.max_tokens == 0 || self.max_tokens > max_seq_len {
            return Err(RequestValidationError::MaxTokensOutOfRange {
                max_tokens: self.max_tokens,
                max_seq_len,
            });
        }
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(RequestValidationError::InvalidTemperature(self.temperature));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(RequestValidationError::InvalidTopP(self.top_p));
        }
        Ok(())
    }
}

/// Convenience standalone function to validate an [`InferenceRequest`].
///
/// # Errors
/// Returns [`RequestValidationError`] if the request parameters fail validation.
pub fn validate_request(
    request: &InferenceRequest,
    max_seq_len: usize,
) -> Result<(), RequestValidationError> {
    request.validate(max_seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> InferenceRequest {
        InferenceRequest {
            id: "req-1".into(),
            prompt: "Hello world".into(),
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
        }
    }

    #[test]
    fn test_valid_request() {
        assert!(sample_request().validate(2048).is_ok());
    }

    #[test]
    fn test_validation_bounds() {
        let mut req = sample_request();

        req.prompt.clear();
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::EmptyPrompt
        );

        req = sample_request();
        req.max_tokens = 0;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::MaxTokensOutOfRange {
                max_tokens: 0,
                max_seq_len: 2048
            }
        );

        req.max_tokens = 4096;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::MaxTokensOutOfRange {
                max_tokens: 4096,
                max_seq_len: 2048
            }
        );

        for bad_temp in [-0.1, 2.1, f32::NAN] {
            req = sample_request();
            req.temperature = bad_temp;
            let err = req.validate(2048).unwrap_err();
            assert!(matches!(err, RequestValidationError::InvalidTemperature(_)));
        }

        for bad_top_p in [0.0, -0.5, 1.01, f32::NAN] {
            req = sample_request();
            req.top_p = bad_top_p;
            let err = req.validate(2048).unwrap_err();
            assert!(matches!(err, RequestValidationError::InvalidTopP(_)));
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let req = sample_request();
        let json = serde_json::to_string(&req).unwrap();
        let parsed: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }
}

