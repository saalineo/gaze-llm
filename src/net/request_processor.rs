use serde::{Deserialize, Serialize};

/// High-level client inference request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    pub id: String,
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl InferenceRequest {
    pub fn validate(&self, max_seq_len: usize) -> Result<(), String> {
        if self.prompt.is_empty() {
            return Err("Prompt cannot be empty".into());
        }
        if self.max_tokens == 0 || self.max_tokens > max_seq_len {
            return Err(format!("max_tokens out of range (1..{max_seq_len})"));
        }
        if !(0.0..=2.0).contains(&self.temperature) {
            return Err("temperature must be between 0.0 and 2.0".into());
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err("top_p must be between (0.0, 1.0]".into());
        }
        Ok(())
    }
}

pub fn validate_request(request: &InferenceRequest, max_seq_len: usize) -> Result<(), String> {
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
        assert_eq!(req.validate(2048).unwrap_err(), "Prompt cannot be empty");

        req = sample_request();
        req.max_tokens = 0;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            "max_tokens out of range (1..2048)"
        );

        req.max_tokens = 4096;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            "max_tokens out of range (1..2048)"
        );

        for bad_temp in [-0.1, 2.1, f32::NAN] {
            req = sample_request();
            req.temperature = bad_temp;
            assert_eq!(
                req.validate(2048).unwrap_err(),
                "temperature must be between 0.0 and 2.0"
            );
        }

        for bad_top_p in [0.0, -0.5, 1.01, f32::NAN] {
            req = sample_request();
            req.top_p = bad_top_p;
            assert_eq!(
                req.validate(2048).unwrap_err(),
                "top_p must be between (0.0, 1.0]"
            );
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
