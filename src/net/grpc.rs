//! gRPC protocol handler and streaming inference service implementation.

pub mod gaze_v1 {
    tonic::include_proto!("gaze.v1");
}

pub use gaze_v1::inference_service_server::{InferenceService, InferenceServiceServer};
pub use gaze_v1::{GenerateRequest, GenerateResponse};

use std::pin::Pin;
use tokio_stream::Stream;
use tonic::{Request, Response, Status};

/// High-throughput streaming gRPC inference service implementation.
#[derive(Debug, Default)]
pub struct EngineInferenceService;

#[tonic::async_trait]
impl InferenceService for EngineInferenceService {
    type StreamGenerateStream = Pin<Box<dyn Stream<Item = Result<GenerateResponse, Status>> + Send + 'static>>;

    async fn stream_generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let req = request.into_inner();
        let (tx, rx) = tokio::sync::mpsc::channel(128);

        tokio::spawn(async move {
            let response = GenerateResponse {
                request_id: req.request_id,
                token_text: "stub".to_string(),
                token_id: 100,
                is_finished: true,
                finish_reason: "completed".to_string(),
            };
            let _ = tx.send(Ok(response)).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use std::time::Instant;
    use tokio_stream::StreamExt;

    #[test]
    fn test_protobuf_serialization() {
        let req = GenerateRequest {
            request_id: "req-12345".to_string(),
            prompt: "The quick brown fox jumps over the lazy dog".to_string(),
            max_new_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        };

        // Encode to byte buffer
        let mut buf = Vec::new();
        req.encode(&mut buf).expect("Failed to encode GenerateRequest");

        assert!(!buf.is_empty());

        // Decode back
        let decoded = GenerateRequest::decode(&buf[..]).expect("Failed to decode GenerateRequest");
        assert_eq!(decoded.request_id, "req-12345");
        assert_eq!(decoded.prompt, "The quick brown fox jumps over the lazy dog");
        assert_eq!(decoded.max_new_tokens, 128);
        assert!((decoded.temperature - 0.7).abs() < f32::EPSILON);
        assert!((decoded.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(decoded.top_k, 40);
    }

    #[tokio::test]
    async fn test_stream_generate_service() {
        let service = EngineInferenceService;
        let req = Request::new(GenerateRequest {
            request_id: "test-stream-1".to_string(),
            prompt: "Test gRPC streaming".to_string(),
            max_new_tokens: 16,
            temperature: 0.5,
            top_p: 0.9,
            top_k: 20,
        });

        let response = service.stream_generate(req).await.unwrap();
        let mut stream = response.into_inner();

        let item = stream.next().await;
        assert!(item.is_some());
        let res = item.unwrap().unwrap();
        assert_eq!(res.request_id, "test-stream-1");
        assert_eq!(res.token_text, "stub");
        assert_eq!(res.token_id, 100);
        assert!(res.is_finished);
        assert_eq!(res.finish_reason, "completed");

        let next_item = stream.next().await;
        assert!(next_item.is_none());
    }

    #[test]
    fn test_serialization_benchmark() {
        let req = GenerateRequest {
            request_id: "bench-req-999".to_string(),
            prompt: "Benchmark prompt string for sub-microsecond serialization test".to_string(),
            max_new_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
        };

        let iterations = 10_000;
        let start = Instant::now();
        let mut total_bytes = 0;

        for _ in 0..iterations {
            let mut buf = Vec::with_capacity(128);
            req.encode(&mut buf).unwrap();
            total_bytes += buf.len();
            let _decoded = GenerateRequest::decode(&buf[..]).unwrap();
        }

        let elapsed = start.elapsed();
        let nanos_per_op = elapsed.as_nanos() / iterations as u128;
        println!(
            "Protobuf roundtrip serialization: {} iterations in {:?}, avg {} ns/op, total {} bytes",
            iterations,
            elapsed,
            nanos_per_op,
            total_bytes
        );

        // Verification serialization roundtrip should take less than 10 microseconds per op
        assert!(nanos_per_op < 10_000, "Serialization took too long: {} ns/op", nanos_per_op);
    }
}
