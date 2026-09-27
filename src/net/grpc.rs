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
        let generate_request = request.into_inner();
        let (sender, receiver) = tokio::sync::mpsc::channel(128);

        tokio::spawn(async move {
            let response = GenerateResponse {
                request_id: generate_request.request_id,
                token_text: "stub".to_string(),
                token_id: 100,
                is_finished: true,
                finish_reason: "completed".to_string(),
            };
            let _ = sender.send(Ok(response)).await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(receiver);
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
        let request = GenerateRequest {
            request_id: "req-12345".to_string(),
            prompt: "The quick brown fox jumps over the lazy dog".to_string(),
            max_new_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
        };

        let mut buffer = Vec::new();
        request.encode(&mut buffer).expect("Failed to encode GenerateRequest");
        assert!(!buffer.is_empty());

        let decoded_request = GenerateRequest::decode(&buffer[..]).expect("Failed to decode GenerateRequest");
        assert_eq!(decoded_request.request_id, "req-12345");
        assert_eq!(decoded_request.prompt, "The quick brown fox jumps over the lazy dog");
        assert_eq!(decoded_request.max_new_tokens, 128);
        assert!((decoded_request.temperature - 0.7).abs() < f32::EPSILON);
        assert!((decoded_request.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(decoded_request.top_k, 40);
    }

    #[tokio::test]
    async fn test_stream_generate_service() {
        let service = EngineInferenceService;
        let request = Request::new(GenerateRequest {
            request_id: "test-stream-1".to_string(),
            prompt: "Test gRPC streaming".to_string(),
            max_new_tokens: 16,
            temperature: 0.5,
            top_p: 0.9,
            top_k: 20,
        });

        let response = service.stream_generate(request).await.expect("RPC failed");
        let mut stream = response.into_inner();

        let stream_item = stream.next().await;
        assert!(stream_item.is_some());
        let generate_response = stream_item
            .unwrap()
            .expect("Stream item should be Ok(GenerateResponse)");
        assert_eq!(generate_response.request_id, "test-stream-1");
        assert_eq!(generate_response.token_text, "stub");
        assert_eq!(generate_response.token_id, 100);
        assert!(generate_response.is_finished);
        assert_eq!(generate_response.finish_reason, "completed");

        let next_item = stream.next().await;
        assert!(next_item.is_none());
    }

    #[test]
    fn test_serialization_benchmark() {
        let request = GenerateRequest {
            request_id: "bench-req-999".to_string(),
            prompt: "Benchmark prompt string for sub-microsecond serialization test".to_string(),
            max_new_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 50,
        };

        let iterations = 10_000;
        let start_time = Instant::now();
        let mut total_bytes = 0;

        for _ in 0..iterations {
            let mut buffer = Vec::with_capacity(128);
            request.encode(&mut buffer).expect("Encode failed");
            total_bytes += buffer.len();
            let _decoded = GenerateRequest::decode(&buffer[..]).expect("Decode failed");
        }

        let elapsed = start_time.elapsed();
        let nanos_per_op = elapsed.as_nanos() / iterations as u128;
        println!(
            "Protobuf roundtrip serialization: {} iterations in {:?}, avg {} ns/op, total {} bytes",
            iterations,
            elapsed,
            nanos_per_op,
            total_bytes
        );

        assert!(nanos_per_op < 10_000, "Serialization took too long: {} ns/op", nanos_per_op);
    }
}
