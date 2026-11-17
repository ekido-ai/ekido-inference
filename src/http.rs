//! HTTP routes and the request-ID middleware.

use std::future::{ready, Ready};

use actix_web::dev::{forward_ready, Service, ServiceRequest, ServiceResponse, Transform};
use actix_web::http::header::{HeaderName, HeaderValue};
use actix_web::{web, Error, HttpResponse};
use serde::Serialize;

pub const REQUEST_ID: &str = "x-request-id";

#[derive(Serialize)]
struct Health {
    ok: bool,
    service: &'static str,
    version: &'static str,
}

async fn healthz() -> HttpResponse {
    HttpResponse::Ok().json(Health {
        ok: true,
        service: "inference",
        version: env!("CARGO_PKG_VERSION"),
    })
}

/// Ready means able to serve. Until a model engine is wired in, fail readiness while preserving liveness.
async fn readyz() -> HttpResponse {
    HttpResponse::ServiceUnavailable().json(serde_json::json!({ "ok": false, "checks": { "engine": { "ok": false, "detail": "not configured yet" } } }))
}

pub fn routes(cfg: &mut web::ServiceConfig) {
    cfg.route("/healthz", web::get().to(healthz))
        .route("/readyz", web::get().to(readyz));
}

/// A caller's request ID is kept if it looks sane (1-128 visible ASCII characters); otherwise a new UUID is made.
pub fn request_id(incoming: Option<&HeaderValue>) -> String {
    incoming
        .and_then(|v| v.to_str().ok())
        .filter(|v| !v.is_empty() && v.len() <= 128 && v.bytes().all(|b| b.is_ascii_graphic()))
        .map(str::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

/// Middleware: puts a request ID on every response and in the log line for the request.
pub struct RequestId;

impl<S, B> Transform<S, ServiceRequest> for RequestId
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Transform = RequestIdMiddleware<S>;
    type InitError = ();
    type Future = Ready<Result<Self::Transform, Self::InitError>>;

    fn new_transform(&self, service: S) -> Self::Future {
        ready(Ok(RequestIdMiddleware {
            service: std::rc::Rc::new(service),
        }))
    }
}

pub struct RequestIdMiddleware<S> {
    service: std::rc::Rc<S>,
}

impl<S, B> Service<ServiceRequest> for RequestIdMiddleware<S>
where
    S: Service<ServiceRequest, Response = ServiceResponse<B>, Error = Error> + 'static,
    B: 'static,
{
    type Response = ServiceResponse<B>;
    type Error = Error;
    type Future =
        std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self::Response, Self::Error>>>>;

    forward_ready!(service);

    fn call(&self, req: ServiceRequest) -> Self::Future {
        let id = request_id(req.headers().get(REQUEST_ID));
        let path = req.path().to_owned();
        let service = self.service.clone();
        Box::pin(async move {
            let mut response = service.call(req).await?;
            tracing::info!(request_id = %id, %path, status = response.status().as_u16(), "request");
            if let Ok(value) = HeaderValue::from_str(&id) {
                response
                    .headers_mut()
                    .insert(HeaderName::from_static(REQUEST_ID), value);
            }
            Ok(response)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn health_and_readiness_answer() {
        let app = test::init_service(App::new().wrap(RequestId).configure(routes)).await;
        let response =
            test::call_service(&app, test::TestRequest::get().uri("/readyz").to_request()).await;
        assert_eq!(
            response.status(),
            actix_web::http::StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(response.headers().contains_key(REQUEST_ID));
        let ready: serde_json::Value = test::read_body_json(response).await;
        assert_eq!(ready["ok"], false);
        assert_eq!(ready["checks"]["engine"]["ok"], false);
        let response =
            test::call_service(&app, test::TestRequest::get().uri("/healthz").to_request()).await;
        assert_eq!(response.status(), actix_web::http::StatusCode::OK);
        assert!(response.headers().contains_key(REQUEST_ID));
        let body: serde_json::Value = test::read_body_json(response).await;
        assert_eq!(body["service"], "inference");
    }

    #[actix_web::test]
    async fn a_callers_request_id_is_kept() {
        let app = test::init_service(App::new().wrap(RequestId).configure(routes)).await;
        let request = test::TestRequest::get()
            .uri("/healthz")
            .insert_header((REQUEST_ID, "abc-123"))
            .to_request();
        let response = test::call_service(&app, request).await;
        assert_eq!(response.headers().get(REQUEST_ID).unwrap(), "abc-123");
    }

    // `test` here is actix-web's macro (imported above), so this test is async like the others.
    #[actix_web::test]
    async fn bad_request_ids_are_replaced() {
        let made = request_id(Some(&HeaderValue::from_static("has space")));
        assert_ne!(made, "has space");
        assert_eq!(made.len(), 36);
        assert_eq!(request_id(None).len(), 36);
        let long = "x".repeat(129);
        assert_ne!(
            request_id(Some(&HeaderValue::from_str(&long).unwrap())),
            long
        );
    }

    #[actix_web::test]
    async fn unknown_paths_are_404() {
        let app = test::init_service(App::new().wrap(RequestId).configure(routes)).await;
        let response =
            test::call_service(&app, test::TestRequest::get().uri("/nope").to_request()).await;
        assert_eq!(response.status().as_u16(), 404);
    }
}
