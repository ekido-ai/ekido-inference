//! Ekido Inference service.
//!
//! Serves customers' finished models behind their API keys. Model execution itself is llama.cpp (C++), run as a
//! local process this service talks to; this service owns routing, keys, metering and limits (tickets C10, C11).
//! For now it only reports its health, so it can be deployed, monitored and routed like the other services.
//!
//! Listens on 127.0.0.1:8020 by default (`EKIDO_INFERENCE_ADDR`). Logs are JSON lines; every response carries an
//! `X-Request-Id`, reusing the caller's when it sends one, so one request can be followed across services.

use actix_web::{App, HttpServer};
use tracing_subscriber::EnvFilter;

mod gateway;
mod http;

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    tracing_subscriber::fmt()
        .json()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();
    let addr =
        std::env::var("EKIDO_INFERENCE_ADDR").unwrap_or_else(|_| "127.0.0.1:8020".to_string());
    tracing::info!(%addr, version = env!("CARGO_PKG_VERSION"), "inference service starting");
    let gateway = actix_web::web::Data::new(gateway::Gateway::from_env());
    HttpServer::new(move || {
        App::new()
            .app_data(gateway.clone())
            .wrap(http::RequestId)
            .configure(http::routes)
            .configure(gateway::routes)
    })
    .bind(addr)?
    .run()
    .await
}
