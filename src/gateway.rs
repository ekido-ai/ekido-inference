//! Customer inference routes. Authorization and usage are owned by Platform;
//! only numeric loopback origins are accepted for the authority and model engines.
use std::{
    net::IpAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use actix_web::{
    http::{header, StatusCode},
    web, HttpRequest, HttpResponse,
};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

const MAX_RESPONSE: usize = 1_000_000;
const MAX_PROMPT: usize = 8_000;

pub struct Gateway {
    config: Option<Config>,
    permits: Arc<Semaphore>,
}

struct Config {
    authority: Url,
    token: String,
    client: Client,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Predict {
    inputs: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Chat {
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<u32>,
    temperature: Option<f64>,
    stream: Option<bool>,
}

#[derive(Deserialize)]
struct Reservation {
    reservation_id: String,
    engine_url: String,
    engine_model: String,
}

struct Completed {
    outputs: Vec<String>,
    input: u64,
    output: u64,
    elapsed: u64,
    finish_reason: Option<String>,
}

impl Gateway {
    pub fn from_env() -> Self {
        let config = (|| {
            let authority = loopback_origin(&std::env::var("EKIDO_INFERENCE_AUTHORITY_URL").ok()?)?;
            let token = std::env::var("EKIDO_INFERENCE_INTERNAL_TOKEN").ok()?;
            if token.len() < 32 || !token.is_ascii() || token.bytes().any(|b| b.is_ascii_control())
            {
                return None;
            }
            let client = Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(Duration::from_secs(3))
                .timeout(Duration::from_secs(120))
                .build()
                .ok()?;
            Some(Config {
                authority,
                token,
                client,
            })
        })();
        Self {
            config,
            permits: Arc::new(Semaphore::new(8)),
        }
    }

    pub async fn ready(&self) -> bool {
        let Some(cfg) = self.config.as_ref() else {
            return false;
        };
        let Ok(url) = cfg.authority.join("/internal/inference/ready") else {
            return false;
        };
        let Ok(response) = cfg
            .client
            .get(url)
            .header("X-Ekido-Inference-Token", &cfg.token)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        else {
            return false;
        };
        if !response.status().is_success() {
            return false;
        }
        let Ok(value) = read_json(response).await else {
            return false;
        };
        let Some(engines) = value.get("engines").and_then(Value::as_array) else {
            return false;
        };
        // Bound both the number and total time spent on health checks.
        for engine in engines.iter().take(4) {
            let Some(origin) = engine.as_str().and_then(loopback_origin) else {
                continue;
            };
            let Ok(url) = origin.join("/health") else {
                continue;
            };
            let Ok(response) = cfg
                .client
                .get(url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            else {
                continue;
            };
            if !response.status().is_success() {
                continue;
            }
            if let Ok(body) = read_json(response).await {
                if body.get("status").and_then(Value::as_str) == Some("ok") {
                    return true;
                }
            }
        }
        false
    }
}

fn loopback_origin(value: &str) -> Option<Url> {
    let parsed = Url::parse(value).ok()?;
    let host: IpAddr = parsed
        .host_str()?
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .ok()?;
    if parsed.scheme() != "http"
        || !host.is_loopback()
        || parsed.port().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return None;
    }
    Some(parsed)
}

fn failure(status: StatusCode, code: &'static str, detail: &'static str) -> Box<HttpResponse> {
    Box::new(HttpResponse::build(status).json(json!({"error": code, "detail": detail})))
}

fn api_key(req: &HttpRequest) -> Option<&str> {
    let value = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, key) = value.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("bearer")
        || key.len() > 160
        || !(key.starts_with("ek_test_") || key.starts_with("ek_live_"))
        || !key
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        return None;
    }
    Some(key)
}

fn valid_model_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 120
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

async fn read_json(mut response: reqwest::Response) -> Result<Value, ()> {
    if response
        .content_length()
        .is_some_and(|n| n > MAX_RESPONSE as u64)
    {
        return Err(());
    }
    let mut data = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| ())? {
        if data.len().saturating_add(chunk.len()) > MAX_RESPONSE {
            return Err(());
        }
        data.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&data).map_err(|_| ())
}

async fn reserve(cfg: &Config, key: &str, model: &str) -> Result<Reservation, Box<HttpResponse>> {
    let url = cfg
        .authority
        .join("/internal/inference/reserve")
        .map_err(|_| {
            failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "UNAVAILABLE",
                "Inference is unavailable.",
            )
        })?;
    let response = cfg
        .client
        .post(url)
        .header("X-Ekido-Inference-Token", &cfg.token)
        .json(&json!({"api_key": key, "model_id": model}))
        .send()
        .await
        .map_err(|_| {
            failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "UNAVAILABLE",
                "Inference is unavailable.",
            )
        })?;
    let status = response.status().as_u16();
    if status != 200 {
        return Err(match status {
            401 => failure(
                StatusCode::UNAUTHORIZED,
                "INVALID_KEY",
                "Invalid or revoked API key.",
            ),
            403 => failure(
                StatusCode::FORBIDDEN,
                "FORBIDDEN",
                "API key needs models.predict.",
            ),
            404 => failure(StatusCode::NOT_FOUND, "NOT_FOUND", "No such model."),
            429 => failure(
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Too many requests.",
            ),
            _ => failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "UNAVAILABLE",
                "Inference is unavailable.",
            ),
        });
    }
    let value = read_json(response).await.map_err(|_| {
        failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "UNAVAILABLE",
            "Inference is unavailable.",
        )
    })?;
    let result: Reservation = serde_json::from_value(value).map_err(|_| {
        failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "UNAVAILABLE",
            "Inference is unavailable.",
        )
    })?;
    if uuid::Uuid::parse_str(&result.reservation_id).is_err()
        || loopback_origin(&result.engine_url).is_none()
        || result.engine_model.is_empty()
        || result.engine_model.len() > 200
    {
        return Err(failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "UNAVAILABLE",
            "Inference is unavailable.",
        ));
    }
    Ok(result)
}

async fn finalize(
    cfg: &Config,
    id: &str,
    ok: bool,
    input: Option<u64>,
    output: Option<u64>,
    elapsed: u64,
) -> bool {
    let Ok(url) = cfg
        .authority
        .join(&format!("/internal/inference/usage/{id}"))
    else {
        return false;
    };
    match cfg
        .client
        .post(url)
        .header("X-Ekido-Inference-Token", &cfg.token)
        .json(
            &json!({"status": if ok { "ok" } else { "error" }, "input_tokens": input,
                      "output_tokens": output, "latency_ms": elapsed.min(i32::MAX as u64)}),
        )
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

async fn generate(
    cfg: &Config,
    reservation: &Reservation,
    messages: &[Message],
    max_tokens: u32,
    temperature: f64,
) -> Result<(String, u64, u64, Option<String>), ()> {
    let url = loopback_origin(&reservation.engine_url)
        .ok_or(())?
        .join("/v1/chat/completions")
        .map_err(|_| ())?;
    let response = cfg
        .client
        .post(url)
        .json(
            &json!({"model": reservation.engine_model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature, "stream": false}),
        )
        .send()
        .await
        .map_err(|_| ())?;
    if !response.status().is_success() {
        return Err(());
    }
    let value = read_json(response).await?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .ok_or(())?;
    let input = value
        .pointer("/usage/prompt_tokens")
        .and_then(Value::as_u64)
        .filter(|v| *v <= i32::MAX as u64)
        .ok_or(())?;
    let output = value
        .pointer("/usage/completion_tokens")
        .and_then(Value::as_u64)
        .filter(|v| *v <= i32::MAX as u64)
        .ok_or(())?;
    let finish_reason = value
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .filter(|v| matches!(*v, "stop" | "length" | "content_filter"))
        .map(str::to_owned);
    Ok((content.to_owned(), input, output, finish_reason))
}

async fn execute(
    req: &HttpRequest,
    gateway: &Gateway,
    model: &str,
    batches: Vec<Vec<Message>>,
    max_tokens: u32,
    temperature: f64,
) -> Result<Completed, Box<HttpResponse>> {
    let Some(key) = api_key(req) else {
        return Err(failure(
            StatusCode::UNAUTHORIZED,
            "INVALID_KEY",
            "Send a valid API key.",
        ));
    };
    let Some(cfg) = gateway.config.as_ref() else {
        return Err(failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "UNAVAILABLE",
            "Inference is not configured.",
        ));
    };
    let Ok(_permit) = gateway.permits.clone().try_acquire_owned() else {
        return Err(failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "BUSY",
            "Inference is busy. Try again.",
        ));
    };
    let reservation = reserve(cfg, key, model).await?;
    let started = Instant::now();
    let (mut outputs, mut input, mut output, mut finish_reason) = (Vec::new(), 0_u64, 0_u64, None);
    for messages in &batches {
        match generate(cfg, &reservation, messages, max_tokens, temperature).await {
            Ok((text, a, b, reason))
                if input + a <= i32::MAX as u64 && output + b <= i32::MAX as u64 =>
            {
                outputs.push(text);
                input += a;
                output += b;
                finish_reason = reason;
            }
            _ => {
                let partial = if outputs.is_empty() {
                    (None, None)
                } else {
                    (Some(input), Some(output))
                };
                let _ = finalize(
                    cfg,
                    &reservation.reservation_id,
                    false,
                    partial.0,
                    partial.1,
                    started.elapsed().as_millis().min(i32::MAX as u128) as u64,
                )
                .await;
                return Err(failure(
                    StatusCode::BAD_GATEWAY,
                    "ENGINE_ERROR",
                    "Model inference failed.",
                ));
            }
        }
    }
    let elapsed = started.elapsed().as_millis().min(i32::MAX as u128) as u64;
    if !finalize(
        cfg,
        &reservation.reservation_id,
        true,
        Some(input),
        Some(output),
        elapsed,
    )
    .await
    {
        return Err(failure(
            StatusCode::SERVICE_UNAVAILABLE,
            "USAGE_UNAVAILABLE",
            "Usage could not be recorded.",
        ));
    }
    Ok(Completed {
        outputs,
        input,
        output,
        elapsed,
        finish_reason,
    })
}

async fn predict(
    req: HttpRequest,
    gateway: web::Data<Gateway>,
    model: web::Path<String>,
    body: web::Json<Predict>,
) -> HttpResponse {
    if !valid_model_id(&model)
        || body.inputs.is_empty()
        || body.inputs.len() > 32
        || body
            .inputs
            .iter()
            .any(|v| v.is_empty() || v.len() > MAX_PROMPT)
    {
        return *failure(
            StatusCode::BAD_REQUEST,
            "INVALID_INPUT",
            "Supply 1 to 32 non-empty inputs of at most 8000 bytes.",
        );
    }
    let batches = body
        .inputs
        .iter()
        .map(|text| {
            vec![Message {
                role: "user".into(),
                content: text.clone(),
            }]
        })
        .collect();
    match execute(&req, &gateway, &model, batches, 128, 0.0).await {
        Ok(done) => HttpResponse::Ok()
            .insert_header((header::CACHE_CONTROL, "no-store"))
            .json(
                json!({"outputs": done.outputs, "usage": {"input_tokens": done.input,
                         "output_tokens": done.output, "latency_ms": done.elapsed}}),
            ),
        Err(response) => *response,
    }
}

async fn chat(
    req: HttpRequest,
    gateway: web::Data<Gateway>,
    body: web::Json<Chat>,
) -> HttpResponse {
    let max_tokens = body.max_tokens.unwrap_or(128);
    let temperature = body.temperature.unwrap_or(0.0);
    if !valid_model_id(&body.model)
        || body.stream.unwrap_or(false)
        || body.messages.is_empty()
        || body.messages.len() > 32
        || max_tokens == 0
        || max_tokens > 1024
        || !temperature.is_finite()
        || !(0.0..=2.0).contains(&temperature)
        || body.messages.iter().any(|m| {
            !matches!(m.role.as_str(), "system" | "user" | "assistant")
                || m.content.is_empty()
                || m.content.len() > MAX_PROMPT
        })
        || body.messages.iter().map(|m| m.content.len()).sum::<usize>() > 64_000
    {
        return *failure(
            StatusCode::BAD_REQUEST,
            "INVALID_INPUT",
            "Invalid chat request.",
        );
    }
    match execute(&req, &gateway, &body.model, vec![body.messages.clone()], max_tokens, temperature).await {
        Ok(done) => HttpResponse::Ok().insert_header((header::CACHE_CONTROL, "no-store"))
            .json(json!({"id": format!("chatcmpl-{}", uuid::Uuid::new_v4()), "object": "chat.completion",
                "created": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
                "model": body.model, "choices": [{"index": 0, "message": {"role": "assistant",
                "content": done.outputs[0]}, "finish_reason": done.finish_reason}],
                "usage": {"prompt_tokens": done.input, "completion_tokens": done.output,
                          "total_tokens": done.input + done.output}})),
        Err(response) => *response,
    }
}

pub fn routes(cfg: &mut web::ServiceConfig) {
    cfg.app_data(web::JsonConfig::default().limit(262_144))
        .route("/v1/models/{model_id}/predict", web::post().to(predict))
        .route("/v1/chat/completions", web::post().to(chat));
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App, HttpServer};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct MockState {
        calls: AtomicUsize,
        metered: AtomicUsize,
        fail_usage: AtomicBool,
    }

    async fn mock_reserve(req: HttpRequest, body: web::Json<Value>) -> HttpResponse {
        if req
            .headers()
            .get("X-Ekido-Inference-Token")
            .and_then(|h| h.to_str().ok())
            != Some("internal-test-token-12345678901234567890")
        {
            return HttpResponse::Unauthorized().finish();
        }
        if body["api_key"] == "ek_test_wrong" {
            return HttpResponse::Unauthorized().finish();
        }
        HttpResponse::Ok().json(json!({"reservation_id": "12345678-1234-4234-8234-123456789012",
            "engine_url": format!("http://{}", req.connection_info().host()), "engine_model": "local"}))
    }

    async fn mock_ready(req: HttpRequest) -> HttpResponse {
        if req
            .headers()
            .get("X-Ekido-Inference-Token")
            .and_then(|h| h.to_str().ok())
            != Some("internal-test-token-12345678901234567890")
        {
            return HttpResponse::Unauthorized().finish();
        }
        HttpResponse::Ok()
            .json(json!({"engines": [format!("http://{}", req.connection_info().host())]}))
    }

    async fn mock_health() -> HttpResponse {
        HttpResponse::Ok().json(json!({"status": "ok"}))
    }

    async fn mock_usage(state: web::Data<Arc<MockState>>, body: web::Json<Value>) -> HttpResponse {
        if state.fail_usage.load(Ordering::SeqCst) {
            return HttpResponse::ServiceUnavailable().finish();
        }
        if body["status"] == "ok" {
            state.metered.fetch_add(1, Ordering::SeqCst);
        }
        HttpResponse::Ok().finish()
    }

    async fn mock_engine(state: web::Data<Arc<MockState>>, body: web::Json<Value>) -> HttpResponse {
        if body["model"] != "local" {
            return HttpResponse::BadRequest().finish();
        }
        state.calls.fetch_add(1, Ordering::SeqCst);
        HttpResponse::Ok().json(
            json!({"choices": [{"message": {"content": "answer"}, "finish_reason": "length"}],
                                      "usage": {"prompt_tokens": 3, "completion_tokens": 2}}),
        )
    }

    async fn setup() -> (Gateway, Arc<MockState>, actix_web::dev::ServerHandle) {
        let state = Arc::new(MockState {
            calls: AtomicUsize::new(0),
            metered: AtomicUsize::new(0),
            fail_usage: AtomicBool::new(false),
        });
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let app_state = state.clone();
        let server = HttpServer::new(move || {
            App::new()
                .app_data(web::Data::new(app_state.clone()))
                .route("/internal/inference/reserve", web::post().to(mock_reserve))
                .route("/internal/inference/ready", web::get().to(mock_ready))
                .route("/internal/inference/usage/{id}", web::post().to(mock_usage))
                .route("/health", web::get().to(mock_health))
                .route("/v1/chat/completions", web::post().to(mock_engine))
        })
        .listen(listener)
        .unwrap()
        .run();
        let handle = server.handle();
        actix_web::rt::spawn(server);
        let config = Config {
            authority: loopback_origin(&format!("http://{addr}")).unwrap(),
            token: "internal-test-token-12345678901234567890".into(),
            client: Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .timeout(Duration::from_secs(5))
                .build()
                .unwrap(),
        };
        (
            Gateway {
                config: Some(config),
                permits: Arc::new(Semaphore::new(2)),
            },
            state,
            handle,
        )
    }

    #[actix_web::test]
    async fn predict_and_chat_are_authorized_and_metered() {
        let (gateway, state, handle) = setup().await;
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(gateway))
                .configure(routes),
        )
        .await;
        let denied = test::TestRequest::post()
            .uri("/v1/models/model1/predict")
            .set_json(json!({"inputs": ["hello"]}))
            .to_request();
        assert_eq!(
            test::call_service(&app, denied).await.status(),
            StatusCode::UNAUTHORIZED
        );
        let request = test::TestRequest::post()
            .uri("/v1/models/model1/predict")
            .insert_header((header::AUTHORIZATION, "Bearer ek_test_valid"))
            .set_json(json!({"inputs": ["one", "two"]}))
            .to_request();
        let response = test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["outputs"], json!(["answer", "answer"]));
        assert_eq!(body["usage"]["input_tokens"], 6);
        assert_eq!(state.calls.load(Ordering::SeqCst), 2);
        assert_eq!(state.metered.load(Ordering::SeqCst), 1);
        let request = test::TestRequest::post()
            .uri("/v1/chat/completions")
            .insert_header((header::AUTHORIZATION, "Bearer ek_test_valid"))
            .set_json(json!({"model": "model1", "messages": [{"role": "user", "content": "hi"}]}))
            .to_request();
        let response = test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value = test::read_body_json(response).await;
        assert_eq!(body["choices"][0]["message"]["content"], "answer");
        assert_eq!(body["choices"][0]["finish_reason"], "length");
        assert_eq!(body["usage"]["total_tokens"], 5);
        handle.stop(false).await;
    }

    #[actix_web::test]
    async fn denied_key_and_unrecorded_usage_never_return_model_output() {
        let (gateway, state, handle) = setup().await;
        let app = test::init_service(
            App::new()
                .app_data(web::Data::new(gateway))
                .configure(routes),
        )
        .await;
        let request = test::TestRequest::post()
            .uri("/v1/models/model1/predict")
            .insert_header((header::AUTHORIZATION, "Bearer ek_test_wrong"))
            .set_json(json!({"inputs": ["secret"]}))
            .to_request();
        assert_eq!(
            test::call_service(&app, request).await.status(),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(state.calls.load(Ordering::SeqCst), 0);
        state.fail_usage.store(true, Ordering::SeqCst);
        let request = test::TestRequest::post()
            .uri("/v1/models/model1/predict")
            .insert_header((header::AUTHORIZATION, "Bearer ek_test_valid"))
            .set_json(json!({"inputs": ["secret"]}))
            .to_request();
        let response = test::call_service(&app, request).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = test::read_body(response).await;
        assert!(!String::from_utf8_lossy(&body).contains("answer"));
        handle.stop(false).await;
    }

    #[actix_web::test]
    async fn only_numeric_loopback_origins_are_allowed() {
        assert!(loopback_origin("http://127.0.0.1:9000").is_some());
        assert!(loopback_origin("http://[::1]:9000").is_some());
        for url in [
            "http://example.com:9000",
            "http://127.0.0.1:9000/path",
            "https://127.0.0.1:9000",
            "http://user@127.0.0.1:9000",
        ] {
            assert!(loopback_origin(url).is_none());
        }
    }

    #[actix_web::test]
    async fn readiness_requires_a_healthy_registered_engine() {
        let (gateway, _, handle) = setup().await;
        assert!(gateway.ready().await);
        handle.stop(false).await;
        assert!(!gateway.ready().await);
    }
}
