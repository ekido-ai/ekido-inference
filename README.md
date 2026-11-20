# ekido-inference

Serves customers' finished models behind their API keys. Rust (actix-web) on 127.0.0.1:8020, wrapping llama.cpp; models load on demand within a memory budget.

Part of [Ekido](https://github.com/ekido-ai): your own small AI model, trained on spare devices and paid for only when it works. The shared library, the API contract and the evidence behind every number live in [ekido-core](https://github.com/ekido-ai/ekido-core).

## Gateway setup

The gateway stays unavailable until the Platform inference authority and at least one local model engine are configured. Set `EKIDO_INFERENCE_AUTHORITY_URL` to the Platform's numeric loopback HTTP origin, such as `http://127.0.0.1:8010`, and set `EKIDO_INFERENCE_INTERNAL_TOKEN` to the same operator-managed secret configured in Platform. Do not expose that token to customers or commit it. Platform's protected model registry associates each customer model with its job and local engine. Configure Platform, Postgres, and the model engine before enabling public routes.

`/v1/models/{id}/predict` accepts up to 32 inputs, and `/v1/chat/completions` supports non-streaming chat. Each request requires a scoped customer API key; Platform checks ownership and reserves the rate limit for each input before the router calls the model. If any reservation fails, the router cancels the earlier reservations and does not invoke the engine. A key may hold at most two concurrent request permits per router process. Successful replies are returned only after token usage is recorded. If a batch fails after earlier inputs succeeded, the client receives an error without partial output, but those earlier inputs remain recorded as successful usage; unstarted reservations are cancelled. Request and response text are not logged by the router. A router crash or a failed usage write can leave a pending reservation requiring reconciliation. Streaming and billing reconciliation are not implemented.

`/readyz` succeeds only when the authority is configured and reports at least one registered local engine that answers its health check. `/healthz` reports process liveness separately.

## Rules

- Every change is a pull request with the full report (the PR template); the **Gate** check refuses it otherwise.
- Testnet and test money only. Never commit keys, `.env` contents, wallet keys or customer data.
- Every number we quote is measured and has its raw file.

Licence: all rights reserved (see LICENSE).
