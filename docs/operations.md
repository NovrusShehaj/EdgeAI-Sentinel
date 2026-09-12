# Operations

## Concurrency

One Uvicorn worker, one ONNX session, one infer executor thread. `session.run` is serialized by a lock. Concurrent HTTP requests queue; they do not share an unsafe session.

Set `INFER_TIMEOUT_S` (default 5). Timeouts increment `sentinel_inferences_total{status="timeout"}` and return HTTP 504. A timed-out ORT call may still finish in the background; restart the process if the session becomes unhealthy.

## Readiness

| Endpoint | Meaning |
|---|---|
| `/live` | Process is running |
| `/ready` | Model file exists and the engine loaded |
| `/health` | Diagnostic JSON, always 200 once the app started |

## Metrics

`/metrics` accepts `X-API-Token` / bearer token or an address in `METRICS_ALLOWED_CIDRS` (loopback and RFC1918 by default).

Useful series:

- `sentinel_inferences_total{status=success\|timeout\|error\|unavailable}`
- `sentinel_inference_latency_ms`
- `sentinel_model_ready`

## Alerts

`monitoring/prometheus/alert_rules.yml` defines scrape-down, readiness failure, and high p95 latency. Alertmanager targets are empty; routing is not configured.

## Common failures

- `/ready` 503: mount the ONNX file and matching manifest, then restart
- `/infer` 401: set a non-empty `API_TOKEN` and send it on the request. The monitoring compose default profile does not need this token; `--profile with-api` still cannot infer if it is empty.
- `/infer` 413: payload larger than `MAX_UPLOAD_BYTES`
- Class mismatch: serving class names must match the model head and `configs/data.yaml`
- Checksum mismatch: the file is not the artifact recorded in the manifest
