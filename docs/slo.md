# Latency and memory SLOs

No hardware-tagged edge SLO is published yet. Do not treat README history or synthetic training logs as targets.

When a real model and named device exist, record:

- Device name and CPU/RAM
- OS, Python, onnxruntime, and image tag
- Model SHA-256 and class names
- Image size and thread count
- p50/p95 latency, RSS, and throughput

Suggested starting budgets after measurement, not before:

- API timeout: `INFER_TIMEOUT_S=5`
- Container memory limit: 1500Mi
- Alert: p95 inference latency above 500ms for 5 minutes

Raspberry Pi and GPU acceleration work is deferred until those measurements exist.
