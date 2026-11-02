# Copilot instructions for EdgeAI Sentinel

## Project shape

EdgeAI Sentinel is a Python 3.10+ YOLOv8 object-detection pipeline for low-SWaP edge deployment:

- `training/train.py` loads `configs/train_config.yaml`, generates the Ultralytics-compatible `configs/data.yaml`, and trains a YOLO checkpoint using PyTorch/Ultralytics.
- `training/export.py` converts a `.pt` checkpoint into deployment artifacts. Static 640x640 ONNX is the primary Raspberry Pi/CPU format; TensorRT export is optional and hardware-specific.
- `edge/inference.py` is the core runtime. It preprocesses BGR OpenCV frames with letterboxing, runs one ONNX Runtime CPU session, performs confidence filtering and NMS, and exposes Prometheus metrics.
- `edge/api.py` wraps one process-wide inference engine in FastAPI. `/health`, `/info`, `/infer`, and `/metrics` support deployment probes, model metadata, image inference, and monitoring. Runtime settings come from environment variables.
- `benchmarks/edge_benchmark.py` measures ONNX latency, throughput, CPU/RAM, temperature, and estimated power across thread counts. `benchmarks/gpu_benchmark.py` measures Ultralytics/PyTorch GPU throughput, latency, and VRAM.
- `orchestration/docker/Dockerfile.edge` builds the non-root edge API image for amd64/arm64. Ansible (`orchestration/ansible/deploy_edge.yml`) deploys it to edge hosts, while `orchestration/kubernetes/edge-daemonset.yml` runs one API pod per labeled K3s node. Prometheus/Grafana configuration is under `monitoring/` and `orchestration/docker/`.

Run commands from the repository root. Many paths in the training config, generated data YAML, deployment manifests, and scripts are relative to the current working directory.

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The normal development environment installs CPU ONNX Runtime. Uncomment/use `onnxruntime-gpu` only on hosts configured for CUDA. Training and GPU benchmarks additionally depend on the PyTorch/Ultralytics stack and a suitable GPU for practical runtimes.

## Build, test, lint, and benchmark commands

There is no package build step; the deployable build is the edge container:

```bash
docker build -f orchestration/docker/Dockerfile.edge .
```

Run the test suite and one test:

```bash
pytest tests/ -v --tb=short
pytest tests/test_inference.py::TestONNXInferenceEngine::test_preprocess_output_shape -v
```

The CI definition uses these quality checks:

```bash
ruff check .
black --check --diff .
```

Use `black .` only when intentionally applying formatting changes. The repository’s CI workflow is stored at `docs/.github/workflows/ci_cd.yml`; it describes lint, test, CPU benchmark, container build/push, and protected-main Ansible deployment stages.

CPU/edge benchmark (works without a real checkpoint because the script can generate a dummy ONNX model):

```bash
python benchmarks/edge_benchmark.py \
  --model models/best.onnx \
  --threads 1 2 4 \
  --output results/edge_benchmark.json
```

GPU benchmark:

```bash
python benchmarks/gpu_benchmark.py \
  --model yolov8n \
  --batch-sizes 1 4 8 16
```

## Model lifecycle

Training is configured in `configs/train_config.yaml`; CLI values override the corresponding training settings:

```bash
python training/train.py --config configs/train_config.yaml --epochs 50
```

The training script writes Ultralytics runs under `runs/train/` and expects the best checkpoint at the selected run’s `weights/best.pt`. Export it for edge use:

```bash
python training/export.py \
  --checkpoint runs/train/exp/weights/best.pt \
  --format onnx \
  --output-dir models/
```

Keep the dataset in YOLO layout (`images/` and `labels/` beneath `data/train` and `data/val`), keep class names/counts consistent across `configs/train_config.yaml`, `configs/data.yaml`, API environment configuration, and inference CLI arguments, and preserve static input shapes when changing the edge model.

## Runtime and deployment conventions

- The inference engine owns a single ONNX Runtime session and serializes frame-id updates with a lock; reuse the engine rather than creating one per request.
- Coordinate handling assumes OpenCV BGR input and letterbox preprocessing. Changes to preprocessing must be matched by postprocessing scale/padding recovery.
- API defaults are defined by `MODEL_PATH`, `CLASS_NAMES`, `CONF_THRESHOLD`, `IOU_THRESHOLD`, `NUM_THREADS`, and `PORT`. The API intentionally starts in a degraded state when the model is absent; `/infer` and `/info` then return `503`.
- `/infer` accepts only JPEG, PNG, or BMP uploads and returns Pydantic-validated detection data. Preserve this response contract when changing inference internals.
- `models/best.onnx` is the runtime artifact expected by the Docker image and deployment manifests. TensorRT engines are device-specific and must be built on the target NVIDIA device.
- The Docker image runs as the non-root `sentinel` user and exposes port `8080`; keep model storage read-only when deploying through Ansible or Kubernetes.
- Kubernetes scheduling expects nodes labeled `role=edge-inference`; the DaemonSet exposes the API through NodePort `30080`. Ansible uses `inventory.ini`, SSH key authentication, and tags such as `setup`, `deploy`, `verify`, and `monitoring`.
- Prometheus metric names and the `/metrics` endpoint are part of the monitoring integration. Add or rename metrics deliberately because `monitoring/prometheus/prometheus.yml` and deployed dashboards consume them.

## Test conventions

Tests use pytest fixtures and build a minimal in-memory-compatible ONNX graph where possible, so inference tests do not require GPU hardware or trained weights. Preserve this pattern for new engine/API tests: use temporary model files and synthetic NumPy/OpenCV images, and skip only when the optional dependency under test is unavailable.

