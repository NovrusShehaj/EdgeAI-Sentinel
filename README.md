# EdgeAI Sentinel

Train, export, run, observe, and deploy a single-class object detector on one edge host.

This repository is **pre-production engineering**, not a production detector. There is no shipped trained ONNX model and no real-world accuracy claim.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-OCI-blue)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-purple)

## Current status

- Classification: development / pre-production candidate for the software loop
- Class contract: one class, `object`, from `configs/data.yaml`
- Dataset in Git: synthetic labels only; images are generated locally
- Model artifacts: mount or generate a versioned ONNX file at runtime
- CI: `.github/workflows/ci.yml` (lint, tests, image build, scans)
- Deploy: `.github/workflows/deploy.yml` is `workflow_dispatch` only

## Repository layout

```
EdgeAI-Sentinel/
├── edge/                         # Inference engine and FastAPI server
├── training/                     # YOLOv8 train/export helpers
├── benchmarks/                   # GPU and edge benchmark scripts
├── tests/                        # Unit and contract tests
├── scripts/                      # Synthetic data and demo ONNX helpers
├── configs/                      # Training and class configuration
├── data/                         # Synthetic labels; images are generated
├── models/                       # Runtime mount point (weights are not committed)
├── monitoring/prometheus/        # Scrape config and alert rules
├── monitoring/grafana/           # Provisioned datasource and dashboard
├── orchestration/docker/         # Edge image and monitoring compose
├── orchestration/ansible/        # One-device deploy playbook
├── orchestration/kubernetes/     # DaemonSet with hostPath models
├── results/                      # Measured artifacts only
├── docs/                         # Operator notes
└── .github/workflows/            # Active CI and manual deploy
```

## Quick start

Python 3.10+ is required. Docker is optional for the container path.

```bash
git clone git@github.com:NovrusShehaj/EdgeAI-Sentinel.git
cd EdgeAI-Sentinel
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
# or: pip install -r requirements-dev.txt
```

### 1. Generate a labeled demo ONNX graph (not a trained model)

```bash
python scripts/generate_demo_onnx.py --output models/demo.onnx --manifest models/model_manifest.json
```

### 2. Run the API locally

```bash
export API_TOKEN='local-dev-token'
export MODEL_PATH=models/demo.onnx
export MODEL_MANIFEST_PATH=models/model_manifest.json
export CLASS_NAMES=object
python -m uvicorn edge.api:app --host 127.0.0.1 --port 8080 --workers 1
```

Check probes:

```bash
curl -sS http://127.0.0.1:8080/live
curl -sS http://127.0.0.1:8080/ready
curl -sS -H "X-API-Token: $API_TOKEN" -F "file=@some.jpg" http://127.0.0.1:8080/infer
```

Without a model, `/live` is 200 and `/ready` is 503.

### 3. Headless camera or video loop

```bash
python -m edge.inference --model models/demo.onnx --source video.mp4 --headless --classes object
```

Do not use `cv2.imshow` on a headless host.

### 4. Tests

```bash
pytest tests/ -v --junitxml=test-results.xml -m "not training"
ruff check edge training tests scripts benchmarks
black --check edge training tests scripts benchmarks
```

### 5. Container

```bash
docker build -f orchestration/docker/Dockerfile.edge -t sentinel-edge:local .
docker run --rm -p 127.0.0.1:8080:8080 \
  -e API_TOKEN="$API_TOKEN" \
  -v "$PWD/models:/app/models:ro,z" \
  --user 1000:1000 \
  sentinel-edge:local
```

The image does not copy a Git `models/` tree. Mount an artifact or `/ready` stays 503. `:ro,z` keeps the mount read-only and applies Docker's shared SELinux label. Docker ignores `:z` when SELinux is not enabled; on Fedora/RHEL, plain `:ro` is denied.

### 6. Monitoring (localhost only)

```bash
export GRAFANA_ADMIN_PASSWORD='choose-a-local-password'
docker compose -f orchestration/docker/docker-compose.monitoring.yml up -d
```

Compose exits if `GRAFANA_ADMIN_PASSWORD` is unset. The default stack does not require `API_TOKEN`. Grafana is `http://127.0.0.1:3000`. Alertmanager routing is not configured. Prometheus, Grafana, and the optional API model bind mounts use `:ro,z` so SELinux hosts can read repository files; Docker ignores `:z` without SELinux.

Optional API sidecar (`--profile with-api`) still rejects inference when `API_TOKEN` is empty; set a non-empty token in the environment. No token is committed.

```bash
export API_TOKEN='local-dev-token'
docker compose -f orchestration/docker/docker-compose.monitoring.yml --profile with-api up -d
```

## Training and data

`python scripts/test_data.py` writes **synthetic** white rectangles for CI and short CPU smoke training. It is not a production dataset.

```bash
python scripts/test_data.py
python training/train.py --config configs/train_config.yaml --epochs 1 --device cpu
python training/export.py --checkpoint runs/train/exp/weights/best.pt --format onnx
```

Do not publish mAP, FPS, or power numbers unless they come from a measured file in `results/` with hardware and software tags.

The committed `results/gpu_benchmark.json` is **CPU-only** Ultralytics `yolov8n` timing. Raspberry Pi and RTX 3060 figures are not present and are not claimed.

## Deployment

One-device Ansible deployment is repository-ready and **externally unverified** until a real host and inventory exist.

```bash
cp inventory.ini.example inventory.local.ini
export API_TOKEN='...'
export SENTINEL_IMAGE=sentinel-edge:local
ansible-playbook orchestration/ansible/deploy_edge.yml -i inventory.local.ini --tags deploy,verify
```

Kubernetes uses a per-node hostPath at `/opt/sentinel/models`, `/live` for liveness, and `/ready` for readiness. Create `sentinel-api-token` before apply. See `docs/deployment.md`.

## Security notes

- `/infer` requires `X-API-Token` or `Authorization: Bearer`
- Uploads are capped by `MAX_UPLOAD_BYTES` and `MAX_IMAGE_SIDE`
- `/metrics` allows a token or private/loopback CIDRs
- Do not expose port 8080 to the public Internet without TLS
- No Grafana password is stored in Git

## License

MIT. See [LICENSE](LICENSE).
