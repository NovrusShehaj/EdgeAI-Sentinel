# EdgeAI Sentinel

**An end-to-end AI/ML pipeline demonstrating edge inference, GPU benchmarking, container orchestration, and infrastructure automation — built for Low-SWaP (Size, Weight, and Power) deployment environments.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Docker](https://img.shields.io/badge/Docker-OCI-blue)
![Ansible](https://img.shields.io/badge/Ansible-Automation-red)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-purple)
![Raspberry Pi](https://img.shields.io/badge/Edge-Raspberry%20Pi%204%2F5-green)

---

## Overview

EdgeAI Sentinel is a production-grade object detection system designed to simulate the kind of AI/ML infrastructure work done at enterprise AI factories. The project covers the full lifecycle of an AI model:

1. **Training** — Fine-tune YOLOv8 on a custom dataset using GPU-accelerated PyTorch
2. **Benchmarking** — Profile GPU memory, throughput, latency, and power consumption
3. **Export** — Convert trained models to ONNX and TensorRT for edge deployment
4. **Edge Deployment** — Run optimized inference on a Raspberry Pi (Low-SWaP platform)
5. **Orchestration** — Automate fleet deployment with Ansible and Docker
6. **Monitoring** — Track inference metrics with Prometheus + Grafana dashboards
7. **CI/CD** — Automated MLOps pipeline via GitHub Actions / GitLab CI

---

## Skills Demonstrated

| Skill | Implementation |
|---|---|
| GPU / HPC Computing | CUDA profiling, memory benchmarking, PyTorch training loop |
| Low-SWaP Edge Platforms | Raspberry Pi 4/5 inference with ONNX Runtime, power profiling |
| OCI / Container Runtimes | Docker + containerd deployment manifests |
| Kubernetes | K3s manifests for edge fleet orchestration |
| Infrastructure Automation | Ansible playbooks for provisioning and deployment |
| Python / Bash Scripting | Training, benchmarking, edge inference, and utility scripts |
| Monitoring / Observability | Prometheus metrics + Grafana dashboard |
| CI/CD MLOps | GitHub Actions pipeline for train → export → deploy |
| Networking + Security | Firewall rules in Ansible, secure API endpoints |

---

## Repository Structure

```
edgeai-sentinel/
├── training/              # Model training scripts (PyTorch / YOLOv8)
│   ├── train.py           # Main training entry point
│   ├── dataset.py         # Dataset loader and augmentation
│   └── export.py          # ONNX and TensorRT export
├── benchmarks/            # GPU and edge hardware benchmarking
│   ├── gpu_benchmark.py   # CUDA profiling (throughput, memory, power)
│   └── edge_benchmark.py  # Raspberry Pi inference benchmarking
├── edge/                  # Edge device inference application
│   ├── inference.py       # ONNX Runtime inference engine
│   ├── camera.py          # Camera stream handler (OpenCV)
│   └── api.py             # FastAPI inference server
├── orchestration/
│   ├── docker/            # Dockerfile for edge and training containers
│   ├── ansible/           # Playbooks for fleet provisioning
│   └── kubernetes/        # K3s manifests for edge cluster
├── monitoring/
│   ├── prometheus/        # Prometheus scrape configs
│   └── grafana/           # Dashboard JSON exports
├── scripts/               # Utility Bash scripts
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter exploration notebooks
├── docs/                  # Architecture docs and diagrams
└── .github/workflows/     # CI/CD pipeline definitions
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker + containerd
- NVIDIA GPU (optional, for training)
- Ansible 2.14+

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/edgeai-sentinel.git
cd edgeai-sentinel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python training/train.py --config configs/train_config.yaml --epochs 50
```

### 3. Run GPU benchmark

```bash
python benchmarks/gpu_benchmark.py --model yolov8n --batch-sizes 1 4 8 16
```

### 4. Export to ONNX

```bash
python training/export.py --checkpoint runs/train/exp/weights/best.pt --format onnx
```

### 5. Deploy to edge device

```bash
# Using Ansible
ansible-playbook orchestration/ansible/deploy_edge.yml -i inventory.ini

# Or manually on the Raspberry Pi
python edge/inference.py --model models/best.onnx --source 0
```

### 6. Launch monitoring stack

```bash
docker-compose -f orchestration/docker/docker-compose.monitoring.yml up -d
# Access Grafana at http://localhost:3000
```

---

## Hardware Reference

| Platform | Use Case | Notes |
|---|---|---|
| NVIDIA GPU (RTX 3060+) | Model training, TensorRT export | CUDA 11.8+ required |
| Raspberry Pi 4 (8GB) | Edge inference, API server | ARM64, ~3W idle |
| Raspberry Pi 5 (8GB) | Faster edge inference | ~4W idle, faster NPU |
| USB Camera / Pi Camera | Video input stream | OpenCV compatible |

---

## Results

Sample benchmark results (Raspberry Pi 4B, 8GB, YOLOv8n ONNX):

| Metric | Value |
|---|---|
| Inference latency (avg) | 142 ms/frame |
| Throughput | 7.0 FPS |
| Peak RAM | 380 MB |
| CPU utilization | 68% |
| Power draw (estimated) | 4.2W |

GPU training results (RTX 3060, batch=16):

| Metric | Value |
|---|---|
| Training throughput | 82 images/sec |
| GPU memory (peak) | 6.4 GB |
| mAP@0.5 (COCO val) | 0.532 |
| Export (ONNX) | ✅ |

---

## License

MIT — see [LICENSE](LICENSE)
