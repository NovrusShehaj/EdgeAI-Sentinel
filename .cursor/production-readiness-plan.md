# EdgeAI-Sentinel Production-Readiness Implementation Plan

Source: Cursor Agent / Grok 4.6 production-readiness audit
Audit branch: `dev/production-readiness`
Current classification: Development
Overall readiness score: 39/100

This plan is implementation-ready but must be executed only after review and approval. The audit phase made no implementation changes beyond creating and switching to the dedicated branch.

## Guiding Constraints

- Preserve the existing train → export → edge inference → observe → deploy pipeline.
- Do not introduce microservices, Kubernetes, service meshes, Kafka, Redis, or other infrastructure without a concrete project requirement.
- Do not publish unsupported model accuracy, latency, power, or hardware claims.
- Preserve existing user work and avoid destructive Git operations.
- Keep implementation work on `dev/production-readiness`.
- Do not commit, push, merge, or open a pull request unless explicitly requested.

## Priority and Effort

- P0 — Production blocker
- P1 — High priority
- P2 — Medium priority
- P3 — Nice to have

Effort values: Small, Medium, Large.

## Phase 1 — Make the Core Loop Actually Run

Depends on: the audit branch.

Completion criteria:

- Docker builds without requiring an absent `models/` directory.
- The API starts.
- `/ready` fails without a model and succeeds with one.
- A real or clearly labeled demo ONNX model loads.
- Class names match the model.
- Letterbox tests pass.
- One-node Uvicorn with a mounted ONNX model returns correct boxes on a known image.

### 1.1 Unify Class Contract and Configuration

Problem:

Training uses one class, `object`, while serving defaults use three classes: `person`, `vehicle`, and `equipment`.

Evidence:

- `configs/train_config.yaml` defines one class.
- `configs/data.yaml` contains the training class configuration.
- `edge/api.py` defines three default class names.
- `edge/inference.py` defines three default class names.
- `orchestration/kubernetes/edge-daemonset.yml` defines three class names.

Risk:

Class IDs can be reported with incorrect labels, making inference results misleading.

Recommendation:

Establish one source of truth for class names and use it consistently across training, inference, API defaults, and deployment manifests.

Implementation:

- Add a small class/config loader.
- Default the API and CLI to `object` until a multi-class model exists.
- Fail startup if configured class names do not match the model head when that can be inspected.
- Pass the same class configuration into Kubernetes and Ansible deployments.

Files likely affected:

- `configs/train_config.yaml`
- `configs/data.yaml`
- `edge/api.py`
- `edge/inference.py`
- `orchestration/kubernetes/edge-daemonset.yml`
- Tests

Validation:

- Test that API defaults match `data.yaml`.
- Test that class-count mismatches fail clearly.
- Run an inference response and verify the class name for a known class ID.

Priority: P0
Effort: Small
Dependencies: None

### 1.2 Fix Letterbox Inverse Mapping

Problem:

`ONNXInferenceEngine.postprocess` reverses letterbox preprocessing incorrectly for non-square images.

Evidence:

- `edge/inference.py`, `ONNXInferenceEngine.postprocess`.
- The current calculation uses `pad_x`, `pad_y`, and a derived `scale` that does not correctly invert the original preprocessing scale.
- A 640×640 image can appear correct while typical camera ratios such as 1280×720 are wrong.

Risk:

Bounding boxes can be silently misplaced or incorrectly scaled in production.

Recommendation:

Invert preprocessing using the original scale and padding values exactly.

Implementation:

- Map coordinates using the original preprocessing scale:
  - `x = (x_pred - pad_x) * x_scale`
  - `y = (y_pred - pad_y) * y_scale`
- Clip coordinates to the original image dimensions.
- Keep the transformation logic explicit and testable.

Files likely affected:

- `edge/inference.py`
- `tests/test_inference.py`

Validation:

- Add known-coordinate tests for 640×640, 1280×720, and 480×640.
- Use a synthetic YOLO-shaped output tensor with known boxes.
- Perform one visual check against a known image.

Priority: P0
Effort: Small
Dependencies: None

### 1.3 Split Liveness and Readiness

Problem:

`/health` returns HTTP 200 even when the model failed to load.

Evidence:

- `edge/api.py` lifespan and `health()` implementation.
- Docker health check uses `/health`.
- Kubernetes liveness and readiness probes use `/health`.
- Ansible health polling does not require `model_loaded`.

Risk:

Traffic can be routed to a process that cannot perform inference, and deployment automation can report a failed model as healthy.

Recommendation:

Provide separate process liveness and model readiness endpoints.

Implementation:

- Add `GET /live`, returning 200 when the process is running.
- Add `GET /ready`, returning 503 unless the inference engine is loaded.
- Keep `/health` as a human diagnostic endpoint.
- Update Docker, Kubernetes, and Ansible probes to use `/ready` where appropriate.

Files likely affected:

- `edge/api.py`
- `orchestration/docker/Dockerfile.edge`
- `orchestration/kubernetes/edge-daemonset.yml`
- `orchestration/ansible/deploy_edge.yml`
- Tests

Validation:

- Without a model: `/live` returns 200 and `/ready` returns 503.
- With a valid model: both return 200.
- Ansible asserts both HTTP status and `model_loaded`.

Priority: P0
Effort: Small
Dependencies: 1.5 for Docker validation

### 1.4 Establish a Model Artifact Strategy

Problem:

No model artifact exists, while Docker and Ansible expect `models/` and `models/best.onnx`.

Evidence:

- No `.pt`, `.onnx`, or `.engine` files are present.
- `orchestration/docker/Dockerfile.edge` contains `COPY models/ ./models/`.
- Ansible copies `models/best.onnx`.

Risk:

Deployment cannot work and model provenance, integrity, and rollback are undefined.

Recommendation:

Do not bake an absent or unverified model into the image. Mount or fetch a versioned model at runtime, with integrity metadata.

Implementation:

- Create `/app/models` in the image.
- Configure `MODEL_PATH` through the environment.
- Add a model manifest containing model version, source, class names, and SHA-256 checksum.
- Use a documented runtime mount or an artifact-fetch step.
- Use Git LFS or external artifact storage for large weights.
- If CI uses a stub, create a valid YOLO-shaped model or use a controlled session mock; do not use an incompatible `Identity` graph as a YOLO model.

Files likely affected:

- `orchestration/docker/Dockerfile.edge`
- CI workflow
- New model manifest
- Optional `.gitattributes`
- Deployment configuration

Validation:

- `docker build` succeeds without a Git-tracked `models/` directory.
- Container starts and `/ready` returns 503 until a model is mounted.
- A mounted model loads successfully.
- Checksum and class metadata are verified.

Priority: P0
Effort: Medium
Dependencies: 1.6 if using a trained export

### 1.5 Fix Docker Runtime User and Dependency Installation

Problem:

Runtime packages are installed into `/root/.local`, then the image switches to user `sentinel`.

Evidence:

- `orchestration/docker/Dockerfile.edge` builder/runtime stages.
- Dependencies are copied from `/root/.local`.
- The runtime uses `USER sentinel` and invokes Uvicorn.

Risk:

The runtime user may not be able to find or execute `uvicorn` and its dependencies.

Recommendation:

Install runtime dependencies into a system location available to the non-root user and add a `.dockerignore`.

Implementation:

- Install runtime dependencies into `/usr/local`.
- Switch to `USER sentinel` only after installation.
- Use an explicit Uvicorn command.
- Ensure the model directory is writable only if the application needs it; otherwise mount it read-only.
- Add `.dockerignore` to exclude local environments, runs, secrets, and unrelated artifacts.

Files likely affected:

- `orchestration/docker/Dockerfile.edge`
- `.dockerignore`

Validation:

- Build the image.
- Run it as non-root.
- Confirm Uvicorn starts.
- Query `/live` and `/ready`.

Priority: P0
Effort: Small
Dependencies: 1.4

### 1.6 Replace Synthetic-Only Validation or Label It Explicitly

Problem:

Current metrics are based on synthetic white rectangles rather than real-world imagery.

Evidence:

- `scripts/test_data.py` generates synthetic data.
- `data/` contains labels but no training images.
- `runs/detect/runs/train/exp6/results.csv` reports approximately 0.995 mAP on the synthetic data.
- README results imply production-like accuracy that is not supported by the artifacts.

Risk:

The project can create false confidence and ship an ineffective detector.

Recommendation:

Either add a real documented dataset or strictly limit synthetic data to CI smoke testing.

Implementation:

- Preferred: add a real public/domain dataset with image provenance, labels, splits, and a revision identifier.
- Alternative: retain synthetic data only for smoke tests and remove production-style metric claims.
- Fix data paths and document dataset provenance.
- Keep synthetic generation behind an explicit command and an `if __name__ == "__main__"` guard.

Files likely affected:

- `data/`
- `scripts/test_data.py`
- `configs/`
- README later

Validation:

- Confirm image and label counts match.
- Run a short CPU smoke-training job.
- For real data, evaluate on held-out images and record reproducible metrics.

Priority: P0
Effort: Medium to Large
Dependencies: None for the synthetic-only CI path; storage and dataset decisions for the real-data path

## Phase 2 — Security Baseline

Depends on: Phase 1.

Completion criteria:

- Unauthenticated inference is rejected.
- Uploads are capped by size and dimensions.
- Monitoring credentials are not committed.
- No unsafe `curl | sh` bootstrap remains.
- Dependencies are pinned or locked.
- Unsafe checkpoint loading is removed.

### 2.1 Authenticate and Limit the API

Problem:

`/infer` and `/metrics` are exposed without authentication, while the API binds to all interfaces.

Evidence:

- `edge/api.py` has no authentication implementation.
- The documented server binds to `0.0.0.0:8080`.
- The Docker and firewall configuration exposes the API port.
- Upload handling reads the full request body.

Risk:

Attackers or unintended clients can consume edge CPU/memory, scrape metrics, or cause denial of service.

Recommendation:

Use an API key or mTLS for LAN deployments, restrict metrics access, and do not expose port 8080 directly to the public Internet without TLS.

Implementation:

- Require an `API_TOKEN` header for inference.
- Compare tokens using a constant-time comparison.
- Reject missing or invalid tokens.
- Add request-body and image-dimension limits.
- Consider TrustedHost middleware and deny-by-default CORS.
- Restrict `/metrics` to the monitoring network or protect it separately.

Files likely affected:

- `edge/api.py`
- Ansible/Kubernetes/compose environment configuration
- Tests

Validation:

- Missing token returns 401.
- Valid token permits inference.
- Oversized payload returns 413.
- Excessive dimensions are rejected.
- Metrics cannot be scraped from an unauthorized network path.

Priority: P0
Effort: Medium
Dependencies: 1.3

### 2.2 Remove Hardcoded Monitoring Credentials

Problem:

Grafana uses a hardcoded password of `sentinel`.

Evidence:

- `orchestration/docker/docker-compose.monitoring.yml`.

Risk:

Monitoring access can be compromised immediately if the stack is exposed.

Recommendation:

Require an environment-provided credential and restrict monitoring ports.

Implementation:

- Use `${GRAFANA_ADMIN_PASSWORD}` without a committed default.
- Document a local environment file that is ignored by Git.
- Bind Grafana and Prometheus to localhost or an internal network unless external exposure is required.

Files likely affected:

- `orchestration/docker/docker-compose.monitoring.yml`
- `.gitignore`

Validation:

- Compose fails closed when the credential is absent.
- No password appears in tracked files.
- Published ports are intentionally scoped.

Priority: P1
Effort: Small
Dependencies: Git-ignore work

### 2.3 Harden Ansible Bootstrap

Problem:

Ansible uses `curl | sh`, grants Docker-group membership, and references a missing template.

Evidence:

- `orchestration/ansible/deploy_edge.yml`.
- Docker installation task.
- User/group configuration.
- `templates/prometheus_target.yml.j2` reference.

Risk:

Supply-chain compromise, effective-root access through the Docker socket, and deployment failure.

Recommendation:

Use distribution packages or a pinned installer artifact, avoid Docker-group access for the application user, and add or remove the missing template task.

Implementation:

- Install Docker through the supported package manager where possible.
- Use rootless or controlled Docker access if application-level access is necessary.
- Add `templates/prometheus_target.yml.j2` if required.
- Keep SSH host-key checking enabled.
- Remove ignored pull failures and fail deployment when the image is unavailable.

Files likely affected:

- `orchestration/ansible/deploy_edge.yml`
- `orchestration/ansible/templates/prometheus_target.yml.j2`
- CI workflow after relocation

Validation:

- `ansible-playbook --syntax-check`
- `ansible-lint`
- Molecule or one VM smoke test

Priority: P1
Effort: Medium
Dependencies: Phase 5 inventory

### 2.4 Safe Checkpoint Loading and Dependency Pinning

Problem:

Checkpoint loading can deserialize unsafe pickle content, and dependency versions float.

Evidence:

- `training/export.py`, `print_model_info`.
- `requirements.txt` uses broad version ranges.
- No lockfile exists.

Risk:

Untrusted checkpoints can execute code, and builds are not reproducible.

Recommendation:

Use safe checkpoint loading and generate a lockfile. Split edge, training, and development dependencies.

Implementation:

- Use `weights_only=True` where supported or use a safe Ultralytics loading path.
- Add `requirements-edge.txt`.
- Add `requirements-train.txt`.
- Add development requirements.
- Generate a lockfile with `pip-compile` or `uv`.
- Install from the lockfile in CI.

Files likely affected:

- `training/export.py`
- `requirements.txt`
- New requirement files
- Lockfiles
- CI workflow

Validation:

- Install dependencies in a clean virtual environment.
- Run `pip-audit` or OSV scanning.
- Verify checkpoint handling with a safe and an invalid/untrusted input.

Priority: P1
Effort: Medium
Dependencies: CI work

## Phase 3 — Make CI/CD Real

Depends on: Phase 1 Docker/tests compiling; Phase 2 lockfile preferred.

Completion criteria:

- A workflow under `.github/workflows/` runs lint, tests, and container build.
- Pull requests do not deploy.
- Test results are actually produced.
- Vulnerability and secret scanning are included.

### 3.1 Move and Simplify the Workflow

Problem:

The only workflow is under `docs/.github/workflows/ci_cd.yml`, which GitHub Actions will not load.

Evidence:

- `docs/.github/workflows/ci_cd.yml`.
- No root `.github/workflows/` directory.

Risk:

CI is inactive and cannot enforce quality or build artifacts.

Recommendation:

Move the workflow to `.github/workflows/ci.yml` and remove automatic deployment from the initial working pipeline.

Implementation:

- Use `.github/workflows/ci.yml`.
- Run lint, formatting checks, tests, and image build.
- Use `pytest --junitxml=test-results.xml` if the result is uploaded.
- Cache dependencies.
- Install edge/development dependencies for tests rather than the full Jupyter/training stack.
- Make benchmarks optional or manually triggered initially.
- Add `pip-audit`, OSV, secret scanning, and image scanning.

Files likely affected:

- `docs/.github/workflows/ci_cd.yml`
- `.github/workflows/ci.yml`

Validation:

- Run with `act` or through a draft pull request.
- Confirm expected jobs execute.
- Confirm test artifacts exist before upload.

Priority: P0
Effort: Medium
Dependencies: 1.4, 1.5, and testing improvements

### 3.2 Use Valid CI Model Fixtures and Remove Default Deployment

Problem:

CI uses an invalid dummy ONNX model and deploys on every `main` push.

Evidence:

- Dummy ONNX generation in the workflow.
- Deploy job condition on the main branch.
- Empty deployment inventory.

Risk:

CI cannot meaningfully validate inference, and an accidental push can trigger a broken production deployment.

Recommendation:

Use a valid small model or controlled session mock, and make deployment an explicitly approved operation.

Implementation:

- Add a valid model fixture or artifact-generation step.
- Separate build and deployment workflows.
- Require `workflow_dispatch` for deployment.
- Require a protected production environment and populated inventory.

Files likely affected:

- `.github/workflows/*`
- `benchmarks/edge_benchmark.py`

Validation:

- Benchmark loads and runs the model successfully.
- Pull requests never deploy.
- Manual deployment requires explicit environment approval.

Priority: P1
Effort: Medium
Dependencies: 3.1

## Phase 4 — Tests That Can Fail the Build

Depends on: 1.2, 1.3, and Phase 3.

Completion criteria:

- Full pytest suite passes locally and in CI.
- Postprocessing, authentication, readiness, and upload limits are covered.
- Metrics registry is isolated.
- Broken setup is not converted into a skip.

### 4.1 Fix Fixtures and Metric Registry

Problem:

Metric names can be registered more than once, and the dummy ONNX fixture has incompatible behavior.

Evidence:

- `_setup_metrics` in `edge/inference.py`.
- `dummy_onnx_model` in tests/benchmark paths.
- Identity graph input/output mismatch.

Risk:

Tests may fail for environmental reasons or fail to exercise the actual inference contract.

Recommendation:

Use an injected or isolated Prometheus registry and a valid model fixture or session mock.

Implementation:

- Add a registry parameter to the engine or isolate metrics at module scope.
- Provide a test registry fixture.
- Replace the Identity graph with a valid YOLO-shaped output fixture or mock `session.run`.
- Ensure tests construct more than one engine instance safely.

Files likely affected:

- `edge/inference.py`
- `tests/test_inference.py`

Validation:

- Initialize two engines in one process.
- Run pytest with Prometheus installed.
- Confirm fixture outputs match the expected inference schema.

Priority: P1
Effort: Medium
Dependencies: 1.2

### 4.2 Add Contract Tests

Problem:

Export, Docker, API readiness, authentication, upload limits, and deployment configuration are not tested.

Recommendation:

Add focused contract tests rather than pursuing arbitrary coverage percentages.

Implementation:

- Test `/live`, `/ready`, `/health`, `/infer`, and `/metrics` behavior.
- Test authentication and upload limits.
- Test known geometry transformations.
- Validate ONNX structure with `onnx.checker`.
- Add optional Docker, Ansible, Kubernetes, and Prometheus checks.

Files likely affected:

- `tests/`
- CI configuration

Validation:

- CI fails if `/ready` returns 200 without a model.
- CI fails if known coordinates are mapped incorrectly.
- CI fails on invalid deployment configuration.

Priority: P1
Effort: Medium
Dependencies: 2.1 and 1.3

## Phase 5 — Deployment That Can Reach One Device

Depends on: Phase 1 image, Phase 2 Ansible hardening, and Phase 3 image publication if required.

Completion criteria:

- One Raspberry Pi or VM runs the container.
- `/ready` returns 200 with a verified model.
- Prometheus can scrape the service if configured.
- Restart and shutdown behavior are documented and validated.

### 5.1 Inventory and Image References

Problem:

Inventory is empty, image references are placeholders, and image pull failures are ignored.

Evidence:

- `inventory.ini` has no real edge devices.
- Kubernetes image uses a placeholder registry/user.
- Ansible pull task ignores errors.

Risk:

Deployment cannot target a real device reliably and can report success when the image is absent.

Recommendation:

Keep an example inventory in Git, keep real inventory outside Git, and use one canonical image reference.

Implementation:

- Add `inventory.ini.example`.
- Keep real inventory ignored or in a controlled secret store.
- Use `sentinel-edge:${GIT_SHA}` for local builds where appropriate.
- Use the same image name in CI, Ansible, and Kubernetes.
- Fail deployment if image acquisition fails.

Files likely affected:

- `inventory.ini`
- New `inventory.ini.example`
- `orchestration/ansible/deploy_edge.yml`
- `orchestration/kubernetes/edge-daemonset.yml`

Validation:

- Ansible syntax check.
- One-device deployment with explicit verification tags.
- Confirm image and model versions on the device.

Priority: P1
Effort: Medium
Dependencies: 1.5 and 2.3

### 5.2 Fix Kubernetes Model Volume

Problem:

A DaemonSet uses one RWO PVC across nodes.

Evidence:

- `orchestration/kubernetes/edge-daemonset.yml` PVC and volume configuration.

Risk:

Pods on multiple nodes cannot reliably mount the same RWO volume.

Recommendation:

Use per-node storage or a suitable shared volume. If fleet storage is not required yet, use a single-replica Deployment.

Implementation:

- Use a documented host model path such as `/opt/sentinel/models`, or select storage based on actual cluster requirements.
- Add a non-root security context.
- Drop unnecessary capabilities.
- Use `/ready` for readiness.

Files likely affected:

- `orchestration/kubernetes/edge-daemonset.yml`

Validation:

- `kubeconform`.
- `kubectl apply --dry-run=client`.
- Validate one-node and multi-node behavior if the DaemonSet remains.

Priority: P1
Effort: Small
Dependencies: 1.3

## Phase 6 — Observability and Reliability

Depends on: Phase 1 API metrics and Phase 5 scrape path.

Completion criteria:

- Monitoring compose starts.
- At least one dashboard exists.
- At least one alert can fire.
- Failure metrics increment.
- Readiness and latency are observable.

### 6.1 Complete Prometheus and Grafana Files

Problem:

Prometheus alert rules and Grafana provisioning are missing.

Evidence:

- `monitoring/prometheus/prometheus.yml` references `alert_rules.yml`.
- `docker-compose.monitoring.yml` mounts a missing Grafana directory.
- Alertmanager targets are empty.

Risk:

Monitoring may fail to start and operators will lack actionable visibility.

Recommendation:

Add a minimal operational monitoring stack before adding advanced dashboards.

Implementation:

- Add alert rules for scrape down, readiness failure, and high p95 latency.
- Add one provisioned Grafana dashboard.
- Use compose service names for local scrape targets.
- Define alert routing or explicitly document that routing is not yet configured.

Files likely affected:

- `monitoring/prometheus/alert_rules.yml`
- `monitoring/grafana/`
- `monitoring/prometheus/prometheus.yml`
- `orchestration/docker/docker-compose.monitoring.yml`

Validation:

- `promtool check config`.
- `promtool check rules`.
- Start the compose stack.
- Confirm the dashboard and an alert are visible.

Priority: P0 for stack startup; P1 for useful alerting
Effort: Medium
Dependencies: 2.2

### 6.2 Add Reliability Controls

Problem:

There are no inference timeouts, failure metrics, or clearly defined ONNX Runtime concurrency controls.

Evidence:

- `edge/inference.py` inference path.
- `edge/api.py` exception handling.
- Lock scope protects frame IDs but not necessarily `session.run`.

Risk:

Hung inference, resource exhaustion, concurrent runtime failures, and poor incident diagnosis.

Recommendation:

Bound inference work and make failure behavior observable.

Implementation:

- Add a bounded executor or timeout wrapper.
- Increment error metrics with safe status labels.
- Serialize `session.run` if required by the selected ONNX Runtime version.
- Document the worker and concurrency model.
- Define behavior when the model becomes unavailable after startup.

Files likely affected:

- `edge/inference.py`
- `edge/api.py`

Validation:

- Timeout test.
- Error-counter test.
- Concurrent-request test.
- Restart and shutdown smoke test.

Priority: P1
Effort: Small
Dependencies: 4.1

## Phase 7 — AI Quality and Performance

Depends on: real data and model artifact.

Completion criteria:

- Reported metrics come from real validation data.
- Edge benchmarks are tagged with hardware and software versions.
- Latency and memory SLOs are documented.
- Model checksum and provenance are recorded.

### 7.1 Retrain and Export a Real ONNX Model

Problem:

Current metrics are synthetic and CPU-only, and README results contradict the available artifacts.

Evidence:

- `results/gpu_benchmark.json` is CPU-only.
- `runs/detect/runs/train/exp6/results.csv` is based on synthetic data.
- README claims unsupported Pi and RTX 3060 results.

Recommendation:

Retrain on real data, export ONNX, benchmark on target hardware, and publish only measured results.

Implementation:

- Record random seed, dataset revision, image size, hardware, and lockfile version.
- Export a versioned ONNX model.
- Add `results/model_manifest.json` with checksum and metadata.
- Add a hardware-tagged edge benchmark JSON.
- Remove unsupported README claims.

Files likely affected:

- `training/`
- `results/`
- `models/`
- README later

Validation:

- Independent held-out validation.
- ONNX checker.
- Latency, memory, and throughput measurements against defined SLOs.
- Compare results with the previous model and document regressions.

Priority: P1
Effort: Large
Dependencies: 1.6

### 7.2 Optimize Edge Inference Only if Measurements Require It

Problem:

Raspberry Pi performance claims are unsupported, and acceleration paths are not established.

Evidence:

- No edge benchmark JSON exists.
- README Pi numbers are not backed by artifacts.

Recommendation:

Measure first. Optimize only when the target SLO is missed.

Implementation:

Evaluate, in order:

- Image sizes 320 or 416.
- Thread-count sweep.
- XNNPACK or hardware-specific execution providers.
- INT8 quantization if accuracy remains acceptable.
- Device-specific acceleration only after baseline measurements.

Files likely affected:

- `edge/inference.py`
- Export tooling
- Benchmark scripts

Validation:

- Before/after benchmark JSON on identical hardware.
- Accuracy regression check.
- Memory and thermal stability test.

Priority: P2
Effort: Medium
Dependencies: 7.1

## Phase 8 — Hygiene, Documentation, and Git

Depends on: Phases 1–3 so documentation describes reality.

Completion criteria:

- README file tree matches disk.
- LICENSE exists.
- `.gitignore` and `.dockerignore` exist.
- Placeholder URLs and registry names are removed.
- Published results match actual artifacts.
- Fresh-clone onboarding works.

### 8.1 Add Ignore Rules, License, and Accurate README

Problem:

No ignore rules or license exist, and README paths/results are stale or unsupported.

Evidence:

- `.gitignore` is missing.
- `LICENSE` is missing.
- README references absent modules and paths.
- README uses placeholder clone URLs and image references.

Recommendation:

Align documentation and repository hygiene with the implemented system.

Implementation:

- Add `.gitignore` for virtual environments, runs, local artifacts, environment files, macOS junk, and secrets.
- Add `.dockerignore`.
- Add the intended MIT license if that remains the project’s license.
- Rewrite results based only on measured artifacts.
- Remove or add references to missing modules.
- Correct clone URLs, image names, checkpoint paths, and deployment assumptions.

Files likely affected:

- `.gitignore`
- `.dockerignore`
- `LICENSE`
- `README.md`
- Possibly committed `runs/`

Validation:

- Fresh clone and clean virtual environment.
- Follow the README quick start.
- Confirm no secrets or unintended artifacts are staged.
- Confirm file tree documentation matches the repository.

Priority: P2, with ignore rules treated as P1 if secret exposure risk increases
Effort: Small
Dependencies: Phase 1 facts

### 8.2 Improve Developer Packaging

Problem:

There is no `pyproject.toml`, editable-install path, or explicit package structure.

Recommendation:

Add minimal packaging without redesigning the project.

Implementation:

- Add `pyproject.toml`.
- Define edge, training, and development dependency extras.
- Add package initialization files where appropriate.
- Provide standard commands for linting, testing, and local startup.

Files likely affected:

- `pyproject.toml`
- `edge/__init__.py`
- `training/__init__.py`

Validation:

- `pip install -e '.[dev]'`.
- Run pytest from a clean environment.
- Verify CLI/module entry points.

Priority: P3
Effort: Small
Dependencies: 2.4

## Recommended Execution Order

1. Stay on `dev/production-readiness`.
2. Execute Phase 1:
   - unify classes
   - fix letterbox mapping
   - add `/ready`
   - fix Docker runtime
   - stop copying an absent `models/` directory
   - define the model artifact strategy
   - clarify data validity
3. Add geometry and metric-registry tests immediately.
4. Move CI to `.github/workflows/ci.yml`.
5. Make CI PR-safe and ensure tests and builds actually execute.
6. Add API authentication and upload limits.
7. Lock dependencies and use safe checkpoint loading.
8. Complete Prometheus/Grafana configuration.
9. Harden Ansible and deploy to one device.
10. Retrain, export, and benchmark using real data.
11. Update documentation, license, ignore rules, and published results.
12. Do not begin Helm, GitLab CI, or a TensorRT fleet before the single-device path works.

## Verification Strategy

### Static Checks

- `ruff check`
- `black --check`
- `ansible-lint`
- `ansible-playbook --syntax-check`
- `kubeconform`
- `kubectl apply --dry-run=client`
- `promtool check config`
- `promtool check rules`
- Hadolint for the Dockerfile
- `pip-audit` or OSV scanning against the lockfile
- Secret scanning
- Container vulnerability scanning

### Unit and Contract Checks

- `pytest tests/ -v --junitxml=test-results.xml`
- No silent skips for broken imports/configuration in the full job.
- Geometry tests for 640×640 and 1280×720.
- `/live` returns 200.
- `/ready` returns 503 without a model.
- `/ready` returns 200 with a valid model.
- `/infer` returns 401 without authentication.
- Oversized uploads return 413.
- Invalid content types return 400.
- Model class names match configuration.
- Multiple engine instances do not collide in the metrics registry.

### Build Validation

- `docker build -f orchestration/docker/Dockerfile.edge .` succeeds without a Git-tracked `models/` directory.
- The container runs as non-root.
- Uvicorn starts successfully.
- `/live`, `/ready`, and authenticated `/infer` behave correctly with a mounted model.

### Security Validation

- No default passwords are tracked.
- No `.env` or credential files are staged.
- CI does not disable SSH host verification.
- Dependency and image vulnerability scans run.
- Model checksum verification works.
- API upload limits cannot be bypassed through oversized bodies or dimensions.

### Integration and Smoke Testing

- Start the API container with a mounted model.
- Send an authenticated small JPEG.
- Start the monitoring compose stack.
- Confirm Prometheus readiness and scrape collection.
- Deploy to one Raspberry Pi or VM.
- Validate startup, inference, restart, and shutdown.
- Confirm readiness becomes unhealthy when the model is unavailable.

### AI Validation

- Run `onnx.checker` on the shipped model.
- Validate held-out images with known ground truth.
- Check IoU and class correctness.
- Benchmark on named hardware.
- Publish only values contained in reproducible result files.

### Manual Validation

- Follow the README from a fresh clone.
- Run the headless edge path without `cv2.imshow`.
- Confirm GitHub Actions executes from `.github/workflows/`.
- Confirm rollback to a prior image/model tag.
- Confirm operator documentation covers common failures.

## Definition of Done

The project may reasonably be called Pre-Production only when all of the following are true:

- A versioned ONNX model exists with checksum.
- Class names match training and serving.
- Letterbox tests pass on square and 16:9 frames.
- `/ready` returns 503 without a model and probes use it.
- The Docker edge image builds.
- The process runs as non-root and Uvicorn is available.
- `.github/workflows/ci.yml` runs lint, tests, and image build.
- `/infer` requires a secret header.
- Uploads are size-limited.
- Dependencies are locked.
- Unsafe checkpoint loading is removed.
- Prometheus configuration and alert rules validate.
- Monitoring compose starts.
- Ansible syntax-checks.
- One documented host can deploy, or deployment is explicitly documented as out of scope.
- README file tree, clone URL, and results match reality.
- `.gitignore`, `.dockerignore`, and `LICENSE` exist.
- No default Grafana password is stored in Git.
- No Identity-shaped YOLO model is used as a production or CI success signal.

Production Candidate additionally requires:

- Real non-synthetic validation metrics.
- Hardware-tagged edge benchmark JSON.
- TLS or a documented air-gapped network model.
- Rollback image/model tags.

Production Ready additionally requires:

- Pinned fleet inventory in a secret store.
- Alert routing.
- Vulnerability scanning on the image.
- An on-call process and operational runbook.

## Recommended Next Action

Review this plan before implementation. If approved, start Phase 1 only:

- Fix postprocessing math.
- Align class names.
- Add `/ready`.
- Stop `COPY models/`.
- Install Python dependencies into `/usr/local`.
- Decide the model-mount versus Git LFS/artifact-storage strategy.

Do not refactor training, add Kubernetes features, or rewrite the README until those paths work.
