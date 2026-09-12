# One-device deployment

Repository-side deployment is implemented. A live Raspberry Pi or VM deploy is **out of scope until a private inventory and image tag exist**.

## Local image

```bash
docker build -f orchestration/docker/Dockerfile.edge -t sentinel-edge:<git-sha> .
docker tag sentinel-edge:<git-sha> sentinel-edge:local
```

Canonical local name: `sentinel-edge`. Pin `sentinel-edge:<git-sha>` for rollback.

## Ansible

1. Copy `inventory.ini.example` to `inventory.local.ini`
2. Set `API_TOKEN` and `SENTINEL_IMAGE`
3. Place `best.onnx` and `model_manifest.json` under `models/` on the controller, or copy them to `/opt/sentinel/models` on the device
4. Run `ansible-playbook orchestration/ansible/deploy_edge.yml -i inventory.local.ini --tags deploy,verify`

The playbook installs Docker from distro packages, does not add `sentinel` to the Docker group, keeps SSH host-key checking enabled, uses `/live` for the container health check, and asserts `/ready` plus `model_loaded`. The model bind is `/opt/sentinel/models:/app/models:ro,z`, matching Compose: `:ro` stays read-only and `:z` is Docker's shared SELinux relabel so Fedora/RHEL can read the host models directory. Docker ignores `:z` when SELinux is not enabled.

## Kubernetes

- Models: hostPath `/opt/sentinel/models`
- Liveness: `/live`
- Readiness: `/ready`
- Image: `sentinel-edge:local` with `IfNotPresent` for local loads
- Secret: `sentinel-api-token` with key `API_TOKEN` (see `orchestration/kubernetes/secret.example.yml`)

```bash
kubectl apply --dry-run=client -f orchestration/kubernetes/edge-daemonset.yml
```

## Local monitoring compose

```bash
export GRAFANA_ADMIN_PASSWORD='...'
docker compose -f orchestration/docker/docker-compose.monitoring.yml config
docker compose -f orchestration/docker/docker-compose.monitoring.yml up -d
```

The default profile interpolates only `GRAFANA_ADMIN_PASSWORD`. `API_TOKEN` is optional at render time because Compose expands every service before profiles apply. `--profile with-api` still cannot infer without a non-empty token; do not commit one.

Repository bind mounts use `:ro,z` (Prometheus config, Grafana provisioning, and the optional API model directory). `:ro` is read-only. `:z` is Docker's shared SELinux relabel so a Fedora/RHEL host can serve those files into the container. Docker and Compose accept `:z` on Ubuntu, Debian, and Docker Desktop and ignore it when SELinux is not enabled. Do not add `:z` to node-exporter's `/proc`, `/sys`, or `/` mounts. A one-off `docker run` of the edge image on SELinux needs the same suffix, for example `-v "$PWD/models:/app/models:ro,z"`.

## Network model

This stack is intended for a trusted LAN or air-gapped site. TLS termination is not implemented. Do not publish port 8080, 3000, or 9090 on the public Internet. Local compose binds Grafana/Prometheus/node-exporter to `127.0.0.1`. Ansible UFW allows the API only from `monitoring_server_ip`.

## Restart and shutdown

- Missing model at start: process stays up, `/ready` is 503, `/infer` is 503
- Model file removed after start: the next infer/ready check fails closed; restart after restoring the file
- Shutdown: Uvicorn lifespan closes the engine executor
- Rollback: run a previous `sentinel-edge:<git-sha>` and the matching manifest/checksum
