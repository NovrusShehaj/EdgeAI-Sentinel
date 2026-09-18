"""Static contract checks for Docker, Ansible, Kubernetes, and Prometheus."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "orchestration" / "docker" / "Dockerfile.edge"
COMPOSE = ROOT / "orchestration" / "docker" / "docker-compose.monitoring.yml"
K8S = ROOT / "orchestration" / "kubernetes" / "edge-daemonset.yml"
ANSIBLE = ROOT / "orchestration" / "ansible" / "deploy_edge.yml"
PROM = ROOT / "monitoring" / "prometheus" / "prometheus.yml"
ALERTS = ROOT / "monitoring" / "prometheus" / "alert_rules.yml"


def test_dockerfile_does_not_copy_models_tree():
    text = DOCKERFILE.read_text()
    assert "COPY models/" not in text
    assert "USER sentinel" in text
    assert "/usr/local" in text
    assert "/ready" in text or "/live" in text
    assert "CLASS_NAMES=object" in text


def test_compose_has_no_default_grafana_password():
    text = COMPOSE.read_text()
    assert "GF_SECURITY_ADMIN_PASSWORD: sentinel" not in text
    assert "GRAFANA_ADMIN_PASSWORD" in text
    assert "127.0.0.1:3000:3000" in text
    assert "127.0.0.1:9090:9090" in text


def test_compose_bind_mounts_use_shared_selinux_relabel():
    text = COMPOSE.read_text()
    assert "../../monitoring/prometheus:/etc/prometheus:ro,z" in text
    assert "../../monitoring/grafana:/etc/grafana/provisioning:ro,z" in text
    assert "${SENTINEL_MODEL_DIR:-../../models}:/app/models:ro,z" in text
    assert "- /proc:/host/proc:ro" in text
    assert "- /sys:/host/sys:ro" in text
    assert "- /:/rootfs:ro" in text
    assert "/host/proc:ro,z" not in text
    assert "/host/sys:ro,z" not in text
    assert "/rootfs:ro,z" not in text
    assert ":/etc/prometheus:ro\n" not in text
    assert ":/etc/grafana/provisioning:ro\n" not in text
    assert ":/app/models:ro\n" not in text


def test_compose_keeps_api_token_optional_at_interpolation():
    text = COMPOSE.read_text()
    assert "${API_TOKEN:?" not in text
    assert "${GRAFANA_ADMIN_PASSWORD:?" in text
    assert "API_TOKEN: ${API_TOKEN:-}" in text
    assert "profiles:" in text and "with-api" in text
    assert "API_TOKEN: sentinel" not in text
    assert "API_TOKEN: change-me" not in text
    assert "API_TOKEN: replace-me" not in text


_COMPOSE_AVAILABLE: bool | None = None


def _docker_compose_available(docker: str) -> bool:
    """Check (and cache) that the docker CLI has a working compose plugin."""
    global _COMPOSE_AVAILABLE
    if _COMPOSE_AVAILABLE is None:
        try:
            probe = subprocess.run(
                [docker, "compose", "version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            _COMPOSE_AVAILABLE = probe.returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _COMPOSE_AVAILABLE = False
    return _COMPOSE_AVAILABLE


def _compose_config(
    *args: str, extra_env: dict[str, str | None]
) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None or not _docker_compose_available(docker):
        pytest.skip("docker compose is not available")
    env = os.environ.copy()
    for key, value in extra_env.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return subprocess.run(
        [docker, "compose", "-f", str(COMPOSE), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _bind_volume(service: dict, target: str) -> dict:
    for volume in service["volumes"]:
        if volume.get("type") == "bind" and volume.get("target") == target:
            return volume
    raise AssertionError(f"bind mount for {target} not found")


def test_compose_config_default_stack_needs_only_grafana_password():
    result = _compose_config(
        "config",
        extra_env={
            "GRAFANA_ADMIN_PASSWORD": "test-only-grafana-password",
            "API_TOKEN": None,
        },
    )
    assert result.returncode == 0, result.stderr
    assert "sentinel-grafana" in result.stdout
    assert "sentinel-api" not in result.stdout
    rendered = yaml.safe_load(result.stdout)
    prometheus = _bind_volume(rendered["services"]["prometheus"], "/etc/prometheus")
    grafana = _bind_volume(rendered["services"]["grafana"], "/etc/grafana/provisioning")
    assert prometheus.get("read_only") is True
    assert grafana.get("read_only") is True
    assert prometheus.get("bind", {}).get("selinux") == "z"
    assert grafana.get("bind", {}).get("selinux") == "z"
    proc = _bind_volume(rendered["services"]["node-exporter"], "/host/proc")
    assert proc.get("read_only") is True
    assert proc.get("bind", {}).get("selinux") in {None, ""}


def test_compose_config_fails_without_grafana_password():
    result = _compose_config(
        "config",
        extra_env={"GRAFANA_ADMIN_PASSWORD": None, "API_TOKEN": None},
    )
    assert result.returncode != 0
    assert "GRAFANA_ADMIN_PASSWORD" in result.stderr


def test_compose_config_with_api_profile_renders_without_default_token():
    result = _compose_config(
        "--profile",
        "with-api",
        "config",
        extra_env={
            "GRAFANA_ADMIN_PASSWORD": "test-only-grafana-password",
            "API_TOKEN": None,
        },
    )
    assert result.returncode == 0, result.stderr
    assert "sentinel-api" in result.stdout
    assert "test-only-grafana-password" in result.stdout
    rendered = yaml.safe_load(result.stdout)
    token = rendered["services"]["sentinel-api"]["environment"]["API_TOKEN"]
    assert token in {"", None}
    models = _bind_volume(rendered["services"]["sentinel-api"], "/app/models")
    assert models.get("read_only") is True
    assert models.get("bind", {}).get("selinux") == "z"


def test_kubernetes_uses_ready_and_host_models():
    documents = list(yaml.safe_load_all(K8S.read_text()))
    daemon = next(
        doc for doc in documents if doc and doc.get("kind") in {"DaemonSet", "Deployment"}
    )
    container = daemon["spec"]["template"]["spec"]["containers"][0]
    assert container["livenessProbe"]["httpGet"]["path"] == "/live"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["securityContext"]["runAsNonRoot"] is True
    assert "ALL" in container["securityContext"]["capabilities"]["drop"]
    volumes = {item["name"]: item for item in daemon["spec"]["template"]["spec"]["volumes"]}
    assert "hostPath" in volumes["models"]
    assert volumes["models"]["hostPath"]["path"] == "/opt/sentinel/models"
    config = next(doc for doc in documents if doc and doc.get("kind") == "ConfigMap")
    assert config["data"]["CLASS_NAMES"] == "object"
    assert not any(doc and doc.get("kind") == "PersistentVolumeClaim" for doc in documents)


def test_ansible_has_no_curl_pipe_and_fails_closed_on_image():
    text = ANSIBLE.read_text()
    assert "curl -fsSL https://get.docker.com | sh" not in text
    assert "ignore_errors: true" not in text
    assert "/ready" in text
    assert "model_loaded" in text
    assert "CLASS_NAMES" in text
    assert "{{ model_dir }}:/app/models:ro,z" in text
    assert '{{ model_dir }}:/app/models:ro"' not in text


def test_prometheus_rules_exist():
    assert ALERTS.is_file()
    alerts = yaml.safe_load(ALERTS.read_text())
    names = {rule["alert"] for group in alerts["groups"] for rule in group["rules"]}
    assert "SentinelTargetDown" in names
    assert "SentinelReadyFailed" in names
    assert "SentinelHighP95Latency" in names
    prom = PROM.read_text()
    assert "alert_rules.yml" in prom
    assert "sentinel-api:8080" in prom or "sentinel_edge" in prom


def test_ci_workflow_is_in_github_path():
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    deploy = ROOT / ".github" / "workflows" / "deploy.yml"
    assert workflow.is_file()
    text = workflow.read_text()
    assert "pytest tests/" in text
    assert "junitxml" in text
    assert "Dockerfile.edge" in text
    assert "pip-audit -r requirements-edge.txt" in text
    assert "pip-audit -r requirements-edge.txt -c" not in text
    assert "PIP_CONSTRAINT: constraints-edge.txt" in text
    deploy_text = deploy.read_text()
    assert "workflow_dispatch" in deploy_text
    assert "pull_request" not in deploy_text
    assert 'ANSIBLE_HOST_KEY_CHECKING: "False"' not in deploy_text
