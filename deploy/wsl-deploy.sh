#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEPLOY_DIR="${ASR_DEPLOY_DIR:-/home/fragt/services/asr-server}"
HOST="${ASR_HOST:-0.0.0.0}"
PORT="${ASR_PORT:-18080}"
ADAPTER="${ASR_ADAPTER:-qwen}"
QWEN_BATCH_SIZE="${ASR_QWEN_BATCH_SIZE:-1}"
IDLE_UNLOAD_SECONDS="${ASR_IDLE_UNLOAD_SECONDS:-180}"
HF_HUB_OFFLINE="${ASR_HF_HUB_OFFLINE:-1}"
TRANSFORMERS_OFFLINE="${ASR_TRANSFORMERS_OFFLINE:-1}"
SMOKE_MODEL_REPO="${ASR_SMOKE_MODEL_REPO:-Qwen/Qwen3-ASR-0.6B-hf}"
SMOKE_AUDIO="${ASR_SMOKE_AUDIO:-test-fixtures/audio/test_short.wav}"

SYNC_SOURCE=1
INSTALL_SYSTEM_PACKAGES=1
RUN_TESTS=1
RUN_MYPY=1
INSTALL_GPU_DEPS=1
RUN_CUDA_CHECK=1
INSTALL_SILERO=0
RUN_QWEN_BACKEND_SMOKE="${ASR_RUN_QWEN_BACKEND_SMOKE:-1}"
ENABLE_SERVICE=1
START_SERVICE=1
RUN_HTTP_SMOKE=1

usage() {
  cat <<'EOF'
Usage:
  deploy/wsl-deploy.sh [options]

Default behavior:
  - sync this repo to /home/fragt/services/asr-server
  - install Arch packages: uv, rsync, git, ffmpeg, libsndfile
  - create Python 3.12 uv environment
  - run pytest and mypy
  - install pinned CUDA 12.8 torch plus HF native Qwen runtime dependencies
  - assert torch is CUDA-enabled
  - run minimal Qwen3-ASR transformers backend smoke
  - install and start the systemd user service on 0.0.0.0:18080
  - run live HTTP smoke tests against 127.0.0.1:18080

Options:
  --deploy-dir DIR              Deployment directory. Default: /home/fragt/services/asr-server
  --host HOST                   Service host. Default: 0.0.0.0
  --port PORT                   Service port. Default: 18080
  --adapter NAME                ASR adapter for the service. Default: qwen
  --idle-unload-seconds SECONDS Unload an idle loaded model after this many seconds. Default: 180
  --mock                        Deploy mock adapter and skip GPU/Qwen checks
  --skip-sync                   Do not sync source tree to deployment directory
  --skip-system-packages        Do not install Arch system packages
  --skip-tests                  Do not run pytest
  --skip-mypy                   Do not run mypy
  --skip-gpu-deps               Do not install requirements/wsl-gpu-cu128.txt
  --skip-cuda-check             Do not run torch CUDA validation
  --install-silero              Install optional silero-vad package
  --skip-qwen-backend-smoke     Do not run scripts/qwen_asr_backend_smoke.py
  --smoke-model REPO            Qwen smoke model repo. Default: Qwen/Qwen3-ASR-0.6B-hf
  --smoke-audio PATH            Qwen smoke audio path. Default: test-fixtures/audio/test_short.wav
  --online-service              Allow the systemd service to contact Hugging Face while loading models
  --no-enable-service           Do not install or enable systemd user service
  --no-start                    Install service but do not start/restart it
  --skip-http-smoke             Do not run live HTTP smoke tests
  -h, --help                    Show this help
EOF
}

log() {
  printf '[asr-deploy] %s\n' "$*"
}

die() {
  printf '[asr-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-dir)
      DEPLOY_DIR="${2:?missing value for --deploy-dir}"
      shift 2
      ;;
    --host)
      HOST="${2:?missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?missing value for --port}"
      shift 2
      ;;
    --adapter)
      ADAPTER="${2:?missing value for --adapter}"
      shift 2
      ;;
    --idle-unload-seconds)
      IDLE_UNLOAD_SECONDS="${2:?missing value for --idle-unload-seconds}"
      shift 2
      ;;
    --mock)
      ADAPTER="mock"
      INSTALL_GPU_DEPS=0
      RUN_CUDA_CHECK=0
      RUN_QWEN_BACKEND_SMOKE=0
      shift
      ;;
    --skip-sync)
      SYNC_SOURCE=0
      shift
      ;;
    --skip-system-packages)
      INSTALL_SYSTEM_PACKAGES=0
      shift
      ;;
    --skip-tests)
      RUN_TESTS=0
      shift
      ;;
    --skip-mypy)
      RUN_MYPY=0
      shift
      ;;
    --skip-gpu-deps)
      INSTALL_GPU_DEPS=0
      shift
      ;;
    --skip-cuda-check)
      RUN_CUDA_CHECK=0
      shift
      ;;
    --install-silero)
      INSTALL_SILERO=1
      shift
      ;;
    --skip-qwen-backend-smoke)
      RUN_QWEN_BACKEND_SMOKE=0
      shift
      ;;
    --smoke-model)
      SMOKE_MODEL_REPO="${2:?missing value for --smoke-model}"
      shift 2
      ;;
    --smoke-audio)
      SMOKE_AUDIO="${2:?missing value for --smoke-audio}"
      shift 2
      ;;
    --online-service)
      HF_HUB_OFFLINE=0
      TRANSFORMERS_OFFLINE=0
      shift
      ;;
    --no-enable-service)
      ENABLE_SERVICE=0
      shift
      ;;
    --no-start)
      START_SERVICE=0
      shift
      ;;
    --skip-http-smoke)
      RUN_HTTP_SMOKE=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

case "${PORT}" in
  8765)
    die "port 8765 is a historical test port and must not be used"
    ;;
  18080)
    ;;
  *)
    log "warning: official deployment port is 18080; requested port is ${PORT}"
    ;;
esac

case "${DEPLOY_DIR}" in
  /mnt/*)
    die "deployment directory must be inside WSL Linux filesystem, not ${DEPLOY_DIR}"
    ;;
esac

if [[ -r /proc/version ]] && ! grep -qiE 'microsoft|wsl' /proc/version; then
  log "warning: this script is intended for WSL Arch Linux"
fi

install_system_packages() {
  if [[ "${INSTALL_SYSTEM_PACKAGES}" != "1" ]]; then
    return
  fi

  if command -v pacman >/dev/null 2>&1; then
    log "installing Arch system packages"
    sudo pacman -Syu --needed --noconfirm uv rsync git ffmpeg libsndfile
    return
  fi

  log "pacman not found; checking required commands"
  command -v uv >/dev/null 2>&1 || die "uv is required"
  command -v rsync >/dev/null 2>&1 || die "rsync is required when sync is enabled"
  command -v git >/dev/null 2>&1 || die "git is required to install Transformers main"
  command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is required"
}

sync_source_tree() {
  if [[ "${SYNC_SOURCE}" != "1" ]]; then
    log "skipping source sync"
    return
  fi

  mkdir -p "${DEPLOY_DIR}"

  local source_real
  local deploy_real
  source_real="$(cd -- "${SOURCE_DIR}" && pwd -P)"
  deploy_real="$(cd -- "${DEPLOY_DIR}" && pwd -P)"

  if [[ "${source_real}" == "${deploy_real}" ]]; then
    log "source already in deployment directory"
    return
  fi

  command -v rsync >/dev/null 2>&1 || die "rsync is required to sync ${SOURCE_DIR} to ${DEPLOY_DIR}"

  log "syncing ${SOURCE_DIR} to ${DEPLOY_DIR}"
  rsync -a \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude 'transcripts/' \
    --exclude 'uploads/' \
    --exclude 'models/' \
    --exclude '.cache/' \
    "${SOURCE_DIR}/" "${DEPLOY_DIR}/"
}

prepare_python_env() {
  cd -- "${DEPLOY_DIR}"

  log "installing Python 3.12 with uv if needed"
  uv python install 3.12

  if [[ "${INSTALL_GPU_DEPS}" == "1" ]]; then
    log "syncing locked Python dependencies"
    uv sync --frozen
  else
    log "syncing locked Python dependencies without pruning installed runtime packages"
    uv sync --frozen --inexact
  fi
}

run_static_checks() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${RUN_TESTS}" == "1" ]]; then
    log "running pytest"
    "${DEPLOY_DIR}/.venv/bin/pytest" -q
  fi

  if [[ "${RUN_MYPY}" == "1" ]]; then
    log "running mypy"
    "${DEPLOY_DIR}/.venv/bin/mypy" asr_server asr_mcp tests scripts
  fi
}

install_gpu_runtime() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${INSTALL_GPU_DEPS}" == "1" ]]; then
    log "checking NVIDIA visibility"
    nvidia-smi

    log "installing pinned CUDA 12.8 torch and HF native Qwen runtime dependencies"
    uv pip install --torch-backend cu128 -r requirements/wsl-gpu-cu128.txt
  fi

  if [[ "${INSTALL_SILERO}" == "1" ]]; then
    log "installing optional silero-vad"
    uv pip install silero-vad==6.2.1
  fi
}

run_cuda_check() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${RUN_CUDA_CHECK}" != "1" ]]; then
    return
  fi

  log "validating torch CUDA runtime"
  "${DEPLOY_DIR}/.venv/bin/python" - <<'PY'
import torch
import torchvision
import torchaudio
import transformers

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.__version__ == "2.11.0+cu128", f"unexpected torch version: {torch.__version__}"
assert torch.version.cuda == "12.8", f"unexpected torch CUDA version: {torch.version.cuda}"
assert torch.cuda.is_available(), "torch cannot see CUDA"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
assert hasattr(transformers, "AutoProcessor"), "transformers missing AutoProcessor"
assert hasattr(transformers, "AutoModelForMultimodalLM"), "transformers missing AutoModelForMultimodalLM"
print("HF native Qwen classes import ok")
PY
}

run_backend_smoke() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${RUN_QWEN_BACKEND_SMOKE}" != "1" ]]; then
    return
  fi

  [[ -f "${SMOKE_AUDIO}" ]] || die "smoke audio not found: ${SMOKE_AUDIO}"

  log "running Qwen3-ASR transformers backend smoke"
  "${DEPLOY_DIR}/.venv/bin/python" scripts/qwen_asr_backend_smoke.py \
    --backend transformers \
    --model "${SMOKE_MODEL_REPO}" \
    --audio "${SMOKE_AUDIO}" \
    --language auto
}

install_user_service() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${ENABLE_SERVICE}" != "1" ]]; then
    log "skipping systemd user service installation"
    return
  fi

  command -v systemctl >/dev/null 2>&1 || die "systemctl is required to install the user service"
  systemctl --user show-environment >/dev/null 2>&1 || die "systemd user manager is not available; enable WSL systemd or use --no-enable-service"

  local service_dir
  local service_file
  local tmp_file
  service_dir="${HOME}/.config/systemd/user"
  service_file="${service_dir}/asr-server.service"
  tmp_file="$(mktemp)"

  mkdir -p "${service_dir}"

  cat >"${tmp_file}" <<EOF
[Unit]
Description=WSL ASR Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${DEPLOY_DIR}
Environment=ASR_HOST=${HOST}
Environment=ASR_PORT=${PORT}
Environment=ASR_ADAPTER=${ADAPTER}
Environment=ASR_QWEN_BATCH_SIZE=${QWEN_BATCH_SIZE}
Environment=ASR_IDLE_UNLOAD_SECONDS=${IDLE_UNLOAD_SECONDS}
Environment=ASR_WORKER_STARTUP_TIMEOUT_SECONDS=30
Environment=ASR_WORKER_LOAD_TIMEOUT_SECONDS=900
Environment=ASR_WORKER_INFERENCE_TIMEOUT_SECONDS=3600
Environment=ASR_WORKER_SHUTDOWN_TIMEOUT_SECONDS=10
Environment=ASR_JOB_SHUTDOWN_GRACE_SECONDS=30
Environment=ASR_MIN_FREE_DISK_MB=5120
Environment=ASR_MAX_SPOOL_MB=12288
Environment=ASR_MAX_WORKSPACE_MB=1280
Environment=ASR_MAX_WORKSPACE_FILES=8
Environment=HF_HUB_OFFLINE=${HF_HUB_OFFLINE}
Environment=TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}
ExecStart=${DEPLOY_DIR}/.venv/bin/uvicorn asr_server.main:app --host ${HOST} --port ${PORT}
Restart=on-failure
RestartSec=5
TimeoutStopSec=60
KillMode=mixed

[Install]
WantedBy=default.target
EOF

  install -m 0644 "${tmp_file}" "${service_file}"
  rm -f "${tmp_file}"

  log "installed ${service_file}"
  systemctl --user daemon-reload
  systemctl --user enable asr-server.service

  if [[ "${START_SERVICE}" == "1" ]]; then
    log "starting asr-server.service"
    systemctl --user restart asr-server.service
  fi
}

run_http_smoke() {
  cd -- "${DEPLOY_DIR}"

  if [[ "${RUN_HTTP_SMOKE}" != "1" ]]; then
    return
  fi

  if [[ "${START_SERVICE}" != "1" || "${ENABLE_SERVICE}" != "1" ]]; then
    log "skipping HTTP smoke because the service was not started by this script"
    return
  fi

  local base_url="http://127.0.0.1:${PORT}"

  log "waiting for ${base_url}/health"
  for _ in $(seq 1 120); do
    if curl --noproxy '*' -fsS "${base_url}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  curl --noproxy '*' -fsS "${base_url}/health" >/dev/null

  log "running live HTTP smoke tests"
  ASR_BASE_URL="${base_url}" "${DEPLOY_DIR}/.venv/bin/pytest" tests/test_http_smoke.py -q
}

install_system_packages
sync_source_tree
prepare_python_env
run_static_checks
install_gpu_runtime
run_cuda_check
run_backend_smoke
install_user_service
run_http_smoke

log "deployment complete"
log "local health: curl --noproxy '*' http://127.0.0.1:${PORT}/health"
log "LAN health:   curl --noproxy '*' http://192.168.31.137:${PORT}/health"
