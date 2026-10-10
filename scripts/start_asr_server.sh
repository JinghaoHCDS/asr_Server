#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${ASR_SERVICE_NAME:-asr-server.service}"
PORT="${ASR_PORT:-18080}"
BASE_URL="${ASR_BASE_URL:-http://127.0.0.1:${PORT}}"
WAIT_SECONDS="${ASR_START_WAIT_SECONDS:-90}"
LAN_BASE_URL="${ASR_PUBLIC_BASE_URL:-http://<WINDOWS_LAN_IP>:${PORT}}"

log() {
  printf '[asr-start] %s\n' "$*"
}

die() {
  printf '[asr-start] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v systemctl >/dev/null 2>&1 || die "systemctl is required; run this inside WSL with systemd enabled"
command -v curl >/dev/null 2>&1 || die "curl is required"
[[ "${PORT}" == "18080" ]] || die "the only supported LAN API port is 18080"
[[ "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || die "ASR_START_WAIT_SECONDS must be a positive integer"
systemctl --user show-environment >/dev/null 2>&1 || die "systemd --user is not available"
systemctl --user cat "${SERVICE_NAME}" >/dev/null 2>&1 || die "${SERVICE_NAME} is not installed; run deploy/wsl-deploy.sh first"

if systemctl --user is-active --quiet "${SERVICE_NAME}"; then
  log "${SERVICE_NAME} is already running"
else
  log "starting ${SERVICE_NAME}"
  systemctl --user start "${SERVICE_NAME}"
fi

log "waiting for ${BASE_URL}/health"
deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    log "service is healthy"
    curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${BASE_URL}/health"
    printf '\n'
    log "local: ${BASE_URL}"
    log "LAN:   ${LAN_BASE_URL}"
    exit 0
  fi
  sleep 1
done

systemctl --user status "${SERVICE_NAME}" --no-pager || true
die "service did not become healthy within ${WAIT_SECONDS}s"
