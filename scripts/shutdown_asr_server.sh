#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="${ASR_SERVICE_NAME:-asr-server.service}"
PORT="${ASR_PORT:-18080}"
BASE_URL="${ASR_BASE_URL:-http://127.0.0.1:${PORT}}"
WAIT_SECONDS="${ASR_SHUTDOWN_WAIT_SECONDS:-90}"

log() {
  printf '[asr-shutdown] %s\n' "$*"
}

die() {
  printf '[asr-shutdown] ERROR: %s\n' "$*" >&2
  exit 1
}

command -v systemctl >/dev/null 2>&1 || die "systemctl is required; run this inside WSL with systemd enabled"
command -v curl >/dev/null 2>&1 || die "curl is required"
[[ "${PORT}" == "18080" ]] || die "the only supported LAN API port is 18080"
[[ "${WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || die "ASR_SHUTDOWN_WAIT_SECONDS must be a positive integer"
systemctl --user show-environment >/dev/null 2>&1 || die "systemd --user is not available"
systemctl --user cat "${SERVICE_NAME}" >/dev/null 2>&1 || die "${SERVICE_NAME} is not installed"

if systemctl --user is-active --quiet "${SERVICE_NAME}"; then
  log "stopping ${SERVICE_NAME}"
  systemctl --user stop "${SERVICE_NAME}"
else
  log "${SERVICE_NAME} is already stopped"
fi

log "waiting for ${SERVICE_NAME} to stop"
deadline=$((SECONDS + WAIT_SECONDS))
while (( SECONDS < deadline )); do
  if ! systemctl --user is-active --quiet "${SERVICE_NAME}"; then
    if ! curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
      log "service is stopped"
      exit 0
    fi
  fi
  sleep 1
done

systemctl --user status "${SERVICE_NAME}" --no-pager || true
die "service did not stop within ${WAIT_SECONDS}s"
