# WSL ASR 服务

这个仓库实现一个局域网 ASR 网关。服务运行在 Windows PC 的 WSL Arch Linux 内，Mac mini 项目通过 HTTP 调用 WSL 里的 GPU ASR 模型。

运行时模型与能力发现以 `GET /v1/models` 为准；`GET /docs` 只是 Swagger UI 和当前实现说明，不作为客户端硬编码依据。

## 目录结构

- `docs/README.md`：文档索引和维护规则。
- `docs/asr-server-prd.md`：核心 API 合约和产品边界。
- `docs/wsl-deployment.md`：WSL Arch Linux 部署、GPU 依赖、systemd 和验收命令。
- `docs/docs-endpoint-capabilities.md`：`/docs` Swagger UI 使用的当前实现能力说明。
- `docs/validation-2026-07-02-wsl-hf-native.md`：Qwen3-ASR HF native WSL 验收记录。
- `docs/validation-2026-07-10-wsl-moss.md`：MOSS-Transcribe-Diarize WSL 验收记录。
- `docs/validation-2026-07-15-wsl-moss-long-form.md`：MOSS 10/30/60/90 分钟原生与自动降级对抗性验收。
- `docs/validation-2026-09-19-upgrade.md`：细粒度对齐、Context、批量交接与长播客验收边界。
- `docs/batch-workflow.md`：单条用途配置、批量恢复、重试和结果落盘命令。
- `prompts/`：给 WSL 服务端代理和 Mac 请求端代理的交接提示词。
- `asr_server/`：FastAPI 应用、模型注册表、生命周期管理器、异步 job 队列和 ASR 适配器。
- `tests/`：不依赖 CUDA 的 API、生命周期、临时文件清理、MOSS gate 和 job 行为测试。
- `scripts/asr_client.py`：Mac 侧验证客户端，会绕过本机代理设置。
- `asr_mcp/`：Mac 侧 stdio MCP bridge，通过现有 HTTP API 调用 WSL ASR Gateway。
- `docs/mcp-bridge.md`：MCP Host 配置、安全限制和验收说明。
- `scripts/qwen_asr_backend_smoke.py`：WSL 侧 Qwen3-ASR 后端最小验收脚本。
- `scripts/moss_backend_smoke.py`：WSL 侧 MOSS 后端最小验收脚本。
- `scripts/wsl_smoke.sh`：WSL 侧启动服务并运行 HTTP smoke test 的脚本。
- `deploy/`：WSL 一键部署脚本、systemd user service 和 Windows 启动脚本模板。
- `test-fixtures/audio/`：给 WSL 侧 ASR 自测使用的音频样本。

## 部署目标

正式部署目录：

```text
/home/fragt/services/asr-server
```

局域网公开 API 入口：

```text
http://192.168.31.137:18080
```

Mac 侧请求这个局域网入口时必须绕过本机代理：

```bash
curl --noproxy '*' http://192.168.31.137:18080/health
```

## 当前状态

当前代码已经实现 FastAPI 网关、Qwen3-ASR HF native 适配器、按需 Qwen3 ForcedAligner、可选 MOSS 与 Anchor Replay、真实 Context prompt、用途配置、异步 FIFO job、精简轮询、长音频切分/覆盖检查，以及 Mac 侧批量恢复与原子落盘。

默认模型仍是 `qwen3-asr-1.7b`。只有设置 `ASR_ENABLE_QWEN_ALIGNMENT=1` 才在 `/v1/models` 声明 Qwen timestamps/forced alignment；只有完成 MOSS smoke 并设置 `ASR_ENABLE_MOSS=1` 才声明播客模型。MOSS 超过 1801 秒会自动分块，`podcast` 配置启用 Anchor Replay；客户端必须检查 `diarization.status` 与 `speaker_scope`，不能把 mixed/chunk-local 标签冒充整期全局身份。

转录过程中产生的上传文件、解码文件和模型临时音频文件只用于当次请求。同步请求在返回前清理临时文件；异步 job 在完成、失败或取消后清理 job 工作目录，只在内存里按 `ASR_JOB_RESULT_TTL_SECONDS` 保留 job 结果。

模型默认 lazy load。转录完成后如果 `ASR_IDLE_UNLOAD_SECONDS` 秒内没有新的同模型转录请求，服务会自动卸载该模型并释放 CUDA cache；默认值为 `180` 秒。

真实 Qwen、MOSS、CUDA 验证和常驻部署只在 WSL Arch Linux 侧完成，Mac mini 只作为轻量客户端和验收机。

## 本地开发

Python 和依赖以 uv 为准：

```bash
uv sync
uv run pytest -q
uv run mypy asr_server asr_mcp tests scripts
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

Mac 侧需要使用 MCP 时安装可选依赖：

```bash
uv sync --extra mcp
ASR_MCP_ALLOWED_ROOTS=/path/to/audio \
  uv run --extra mcp python -m asr_mcp.server
```

MCP bridge 使用 stdio，不监听新的网络端口；它通过 `ASR_MCP_BASE_URL` 调用现有 `18080` HTTP API。完整配置见 `docs/mcp-bridge.md`。

WSL 真实部署优先使用一键脚本：

```bash
deploy/wsl-deploy.sh
```

常驻 systemd 服务和 Windows 启动任务使用部署目录里的 `.venv/bin/uvicorn` 直接启动，不使用 `uv run uvicorn`，避免服务启动时因依赖同步访问 PyPI 镜像而卡住。

Python 通过 `.python-version` 和 `pyproject.toml` 固定为 3.12；具体 Python 包版本锁定在 `uv.lock`。不要在 Mac mini 上安装 CUDA、torch GPU 包、Qwen/MOSS 模型包或模型缓存。

## Mac 侧验证客户端

辅助客户端使用 `httpx.Client(trust_env=False)` 禁用环境代理。

```bash
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 check
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 transcribe /path/to/audio.wav --model qwen3-asr-1.7b --backend auto
uv run python scripts/asr_client.py --base-url http://192.168.31.137:18080 transcribe /path/to/audio.wav --context "Qwen3-ASR, Silero VAD, Hugging Face" --max-new-tokens 512
```

MOSS 已启用时，可以用同步接口请求 `verbose_json` 查看段级说话人结果：

```bash
curl --noproxy '*' -sS \
  -F file=@/path/to/audio.wav \
  -F model=moss-transcribe-diarize-0.9b \
  -F response_format=verbose_json \
  http://192.168.31.137:18080/v1/audio/transcriptions
```

接口使用原则：

- 短音频或脚本验证可以继续使用 `POST /v1/audio/transcriptions` 同步接口。
- 长音频或需要前端进度展示时使用 `POST /v1/audio/transcription-jobs`。
- job 进度只承诺服务端阶段和 chunk 级真实进度，不承诺单个模型 chunk 内部百分比。
- 客户端必须读取 `/v1/models` 后再决定模型、后端和能力，不要硬编码 MOSS、timestamps、forced alignment 或 streaming。

口播、长播客、带关键词请求、批量清单、恢复/重试和输出目录的可执行命令见 [docs/batch-workflow.md](docs/batch-workflow.md)。
