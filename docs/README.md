# 文档索引

当前项目文档只保留运行、维护和验收需要的稳定入口。已经完成的实施计划不再放在 `docs/` 顶层；需要追溯时使用 Git 历史。

## 必读入口

- `asr-server-prd.md`：核心 API 合约、生命周期语义、错误信封和安全边界。
- `wsl-deployment.md`：WSL Arch Linux 部署步骤、GPU 依赖、MOSS gate、systemd 和 smoke test。
- `docs-endpoint-capabilities.md`：`/docs` Swagger UI 的说明文本来源，以及当前 API 能力声明。
- `mcp-bridge.md`：Mac 侧 stdio MCP bridge 的配置、安全边界和验收方法。
- `batch-workflow.md`：口播、播客、关键词、批量恢复和本地文件夹交接的可执行流程。
- `validation-template.md`：每次 WSL 真实模型验收的记录模板。

## 当前验收记录

- `validation-2026-07-02-wsl-hf-native.md`：Qwen3-ASR 0.6B/1.7B 的 HF native `transformers` 后端验收。
- `validation-2026-07-10-wsl-moss.md`：MOSS-Transcribe-Diarize 独立 smoke、HTTP 路径和卸载验收。
- `validation-2026-07-15-wsl-moss-long-form.md`：MOSS 10/30/60/90 分钟原生推理边界、缺尾保护与自动降级验收。
- `validation-2026-07-15-wsl-moss-anchor-replay.md`：MOSS Anchor Replay 四说话人跨块解析与对抗性验收。
- `validation-2026-09-19-upgrade.md`：细粒度对齐、Context、批量交付及本轮长播客限制的分级验收记录。
- `review-2026-07-10-stability-refactor.md`：单 GPU、Worker 可靠性、Path 音频管线、MOSS 正确性与 Job 生命周期的对抗性审查和重构记录。

## 维护规则

- 运行时模型和能力以 `GET /v1/models` 为准；客户端不要从 README、PRD 或 `/docs` 硬编码能力。
- 新增或关闭模型时，同步更新 `README.md`、`wsl-deployment.md`、`docs-endpoint-capabilities.md` 和相关提示词。
- 真实 WSL 验收通过后写入新的 `docs/validation-YYYY-MM-DD-*.md`，不要覆盖旧记录。
- 已执行完成的迁移计划、部署计划和临时调研文档应删除或归档，不继续作为交接入口。
