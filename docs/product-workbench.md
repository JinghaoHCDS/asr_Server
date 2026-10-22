# 局域网工作台与 Qwen quality_v1

批准日期：2026-10-22。本文是 `docs/asr-server-prd.md` 的产品化扩展，覆盖旧版“不做 Web UI / 不引入数据库”的阶段性边界。

## 产品与运行边界

默认 Qwen 1.7B、transformers、官方 ForcedAligner；现有 MOSS 保持原策略。网页在 API 的同一 `18080` 端口提供上传音视频、排队、个人任务、回听改字及 TXT/SRT/VTT/JSON 导出，不依赖公网或云服务。前端为 React/TypeScript/Vite 静态产物，生产机器无需运行 Node 服务。

面向中文口播、课程、访谈、播客和中英混说。最大六小时是输入上限，不代表六小时识别质量已验收。`quality_v1` 初始 60 秒是实验候选；真实开发集、独立验收集通过发布门槛前，旧调用默认保持 `legacy`。

网页 `/api/**` 以个人用户名密码和 HttpOnly Cookie 认证；管理员创建、停用账户及重置密码。普通用户只能访问自己的任务、音频、上传和版本，服务端逐次检查归属。原 `/v1/**` 的可选 Bearer Token 不赋予个人工作台数据访问权；CLI/MCP 保持原行为。

## 质量流程

1. 统一原音格式并保留绝对时间轴，视频只保留提取出的可回听音轨。
2. 完整覆盖切分，在目标附近寻找低能量切点，不因检测到静音就丢弃输入。候选目标 30/60/90/120/180 秒，比较零重叠和边界两侧上下文，单次输入不超过 180 秒。
3. 每块使用任务冻结的词库快照；默认不传播上一块全文。自动生成预算 4096，EOS 提前结束；显式限制保持原值。
4. 截断、疑似循环、空白语音区、边界冲突及疑似术语问题有界复核：最多两轮，总复核音频预算不超过首次输入时长的 30%。未完成区间不得静默变成 completed。
5. 使用真实字词对齐确定边界归属；不以字符串模糊匹配直接删掉真实重复。真实 timestamps 来源始终可见，mock 不作为时间精度证据。
6. 返回模型原稿、规范稿、改词记录和待复核区间。对齐成功不等于文字正确，不提供未经校准的准确率百分比。

`quality_v1.2` 修复空稿区间判断：每个区间及复核子区间仍先运行 ASR，纯零 PCM 可以交付空文本；只在源音频尾部，且 PCM16 峰值不超过 4、非零采样比例不超过 5%、末尾全零至少持续 1 秒及该区间的一半时，允许把空结果作为近数字静音尾段交付。后者必须保留 `near_silent_tail` 待复核标记及采样证据。常量非零低音量、更强瞬时信号和非尾部输入不通过该豁免；这不是通用语音检测，也不据此跳过低能量音频。

流程版本参与检查点指纹。升级后不得拼接旧流程结果；旧检查点版本不匹配时明确拒绝恢复，重新运行需保留原失败记录并从原音开始。

公共词库分通用和 AI/技术领域。任务本次热词优先于选定领域，再优先于通用词库。词条包含 `canonical / aliases / domain / auto_normalize`；只有已人工确认且无歧义的别名允许自动规范，疑似同音词通过回听与复核处理。不存在于正文的热词不补入。词库版本更新不改变正在执行或已完成任务。

## 持久化、恢复与容量

工作台使用 SQLite 和本地数据目录，保存账号、会话、上传、队列、检查点、词库快照和结果版本。旧 API 内存 job 不变，但两者共用一个 FIFO worker 和模型管理器，GPU 仍串行。

上传块大小 8 MiB，服务端顺序确认 offset；断线后查询已接收 offset，用户重选同一文件后续传。提交使用幂等键，重复提交不产生重复任务。浏览器关闭后任务继续。

每个有效分块原子保存检查点，包含原音指纹、模型 revision、流程版本、词库和参数。重启恢复队列顺序；未结束模型调用重做，已完成分块可复用。版本不一致时拒绝混用，明确要求重新运行。

默认容量：单个上传 8 GiB、每用户保留区 16 GiB、全局 64 GiB、磁盘余量 5 GiB、最长六小时。限制均可配置。到达容量限制拒绝新上传，不删除未到期任务来凑空间。已完成/失败/取消任务在终态七天后清理，可提前删除。未提交上传也需要过期回收，避免永久占用配额。

保留数据与推理临时目录分离；临时分片和预处理副本用后删除。主动取消等待模型安全边界，不强杀当前推理。删除包括音频、检查点、规范稿、原稿和全部编辑版本，不能让晚到的模型结果复活已删除任务。

文本按片段修改，携带 `base_version` 做乐观锁。只重新对齐受影响片段，新版本对齐未完成时禁止导出新 SRT/VTT；可回到已完成的旧版本导出。编辑不隐式改变公共词库。

## API 扩展

- `/v1/audio/transcriptions`、`/v1/audio/transcription-jobs` 增加可选 `quality_profile=legacy|quality_v1`，默认 legacy；MOSS 不接受 Qwen 专属 profile。
- `/api/auth/me|login|logout`：个人会话；`/api/capabilities`：启用模型、质量流程、领域、输入及保留限制。
- `/api/uploads`：创建；`GET/PATCH /api/uploads/{id}`：查询/顺序续传，PATCH 用 `Upload-Offset`；`POST /api/uploads/{id}/complete`：冻结参数并提交任务。
- `/api/jobs`：个人列表；`/api/jobs/{id}`：查询/删除；`POST .../cancel|retry`：取消/重试；`GET .../audio`：带归属校验的 Range 音频。
- `PATCH /api/jobs/{id}/segments/{segment_id}`：带版本的改字；`GET .../export?format=txt|srt|vtt|json&version=N`：指定版本导出。
- `/api/admin/users`、`/api/admin/users/{id}`：账户；`/api/admin/glossary`：词库；`/api/admin/status`：队列与容量。

错误沿用现有 error envelope。前端不显示服务器文件路径、模型内部状态细节或虚构模型内部进度。实际模型能力仍从 registry 的 `/v1/models` 同源发现。

## 开发、运行与管理员

```bash
uv sync
cd frontend
npm ci
npm run build
cd ..
uv run python -m scripts.workbench_admin --help
uv run uvicorn asr_server.main:app --host 0.0.0.0 --port 18080
```

管理员使用本地 CLI 创建，密码通过交互或 stdin 输入，不预置默认密码、不打印会话 Token。CLI 与服务必须使用相同的 `ASR_WORKBENCH_DATA_DIR`。真实环境保持现有 WSL CUDA 安装流程；Mac 不安装 GPU 模型依赖。

新增环境配置：

| 设置 | 默认 |
| --- | --- |
| `ASR_WORKBENCH_ENABLED` | `1` |
| `ASR_WORKBENCH_DATA_DIR` | `data/workbench` |
| `ASR_WORKBENCH_RETENTION_DAYS` | `7` |
| `ASR_WORKBENCH_MAX_UPLOAD_BYTES` | `8589934592` |
| `ASR_WORKBENCH_USER_QUOTA_BYTES` | `17179869184` |
| `ASR_WORKBENCH_TOTAL_QUOTA_BYTES` | `68719476736` |

`ASR_MIN_FREE_DISK_MB` 同时保护临时数据与保留数据。`.env` 仅由部署生成的 systemd unit 自动读取，直接 Python 和管理员 CLI 需显式设置环境。

## 发布与回滚

先运行 mock 回归、类型检查、前端构建、真实浏览器流程，再在 WSL 独立代码目录和数据目录完成 CUDA、Qwen backend smoke、1.7B+ForcedAligner及完整工作台验收。线上服务不在测试过程中被替换。

切换前核对线上与本地文件差异、等待活跃任务结束，停止写入后使用 SQLite backup API 生成一致备份，保存旧源码和依赖记录。回滚必须同时恢复匹配版本的代码、静态产物和数据库；不能让旧代码读取未知新 schema。新流程失败时旧 API 默认值仍保持 legacy。真实质量门槛见 [评测说明](../evaluation/README.md)。

### 发布包与离线数据备份命令

```bash
# 必须先完成frontend构建；产物含逐文件SHA-256清单，不含数据/secret/node_modules
uv run python -m scripts.workbench_release bundle --output /path/to/new-release.tar.gz
# 停止使用该data-dir的服务后备份；若仍持有service.lock则拒绝操作
uv run python -m scripts.workbench_release backup --data-dir /path/to/data/workbench --output /path/to/new-backup
# 恢复到新目录或空目录；逐文件校验、SQLite integrity_check，并报告源码匹配状态
uv run python -m scripts.workbench_release restore --backup /path/to/new-backup --data-dir /path/to/restored-workbench
```

备份包含账号密码哈希、保留音频与结果，应保存在 Git 之外。首次恢复需核对 `required_source.source_digest` 与配套源码包；它包含未提交改动，不能只看 Git commit。工具拒绝覆盖既有备份、不自动重启服务或切换线上目录。
