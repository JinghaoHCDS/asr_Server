# WSL ASR 验收记录模板

部署路径和 LAN IP 由用户环境决定；示例 `YOUR_WINDOWS_LAN_IP` 与 `/path/to/` 必须替换。历史 Mac / Arch 验收不代表其他平台已实测。

## 环境

- 日期：
- WSL 发行版：
- 项目路径：`~/services/asr-server`
- Git commit：
- Python 版本：
- uv 版本：
- torch 版本：
- torch CUDA 版本：
- GPU 名称：
- GPU capability：
- `nvidia-smi` 摘要：

## 后端预验收

### Qwen transformers

- 命令：
- 模型 ID：
- 音频文件：
- 输出语言：
- 文本前 200 字：
- 是否通过：
- 失败日志：

### MOSS transformers

- 是否启用 `ASR_ENABLE_MOSS`：
- 命令：
- 模型 ID：`OpenMOSS-Team/MOSS-Transcribe-Diarize`
- 服务模型 ID：`moss-transcribe-diarize-0.9b`
- MOSS package 版本或 commit：
- Hugging Face snapshot：
- 音频文件：
- 输出语言：
- 文本前 200 字：
- parsed segments 数量：
- 第一条 segment：
- 安装 MOSS 依赖后 CUDA torch 是否仍可用：
- 是否通过：
- 失败日志：

## 服务端 API 验收

- 服务地址：
- `/health` 摘要：
- `/v1/models` 模型和后端列表：
- 默认模型：
- `ASR_ENABLE_MOSS=0` 或未设置时 MOSS 是否不在 `/v1/models`：
- `ASR_ENABLE_MOSS=1` 时 MOSS 是否出现在 `/v1/models`：
- `qwen3-asr-0.6b` + `transformers` 文本前 200 字：
- `qwen3-asr-1.7b` + `transformers` 文本前 200 字：
- `moss-transcribe-diarize-0.9b` + `transformers` 文本前 200 字：
- MOSS `verbose_json` segments 摘要：
- `backend=vllm` 未声明后端错误码：
- `timestamps=word` 未声明能力错误码：
- 卸载活跃请求时的状态：
- `unloading_scheduled` 新请求错误码：

## 转录耗时记录

- 是否返回 `timings.total_ms`：
- 是否返回 `timings.load_ms`：
- 是否返回 `timings.inference_ms`：
- 短音频总耗时：
- 长音频总耗时：
- 是否避免记录完整音频内容：
- 同步请求结束后是否清理临时音频文件：
- 转录结束 3 分钟无新请求后是否自动卸载模型：
- 新请求是否会重置空闲卸载计时：

## 长音频切分与合并

- 使用音频文件：`/path/to/your-long-audio.mp3`
- 切分策略：
- VAD backend：
- chunk 数量：
- overlap 秒数：
- 平均 chunk 时长：
- 最长 chunk 时长：
- 空白/低质量 chunk 数：
- 是否返回 chunk 元数据：
- 合并文本前 200 字：
- 是否通过：

## 异步 job 与进度验收

- `POST /v1/audio/transcription-jobs` 是否返回 `202`：
- 返回的 `job_id`：
- 返回的 `status_url`：
- 短音频 job 模型 ID：
- 短音频 job backend：
- 短音频 job 最终状态：
- 短音频 job 文本前 200 字：
- 长音频 job 使用音频：`/path/to/your-long-audio.mp3`
- 长音频 job 模型 ID：
- 长音频 job backend：
- 长音频 job 状态流转时间线：
- queued/preprocessing/splitting 阶段轮询样例 JSON：
- transcribing 阶段轮询样例 JSON：
- completed 阶段轮询样例 JSON：
- `progress.total_chunks`：
- `progress.completed_chunks` 是否真实增长：
- `progress.current_chunk` 是否真实变化：
- `queue_position` 验收结果：
- 同时提交两个 job 时是否只运行一个：
- 第二个 job 是否显示 queued：
- job 失败错误对象是否使用统一 `code/message/details`：
- queued job 取消结果：
- running job 取消是否等待当前 chunk 结束：
- job 完成/失败/取消后临时文件是否清理：
- job result TTL 配置：
- job TTL 是否只保留内存结果、不保留音频文件：
- 进程重启后内存 job 丢失行为是否记录：

## Silero VAD 对比

- energy VAD chunk 数：
- energy VAD 总转录耗时：
- Silero VAD chunk 数：
- Silero VAD 总转录耗时：
- Silero 是否 fallback：
- Silero fallback 原因：
- 观察到的切断句/重复文本问题：
- 结论：

## context / 热词验收

- 测试音频：
- context 内容摘要：
- context 字符数：
- 易错专有名词清单：
- 无 context 命中数：
- 有 context 命中数：
- 是否出现 context 幻觉插词：
- 结论：

## max_new_tokens 验收

- 测试音频：
- Qwen 默认 `max_new_tokens`：
- MOSS 默认 `max_new_tokens`：
- 对照 `max_new_tokens`：
- 默认输出是否截断：
- 对照输出是否截断：
- 推理耗时变化：
- 峰值显存变化：
- 结论：

## Qwen batch transcription 验收

- 测试音频：`/path/to/your-long-audio.mp3`
- batch size 1 总耗时：
- batch size 2 总耗时：
- batch size 4 总耗时：
- 0.6B 稳定 batch size：
- 1.7B 稳定 batch size：
- batch fallback 次数：
- CUDA OOM 或异常日志：
- 推荐配置：

## MOSS speaker segments 验收

- 测试音频：
- 是否使用 `response_format=verbose_json`：
- `segments[].speaker` 是否存在：
- `segments[].start/end` 是否存在：
- `response_format=json` 是否不默认展开 segments：
- `timestamps=word` 是否返回 `capability_not_supported`：
- `backend=vllm` 是否返回 `capability_not_supported`：
- 长音频切分后是否出现 `moss_speaker_labels_are_chunk_local` warning：
- 跨 chunk speaker label 是否未被误写成全局身份：
- 卸载后 GPU 显存是否释放：

## Mac 局域网验收

- Mac 请求是否绕过代理：
- `curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/health` 结果：
- `curl --noproxy '*' http://YOUR_WINDOWS_LAN_IP:18080/v1/models` 结果：
- Mac 上传真实音频转录摘要：
- Mac MOSS verbose_json 摘要：
- Mac 创建 job 请求摘要：
- Mac 轮询 job 到 completed 的状态摘要：
- Mac 侧观察到的 chunk 进度变化：

## 结论

- 通过/不通过：
- 阻塞点：
- 后续动作：
