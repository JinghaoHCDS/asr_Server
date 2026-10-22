# Qwen 质量评测

`quality_v1` 是候选流程，默认 60 秒分块未被证明最优。评分工具使用真实字符编辑距离，分别报告替换、删除、插入；原始识别与规范后文本分开评分。mock、缺少时长档位、缺少人工标注均不能通过发布门槛。

## 素材清单

复制 `corpus.sample.json` 到仓库外或忽略的 `data/`。实际音频、参考稿和评测输出不提交 Git。每条记录格式：

```json
{
  "id": "course-dev-001",
  "split": "dev",
  "audio": "audio/course.wav",
  "sha256": "这里填实际音频的64位SHA256",
  "reference": "reference/course.txt",
  "source": "本地自有素材或可核对的公开来源URL",
  "license": "明确许可或用户授权说明",
  "duration_seconds": 1800,
  "language": "auto",
  "full_reference": true,
  "terms": ["Qwen3-ASR"],
  "absent_terms": ["ABSENTTERMZXQ"],
  "approved_changes": [{"before": "千问 ASR", "after": "Qwen3-ASR"}],
  "boundary_regions": [{"start": 55, "end": 65, "reference_text": "该区间人工核对的全文"}],
  "paired_regions": [{"start": 120, "end": 150, "reference_text": "该区间人工核对的全文"}],
  "alignment_reference": [{"index": 0, "text": "你", "start": 0.8, "end": 0.96}]
}
```

`reference` 必须是真正人工核对文本，不能把一次模型输出当作参考真值。只有完整参考稿的素材参与全文 CER；长素材抽查区间使用独立裁剪素材条目，另保留完整源素材任务做 paired 对照。`alignment_reference.index` 指定模型输出单元顺序，防止评分时不断找最近同字掩盖错误。

开发集和验收集严格分离。必须覆盖口播、课程、访谈、播客、快速语音、中英混说、背景音乐、低能量、长沉默、真实重复、密集热词。长任务覆盖 30 分钟、2 小时、4 小时、6 小时，至少一条 >=5.5 小时完整人工参考。

公开素材需在清单中填写来源和许可；当前仓库没有提供已下载、已授权及完整标注的长音频集。已有 13 秒 fixture 仅可做 smoke，不能证明术语改善或长音频质量。

## 运行

```bash
uv run python scripts/evaluate_quality.py validate data/corpus.json --split dev
# 仅WSL真实CUDA环境运行；不改在线服务，不需另开监听端口
uv run python scripts/evaluate_quality.py run data/corpus.json --split dev --output data/eval-dev
uv run python scripts/evaluate_quality.py score data/corpus.json --split dev --output data/eval-dev
# 冻结参数后在独立验收集重新运行
uv run python scripts/evaluate_quality.py run data/corpus.json --split acceptance --output data/eval-acceptance
uv run python scripts/evaluate_quality.py score data/corpus.json --split acceptance --output data/eval-acceptance
```

`--profiles legacy legacy_budget4096 quality_60_0` 可缩小实验。完整候选矩阵覆盖 30/60/90/120/180 秒、0/1 秒上下文，并包含 `quality_60_0_no_review`、`quality_60_0_no_glossary` 消融对照。`--mock` 仅检查工程路径，报告标记 mock，永远不会选出通过验收的参数。score 未达到门槛返回 exit 2；输入无效返回 exit 1。报告写入 `report.json`，不会自动修改服务默认值。

字符归一规则固定为 Unicode NFKC、casefold、忽略空白与标点，不把繁简转换或热词替换掺进原始 CER。热词召回按出现次数计数；规范正确性使用人工批准的修改标签，不靠词库命中推定正确。字幕统计起止两端绝对误差 P95，缺失或字不匹配的人工标注单元也会导致 gate 不通过。

## 发布门槛

- 每时长档长任务对同源短片段 CER 增量 <=1 个百分点。
- 独立验收集原始 CER 不劣于 legacy，边界删除与插入分别报告。
- 已纳入词库目标词召回 >=95%，自动规范精度 >=99%，反例词诱导插入为零。
- 无未处理区间、静默截断或丢尾；所有未解决问题可定位。
- 字幕人工标注完整匹配，起止误差 P95 <=500 ms。
- 候选按 CER、术语错误、边界错误、处理耗时排序，仅从全部通过项中选择。

还需独立记录首块耗时、实时率、复核次数/音频预算、峰值显存与磁盘。质量报告不替代恢复、隔离、取消、配额、导出及真实浏览器验收。

## 公开短音频基线：AISHELL-1

[OpenSLR SLR33 官方页面](https://www.openslr.org/33/) 声明 AISHELL-1 为 Apache License 2.0，提供音频和人工转录。完整音频包约 15 GB，本次没有自动下载。它以干净环境的短句为主，仅作为通用普通话基线，不能替代真实长播客、术语、混说及字幕标注。

获得并解压该数据集（包括 `wav/dev`、`wav/test` 内部压缩包）后：

```bash
uv run python -m scripts.import_aishell /path/to/data_aishell --partition dev --limit 100 --output data/aishell-dev
uv run python -m scripts.import_aishell /path/to/data_aishell --partition test --limit 100 --output data/aishell-acceptance
uv run python -m scripts.evaluate_quality validate data/aishell-dev/corpus.json
```

导入器不复制大音频，只冻结来源、许可、绝对音频路径、SHA-256、时长和参考文本。dev 与 test 分别映射开发和独立验收集，不混用；目标领域仍应人工复核参考稿。没有长素材、目标术语及时间标注时，完整发布门槛应继续返回 `not_accepted`。
