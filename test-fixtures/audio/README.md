# Audio Test Fixtures

These user-provided audio files are stored for WSL-side ASR validation.

| File | Purpose | SHA-256 |
| --- | --- | --- |
| `test_short.wav` | Short transcription smoke test | `9121c5126488b541dc75aaced93ffe28a8e80207c87aa040a74e10ef1cec73d2` |

WSL validation example:

```bash
uv run python scripts/asr_client.py \
  --base-url http://127.0.0.1:18080 \
  transcribe test-fixtures/audio/test_short.wav \
  --model qwen3-asr-1.7b \
  --backend auto
```


Long-media regressions generate a synthetic MP3 in memory. For real WSL long-form validation, supply your own permitted audio outside the repository; do not commit podcast downloads or private recordings.
