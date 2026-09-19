from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from asr_server.execution import MOSS_EXECUTION_POLICY, QWEN_EXECUTION_POLICY, ModelExecutionPolicy

Backend = Literal["auto", "transformers", "vllm"]
ModelStatus = Literal["unloaded", "loading", "loaded", "unloading_scheduled", "unloading", "error"]
MOSS_MODEL_ID = "moss-transcribe-diarize-0.9b"


QWEN_LANGUAGES = [
    "auto",
    "zh",
    "en",
    "yue",
    "ar",
    "de",
    "fr",
    "es",
    "pt",
    "id",
    "it",
    "ko",
    "ru",
    "th",
    "vi",
    "ja",
    "tr",
    "hi",
    "ms",
    "nl",
    "sv",
    "da",
    "fi",
    "pl",
    "cs",
    "fil",
    "fa",
    "el",
    "hu",
    "mk",
    "ro",
]

QWEN_CHINESE_DIALECTS = [
    "Anhui",
    "Dongbei",
    "Fujian",
    "Gansu",
    "Guizhou",
    "Hebei",
    "Henan",
    "Hubei",
    "Hunan",
    "Jiangxi",
    "Ningxia",
    "Shandong",
    "Shaanxi",
    "Shanxi",
    "Sichuan",
    "Tianjin",
    "Yunnan",
    "Zhejiang",
    "Cantonese-Hong-Kong-accent",
    "Cantonese-Guangdong-accent",
    "Wu",
    "Minnan",
]


@dataclass(frozen=True)
class ModelCapabilities:
    transcription: bool
    streaming: bool
    timestamps: list[str]
    forced_alignment: bool
    languages: list[str]
    chinese_dialects: list[str]
    backends: list[str]
    diarization: bool = False
    segment_timestamps: bool = False
    execution_modes: list[str] = field(default_factory=lambda: ["chunked"])
    auto_execution_mode: str = "chunked"
    max_new_tokens: int = 4_096
    speaker_scopes: list[str] = field(default_factory=list)
    validated_native_max_seconds: float | None = None
    automatic_fallback_chunk_seconds: float | None = None
    speaker_resolution_modes: list[str] = field(default_factory=lambda: ["off"])
    speaker_resolution_method: str | None = None
    validated_anchor_replay_speakers: int | None = None

    def to_api(self) -> dict[str, object]:
        return {
            "transcription": self.transcription,
            "streaming": self.streaming,
            "timestamps": self.timestamps,
            "forced_alignment": self.forced_alignment,
            "languages": self.languages,
            "chinese_dialects": self.chinese_dialects,
            "backends": self.backends,
            "diarization": self.diarization,
            "segment_timestamps": self.segment_timestamps,
            "execution_modes": self.execution_modes,
            "auto_execution_mode": self.auto_execution_mode,
            "max_new_tokens": self.max_new_tokens,
            "speaker_scopes": self.speaker_scopes,
            "validated_native_max_seconds": self.validated_native_max_seconds,
            "automatic_fallback_chunk_seconds": self.automatic_fallback_chunk_seconds,
            "speaker_resolution_modes": self.speaker_resolution_modes,
            "speaker_resolution_method": self.speaker_resolution_method,
            "validated_anchor_replay_speakers": self.validated_anchor_replay_speakers,
            "context_prompt": True,
        }


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    provider: str
    default: bool
    capabilities: ModelCapabilities
    revision: str
    execution_policy: ModelExecutionPolicy


def default_models(
    default_model_id: str = "qwen3-asr-1.7b",
    *,
    enable_moss: bool = False,
    enable_qwen_alignment: bool = False,
) -> dict[str, ModelDefinition]:
    if default_model_id not in {"qwen3-asr-1.7b", "qwen3-asr-0.6b"}:
        raise ValueError("ASR_DEFAULT_MODEL must be qwen3-asr-1.7b or qwen3-asr-0.6b")
    qwen_capabilities = ModelCapabilities(
        transcription=True,
        streaming=False,
        timestamps=["auto", "word", "char"] if enable_qwen_alignment else [],
        forced_alignment=enable_qwen_alignment,
        languages=QWEN_LANGUAGES,
        chinese_dialects=QWEN_CHINESE_DIALECTS,
        backends=["transformers"],
    )
    models: dict[str, ModelDefinition] = {
        "qwen3-asr-1.7b": ModelDefinition(
            id="qwen3-asr-1.7b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-1.7b",
            capabilities=qwen_capabilities,
            revision="057a3b044fcd31c433e7971ab40d68d20e7eae6d",
            execution_policy=QWEN_EXECUTION_POLICY,
        ),
        "qwen3-asr-0.6b": ModelDefinition(
            id="qwen3-asr-0.6b",
            provider="QwenLM",
            default=default_model_id == "qwen3-asr-0.6b",
            capabilities=qwen_capabilities,
            revision="6aa69c382e2b426eee1f5870d4c95859a74b6445",
            execution_policy=QWEN_EXECUTION_POLICY,
        ),
    }
    if enable_moss:
        models[MOSS_MODEL_ID] = ModelDefinition(
            id=MOSS_MODEL_ID,
            provider="OpenMOSS-Team",
            default=False,
            capabilities=ModelCapabilities(
                transcription=True,
                streaming=False,
                timestamps=[],
                forced_alignment=False,
                languages=["auto"],
                chinese_dialects=[],
                backends=["transformers"],
                diarization=True,
                segment_timestamps=True,
                execution_modes=["native_long_form", "chunked"],
                auto_execution_mode="native_long_form",
                max_new_tokens=MOSS_EXECUTION_POLICY.max_new_tokens,
                speaker_scopes=["global", "chunk", "mixed"],
                validated_native_max_seconds=MOSS_EXECUTION_POLICY.validated_native_max_seconds,
                automatic_fallback_chunk_seconds=MOSS_EXECUTION_POLICY.fallback_chunk_seconds,
                speaker_resolution_modes=["off", "auto", "required"],
                speaker_resolution_method="moss_anchor_replay",
                validated_anchor_replay_speakers=4,
            ),
            revision="d7231bbae2587a4af278735eb765b318c4f64edd",
            execution_policy=MOSS_EXECUTION_POLICY,
        )
    return models
