from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from smolagents import (
    InferenceClientModel,
    OpenAIModel,
    LiteLLMModel,
    TransformersModel,
    AzureOpenAIModel,
    AmazonBedrockModel,
)


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    api_key_env: Optional[str] = None
    api_base: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None else default


def _is_openai_family(provider: str) -> bool:
    return provider.lower() in ("openai", "gemini", "openrouter")


def load_model_config_from_env() -> ModelConfig:
    """
    Minimal env-based config with provider-specific safeguards.

    Key fixes:
    - Do not inject tool_choice by default for OpenAI-family text-only calls.
    - Map BRONCHO_MAX_TOKENS to max_completion_tokens for OpenAI-family
      providers, because some models reject max_tokens.
    """
    provider = _env("BRONCHO_PROVIDER", "hf").lower()
    model_id = _env("BRONCHO_MODEL", "Qwen/Qwen3.5-27B")

    extra: Dict[str, Any] = {}

    # Standard knobs
    if _env("BRONCHO_TEMPERATURE") is not None:
        extra["temperature"] = float(_env("BRONCHO_TEMPERATURE"))  # type: ignore[arg-type]

    if _env("BRONCHO_MAX_TOKENS") is not None:
        tok = int(_env("BRONCHO_MAX_TOKENS"))  # type: ignore[arg-type]
        if _is_openai_family(provider):
            extra["max_completion_tokens"] = tok
        else:
            extra["max_tokens"] = tok

    # Tool choice: only pass through when explicitly set.
    # For OpenAI-family providers, avoid injecting it into plain text calls,
    # because the API rejects tool_choice if no tools are provided.
    tool_choice = _env("BRONCHO_TOOL_CHOICE")
    if tool_choice in ("auto", "required", "none"):
        if not _is_openai_family(provider):
            extra["tool_choice"] = tool_choice

    ptc = _env("BRONCHO_PARALLEL_TOOL_CALLS")
    if ptc is not None:
        extra["parallel_tool_calls"] = ptc.strip().lower() in ("1", "true", "yes", "y")

    if provider == "openai":
        return ModelConfig(provider, model_id, api_key_env="OPENAI_API_KEY", extra=extra)

    if provider == "gemini":
        return ModelConfig(
            provider,
            model_id,
            api_key_env="GEMINI_API_KEY",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            extra=extra,
        )

    if provider == "openrouter":
        return ModelConfig(
            provider,
            model_id,
            api_key_env="OPENROUTER_API_KEY",
            api_base="https://openrouter.ai/api/v1",
            extra=extra,
        )

    if provider == "litellm":
        return ModelConfig(
            provider,
            model_id,
            api_key_env=_env("LITELLM_KEY_ENV", "LITELLM_API_KEY"),
            extra=extra,
        )

    if provider == "transformers":
        return ModelConfig(provider, model_id, extra=extra)

    if provider == "azure":
        return ModelConfig(provider, model_id, api_key_env="AZURE_OPENAI_API_KEY", extra=extra)

    if provider == "bedrock":
        return ModelConfig(provider, model_id, extra=extra)

    return ModelConfig(provider="hf", model_id=model_id, api_key_env="HF_TOKEN", extra=extra)


# --- model factory ------------------------------------------------------------

def create_model(cfg: Optional[ModelConfig] = None):
    """Return a smolagents Model instance based on cfg/provider."""
    cfg = cfg or load_model_config_from_env()
    extra = cfg.extra or {}
    p = cfg.provider.lower()

    if p == "hf":
        token = _env(cfg.api_key_env or "HF_TOKEN") or _env("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HF token missing: set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")
        return InferenceClientModel(model_id=cfg.model_id, token=token, **extra)

    if p in ("openai", "gemini", "openrouter"):
        key = _env(cfg.api_key_env or "OPENAI_API_KEY")
        if not key:
            raise ValueError(f"API key missing: set {cfg.api_key_env}")
        return OpenAIModel(model_id=cfg.model_id, api_key=key, api_base=cfg.api_base, **extra)

    if p == "litellm":
        key = _env(cfg.api_key_env or "LITELLM_API_KEY")
        if not key:
            raise ValueError(f"API key missing: set {cfg.api_key_env}")
        return LiteLLMModel(model_id=cfg.model_id, api_key=key, **extra)

    if p == "transformers":
        return TransformersModel(model_id=cfg.model_id, **extra)

    if p == "azure":
        key = _env(cfg.api_key_env or "AZURE_OPENAI_API_KEY")
        if not key:
            raise ValueError("AZURE_OPENAI_API_KEY missing")
        return AzureOpenAIModel(model_id=cfg.model_id, api_key=key, **extra)

    if p == "bedrock":
        return AmazonBedrockModel(model_id=cfg.model_id, **extra)

    raise ValueError(f"Unsupported provider: {cfg.provider}")
