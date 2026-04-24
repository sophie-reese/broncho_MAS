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

# --- simple config layer ------------------------------------------------------

@dataclass
class ModelConfig:
    provider: str               # "hf" | "openai" | "gemini" | "openrouter" | "litellm" | "transformers" | "azure" | "bedrock"
    model_id: str
    api_key_env: Optional[str] = None  # which env var holds key (if needed)
    api_base: Optional[str] = None     # for OpenAI-compatible endpoints
    extra: Optional[Dict[str, Any]] = None  # temperature, max_tokens, tool_choice, etc.


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None else default


def load_model_config_from_env() -> ModelConfig:
    """
    Minimal, pragmatic env-based config.

    IMPORTANT (tools/tool_choice):
    - smolagents.ToolCallingAgent often uses ReAct-style tool use (Action: ...).
    - Some OpenAI-compatible backends will hard-reject if tool_choice="required"
      but the model did not emit *native* tool_calls.
    - Therefore we default tool_choice to "auto" unless you override it.
      Override with: BRONCHO_TOOL_CHOICE=required|auto|none
    """
    provider = _env("BRONCHO_PROVIDER", "hf").lower()
    model_id = _env("BRONCHO_MODEL", "Qwen/Qwen3.5-27B")

    extra: Dict[str, Any] = {}

    # Standard knobs
    if _env("BRONCHO_TEMPERATURE") is not None:
        extra["temperature"] = float(_env("BRONCHO_TEMPERATURE"))  # type: ignore
    if _env("BRONCHO_MAX_TOKENS") is not None:
        extra["max_tokens"] = int(_env("BRONCHO_MAX_TOKENS"))  # type: ignore

    # ---- Tools / tool-choice knobs (critical for your current crash) ----
    # Default to "auto" unless explicitly set.
    tool_choice = _env("BRONCHO_TOOL_CHOICE", "auto")
    # Only pass through known values
    if tool_choice in ("auto", "required", "none"):
        extra["tool_choice"] = tool_choice

    # Some providers/models are picky about parallel tool calls
    ptc = _env("BRONCHO_PARALLEL_TOOL_CALLS")
    if ptc is not None:
        extra["parallel_tool_calls"] = ptc.strip().lower() in ("1", "true", "yes", "y")

    # provider-specific defaults
    if provider == "openai":
        return ModelConfig(provider, model_id, api_key_env="OPENAI_API_KEY", extra=extra)

    if provider == "gemini":
        # Gemini is OpenAI-compatible via api_base
        return ModelConfig(
            provider, model_id,
            api_key_env="GEMINI_API_KEY",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            extra=extra
        )

    if provider == "openrouter":
        return ModelConfig(
            provider, model_id,
            api_key_env="OPENROUTER_API_KEY",
            api_base="https://openrouter.ai/api/v1",
            extra=extra
        )

    if provider == "litellm":
        return ModelConfig(provider, model_id, api_key_env=_env("LITELLM_KEY_ENV", "LITELLM_API_KEY"), extra=extra)

    if provider == "transformers":
        return ModelConfig(provider, model_id, extra=extra)

    if provider == "azure":
        return ModelConfig(provider, model_id, api_key_env="AZURE_OPENAI_API_KEY", extra=extra)

    if provider == "bedrock":
        return ModelConfig(provider, model_id, extra=extra)

    # default HF
    return ModelConfig(provider="hf", model_id=model_id, api_key_env="HF_TOKEN", extra=extra)


# --- model factory ------------------------------------------------------------

def create_model(cfg: Optional[ModelConfig] = None):
    """
    Return a smolagents Model instance based on cfg/provider.
    """
    cfg = cfg or load_model_config_from_env()
    extra = cfg.extra or {}
    p = cfg.provider.lower()

    if p == "hf":
        # Hugging Face Inference API model wrapper
        token = _env(cfg.api_key_env or "HF_TOKEN") or _env("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HF token missing: set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")

        # NOTE: passing tool_choice="auto" here is the key fix for the
        # 'Tool choice is required, but model did not call a tool' crash
        # when the backend enforces native tool_calls.
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
