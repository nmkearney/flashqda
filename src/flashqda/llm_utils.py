from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openai import OpenAI, OpenAIError

from flashqda.log_utils import update_log

# -----------------
# Provider registry
# -----------------

def _build_openai_client(api_key: Optional[str] = None,
                         base_url: Optional[str] = None) -> OpenAI:
    """
    Build an OpenAI SDK client. This works for:
    - OpenAI cloud
    - local OpenAI-compatible endpoints (e.g., Ollama, LM Studio, vLLM)
    """
    key = api_key or os.getenv("LLM_API_KEY") or "not-needed"
    return OpenAI(api_key=key, base_url=base_url)

def _provider_openai(config) -> Dict[str, Any]:
    """
    OpenAI cloud provider.
    """
    return {
        "client": make_openai_client(config),
        "model": getattr(config, "model", "gpt-4o"),
        "supports_json_mode": True,
        "supports_json_schema": True,
        "provider_name": "openai",
    }
 
def _provider_ollama(config) -> Dict[str, Any]:
    """
    Ollama via OpenAI-compatible local endpoint.
    """
    return {
        "client": make_openai_client(config),
        "model": getattr(config, "model", "llama3"),
        "supports_json_mode": True,
        "supports_json_schema": True,
        "provider_name": "ollama",
    }

def _provider_openai_compatible(config) -> Dict[str, Any]:
    """
    Generic OpenAI-compatible endpoint.
    """
    return {
        "client": make_openai_client(config),
        "model": getattr(config, "model", "unknown-model"),
        "supports_json_mode": False,
        "supports_json_schema": True,
        "provider_name": "openai_compatible",
    }

def _provider_anthropic(config) -> Dict[str, Any]:
    """
    Anthropic Claude via the anthropic SDK.
    send_to_llm handles this provider through a dedicated path (_send_to_anthropic).
    """
    return {
        "client": None,
        "model": getattr(config, "model", "claude-sonnet-4-6"),
        "supports_json_mode": False,
        "supports_json_schema": False,
        "provider_name": "anthropic",
    }

PROVIDER_REGISTRY: Dict[str, Callable[[Any], Dict[str,Any]]] = {
    "openai": _provider_openai,
    "ollama": _provider_ollama,
    "openai_compatible": _provider_openai_compatible,
    "anthropic": _provider_anthropic,
}

# ----------------------
# Initialization helpers
# ----------------------

def get_llm_api_key(config=None, api_key=None, api_key_filename="llm_api_key.txt", project_root=None):
    """
    Resolve provider-aware API key / env setup.

    For local providers like Ollama, no real API key is required.
    For OpenAI-compatible local servers, a dummy key may still be needed by the SDK.
    This function no longer creates or stores a global client.
    """

    provider = getattr(config, "provider", "openai") if config is not None else "openai"

    # Anthropic: check ANTHROPIC_API_KEY first, fall back to LLM_API_KEY
    if provider == "anthropic":
        key = (
            api_key
            or getattr(config, "api_key", None)
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("LLM_API_KEY")
        )
        if not key:
            raise RuntimeError(
                "No API key found for Anthropic. Set the ANTHROPIC_API_KEY environment "
                "variable or pass api_key to PipelineConfig."
            )
        os.environ["LLM_API_KEY"] = key
        return key

    # Local / OpenAI-compatible providers: key may be optional
    if provider in ("ollama", "openai_compatible"):
        key = (
            api_key
            or getattr(config, "api_key", None)
            or os.getenv("LLM_API_KEY")
            or ("ollama" if provider == "ollama" else "not-needed")
        )
        os.environ["LLM_API_KEY"] = key
        return key

    # OpenAI cloud provider: require a real key
    if api_key:
        key = api_key
    else:
        base_path = Path(project_root) if project_root else Path.cwd()
        key_path = base_path / api_key_filename

        if key_path.exists():
            with open(key_path, "r") as f:
                key = f.read().strip()
            if not key:
                raise RuntimeError(f"LLM API key file at {key_path} is empty.")
        else:
            key = getattr(config, "api_key", None) if config is not None else None
            key = key or os.getenv("LLM_API_KEY")
            if not key:
                raise OpenAIError(
                    f"No API key provided, and file not found at {key_path}, "
                    "and LLM_API_KEY environment variable not set."
                )

    os.environ["LLM_API_KEY"] = key
    return key

def get_client_kwargs(config=None):
    provider = getattr(config, "provider", "openai") if config is not None else "openai"

    if provider == "openai":
        return {
            "api_key": getattr(config, "api_key", None) or os.getenv("LLM_API_KEY"),
            "base_url": None,
        }

    if provider == "ollama":
        return {
            "api_key": getattr(config, "api_key", None) or "ollama",
            "base_url": getattr(config, "base_url", None) or "http://localhost:11434/v1",
        }

    if provider == "openai_compatible":
        base_url = getattr(config, "base_url", None)
        if not base_url:
            raise ValueError("base_url is required for provider='openai_compatible'")
        return {
            "api_key": getattr(config, "api_key", None) or os.getenv("LLM_API_KEY") or "not-needed",
            "base_url": base_url,
        }

    raise ValueError(f"Unsupported provider: {provider}")

def make_openai_client(config=None):
    return OpenAI(**get_client_kwargs(config))

# -------------
# JSON handling
# -------------

def validate_required_keys(data: dict, required_keys: list[str]) -> bool:
    return isinstance(data, dict) and all(k in data for k in required_keys)

def extract_json_from_text(text: str) -> Any:
    """
    Parse JSON from a model response.
    First tries direct json.loads().
    Then tries to extract the first JSON object or array from surrounding text.
    """
    if text is None:
        raise ValueError("Response text is None.")
    ext = text.strip()

    # 1. Direct parase
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # 2. Remove fenced code block markers if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Extract first {...} or [...]
    obj_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    arr_match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)

    candidates = []
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    
    raise ValueError(f"Could not parse JSON from model response: {text[:500]}")

# --------------------
# Core provider router
# --------------------

def get_provider_settings(config) -> Dict[str, Any]:
    """
    Resolve provider-specific settings from config.
    """
    provider = getattr(config, "provider", "openai")
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported providers: {sorted(PROVIDER_REGISTRY.keys())}"
        )
    return PROVIDER_REGISTRY[provider](config)

def _send_to_anthropic(
    system_prompt: str,
    user_prompt: str,
    config,
    response_format: Optional[Dict[str, Any]],
    max_retries: int,
    sleep_seconds: int,
) -> str:
    """
    Send a prompt to an Anthropic Claude model.
    JSON output is enforced via prompt instructions (Anthropic's structured output
    uses tool use, which is not required for FlashQDA's use case).
    """
    try:
        import anthropic as _anthropic
    except ImportError:
        raise ImportError(
            "The 'anthropic' package is required for provider='anthropic'. "
            "Install it with: pip install anthropic  or  pip install 'flashqda[anthropic]'"
        )

    api_key = (
        getattr(config, "api_key", None)
        or os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    client = _anthropic.Anthropic(api_key=api_key)
    model = getattr(config, "model", "claude-sonnet-4-6")
    temperature = getattr(config, "temperature", 0.0)
    max_tokens = getattr(config, "max_tokens", 4096)

    final_system_prompt = system_prompt
    if response_format is not None:
        final_system_prompt = (
            f"{system_prompt}\n\n"
            "Return only valid JSON. "
            "Do not include markdown fences, comments, or explanatory text."
        )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=final_system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                raise RuntimeError(
                    f"Anthropic API call failed after {max_retries} attempts "
                    f"(model={model}): {e}"
                )


def send_to_llm(
    system_prompt: str,
    user_prompt: str,
    config,
    response_format: Optional[Dict[str, Any]] = None,
    max_retries = 3,
    sleep_seconds: int = 5,
) -> str:
    """
    Send a prompt to the configured LLM provider.
    If JSON mode is requested but the provider does not support it, the function falls back to prompt-only JSON instructions.
    Anthropic Claude models are handled via a dedicated path using the anthropic SDK.
    """
    if getattr(config, "provider", "openai") == "anthropic":
        return _send_to_anthropic(
            system_prompt, user_prompt, config, response_format, max_retries, sleep_seconds
        )

    provider_settings = get_provider_settings(config)
    client = provider_settings["client"]
    model = provider_settings["model"]

    use_json_mode = bool(getattr(config, "use_json_mode", True))
    temperature = getattr(config, "temperature", 0.0)
    timeout = getattr(config, "timeout", 15)

    final_system_prompt = system_prompt
    request_kwargs = {}

    if response_format is not None and use_json_mode:
        request_kwargs["response_format"] = response_format
    elif response_format is not None:
        # JSON fallback: reinforce via prompt, do not pass response_format
        final_system_prompt = (
            f"{system_prompt}\n\n"
            "Return only valid JSON. " 
            "Do not include markdown fences, comments, or explanatory text."
        )

    messages = [
        {"role": "system", "content": final_system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
                **request_kwargs,
            )
            content = response.choices[0].message.content or ""
            return content.strip()
        except OpenAIError as e:
            if attempt < max_retries - 1:
                time.sleep(sleep_seconds)
            else:
                provider_name = provider_settings.get("provider_name", "unknown")
                raise RuntimeError(
                    f"LLM API call failed after {max_retries} attempts: "
                    f"(provider={provider_name}, model={model}): {e}"
                )
            
def make_embedding_client(config=None):
    return OpenAI(
        api_key=getattr(config, "embedding_api_key", None)
                or getattr(config, "api_key", None)
                or os.getenv("LLM_API_KEY")
                or "not-needed",
        base_url=getattr(config, "embedding_base_url", None)
                 or getattr(config, "base_url", None),
    )