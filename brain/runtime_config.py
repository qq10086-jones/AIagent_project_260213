import json
import os
from pathlib import Path


def _coerce_bool(value, default=False):
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _load_runtime_config():
    candidates = [
        os.getenv("RUNTIME_CONFIG_PATH", "").strip(),
        str(Path.cwd() / "configs" / "runtime" / "runtime_defaults.json"),
        str(Path.cwd().parent / "configs" / "runtime" / "runtime_defaults.json"),
        "/workspace/configs/runtime/runtime_defaults.json",
    ]
    for p in candidates:
        if not p:
            continue
        try:
            fp = Path(p)
            if fp.exists():
                return json.loads(fp.read_text(encoding="utf-8")), str(fp)
        except Exception:
            continue
    return {}, None


RUNTIME_CONFIG, RUNTIME_CONFIG_PATH_LOADED = _load_runtime_config()
_BRAIN = (RUNTIME_CONFIG or {}).get("brain", {})

LLM_PROVIDER_DEFAULT = os.getenv("LLM_PROVIDER", _BRAIN.get("llm_provider", "dashscope")).lower()
QWEN_BASE_URL_DEFAULT = os.getenv(
    "QWEN_BASE_URL",
    _BRAIN.get("qwen_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
)
QWEN_MODEL_DEFAULT = os.getenv("QWEN_MODEL", _BRAIN.get("qwen_model", "qwen-plus"))
LOCAL_MODEL_DEFAULT = os.getenv("QUANT_LLM_MODEL", _BRAIN.get("quant_llm_model", "deepseek-r1:1.5b"))
OLLAMA_BASE_URL_DEFAULT = os.getenv("OLLAMA_BASE_URL", _BRAIN.get("ollama_base_url", "http://host.docker.internal:11434"))
LLM_TIMEOUT_SEC_DEFAULT = int(os.getenv("BRAIN_LLM_TIMEOUT_SEC", str(_BRAIN.get("llm_timeout_sec", 60))))
ENABLE_BROWSER_FACTS_DEFAULT = _coerce_bool(
    os.getenv("ENABLE_BROWSER_FACTS", _BRAIN.get("enable_browser_facts", "auto")),
    default=False,
)
