"""Работа с LLM-провайдерами: URL, ключи, список и загрузка моделей."""
import os
import re
import json
import time
import threading
from urllib.parse import urlparse
import requests
from . import config
from .config import MODEL_TIMEOUT, PROVIDERS
from .utils import _load_json


def _provider_api_key(provider: str = None) -> str:
    selected = str(provider or config.PROVIDER or "custom").strip().lower()
    settings_path = config.SETTINGS_FILE
    if settings_path:
        data = _load_json(settings_path, {})
        keys = data.get("provider_api_keys") if isinstance(data, dict) else {}
        if isinstance(keys, dict):
            saved = str(keys.get(selected) or "").strip()
            if saved:
                return saved
    env_names = {
        "openai": ("OPENAI_API_KEY",),
        "xai": ("XAI_API_KEY",),
        "ollama": ("OLLAMA_API_KEY",),
        "lmstudio": ("LM_API_TOKEN",),
        "custom": ("AGENT_API_KEY",),
    }
    for name in env_names.get(selected, ()) + ("AGENT_API_KEY", "LM_API_TOKEN", "OLLAMA_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _provider_api_key_configured(provider: str = None) -> bool:
    return bool(_provider_api_key(provider))


def _provider_info_list() -> list:
    data = _load_json(config.SETTINGS_FILE, {})
    saved_urls = data.get("provider_urls") if isinstance(data, dict) else {}
    if not isinstance(saved_urls, dict):
        saved_urls = {}
    result = []
    for key, info in PROVIDERS.items():
        result.append({
            "id": key,
            "label": info["label"],
            "default_url": info["default_url"],
            "api_url": str(saved_urls.get(key) or (config.API_URL if key == config.PROVIDER else "") or info["default_url"]),
            "api_key_configured": _provider_api_key_configured(key),
        })
    return result


def _uses_xai() -> bool:
    return config.PROVIDER == "xai" or "api.x.ai" in (config.API_URL or "").lower()


def _uses_openai() -> bool:
    return config.PROVIDER == "openai" or "api.openai.com" in (config.API_URL or "").lower()


def _api_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    token = _provider_api_key()
    if token:
        headers["Authorization"] = "Bearer " + token
    return headers


def _api_base_url() -> str:
    try:
        u = urlparse(config.API_URL)
        if not u.scheme or not u.netloc:
            return ""
        return f"{u.scheme}://{u.netloc}"
    except Exception:
        return ""


def _lmstudio_base_url() -> str:
    return _api_base_url()


def _lmstudio_headers() -> dict:
    token = _provider_api_key("lmstudio")
    return {"Authorization": "Bearer " + token} if token else {}


def _ollama_base_url() -> str:
    return _api_base_url()


MODEL_SWITCH_LOCK = threading.RLock()
MODEL_RUNTIME_LOCK = threading.RLock()


def _is_lmstudio() -> bool:
    if config.PROVIDER == "lmstudio":
        return True
    if config.PROVIDER in ("ollama", "openai", "xai"):
        return False
    base = _api_base_url().lower()
    return bool(base) and (
        ":1234" in base
        or os.getenv("AGENT_LMSTUDIO", "").lower() in ("1", "true", "yes")
    ) and "api.x.ai" not in base and ":11434" not in base


def _is_ollama() -> bool:
    if config.PROVIDER == "ollama":
        return True
    if config.PROVIDER in ("lmstudio", "openai", "xai"):
        return False
    base = _api_base_url().lower()
    return bool(base) and (":11434" in base or "ollama" in base)


_MODEL_CTX_CACHE: dict = {}
_DEFAULT_CTX = int(os.getenv("AGENT_CTX_SIZE", "8192"))


def _ollama_model_context(model: str) -> int:
    """Узнать окно контекста модели Ollama через /api/show."""
    model = str(model or "").strip()
    if not model:
        return _DEFAULT_CTX
    if model in _MODEL_CTX_CACHE:
        return _MODEL_CTX_CACHE[model]
    base = _api_base_url()
    if not base:
        return _DEFAULT_CTX
    try:
        r = requests.post(
            base.rstrip("/") + "/api/show",
            json={"name": model},
            headers=_api_headers(),
            timeout=12,
        )
        if r.status_code == 200:
            data = r.json() or {}

            info = data.get("model_info") or {}
            ctx = None
            if isinstance(info, dict):
                for k, v in info.items():
                    if "context_length" in str(k).lower() or str(k).endswith(".context_length"):
                        try:
                            ctx = int(v)
                            break
                        except Exception:
                            pass
            if not ctx:

                params = str(data.get("parameters") or "")
                m = re.search(r"num_ctx\s+(\d+)", params, re.I)
                if m:
                    ctx = int(m.group(1))
            if ctx and ctx > 0:
                _MODEL_CTX_CACHE[model] = ctx
                return ctx
    except Exception:
        pass
    _MODEL_CTX_CACHE[model] = _DEFAULT_CTX
    return _DEFAULT_CTX


def _model_context_size(model: str = None) -> int:
    model = str(model or config.MODEL_NAME).strip()
    if _is_ollama():
        return _ollama_model_context(model)
    if model in _MODEL_CTX_CACHE:
        return _MODEL_CTX_CACHE[model]
    return _DEFAULT_CTX


def _normalize_api_url(url: str) -> str:
    """Убирает хвостовые слэши у базового URL."""
    return (url or "").strip().rstrip("/")


def _chat_endpoint_url() -> str:
    """URL для chat completions. Для Ollama предпочитаем нативный /api/chat."""
    url = _normalize_api_url(config.API_URL)
    if not url:
        return url
    if _is_ollama():
        base = _api_base_url()
        if base:
            return base.rstrip("/") + "/api/chat"

        try:
            u = urlparse(url)
            if u.scheme and u.netloc:
                return f"{u.scheme}://{u.netloc}/api/chat"
        except Exception:
            pass
    return url


def _ollama_openai_endpoint() -> str:
    """Запасной OpenAI-compatible endpoint Ollama."""
    base = _api_base_url()
    if base:
        return base.rstrip("/") + "/v1/chat/completions"
    url = _normalize_api_url(config.API_URL)
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url.rstrip("/") + "/v1/chat/completions"


def _lmstudio_models_raw():
    if not _is_lmstudio():
        return []
    base = _lmstudio_base_url()
    r = requests.get(base + "/api/v1/models", headers=_lmstudio_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()
    return data.get("models") or []


def _ollama_models_raw():
    """Список моделей Ollama через /api/tags (native) с fallback на /v1/models."""
    if not _is_ollama():
        return []
    base = _ollama_base_url()
    headers = _api_headers()

    try:
        r = requests.get(base + "/api/tags", headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models") or []
            out = []
            for m in models:
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name") or m.get("model") or "")
                if not name:
                    continue
                details = m.get("details") or {}
                out.append({
                    "key": name,
                    "name": name,
                    "params": details.get("parameter_size") or details.get("family"),
                    "quantization": details.get("quantization_level"),
                    "loaded": True,                          
                    "instances": [],
                })
            if out:
                return out
    except Exception:
        pass

    try:
        r = requests.get(base + "/v1/models", headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for m in (data.get("data") or []):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or "")
            if mid:
                out.append({
                    "key": mid,
                    "name": mid,
                    "params": None,
                    "quantization": None,
                    "loaded": True,
                    "instances": [],
                })
        return out
    except Exception:
        return []


def _openai_style_models():
    """Список моделей через OpenAI-compatible /v1/models (xAI, OpenAI, custom)."""
    if _is_lmstudio() or _is_ollama():
        return []
    base = _api_base_url()
    if not base:
        return [{"key": config.MODEL_NAME, "name": config.MODEL_NAME, "loaded": True, "params": None, "quantization": None, "instances": []}]
    try:
        r = requests.get(base + "/v1/models", headers=_api_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        out = []
        for m in (data.get("data") or data.get("models") or []):
            if not isinstance(m, dict):
                continue
            mid = str(m.get("id") or m.get("key") or "")
            if mid:
                out.append({
                    "key": mid,
                    "name": mid,
                    "params": None,
                    "quantization": None,
                    "loaded": True,
                    "instances": [],
                })
        return out or [{"key": config.MODEL_NAME, "name": config.MODEL_NAME, "loaded": True, "params": None, "quantization": None, "instances": []}]
    except Exception:
        return [{"key": config.MODEL_NAME, "name": config.MODEL_NAME, "loaded": True, "params": None, "quantization": None, "instances": []}]


def _available_models():
    """Модели текущего провайдера."""
    if _is_lmstudio():
        models = []
        for m in _lmstudio_models_raw():
            if not isinstance(m, dict) or m.get("type") != "llm":
                continue
            instances = m.get("loaded_instances") or []
            models.append({
                "key": str(m.get("key") or ""),
                "name": str(m.get("display_name") or m.get("key") or ""),
                "params": m.get("params_string"),
                "quantization": ((m.get("quantization") or {}).get("name") if isinstance(m.get("quantization"), dict) else None),
                "loaded": bool(instances),
                "instances": [str(x.get("id")) for x in instances if isinstance(x, dict) and x.get("id")],
            })
        return [m for m in models if m["key"]]
    if _is_ollama():
        return _ollama_models_raw()
    return _openai_style_models()


def _loaded_llm_instances(models=None):
    """Свежий список всех загруженных LLM-инстансов LM Studio."""
    models = _lmstudio_models_raw() if models is None else models
    out = []
    for m in models:
        if not isinstance(m, dict) or m.get("type") != "llm":
            continue
        key = str(m.get("key") or "")
        for inst in (m.get("loaded_instances") or []):
            if isinstance(inst, dict) and inst.get("id"):
                out.append({"model": key, "id": str(inst["id"])})
    return out


def _lm_unload(instance_id: str):
    base = _lmstudio_base_url()
    headers = {**_lmstudio_headers(), "Content-Type": "application/json"}
    rr = requests.post(
        base + "/api/v1/models/unload",
        headers=headers,
        json={"instance_id": str(instance_id)},
        timeout=60,
    )
    rr.raise_for_status()
    return rr


def _wait_no_loaded_llm(timeout: float = 90.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _loaded_llm_instances():
            return True
        time.sleep(0.15)
    return False


def _wait_exactly_one_model(model_key: str, timeout: float = 90.0):
    """Дожидается состояния: загружен ровно один LLM и это model_key."""
    deadline = time.monotonic() + timeout
    model_key = str(model_key)
    while time.monotonic() < deadline:
        loaded = _loaded_llm_instances()
        if len(loaded) == 1 and loaded[0]["model"] == model_key:
            return loaded[0]["id"]
        time.sleep(0.15)
    raise RuntimeError("LM Studio не перешёл в состояние с одной загруженной моделью")


def _set_active_model(model_key: str):
    """Переключает LM Studio в строгое состояние: ровно один LLM-инстанс."""
    model_key = str(model_key or "").strip()
    if not model_key:
        raise ValueError("Не указана модель")

    if not _is_lmstudio():
        config.MODEL_NAME = model_key
        return model_key

    with MODEL_RUNTIME_LOCK:
        with MODEL_SWITCH_LOCK:
            models = _lmstudio_models_raw()
            target = next(
                (m for m in models if m.get("type") == "llm" and str(m.get("key")) == model_key),
                None,
            )
            if target is None:
                raise ValueError("Модель не найдена среди доступных в LM Studio")

            loaded = _loaded_llm_instances(models)
            if len(loaded) == 1 and loaded[0]["model"] == model_key:
                config.MODEL_NAME = model_key
                return model_key

            for inst in loaded:
                _lm_unload(inst["id"])
            if not _wait_no_loaded_llm():
                raise RuntimeError("LM Studio не успел выгрузить предыдущие модели")

            base = _lmstudio_base_url()
            headers = {**_lmstudio_headers(), "Content-Type": "application/json"}
            rr = requests.post(
                base + "/api/v1/models/load",
                headers=headers,
                json={"model": model_key},
                timeout=MODEL_TIMEOUT,
            )
            rr.raise_for_status()
            load_data = rr.json() if rr.content else {}
            loaded_id = str(load_data.get("instance_id") or "").strip()

            _wait_exactly_one_model(model_key)

            current = _loaded_llm_instances()
            candidates = [x for x in current if x["model"] == model_key]
            keep = loaded_id if any(x["id"] == loaded_id for x in candidates) else (candidates[0]["id"] if candidates else "")
            for inst in current:
                if inst["id"] != keep:
                    _lm_unload(inst["id"])
            _wait_exactly_one_model(model_key)

            config.MODEL_NAME = model_key
            return model_key


def _resolve_model(model_key):
    """Нет такой модели — берём уже загруженную или первую доступную."""
    model_key = str(model_key or config.MODEL_NAME).strip()
    if not _is_lmstudio():
        return model_key
    models = [m for m in _lmstudio_models_raw() if isinstance(m, dict) and m.get("type") == "llm"]
    keys = [str(m.get("key")) for m in models]
    if model_key in keys:
        return model_key
    loaded = _loaded_llm_instances(models)
    if loaded:
        return loaded[0]["model"]
    if keys:
        return keys[0]
    raise RuntimeError("В LM Studio нет доступных LLM-моделей")


def _ensure_model(model_key: str):
    """Перед inference гарантирует, что в памяти ровно одна нужная LLM."""
    model_key = str(model_key or config.MODEL_NAME).strip() or config.MODEL_NAME
    if not _is_lmstudio():
        config.MODEL_NAME = model_key
        return config.MODEL_NAME
    with MODEL_RUNTIME_LOCK:
        model_key = _resolve_model(model_key)
        loaded = _loaded_llm_instances()
        if len(loaded) != 1 or loaded[0]["model"] != model_key:
            _set_active_model(model_key)
        config.MODEL_NAME = model_key
        return config.MODEL_NAME


def _http_error_message(e) -> str:
    code = getattr(getattr(e, "response", None), "status_code", 0) or 0
    detail = ""
    try:
        body = e.response.json()
        err = body.get("error") or body.get("code") or ""
        if isinstance(err, dict):
            err = err.get("message") or err.get("code") or ""
        detail = str(err)
        code_s = str(body.get("code") or "")
    except Exception:
        code_s = ""
        detail = ""
    blob = (detail + " " + code_s).lower()
    if code == 403 and any(x in blob for x in ("credit", "spending", "subscription", "personal-team")):
        return ("Модель сейчас недоступна: закончились кредиты Grok. "
                "После пополнения можно прикрепить файл и попросить точечную правку.")
    if code in (401, 403):
        return "Модель отклонила запрос (" + str(code) + "). Проверьте доступ к API."
    tail = _public_error(detail or e)
    return f"Ошибка модели ({code}). {tail}".strip()


def _public_error(e) -> str:
    message = str(e)
    keys = set()
    for provider in PROVIDERS:
        try:
            key = _provider_api_key(provider)
            if key:
                keys.add(key)
        except Exception:
            pass
    for key in keys:
        message = message.replace(key, "***")
    return message[:400]
