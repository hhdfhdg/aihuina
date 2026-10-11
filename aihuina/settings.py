"""Сохранение и загрузка настроек провайдера и рабочей папки."""
import os
import json
from pathlib import Path
from . import config
from .config import PROVIDERS, SETTINGS_FILE, _detect_provider
from .providers import _provider_api_key_configured, _provider_info_list
from .utils import _load_json, _save_json


def _apply_provider_settings(data: dict = None):
    """Применяет сохранённые provider/url/model из settings.json (приоритет над дефолтами)."""
    data = data if data is not None else _load_json(SETTINGS_FILE, {})
    if not isinstance(data, dict):
        return
    prov = str(data.get("provider") or "").strip().lower()
    url = str(data.get("api_url") or "").strip()
    model = str(data.get("model") or "").strip()
    provider_urls = data.get("provider_urls") if isinstance(data.get("provider_urls"), dict) else {}

    if prov in PROVIDERS:
        config.PROVIDER = prov
    if not url and config.PROVIDER in provider_urls:
        url = str(provider_urls.get(config.PROVIDER) or "").strip()
    if not url and config.PROVIDER in PROVIDERS:
        url = PROVIDERS[config.PROVIDER]["default_url"]
    if url:
        config.API_URL = url
        if prov not in PROVIDERS:
            config.PROVIDER = _detect_provider(url)
    if model:
        config.MODEL_NAME = model


def _save_provider_settings():
    """Пишет текущий провайдер, URL и модель на диск — после рестарта не сбрасывается."""
    data = _load_json(SETTINGS_FILE, {})
    if not isinstance(data, dict):
        data = {}
    provider_urls = data.get("provider_urls") if isinstance(data.get("provider_urls"), dict) else {}
    provider_urls = dict(provider_urls)
    if config.PROVIDER:
        provider_urls[config.PROVIDER] = config.API_URL
    data["provider"] = config.PROVIDER
    data["api_url"] = config.API_URL
    data["model"] = config.MODEL_NAME
    data["provider_urls"] = provider_urls
    _save_json(SETTINGS_FILE, data)


def _set_provider(provider: str, api_url: str = None, model: str = None, api_key: str = None, clear_api_key: bool = False):
    provider = str(provider or "").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError("Неизвестный провайдер: " + provider)

    data = _load_json(SETTINGS_FILE, {})
    if not isinstance(data, dict):
        data = {}
    provider_urls = data.get("provider_urls") if isinstance(data.get("provider_urls"), dict) else {}
    provider_keys = data.get("provider_api_keys") if isinstance(data.get("provider_api_keys"), dict) else {}
    previous_provider = config.PROVIDER

    config.PROVIDER = provider
    url_in = str(api_url).strip() if api_url is not None and str(api_url).strip() else ""
    if url_in:
        config.API_URL = url_in
    else:
        saved = str(provider_urls.get(provider) or "").strip()
        config.API_URL = saved or PROVIDERS[provider]["default_url"]

    if api_key is not None:
        supplied_key = str(api_key).strip()
        if supplied_key:
            provider_keys[provider] = supplied_key
        else:
            provider_keys.pop(provider, None)
    elif clear_api_key:
        provider_keys.pop(provider, None)
    data["provider_api_keys"] = provider_keys
    _save_json(SETTINGS_FILE, data)

    if model is not None and str(model).strip():
        config.MODEL_NAME = str(model).strip()
    elif previous_provider != provider or not config.MODEL_NAME:
        config.MODEL_NAME = PROVIDERS[provider]["default_model"]

    _save_provider_settings()
    return {
        "provider": config.PROVIDER,
        "api_url": config.API_URL,
        "model": config.MODEL_NAME,
        "label": PROVIDERS[config.PROVIDER]["label"],
        "api_key_configured": _provider_api_key_configured(config.PROVIDER),
        "providers": _provider_info_list(),
    }


def _initial_workdir() -> Path:
    saved = _load_json(SETTINGS_FILE, {}).get("workdir")
    if saved and Path(saved).is_dir():
        return Path(saved).resolve()
    p = Path(os.getenv("AGENT_WORKDIR", "./agent_workspace")).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


DEFAULT_WORKDIR = _initial_workdir()

_apply_provider_settings()


_apply_provider_settings()
