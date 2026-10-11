"""Потоковый вызов LLM-API и разбор ответов."""
import json
import time
import requests
from . import config
from .config import MODEL_TIMEOUT, TEMPERATURE
from .providers import (
    _api_headers,
    _chat_endpoint_url,
    _is_ollama,
    _normalize_api_url,
    _ollama_openai_endpoint,
    _uses_openai,
    _uses_xai,
)


def _reasoning_value(value):
    """Нормализует разные форматы reasoning/thinking из OpenAI-compatible API."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "reasoning", "thinking"):
                    v = item.get(key)
                    if isinstance(v, str):
                        parts.append(v)
                        break
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "thinking"):
            v = value.get(key)
            if isinstance(v, str):
                return v
    return ""


def stream_model(messages, temperature=TEMPERATURE, retries: int = 2, model_name: str = None,
                 cancel_event=None, on_response=None, usage_out=None):
    """
    Стримит ответ модели и НИКОГДА не смешивает reasoning с обычным ответом.
    Все поля reasoning приводятся к внутреннему тегу <reasoning>.
    Ollama: сначала нативный /api/chat (NDJSON), при 404 — OpenAI-compatible /v1/chat/completions.
    """
    model = model_name or config.MODEL_NAME
    headers = _api_headers()
    last_err = None

    def _yield_openai_sse(resp):
        in_reasoning = False
        for raw_line in resp.iter_lines(decode_unicode=False):
            if cancel_event is not None and cancel_event.is_set():
                break
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace")
            line = line.lstrip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                if in_reasoning:
                    yield "</reasoning>"
                    in_reasoning = False
                return
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if usage_out is not None and isinstance(chunk.get("usage"), dict):
                u = chunk["usage"]
                try:
                    usage_out["prompt_tokens"] = int(u.get("prompt_tokens") or usage_out.get("prompt_tokens") or 0)
                    usage_out["completion_tokens"] = int(u.get("completion_tokens") or usage_out.get("completion_tokens") or 0)
                    usage_out["total_tokens"] = int(u.get("total_tokens") or (usage_out["prompt_tokens"] + usage_out["completion_tokens"]))
                except Exception:
                    pass
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}
                reasoning = ""
                for source in (delta, message):
                    if not isinstance(source, dict):
                        continue
                    for key in (
                        "reasoning_content",
                        "reasoning",
                        "thinking",
                        "analysis",
                        "analysis_content",
                    ):
                        value = _reasoning_value(source.get(key))
                        if value:
                            reasoning += value
                if reasoning:
                    if not in_reasoning:
                        in_reasoning = True
                        yield "<reasoning>"
                    yield reasoning
                content = delta.get("content")
                if content is None:
                    content = message.get("content")
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            value = item.get("text")
                            if isinstance(value, str):
                                parts.append(value)
                    content = "".join(parts)
                if content:
                    if in_reasoning:
                        yield "</reasoning>"
                        in_reasoning = False
                    yield str(content)
        if in_reasoning:
            yield "</reasoning>"

    def _yield_ollama_ndjson(resp):
        """Нативный стрим Ollama: по строке JSON на чанк."""
        in_reasoning = False
        try:
            line_iter = resp.iter_lines(decode_unicode=False)
        except Exception:
            return
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            try:
                raw_line = next(line_iter)
            except StopIteration:
                break
            except Exception:
                break
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                line = raw_line.decode("utf-8", errors="replace")
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = chunk.get("message") if isinstance(chunk.get("message"), dict) else {}
            thinking = ""
            for key in ("thinking", "reasoning", "reasoning_content"):
                v = msg.get(key) if msg else None
                if v is None:
                    v = chunk.get(key)
                v = _reasoning_value(v)
                if v:
                    thinking += v
            if thinking:
                if not in_reasoning:
                    in_reasoning = True
                    yield "<reasoning>"
                yield thinking
            content = ""
            if msg and isinstance(msg.get("content"), str):
                content = msg.get("content") or ""
            elif isinstance(chunk.get("response"), str):
                content = chunk.get("response") or ""
            if content:
                if in_reasoning:
                    yield "</reasoning>"
                    in_reasoning = False
                yield content
            if chunk.get("done"):
                if usage_out is not None:
                    prompt_n = int(chunk.get("prompt_eval_count") or 0)
                    out_n = int(chunk.get("eval_count") or 0)
                    total = prompt_n + out_n
                    if total > 0:
                        usage_out["prompt_tokens"] = prompt_n
                        usage_out["completion_tokens"] = out_n
                        usage_out["total_tokens"] = total
                break
        if in_reasoning:
            yield "</reasoning>"


    if _is_ollama():
        ollama_url = _chat_endpoint_url()

        ollama_messages = []
        for m in messages or []:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or "user"
            content = m.get("content")
            if content is None:
                continue
            if not isinstance(content, str):
                content = str(content)
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            ollama_messages.append({"role": role, "content": content})
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": True,
            "options": {"temperature": float(temperature)},
        }
        for attempt in range(retries + 1):
            try:
                with requests.post(
                    ollama_url,
                    json=payload,
                    headers=headers,
                    timeout=MODEL_TIMEOUT,
                    stream=True,
                ) as r:
                    if on_response:
                        on_response(r)
                    if r.status_code == 404:

                        break
                    r.raise_for_status()
                    try:
                        for piece in _yield_ollama_ndjson(r):
                            yield piece
                    except Exception:
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        raise
                    return
            except requests.exceptions.ConnectionError as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                last_err = e
                code = getattr(getattr(e, "response", None), "status_code", 0) or 0
                if code == 404:
                    break
                if attempt < retries and 500 <= code < 600:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise

        openai_url = _ollama_openai_endpoint()
        payload_oa = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        for attempt in range(retries + 1):
            try:
                with requests.post(
                    openai_url,
                    json=payload_oa,
                    headers=headers,
                    timeout=MODEL_TIMEOUT,
                    stream=True,
                ) as r:
                    if on_response:
                        on_response(r)
                    r.raise_for_status()
                    try:
                        for piece in _yield_openai_sse(r):
                            yield piece
                    except Exception:
                        if cancel_event is not None and cancel_event.is_set():
                            return
                        raise
                    return
            except requests.exceptions.ConnectionError as e:
                last_err = e
                if attempt < retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise
            except requests.exceptions.HTTPError as e:
                last_err = e
                if attempt < retries and 500 <= getattr(e.response, "status_code", 0) < 600:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise
        if last_err:
            raise last_err
        return


    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    if _uses_xai() or _uses_openai():
        payload["stream_options"] = {"include_usage": True}
    if _uses_xai():
        payload["max_tokens"] = 16384
    endpoint = _normalize_api_url(config.API_URL)

    for attempt in range(retries + 1):
        try:
            with requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=MODEL_TIMEOUT,
                stream=True,
            ) as r:
                if on_response:
                    on_response(r)
                r.raise_for_status()
                try:
                    for piece in _yield_openai_sse(r):
                        yield piece
                except Exception:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    raise
                return
        except requests.exceptions.ConnectionError as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise
        except requests.exceptions.HTTPError as e:
            last_err = e
            if attempt < retries and 500 <= getattr(e.response, "status_code", 0) < 600:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise
    if last_err:
        raise last_err
