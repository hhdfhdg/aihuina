"""Чат: состояние, история, сохранение и список чатов."""
import re
import json
import time
import uuid
import threading
import contextlib
from pathlib import Path
from urllib.parse import quote
from . import config
from .agent import Agent, _event_to_json, _shorten, attachments_note
from .config import ACTIVE_WORKSPACE, CHAT_DIR, MAX_ITERATIONS, WORKSPACE_DIR, resolve_effort
from .file_tools import root_id
from .prompts import build_system_prompt
from .providers import (
    MODEL_RUNTIME_LOCK,
    _ensure_model,
    _is_lmstudio,
    _model_context_size,
    _public_error,
    _set_active_model,
)
from .settings import _save_provider_settings
from .utils import _load_json, _save_json
from .workspace import _chat_workspace, _copy_tree, sync_active_to_chat, sync_chat_to_active


CHATS: dict = {}
CHATS_LOCK = threading.Lock()
CLIENTS: set = set()
CLIENTS_LOCK = threading.Lock()


class Chat:
    def __init__(self, cid=None, workdir=None):
        self.id = cid or str(uuid.uuid4())
        self.title = "Новый чат"
        self.created = self.updated = time.time()
        self.model_name = config.MODEL_NAME


        self.workspace_root = _chat_workspace(self.id)

        self.workdir = ACTIVE_WORKSPACE
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(self.workdir, self.model_name)
        self.events: list = []
        self.listeners: set = set()
        self.lock = threading.RLock()
        self.running = False
        self.worker = None
        self._active = False                               

        self.context_info = {"used": 0, "limit": _model_context_size(self.model_name), "percent": 0}
        self.effort = "normal"

    def info(self):
        return {"id": self.id, "title": self.title, "workdir": str(self.workdir),
                "workspace_root": str(self.workspace_root),
                "model": self.model_name,
                "running": self.running, "updated": self.updated,
                "context": self.context_info,
                "effort": self.effort}

    def estimate_context(self):
        """Оценка занятости контекста по истории сообщений ЭТОГО чата."""
        limit = _model_context_size(self.model_name)
        try:
            chars = 0
            with self.agent.lock:
                for m in self.agent.messages:
                    if m.get("role") == "system":
                        continue
                    chars += len(m.get("content") or "")
            used = max(0, chars // 4)
        except Exception:
            used = int((self.context_info or {}).get("used") or 0)
        pct = min(100.0, round(100.0 * used / max(limit, 1), 1))
        self.context_info = {
            "used": used,
            "limit": limit,
            "percent": pct,
            "prompt_tokens": used,
            "completion_tokens": int((self.context_info or {}).get("completion_tokens") or 0),
        }
        return self.context_info

    def activate(self):
        """Подгрузить файлы чата во временный Workspace и сделать его рабочим."""
        if self.running:
            return self.workdir

        with CHATS_LOCK:
            for c in CHATS.values():
                if c is not self and getattr(c, "_active", False):
                    try:
                        sync_active_to_chat(c.id)
                    except Exception:
                        pass
                    c._active = False
        sync_chat_to_active(self.id)
        self.workdir = ACTIVE_WORKSPACE
        self.agent.set_workdir(self.workdir)
        self._active = True
        root_id(self.workdir)
        return self.workdir

    def persist_files(self):
        """Скопировать текущий Workspace обратно в папку чата."""
        try:
            sync_active_to_chat(self.id)
        except Exception as e:
            print(f"[!] sync active→chat {self.id}: {e}")

    def record(self, obj):
        t = obj["type"]
        if t in ("assistant_delta", "reasoning_delta") and self.events \
                and self.events[-1]["type"] == t:
            self.events[-1]["text"] += obj["text"]
        else:
            self.events.append(dict(obj))

    def broadcast(self, obj):
        for fn in list(self.listeners):
            try:
                fn(obj)
            except Exception:
                self.listeners.discard(fn)

    def put(self, ev):
        """Приёмник событий агента: пишет в историю чата и рассылает подписчикам."""
        obj = _event_to_json(ev)
        if obj is None:
            return
        finished = obj["type"] == "done"
        with self.lock:

            if finished and not self.running:
                return
            if finished:
                self.running = False
                self.updated = time.time()

            if obj.get("type") == "context" and isinstance(obj.get("info"), dict):
                self.context_info = obj["info"]
                self.broadcast(obj)
                return

            if obj.get("type") == "user" and obj.get("silent"):
                self.broadcast(obj)
                return
            self.record(obj)
            self.broadcast(obj)
        if finished:
            self.save()
            broadcast_chats()

    def save(self):
        try:
            with self.lock:
                data = {
                    "id": self.id, "title": self.title, "created": self.created,
                    "updated": self.updated, "workdir": str(self.workdir),
                    "workspace_root": str(self.workspace_root),
                    "model": self.model_name,
                    "messages": self.agent.messages, "events": self.events,
                    "context": self.context_info,
                    "effort": getattr(self, "effort", "normal"),
                }
            _save_json(CHAT_DIR / f"{self.id}.json", data)
        except Exception as e:
            print(f"[!] Не удалось сохранить чат {self.id}: {e}")

    @classmethod
    def load(cls, path: Path):
        d = _load_json(path, None)
        if not d or "id" not in d:
            return None
        cid = str(d["id"])

        saved_root = Path(d.get("workspace_root") or _chat_workspace(cid)).resolve()

        legacy = (WORKSPACE_DIR / re.sub(r"[^a-zA-Z0-9_-]", "", cid)).resolve()
        if not any(saved_root.iterdir()) if saved_root.is_dir() else True:
            if legacy.is_dir() and any(legacy.iterdir()):
                try:
                    _copy_tree(legacy, saved_root)
                except Exception:
                    pass
        saved_root.mkdir(parents=True, exist_ok=True)

        c = cls(cid)
        c.workspace_root = saved_root
        c.workdir = ACTIVE_WORKSPACE
        c.agent.set_workdir(c.workdir)
        c.title = d.get("title", "Чат")
        c.model_name = str(d.get("model") or config.MODEL_NAME)
        c.agent.set_model(c.model_name)
        c.created = d.get("created", time.time())
        c.updated = d.get("updated", c.created)
        c.agent.messages = d.get("messages") or c.agent.messages
        if c.agent.messages and c.agent.messages[0].get("role") == "system":
            c.agent.messages[0] = {"role": "system", "content": build_system_prompt(c.workdir)}
        c.events = d.get("events") or []
        last_user = max((i for i, e in enumerate(c.events) if e.get("type") == "user"), default=-1)
        if last_user != -1 and not any(e.get("type") == "done" for e in c.events[last_user:]):
            c.events.append({"type": "error", "text": "Ход был прерван перезапуском сервера"})
            c.events.append({"type": "done"})
        if isinstance(d.get("context"), dict) and d["context"]:
            c.context_info = d["context"]
        else:
            try:
                c.estimate_context()
            except Exception:
                c.context_info = {"used": 0, "limit": _model_context_size(c.model_name), "percent": 0}
        c.effort = str(d.get("effort") or "normal")
        root_id(c.workdir)
        return c

    def set_workdir(self, path: Path):
        """Любая существующая или новая папка на диске."""
        path = Path(path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise ValueError("Путь не является папкой")
        self.workdir = path
        self.agent.set_workdir(self.workdir)
        root_id(self.workdir)
        self.save()

    def set_model(self, model_name):
        model_name = str(model_name or "").strip()
        if not model_name:
            raise ValueError("Не указана модель")
        if self.running:
            raise RuntimeError("Дождитесь окончания ответа")
        with MODEL_RUNTIME_LOCK:
            _set_active_model(model_name)
            self.model_name = model_name
            self.agent.set_model(model_name)
            config.MODEL_NAME = model_name
            self.updated = time.time()
            self.save()
            try:
                _save_provider_settings()
            except Exception:
                pass
        return self.model_name

    def _prepare_files(self, files):
        out = []
        wd = self.workdir.resolve()
        rid = root_id(wd)
        for f in files or []:
            try:
                p = (wd / str(f.get("path", ""))).resolve()
                rel = p.relative_to(wd).as_posix()
                if not p.is_file():
                    continue
                out.append({"name": f.get("name") or p.name, "path": rel,
                            "size": p.stat().st_size,
                            "url": f"/files/{rid}/" + quote(rel)})
            except Exception:
                continue
        return out

    def start_turn(self, text: str, think: bool, files, mode: str = "", effort: str = "normal") -> bool:
        with self.lock:
            if self.running:
                return False
            self.running = True
            try:
                self.activate()
            except Exception as e:
                print(f"[!] activate: {e}")
            infos = self._prepare_files(files)
            if not (text or "").strip() and mode in ("regen", "continue"):
                _prev_users = [e for e in self.events if e.get("type") == "user"]
                text = str(_prev_users[-1].get("text") or "") if _prev_users else ""
            text = (text or "").strip() or (
                "Изучи прикреплённые файлы и расскажи, что в них." if infos
                else "Файлы не прикреплены. Напиши, что нужно сделать."
            )
            mode = str(mode or "").strip().lower()
            if effort:
                self.effort = str(effort).strip().lower()
            effort_cfg = resolve_effort(getattr(self, "effort", None) or effort or "normal")


            if mode == "regen":
                with self.agent.lock:
                    msgs = self.agent.messages

                    while len(msgs) > 1 and msgs[-1].get("role") != "user":
                        msgs.pop()

                last_user = max((i for i, e in enumerate(self.events) if e.get("type") == "user"), default=-1)
                if last_user >= 0:
                    self.events = self.events[: last_user + 1]

                if last_user >= 0:

                    if text and text.strip():
                        self.events[last_user]["text"] = text.strip()
                        with self.agent.lock:
                            for j in range(len(self.agent.messages) - 1, 0, -1):
                                if self.agent.messages[j].get("role") == "user":
                                    self.agent.messages[j]["content"] = text.strip()
                                    break
                    else:
                        text = self.events[last_user].get("text") or text
                infos = []

            if mode == "continue":
                text = text or "Продолжи с того места, где остановился. Не повторяй уже написанное."

            if self.title == "Новый чат" and mode not in ("continue", "regen"):
                self.title = _shorten(text)
            self.updated = time.time()
            self._turn_mode = mode
            self._turn_effort = effort_cfg


        silent = mode in ("continue", "regen")
        if silent:

            if mode == "continue":
                self.put(("user", text, infos, think, True))               
            else:

                pass
        else:
            self.put(("user", text, infos, think, False))
        self.save()
        broadcast_chats()
        model_text = text + attachments_note(self.workdir, infos)

        def run():
            lock = MODEL_RUNTIME_LOCK if _is_lmstudio() else contextlib.nullcontext()
            try:
                with lock:
                    real = _ensure_model(self.model_name)
                    self.model_name = real
                    self.agent.set_model(real)
                    self.agent.run_turn(
                        model_text, self,
                        think=think or effort_cfg.get("force_think"),
                        anchor=text,
                        max_iterations=int(effort_cfg.get("max_iterations") or MAX_ITERATIONS),
                        effort=effort_cfg,
                        mode=mode,
                    )
            except Exception as e:
                self.put(("error", "Ошибка запуска модели: " + _public_error(e)))
            finally:
                try:
                    self.agent._flush_files(self)
                except Exception:
                    pass
                try:
                    self.persist_files()
                except Exception:
                    pass
                if self.running:
                    self.put(("done",))

        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        return True

    def cancel(self):
        self.agent.cancel()
        def _ensure_stopped():
            time.sleep(0.5)
            with self.lock:
                if self.running and self.agent.cancel_flag.is_set():
                    self.put(("error", "[Отменено пользователем]"))
                    if self.running:
                        self.put(("done",))
        threading.Thread(target=_ensure_stopped, daemon=True).start()


def chat_list():
    with CHATS_LOCK:
        items = list(CHATS.values())
    items.sort(key=lambda c: c.updated, reverse=True)
    return [{"id": c.id, "title": c.title or "Новый чат", "updated": c.updated, "running": c.running}
            for c in items]


def broadcast_chats():
    lst = chat_list()
    with CLIENTS_LOCK:
        fns = list(CLIENTS)
    for fn in fns:
        try:
            fn({"type": "chats", "list": lst})
        except Exception:
            pass


def new_chat() -> Chat:
    """Создаёт изолированный чат с чистой историей и нулевым контекстом."""
    c = Chat()

    c.agent.messages = [{"role": "system", "content": build_system_prompt(c.workdir)}]
    c.events = []
    c.context_info = {
        "used": 0,
        "limit": _model_context_size(c.model_name),
        "percent": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    c.effort = "normal"
    with CHATS_LOCK:
        for k in [k for k, v in CHATS.items() if not v.events and not v.listeners and not v.running]:
            del CHATS[k]
        CHATS[c.id] = c
    return c


def load_all_chats():
    for f in CHAT_DIR.glob("*.json"):
        c = Chat.load(f)
        if c:
            CHATS[c.id] = c
