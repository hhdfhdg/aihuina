"""HTTP-обработчик: статика, API загрузки/просмотра файлов, WebSocket."""
import re
import json
import shutil
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from urllib.parse import quote, unquote, urlparse, parse_qs
from . import config
from .agent import _shorten
from .chat import (
    CHATS,
    CHATS_LOCK,
    CLIENTS,
    CLIENTS_LOCK,
    Chat,
    broadcast_chats,
    chat_list,
    new_chat,
)
from .config import (
    ACTIVE_WORKSPACE,
    CHAT_DIR,
    CHAT_FILES_DIR,
    MAX_UPLOAD,
    PROVIDERS,
    WORKSPACE_DIR,
)
from .file_tools import ROOTS
from .providers import (
    MODEL_RUNTIME_LOCK,
    _available_models,
    _is_lmstudio,
    _is_ollama,
    _model_context_size,
    _provider_info_list,
    _public_error,
)
from .settings import DEFAULT_WORKDIR, _set_provider
from .utils import _inside
from .webpage import INDEX_HTML
from .websocket import ws_handshake, ws_recv_frame, ws_send_close, ws_send_text


def _sanitize_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ")
    name = re.sub(r"[\s()\[\]{}]+", "_", name).strip("_")
    return name or "file"


def _browse(path_str: str, root=None):
    """Список подпапок для окна выбора рабочей папки (не выходит за root)."""
    p = Path(path_str).expanduser() if path_str else DEFAULT_WORKDIR
    try:
        p = p.resolve()
    except Exception:
        return {"error": "Некорректный путь"}
    while not p.is_dir() and p.parent != p:
        p = p.parent
    dirs = []
    try:
        for d in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            try:
                if d.is_dir() and not d.name.startswith("."):
                    dirs.append({"name": d.name, "path": str(d)})
            except OSError:
                continue
    except PermissionError:
        return {"error": "Нет доступа к этой папке"}
    parent = str(p.parent) if p.parent != p else None
    if root is not None:
        root = Path(root)
    if root is not None and (parent is None or not _inside(root, Path(parent))):
        parent = None
    return {"path": str(p), "parent": parent, "dirs": dirs}


CONTENT_TYPES = {
    ".txt": "text/plain; charset=utf-8", ".md": "text/markdown; charset=utf-8",
    ".json": "application/json", ".csv": "text/csv; charset=utf-8",
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript", ".py": "text/x-python; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
    ".pdf": "application/pdf", ".zip": "application/zip",
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args): pass

    def _send_bytes(self, body: bytes, ctype: str, extra=None, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                         "application/json; charset=utf-8", code=code)

    def do_GET(self):
        u = urlparse(self.path)
        path = u.path

        if path in ("/", "/index.html") or path.startswith("/chat/"):
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8",
                             {"Cache-Control": "no-store"})
            return

        if path == "/info":
            self._json({
                "model": config.MODEL_NAME,
                "provider": config.PROVIDER,
                "provider_label": PROVIDERS.get(config.PROVIDER, {}).get("label", config.PROVIDER),
                "api_url": config.API_URL,
                "workspace": "per-chat",
                "model_switching": _is_lmstudio() or _is_ollama(),
                "context_limit": _model_context_size(config.MODEL_NAME),
                "providers": _provider_info_list(),
            })
            return

        if path == "/api/models":
            try:
                models = _available_models()
                self._json({
                    "models": models,
                    "active": config.MODEL_NAME,
                    "provider": config.PROVIDER,
                    "supported": True,
                    "model_switching": _is_lmstudio() or _is_ollama(),
                })
            except Exception as e:
                self._json({
                    "error": _public_error(e),
                    "models": [{"key": config.MODEL_NAME, "name": config.MODEL_NAME, "loaded": True}],
                    "active": config.MODEL_NAME,
                    "provider": config.PROVIDER,
                    "supported": False,
                }, 502)
            return

        if path == "/api/provider":
            self._json({
                "provider": config.PROVIDER,
                "api_url": config.API_URL,
                "model": config.MODEL_NAME,
                "providers": _provider_info_list(),
            })
            return

        if path == "/api/browse":
            qs = parse_qs(u.query)
            cid = str(qs.get("chat", [""])[0] or "")
            chat = CHATS.get(cid)
            if not chat:
                self._json({"error": "chat_not_found"}, 404)
                return

            requested = Path(unquote(qs.get("path", [""])[0] or str(ACTIVE_WORKSPACE))).expanduser().resolve()
            if not requested.exists():
                requested = ACTIVE_WORKSPACE.resolve()

            self._json(_browse(str(requested), requested.anchor if hasattr(requested, 'anchor') else str(requested.drive or '/')))
            return

        if path.startswith("/files/"):
            rid, _, rel = path[len("/files/"):].partition("/")
            root = ROOTS.get(rid)
            if not root or not rel:
                self.send_error(404); return
            try:
                p = (root / unquote(rel)).resolve()
                p.relative_to(root.resolve())
                if not p.is_file():
                    self.send_error(404); return
                data = p.read_bytes()
                self._send_bytes(data, CONTENT_TYPES.get(p.suffix.lower(), "application/octet-stream"), {
                    "Content-Disposition": f"attachment; filename*=UTF-8''{quote(p.name)}",
                    "Cache-Control": "no-store",
                })
            except ValueError:
                self.send_error(403)
            except Exception as e:
                self.send_error(500, str(e))
            return

        if path == "/ws":
            self._handle_ws(); return
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/provider":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                provider = str(body.get("provider") or "").strip().lower()
                api_url = body.get("api_url")
                model = body.get("model")
                api_key = body.get("api_key") if "api_key" in body else None
                clear_api_key = bool(body.get("clear_api_key"))
                if any(c.running for c in CHATS.values()):
                    self._json({"error": "busy", "message": "Дождитесь окончания ответа"}, 409)
                    return
                result = _set_provider(provider, api_url=api_url, model=model, api_key=api_key, clear_api_key=clear_api_key)

                with CHATS_LOCK:
                    for c in CHATS.values():
                        if not c.running:
                            c.model_name = config.MODEL_NAME
                            c.agent.set_model(config.MODEL_NAME)
                broadcast_chats()
                self._json({"ok": True, **result})
            except Exception as e:
                self._json({"error": _public_error(e)}, 500)
            return

        if u.path == "/api/model/select":
            try:
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n) if n > 0 else b"{}"
                body = json.loads(raw.decode("utf-8"))
                model = str(body.get("model") or "").strip()
                cid = str(body.get("chat") or "")
                chat = CHATS.get(cid)
                if not chat:
                    self._json({"error": "chat_not_found"}, 404); return

                with MODEL_RUNTIME_LOCK:
                    if chat.running:
                        self._json({"error": "chat_running", "message": "Дождитесь окончания ответа"}, 409); return
                    if any(c.running for c in CHATS.values()):
                        self._json({"error": "another_chat_running", "message": "Сейчас отвечает другой чат. Дождитесь окончания."}, 409); return
                    chat.set_model(model)

                chat.broadcast({"type": "model", "model": chat.model_name})
                broadcast_chats()
                self._json({"ok": True, "model": chat.model_name})
            except Exception as e:
                self._json({"error": _public_error(e)}, 500)
            return

        if u.path != "/upload":
            self.send_error(404); return
        qs = parse_qs(u.query)
        chat = CHATS.get(qs.get("chat", [""])[0])
        name = _sanitize_filename(unquote(qs.get("name", ["file"])[0]))
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            n = 0
        if not chat:
            self.close_connection = True
            self._json({"error": "chat_not_found"}, 404); return
        if n < 0 or n > MAX_UPLOAD:
            self.close_connection = True
            self._json({"error": "bad_size"}, 413); return
        try:
            updir = Path(chat.workspace_root)
            updir.mkdir(parents=True, exist_ok=True)
            dest = updir / name
            stem, suf, i = dest.stem, dest.suffix, 1
            while dest.exists():
                dest = updir / f"{stem}_{i}{suf}"; i += 1
            left = n
            with open(dest, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(65536, left))
                    if not chunk:
                        break
                    fh.write(chunk); left -= len(chunk)
            if left:
                dest.unlink(missing_ok=True)
                self.close_connection = True
                self._json({"error": "incomplete"}, 400); return
            try:
                tmp_dir = Path(chat.workdir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dest, tmp_dir / dest.name)
            except Exception:
                pass
            try:
                chat.persist_files()
            except Exception:
                pass
            self._json({"path": dest.name, "name": dest.name, "size": n})
        except Exception as e:
            self._json({"error": str(e)}, 500)


    def _handle_ws(self):
        if not ws_handshake(self): return
        sock = self.connection
        sock.settimeout(None)
        send_lock = threading.Lock()
        cur = {"chat": None}

        def send(obj):
            with send_lock:
                ws_send_text(sock, json.dumps(obj, ensure_ascii=False))

        def listener(obj):
            send(obj)

        def attach(chat: Chat):
            old = cur["chat"]
            if old is not None:
                old.listeners.discard(listener)


                if old is not chat:
                    try:
                        old.persist_files()
                    except Exception:
                        pass
                    try:
                        old.save()
                    except Exception:
                        pass
                    old._active = False
            cur["chat"] = chat
            try:
                chat.activate()
            except Exception as e:
                print(f"[!] activate on open: {e}")
            try:
                chat.estimate_context()
            except Exception:
                pass
            with chat.lock:
                send({"type": "opened", "chat": chat.info(), "events": list(chat.events)})
                chat.listeners.add(listener)

        with CLIENTS_LOCK:
            CLIENTS.add(send)

        try:
            send({"type": "chats", "list": chat_list()})
            while True:
                opcode, payload = ws_recv_frame(sock)
                if opcode is None: break
                if opcode == 0x8:
                    ws_send_close(sock); break
                if opcode == 0x9:
                    try:
                        with send_lock:
                            sock.sendall(b"\x8a\x00")
                    except Exception:
                        break
                    continue
                if opcode != 0x1: continue

                try:
                    msg = json.loads(payload.decode("utf-8"))
                except Exception:
                    continue
                mtype = msg.get("type")
                chat = cur["chat"]

                if mtype == "open":
                    target = CHATS.get(str(msg.get("id") or ""))
                    attach(target if target else new_chat())
                    send({"type": "chats", "list": chat_list()})

                elif mtype == "new":
                    attach(new_chat())

                elif mtype == "cancel" and chat:
                    chat.cancel()
                elif mtype == "workdir" and chat:
                    path = str(msg.get("path") or "").strip()
                    sub = str(msg.get("sub") or "").strip()
                    if msg.get("create") and sub:
                        path = str(Path(path) / sub) if path else sub
                    if path:
                        try:
                            chat.set_workdir(Path(path))
                            send({"type": "workdir", "path": str(chat.workdir), "chat": chat.info()})
                        except Exception as e:
                            send({"type": "error", "text": "Папка: " + str(e)})

                elif mtype == "chat" and chat:
                    text = str(msg.get("text", "")).strip()
                    files = msg.get("files") if isinstance(msg.get("files"), list) else []
                    mode = str(msg.get("mode") or "").strip().lower()
                    effort = str(msg.get("effort") or "normal").strip()
                    if not text and not files and mode not in ("continue", "regen"):
                        continue
                    ok = chat.start_turn(
                        text, bool(msg.get("think", False)), files,
                        mode=mode, effort=effort,
                    )
                    if not ok:
                        send({"type": "toast", "text": "Подождите, модель ещё отвечает"})

                elif mtype == "set_workdir" and chat:
                    if chat.running:
                        send({"type": "toast", "text": "Дождитесь окончания ответа"}); continue
                    try:
                        base = Path(str(msg.get("path", ""))).expanduser()
                        if msg.get("sub"):
                            sub = str(msg["sub"]).strip()
                            if not sub or re.search(r'[\\/:*?"<>|]', sub) or sub in (".", ".."):
                                send({"type": "toast", "text": "Недопустимое имя папки"}); continue
                            base = base / sub
                            if msg.get("create"):
                                base.mkdir(parents=True, exist_ok=True)
                        if not base.is_dir():
                            send({"type": "toast", "text": "Папка не существует"}); continue
                        chat.set_workdir(base)
                        chat.broadcast({"type": "workdir", "path": str(chat.workdir)})
                    except Exception as e:
                        send({"type": "toast", "text": f"Не удалось выбрать папку: {e}"})

                elif mtype == "rename":
                    target = CHATS.get(str(msg.get("id") or ""))
                    title = _shorten(str(msg.get("title", "")), 80)
                    if target and title:
                        target.title = title
                        target.save()
                        broadcast_chats()

                elif mtype == "delete":
                    cid = str(msg.get("id") or "")
                    target = CHATS.get(cid)
                    if target:
                        target.cancel()
                        with CHATS_LOCK:
                            CHATS.pop(cid, None)
                        try: (CHAT_DIR / f"{cid}.json").unlink(missing_ok=True)
                        except Exception: pass
                        try:
                            if _inside(CHAT_FILES_DIR, target.workspace_root) or _inside(WORKSPACE_DIR, target.workspace_root):
                                shutil.rmtree(target.workspace_root, ignore_errors=True)
                        except Exception: pass
                        if chat is target:
                            cur_chat = new_chat()
                            attach(cur_chat)
                        broadcast_chats()
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            with CLIENTS_LOCK:
                CLIENTS.discard(send)
            if cur["chat"] is not None:
                cur["chat"].listeners.discard(listener)
            try: sock.close()
            except Exception: pass
