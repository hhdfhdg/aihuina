"""Выполнение действий модели (инструменты, файлы, правки)."""
from .edit import _edit_error, _make_diff, apply_edit_request, edit_existing, tool_edit_file
from .file_tools import (
    _safe_path,
    tool_bash,
    tool_list_files,
    tool_move_file,
    tool_read_file,
    tool_send_file,
    tool_write_file,
)


def _short_arg(s, n=200) -> str:
    s = s if isinstance(s, str) else str(s)
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"… [{len(s)} симв.]"


def perform_action(kind: str, payload: dict) -> dict:
    payload = payload or {}
    if kind == "write":
        path = str(payload.get("path") or "")
        content = payload.get("content")
        content = content if isinstance(content, str) else ("" if content is None else str(content))
        existed = False
        old = None
        try:
            p = _safe_path(path)
            if p.is_file():
                existed = True
                if p.stat().st_size <= 200_000 and b"\x00" not in p.read_bytes()[:4096]:
                    old = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            existed = False
            old = None
        msg = tool_write_file(path, content)
        ok = msg.startswith("[OK]")
        diff, summary = "", ("новый файл" if not existed else "перезаписан целиком")
        if ok and existed:
            msg += " Файл перезаписан целиком. Для точечных правок используй <edit>."
        if ok and old is not None:
            try:
                new = _safe_path(path).read_text(encoding="utf-8", errors="replace")
                plus, minus, diff = _make_diff(old, new, limit_lines=80)
                summary = f"перезаписан целиком · +{plus} −{minus}"
            except Exception:
                diff = ""
        return {
            "ok": ok, "text": msg, "path": path if ok else None,
            "action": "edited" if existed else "created",
            "summary": summary, "diff": diff if ok else "",
            "show_file": ok,
            "log_name": "write_file",
            "log_args": {"path": path, "content": _short_arg(content, 240)},
        }
    if kind == "edit":
        return apply_edit_request(payload.get("path") or "", body=payload.get("body"), attrs=payload.get("attrs"))
    if kind == "append":
        return edit_existing(payload.get("path") or "", append=payload.get("body") or "")
    if kind == "insert":
        attrs = payload.get("attrs") or {}
        return apply_edit_request(
            payload.get("path") or attrs.get("path") or "",
            body=payload.get("body") or "",
            attrs=attrs,
        )
    name = str(payload.get("name") or "")
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    if payload.get("bad"):
        return _edit_error("", "[ОШИБКА] Не разобрал JSON инструмента: " + str(payload.get("bad")))
    if name in ("edit_file", "str_replace", "replace"):
        return apply_edit_request(
            args.get("path") or "",
            old=args.get("old", args.get("search")),
            new=args.get("new", args.get("replace")),
            content=args.get("content"),
            start=args.get("start"),
            end=args.get("end"),
            append=args.get("append"),
            after=args.get("after"),
            before=args.get("before"),
            lines=args.get("lines"),
        )
    if name == "write_file":
        return perform_action("write", {"path": args.get("path", ""), "content": args.get("content", "")})
    msg = execute_tool(name, args) if name else "[ОШИБКА] Пустой инструмент"
    if not isinstance(msg, str):
        msg = str(msg)
    ok = not (msg.startswith("[ОШИБКА]") or msg.startswith("[ЗАПРЕЩЕНО]"))
    file_ok = msg.startswith("[OK]") and name in ("send_file", "write_file")
    path = args.get("path") if isinstance(args.get("path"), str) else None
    log_args = dict(args)
    if isinstance(log_args.get("content"), str):
        log_args["content"] = _short_arg(log_args["content"], 240)
    if isinstance(log_args.get("command"), str) and len(log_args["command"]) > 300:
        log_args["command"] = log_args["command"][:300] + "…"
    return {
        "ok": ok,
        "text": msg,
        "path": path if file_ok else None,
        "action": "created",
        "summary": "",
        "diff": "",
        "show_file": bool(file_ok and path),
        "log_name": name or "tool",
        "log_args": log_args,
    }


TOOL_IMPL = {
    "bash":       tool_bash,
    "read_file":  tool_read_file,
    "write_file": tool_write_file,
    "edit_file":  tool_edit_file,
    "list_files": tool_list_files,
    "move_file":  tool_move_file,
    "send_file":  tool_send_file,
}


def execute_tool(name: str, args: dict) -> str:
    if name not in TOOL_IMPL:
        return f"[ОШИБКА] Неизвестный инструмент: {name}"
    if not isinstance(args, dict):
        args = {}
    try:
        return TOOL_IMPL[name](**args)
    except TypeError as e:
        return f"[ОШИБКА] Неверные аргументы для {name}: {e}"
    except Exception as e:
        return f"[ОШИБКА] {name}: {e}"
