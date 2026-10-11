"""Движок точечного редактирования файлов (hunks, строки, замены)."""
import re
import difflib
from pathlib import Path
from .file_tools import _resolve_file, _wd


class EditError(Exception):
    pass


_HUNK_RE = re.compile(
    r"<{3,}\s*SEARCH[^\n]*\n(.*?)\n={3,}[^\n]*\n(.*?)>{3,}[^\n]*",
    re.DOTALL | re.IGNORECASE,
)
_OLDNEW_RE = re.compile(
    r"<old>(.*?)</old>\s*<new>(.*?)</new>",
    re.DOTALL | re.IGNORECASE,
)


def _attrs(s: str) -> dict:
    return {
        k.lower(): v
        for k, v in re.findall(r"""([A-Za-z_]+)\s*=\s*["']([^"']*)["']""", s or "")
    }


def _guess_write_path(attr_str: str, content: str = "") -> str:
    """path из атрибутов <write> или эвристика по содержимому."""
    attrs = _attrs(attr_str or "")
    path = (attrs.get("path") or attrs.get("file") or attrs.get("name") or "").strip()
    if path:
        return path
    low = (content or "")[:800].lower()
    if "<html" in low or "<!doctype" in low:
        return "index.html"
    if "def " in low or "import " in low or "class " in low:
        return "main.py"
    if "{" in low and "function" in low:
        return "script.js"
    return "file.txt"


def _norm_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def _strip_wrap(s: str) -> str:
    s = _norm_newlines(s)
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


def _strip_num_prefix(block: str):
    """Убирает префиксы '12| ' если они стоят на КАЖДОЙ строке."""
    lines = block.split("\n")
    out = []
    for ln in lines:
        m = re.match(r"^\s*\d+\|\s?(.*)$", ln)
        if not m:
            return None
        out.append(m.group(1))
    return "\n".join(out) if out else None


def _split_keep(text: str):
    text = _norm_newlines(text)
    if text == "":
        return [], False
    trailing = text.endswith("\n")
    lines = text.split("\n")
    if trailing:
        lines = lines[:-1]
    return lines, trailing


def _join_lines(lines, trailing: bool) -> str:
    s = "\n".join(lines)
    if trailing and s and not s.endswith("\n"):
        s += "\n"
    return s


def _line_keys(lines):
    return [ln.replace("\t", "    ").rstrip() for ln in lines]


def _dedent_key(lines):
    expanded = [ln.replace("\t", "    ") for ln in lines]
    inds = []
    for ln in expanded:
        if ln.strip():
            inds.append(len(ln) - len(ln.lstrip(" ")))
    base = min(inds) if inds else 0
    key = []
    for ln in expanded:
        if not ln.strip():
            key.append("")
        else:
            key.append(ln[base:] if len(ln) >= base else ln.lstrip(" "))
    return key, base


def _closest_snippet(original: str, old: str) -> str:
    o_lines = original.splitlines()
    q_lines = [ln.rstrip() for ln in old.splitlines() if ln.strip()]
    if not q_lines or not o_lines or len(o_lines) > 4000:
        return ""
    n = min(len(q_lines), 6)
    query = "\n".join(q_lines[:n])
    best, best_i = 0.0, 0
    for i in range(max(1, len(o_lines) - n + 1)):
        window = "\n".join(ln.rstrip() for ln in o_lines[i:i + n])
        score = difflib.SequenceMatcher(None, query, window).ratio()
        if score > best:
            best, best_i = score, i
    if best < 0.55:
        return ""
    snippet = "\n".join(o_lines[best_i:best_i + n])
    return (
        f"Похожий фрагмент (строки {best_i + 1}–{best_i + n}, сходство {best:.0%}):\n"
        f"{snippet[:900]}"
    )


def _replace_once(text: str, old: str, new: str, label: int = 1) -> str:
    text = _norm_newlines(text)
    old_n = _strip_wrap(old)
    new_n = _strip_wrap(new)
    cands = [old_n]
    numbered = _strip_num_prefix(old_n)
    if numbered is not None and numbered != old_n:
        cands.append(numbered)
    new_numbered = _strip_num_prefix(new_n)
    if all(c == "" for c in cands):
        raise EditError(f"[ОШИБКА] Блок {label}: пустой фрагмент SEARCH.")

    def _finish(updated: str) -> str:
        if text.endswith("\n") and updated and not updated.endswith("\n"):
            updated += "\n"
        return updated

    for cand in cands:
        if cand == "":
            continue
        positions = []
        start = 0
        while True:
            i = text.find(cand, start)
            if i < 0:
                break
            positions.append(i)
            start = i + 1
        if len(positions) > 1:
            raise EditError(
                f"[ОШИБКА] Блок {label}: фрагмент встречается {len(positions)} раз. "
                "Добавь соседние строки, чтобы он был уникальным."
            )
        if len(positions) == 1:
            i = positions[0]
            use_new = new_n
            if cand != old_n and new_numbered is not None:
                use_new = new_numbered
            return _finish(text[:i] + use_new + text[i + len(cand):])

    content_lines, trailing = _split_keep(text)
    old_lines = old_n.split("\n")
    n = len(old_lines)
    keys = _line_keys(old_lines)
    src_keys = _line_keys(content_lines)
    hits = []
    if n and n <= len(src_keys):
        for i in range(len(src_keys) - n + 1):
            if src_keys[i:i + n] == keys:
                hits.append(i)
    new_lines = [] if new_n == "" else new_n.split("\n")

    def _apply_at(i, repl_lines):
        updated = content_lines[:i] + repl_lines + content_lines[i + n:]
        return _finish(_join_lines(updated, trailing))

    if len(hits) == 1:
        return _apply_at(hits[0], new_lines)
    if len(hits) > 1:
        raise EditError(
            f"[ОШИБКА] Блок {label}: фрагмент встречается {len(hits)} раз. "
            "Добавь соседние строки, чтобы он был уникальным."
        )

    def _lead_len(s: str) -> int:
        exp = s.replace("\t", "    ")
        return len(exp) - len(exp.lstrip(" "))

    def _reindent(window, search_lines, repl_lines):
        if not repl_lines:
            return []
        out = []
        for i, nl in enumerate(repl_lines):
            if not str(nl).strip():
                out.append("")
                continue
            src = window[i] if i < len(window) else (window[-1] if window else "")
            raw_lead = src[:len(src) - len(src.lstrip(" \t"))]
            search = search_lines[i] if i < len(search_lines) else ""
            extra = max(0, _lead_len(nl) - _lead_len(search))
            out.append(raw_lead + (" " * extra) + nl.lstrip())
        return out

    want = [ln.replace("\t", "    ").strip() for ln in old_lines]
    hits = []
    if n and n <= len(content_lines):
        for i in range(len(content_lines) - n + 1):
            window = content_lines[i:i + n]
            got = [ln.replace("\t", "    ").strip() for ln in window]
            if got == want:
                hits.append(i)
    if len(hits) == 1:
        i = hits[0]
        return _apply_at(i, _reindent(content_lines[i:i + n], old_lines, new_lines))
    if len(hits) > 1:
        raise EditError(
            f"[ОШИБКА] Блок {label}: похожий фрагмент встречается {len(hits)} раз. "
            "Добавь больше контекста."
        )

    old_key, _ = _dedent_key(old_lines)
    hits = []
    bases = []
    if n and n <= len(content_lines):
        for i in range(len(content_lines) - n + 1):
            window = content_lines[i:i + n]
            wkey, base = _dedent_key(window)
            if wkey == old_key:
                hits.append(i)
                bases.append((base, window))
    if len(hits) == 1:
        i = hits[0]
        base, window = bases[0]
        new_rel, _ = _dedent_key(new_lines or [""])
        if new_n == "":
            new_rel = []
        uses_tabs = any(ln.startswith("\t") for ln in window if ln.strip())
        if uses_tabs and base % 4 == 0:
            indent = "\t" * (base // 4)
        else:
            indent = " " * base
        repl = [(indent + ln) if ln.strip() else "" for ln in new_rel]
        return _apply_at(i, repl)
    if len(hits) > 1:
        raise EditError(
            f"[ОШИБКА] Блок {label}: похожий фрагмент встречается {len(hits)} раз. "
            "Добавь больше контекста."
        )

    hint = _closest_snippet(text, old_n)
    msg = (
        f"[ОШИБКА] Блок {label}: фрагмент не найден. Скопируй его из файла дословно, "
        "без номеров строк N| и без изменения отступов."
    )
    if hint:
        msg += "\n" + hint
    raise EditError(msg)


def _as_int(v, label: str) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        raise EditError(f"[ОШИБКА] {label} должен быть номером строки, получено: {v}")


def _parse_span(spec: str, n_lines: int):
    spec = str(spec).strip().replace("–", "-").replace("—", "-").replace(":", "-")
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", spec)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
    else:
        m = re.fullmatch(r"(\d+)", spec)
        if not m:
            raise EditError(f"[ОШИБКА] Непонятный диапазон строк: {spec}. Пример: 10-14")
        a = b = int(m.group(1))
    if a < 1 or b < a:
        raise EditError(f"[ОШИБКА] Неверный диапазон строк: {spec}")
    if n_lines == 0 or a > n_lines:
        raise EditError(f"[ОШИБКА] В файле {n_lines} строк, а запрошена строка {a}.")
    return a, min(b, n_lines)


def _make_diff(old: str, new: str, limit_lines: int = 160):
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=2))
    plus = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
    minus = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
    body = [ln for ln in diff if not ln.startswith("---") and not ln.startswith("+++")]
    if len(body) > limit_lines:
        body = body[:limit_lines] + ["… [diff обрезан]"]
    return plus, minus, "\n".join(body)


def _read_existing(path: str):
    if not path or not isinstance(path, str):
        raise EditError("[ОШИБКА] Не указан path.")
    p = _resolve_file(path)
    if not p.exists() or not p.is_file():
        name = Path(path).name
        found = []
        try:
            found = [str(f.relative_to(_wd().resolve())).replace("\\", "/")
                     for f in _wd().rglob(name) if f.is_file()][:5]
        except Exception:
            found = []
        hint = f" Похожие: {', '.join(found)}." if found else ""
        raise EditError(
            f"[ОШИБКА] Файл '{path}' не найден.{hint} "
            "Новый файл создавай через <write>, существующий правь через <edit>."
        )
    if p.stat().st_size > 1_000_000:
        raise EditError(f"[ОШИБКА] Файл слишком большой для правки (>1 МБ): {path}")
    data = p.read_bytes()
    if b"\x00" in data[:8192]:
        raise EditError(f"[ОШИБКА] '{path}' похож на бинарный. Точечно правятся только текстовые файлы.")
    return p, data.decode("utf-8", errors="replace")


def _edit_result(path: str, old: str, new: str, action: str, word: str) -> dict:
    base = {
        "path": path, "action": action, "log_name": "edit_file",
        "log_args": {"path": path},
    }
    if old == new:
        base.update({
            "ok": True, "changed": False, "summary": "без изменений", "diff": "",
            "show_file": False,
            "text": f"[OK] {path}: текст не изменился — правка совпала с текущим содержимым.",
        })
        return base
    plus, minus, diff = _make_diff(old, new)
    summary = f"{word} · +{plus} −{minus}"
    short = diff if len(diff) <= 1800 else diff[:1800] + "\n… [diff обрезан]"
    text = f"[OK] {path}: {summary}"
    if short.strip():
        text += f"\n<diff>\n{short}\n</diff>"
    base.update({
        "ok": True, "changed": True, "summary": summary, "diff": diff,
        "show_file": True, "text": text, "log_args": {"path": path, "summary": summary},
    })
    return base


def _edit_error(path: str, text: str) -> dict:
    if not str(text).startswith("["):
        text = "[ОШИБКА] " + str(text)
    return {
        "ok": False, "changed": False, "path": path, "action": "edited",
        "summary": "", "diff": "", "text": str(text), "show_file": False,
        "log_name": "edit_file", "log_args": {"path": path},
    }


def edit_existing(path: str, *, hunks=None, lines=None, content=None,
                  after=None, before=None, append=None) -> dict:
    try:
        p, original = _read_existing(path)
        try:
            path = p.relative_to(_wd().resolve()).as_posix()
        except Exception:
            pass
        text = original
        if append is not None:
            chunk = _strip_wrap(append if isinstance(append, str) else str(append))
            if chunk == "":
                raise EditError("[ОШИБКА] Нечего дописывать: пустой блок.")
            if text and not text.endswith("\n"):
                text += "\n"
            text = text + chunk + "\n"
            action, word = "appended", "дописано"
        elif after is not None or before is not None:
            file_lines, trailing = _split_keep(text)
            new_n = _strip_wrap(content or "")
            new_lines = [] if new_n == "" else new_n.split("\n")
            if after is not None:
                n = _as_int(after, "after")
                if n < 0:
                    raise EditError("[ОШИБКА] after не может быть отрицательным. 0 — вставить в начало.")
                idx = min(n, len(file_lines))
                word = f"вставлено после строки {n}"
            else:
                n = _as_int(before, "before")
                if n < 1:
                    raise EditError("[ОШИБКА] before должен быть ≥ 1.")
                idx = min(max(n - 1, 0), len(file_lines))
                word = f"вставлено перед строкой {n}"
            updated = file_lines[:idx] + new_lines + file_lines[idx:]
            text = "\n".join(updated)
            if trailing or original.endswith("\n") or not original:
                if text and not text.endswith("\n"):
                    text += "\n"
            action = "edited"
        elif lines:
            file_lines, trailing = _split_keep(text)
            a, b = _parse_span(str(lines), len(file_lines))
            new_n = _strip_wrap(content or "")
            new_lines = [] if new_n == "" else new_n.split("\n")
            updated = file_lines[:a - 1] + new_lines + file_lines[b:]
            text = "\n".join(updated)
            if trailing and text and not text.endswith("\n"):
                text += "\n"
            action, word = "edited", f"строки {a}–{b}"
        elif hunks:
            for i, pair in enumerate(hunks, 1):
                old, new = pair[0], pair[1] if len(pair) > 1 else ""
                text = _replace_once(text, old, new if isinstance(new, str) else str(new), i)
            action, word = "edited", "заменено"
        else:
            raise EditError(
                "[ОШИБКА] В правке нет изменений. Нужен блок "
                "<<<<<<< SEARCH / ======= / >>>>>>> REPLACE, "
                'либо lines="10-14", либо <append> / <insert>.'
            )
        if text != original:
            p.write_text(text, encoding="utf-8")
        return _edit_result(path, original, text, action, word)
    except EditError as e:
        return _edit_error(path, str(e))
    except Exception as e:
        return _edit_error(path, f"[ОШИБКА] {e}")


def _hunks_from_body(body: str):
    hunks = [(a, b) for a, b in _HUNK_RE.findall(body or "")]
    if hunks:
        return hunks
    return [(a, b) for a, b in _OLDNEW_RE.findall(body or "")]


def apply_edit_request(path: str = "", *, body=None, attrs=None, old=None, new=None,
                       content=None, start=None, end=None, append=None,
                       after=None, before=None, lines=None) -> dict:
    attrs = dict(attrs or {})
    path = (path or attrs.get("path") or "").strip()
    try:
        if body is not None:
            hunks = _hunks_from_body(body)
            spec = attrs.get("lines")
            if spec:
                return edit_existing(path, lines=spec, content=body)
            if attrs.get("after") is not None and str(attrs.get("after")) != "":
                return edit_existing(path, after=attrs.get("after"), content=body)
            if attrs.get("before") is not None and str(attrs.get("before")) != "":
                return edit_existing(path, before=attrs.get("before"), content=body)
            return edit_existing(path, hunks=hunks)
        if append is not None:
            return edit_existing(path, append=append if isinstance(append, str) else str(append))
        if after is not None or before is not None:
            return edit_existing(
                path, after=after, before=before,
                content=content if content is not None else (new or ""),
            )
        if lines or start is not None:
            if start is not None:
                a = _as_int(start, "start")
                b = _as_int(end, "end") if end is not None else a
                lines = f"{a}-{b}"
            return edit_existing(
                path, lines=str(lines),
                content=content if content is not None else (new or ""),
            )
        if old is not None:
            return edit_existing(path, hunks=[(old, "" if new is None else new)])
        return _edit_error(
            path,
            "[ОШИБКА] edit_file: укажи old и new, либо lines и content, "
            "либо append, либо after/before и content.",
        )
    except EditError as e:
        return _edit_error(path, str(e))


def tool_edit_file(path: str, old: str = None, new: str = None, content: str = None,
                   start=None, end=None, append=None, after=None, before=None,
                   lines: str = None) -> str:
    return apply_edit_request(
        path, old=old, new=new, content=content, start=start, end=end,
        append=append, after=after, before=before, lines=lines,
    )["text"]
