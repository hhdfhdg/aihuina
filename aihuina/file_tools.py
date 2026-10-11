"""Инструменты для работы с файлами и командами оболочки."""
import os
import re
import hashlib
import shutil
import threading
import subprocess
from pathlib import Path
from .config import BLOCKED_PATTERNS, CMD_TIMEOUT, MAX_OUTPUT_CHARS, ROOTS_FILE
from .settings import DEFAULT_WORKDIR
from .utils import _load_json, _save_json


_ctx = threading.local()


def _wd() -> Path:
    return getattr(_ctx, "workdir", None) or DEFAULT_WORKDIR


ROOTS: dict = {}
_roots_lock = threading.Lock()


def root_id(root: Path) -> str:
    root = Path(root).resolve()
    rid = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:10]
    with _roots_lock:
        if rid not in ROOTS:
            ROOTS[rid] = root
            try:
                _save_json(ROOTS_FILE, sorted(str(v) for v in ROOTS.values()))
            except Exception:
                pass
    return rid


for _r in _load_json(ROOTS_FILE, []):
    try:
        if Path(_r).is_dir():
            root_id(Path(_r))
    except Exception:
        pass


def _safe_path(rel_path: str) -> Path:
    """Резолвит путь внутри рабочей папки и защищает от выхода наружу."""
    if not rel_path or not isinstance(rel_path, str):
        raise ValueError("Пустой путь")
    wd = _wd().resolve()
    p = (wd / rel_path).resolve()
    try:
        p.relative_to(wd)
    except ValueError:
        raise ValueError(f"Путь '{rel_path}' выходит за пределы рабочей директории")
    return p


def _norm_name(s: str) -> str:
    return re.sub(r"[\s_()\-]+", "", s.lower())


def _resolve_file(rel_path: str) -> Path:
    """Путь к файлу; если точного нет — ищет похожее имя (index.html -> uploads/index (6).html)."""
    p = _safe_path(rel_path)
    if p.is_file():
        return p
    try:
        wd = _wd().resolve()
        want = Path(str(rel_path).replace("\\", "/")).name
        wn, ws, wext = _norm_name(want), _norm_name(Path(want).stem), Path(want).suffix.lower()
        cands = []
        for f in wd.rglob("*"):
            try:
                if f.is_file() and not any(x.startswith(".") for x in f.relative_to(wd).parts):
                    cands.append(f)
            except Exception:
                continue
        pool = [f for f in cands if _norm_name(f.name) == wn]
        if not pool and ws:
            pool = [f for f in cands if f.suffix.lower() == wext and _norm_name(f.stem).startswith(ws)]
        if pool:
            pool.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            return pool[0]
    except Exception:
        pass
    return p


_CASUAL_RE = re.compile(
    r"^(привет|здравствуй|здравствуйте|добрый\s+(день|вечер|утро)|хай|hello|hi|hey|"
    r"как\s+дела|что\s+нового|спасибо|благодарю|пока|до\s+свидания|"
    r"ты\s+кто|кто\s+ты|что\s+ты\s+умеешь|help|помощь)\b",
    re.I,
)


def _is_casual_chat(text: str) -> bool:
    """True если сообщение — приветствие / small-talk без задачи на код."""
    t = (text or "").strip()
    if not t:
        return False
    low = t.lower()
    task_words = (
        "создай", "напиши", "сделай", "исправ", "почин", "добав", "измен", "удал",
        "файл", "код", "скрипт", "программ", "сайт", "игру", "игрушку", "лендинг",
        "функци", "класс", "html", "python", "js", "css", "запусти", "прочитай",
        "открой", "скачай", "отправь", "пришли", "переимен", "рефактор", "тест", "баг", "ошибк",
    )
    if any(w in low for w in task_words):
        return False
    if _CASUAL_RE.search(t):
        return True
    words = re.findall(r"[\wа-яё]+", low, re.I)
    return len(words) <= 3 and len(t) < 40


def _truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[обрезано, всего {len(text)} символов]"


def _looks_broken(cmd: str) -> bool:
    for a, b in (("(", ")"), ("[", "]"), ("{", "}"), ('"', '"'), ("'", "'")):
        if cmd.count(a) != cmd.count(b):
            return True
    return False


def tool_bash(command: str, timeout: int = CMD_TIMEOUT) -> str:
    if not isinstance(command, str) or not command.strip():
        return "[ОШИБКА] Пустая команда"
    stripped = command.strip()

    if stripped.startswith(("python -c", "python3 -c")) and _looks_broken(stripped):
        return ("[ОШИБКА] Команда обрезана или содержит несбалансированные кавычки. "
                "Не используй `python -c` для многострочного кода — "
                "создай .py файл через <write> и запусти `python имя_файла.py`.")

    low = command.lower()
    for bad in BLOCKED_PATTERNS:
        if bad.lower() in low:
            return f"[ЗАПРЕЩЕНО] Команда содержит запрещённый шаблон: {bad}"

    try:
        timeout = int(timeout) if timeout else CMD_TIMEOUT
        timeout = max(1, min(timeout, 300))
        result = subprocess.run(
            command, shell=True, cwd=str(_wd()),
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        parts = [f"exit_code: {result.returncode}"]
        if result.stdout: parts.append(f"stdout:\n{result.stdout}")
        if result.stderr: parts.append(f"stderr:\n{result.stderr}")
        return _truncate("\n".join(parts))
    except subprocess.TimeoutExpired:
        return f"[ОШИБКА] Таймаут {timeout} сек"
    except Exception as e:
        return f"[ОШИБКА] {e}"


def _number_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text == "":
        return "(пустой файл, 0 строк)"
    raw = text[:-1] if text.endswith("\n") else text
    lines = raw.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}| {ln}" for i, ln in enumerate(lines, 1))


def tool_read_file(path: str) -> str:
    try:
        p = _resolve_file(path)
        if not p.is_file():
            return f"[ОШИБКА] Файл не найден: {path}. Список файлов:\n{tool_list_files()}"
        if p.stat().st_size > 1_000_000:
            return f"[ОШИБКА] Файл слишком большой (>1 МБ): {path}"
        body = p.read_text(encoding="utf-8", errors="replace")
        numbered = _number_lines(body)
        return _truncate(
            "Строки пронумерованы как N| — это только для ориентира. "
            "В SEARCH копируй код БЕЗ префикса N|.\n" + numbered
        )
    except Exception as e:
        return f"[ОШИБКА] {e}"


_CSS_QUOTED_NUM_RE = re.compile(r'(:\s*)"([-+.]?\d[^"\n;{}]*)"')
_STYLE_BLOCK_RE = re.compile(r'(<style\b[^>]*>)([\s\S]*?)(</style>)', re.I)


def _fix_css_quotes(text: str) -> str:
    """Убирает кавычки у числовых CSS-значений внутри <style>: height: \"100vh\" -> height: 100vh."""
    def fix_block(m):
        return m.group(1) + _CSS_QUOTED_NUM_RE.sub(lambda q: q.group(1) + q.group(2), m.group(2)) + m.group(3)
    return _STYLE_BLOCK_RE.sub(fix_block, text)


def tool_write_file(path: str, content: str = "") -> str:
    try:
        p = _safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = content if isinstance(content, str) else str(content)
        if str(path).lower().endswith((".html", ".htm")):
            text = _fix_css_quotes(text)
        p.write_text(text, encoding="utf-8")
        return f"[OK] Записано {len(text)} символов в {path}"
    except Exception as e:
        return f"[ОШИБКА] {e}"


def tool_list_files(subdir: str = "") -> str:
    try:
        wd = _wd()
        base = _safe_path(subdir) if subdir else wd
        if not base.exists():
            return f"[ОШИБКА] Директория не найдена: {subdir}"
        lines = []
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            depth = len(Path(root).relative_to(base).parts)
            if depth > 3:
                dirs[:] = []
                continue
            indent = "  " * depth
            rel = Path(root).relative_to(base)
            lines.append(f"{indent}{rel if str(rel) != '.' else '.'}/")
            for f in sorted(files):
                fp = Path(root) / f
                try:
                    size = fp.stat().st_size
                except Exception:
                    size = 0
                lines.append(f"{indent}  {f}  ({size} байт)")
            if len(lines) > 300:
                lines.append("...[список обрезан]")
                break
        return "\n".join(lines) if lines else "(пусто)"
    except Exception as e:
        return f"[ОШИБКА] {e}"


def tool_move_file(source: str, destination: str, overwrite: bool = False) -> str:
    """Перемещает или переименовывает файл внутри рабочей папки."""
    try:
        src = _safe_path(source)
        dst = _safe_path(destination)
        if not src.exists() or not src.is_file():
            return f"[ОШИБКА] Исходный файл не найден: {source}"
        if src == dst:
            return f"[OK] Файл уже находится по этому пути: {destination}"
        if dst.exists() and not overwrite:
            return (f"[ОШИБКА] Назначение уже существует: {destination}. "
                    "Укажи другое имя или передай overwrite=true, если нужно заменить файл.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"[OK] Файл перемещён: {source} -> {destination} ({dst.stat().st_size} байт)"
    except Exception as e:
        return f"[ОШИБКА] Не удалось переместить файл: {e}"


def tool_send_file(path: str) -> str:
    try:
        p = _resolve_file(path)
        if not p.is_file():
            name = Path(path).name
            found = [str(f.relative_to(_wd().resolve())).replace("\\", "/")
                     for f in _wd().rglob(name) if f.is_file()][:5]
            if found:
                return (f"[ОШИБКА] Файл не найден по пути '{path}', "
                        f"но есть такой: {', '.join(found)}. Повтори send_file с верным путём.")
            return (f"[ОШИБКА] Файл '{path}' не существует. Сначала создай его через <write>, "
                    f"потом повтори send_file. Сейчас в рабочей папке:\n{tool_list_files()}")
        return f"[OK] Файл подготовлен к отправке: {path} ({p.stat().st_size} байт)"
    except Exception as e:
        return f"[ОШИБКА] {e}"
