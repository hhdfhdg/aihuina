"""Рабочие папки чатов и синхронизация с активным рабочим пространством."""
import os
import re
import uuid
import shutil
from pathlib import Path
from .config import ACTIVE_WORKSPACE, CHAT_FILES_DIR
from .file_tools import root_id


def _chat_workspace(chat_id: str) -> Path:
    """Папка постоянного хранения файлов чата (не туда, где работает модель)."""
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(chat_id)) or uuid.uuid4().hex[:12]
    p = (CHAT_FILES_DIR / safe_id).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _clear_dir(path: Path):
    """Очищает содержимое директории, саму папку оставляет."""
    path = Path(path)
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


def _copy_tree(src: Path, dst: Path):
    """Копирует все файлы/папки из src в dst (без скрытых служебных)."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        rel = Path(root).relative_to(src)
        target_root = dst / rel
        target_root.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name.startswith("."):
                continue
            s = Path(root) / name
            d = target_root / name
            try:
                shutil.copy2(s, d)
            except Exception:
                try:
                    d.write_bytes(s.read_bytes())
                except Exception:
                    pass


def sync_chat_to_active(chat_id: str):
    """При открытии чата: файлы чата → временный Workspace (модель работает здесь)."""
    storage = _chat_workspace(chat_id)
    ACTIVE_WORKSPACE.mkdir(parents=True, exist_ok=True)
    _clear_dir(ACTIVE_WORKSPACE)
    _copy_tree(storage, ACTIVE_WORKSPACE)
    root_id(ACTIVE_WORKSPACE)
    return ACTIVE_WORKSPACE


def sync_active_to_chat(chat_id: str):
    """После работы: временный Workspace → постоянная папка чата."""
    storage = _chat_workspace(chat_id)
    storage.mkdir(parents=True, exist_ok=True)
    _clear_dir(storage)
    _copy_tree(ACTIVE_WORKSPACE, storage)
    return storage
