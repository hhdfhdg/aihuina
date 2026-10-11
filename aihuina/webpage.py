"""Загрузка HTML-интерфейса из файла web/index.html."""
from pathlib import Path


"""Загрузка HTML-интерфейса из файла web/index.html."""
from pathlib import Path

INDEX_HTML = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")
