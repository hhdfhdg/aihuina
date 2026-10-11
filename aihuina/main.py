"""Точка входа: баннер, запуск веб-сервера или CLI."""
import io
import sys
from http.server import ThreadingHTTPServer
from . import config
from .chat import CHATS, load_all_chats
from .cli import cli_main
from .config import CHAT_DIR, PROVIDERS, WEB_HOST, WEB_PORT
from .server import Handler
from .settings import DEFAULT_WORKDIR


def banner():
    print("╔══════════════════════════════════════════════════╗", flush=True)
    print("║      DeepSeek-style Agent  ·  веб-чат            ║", flush=True)
    print("╚══════════════════════════════════════════════════╝", flush=True)
    print(f"  URL:     http://127.0.0.1:{WEB_PORT}", flush=True)
    print(f"  Хост:    {WEB_HOST}", flush=True)
    print(f"  API:     {config.API_URL}", flush=True)
    print(f"  Провайдер: {PROVIDERS.get(config.PROVIDER, {}).get('label', config.PROVIDER)}", flush=True)
    print(f"  Модель:  {config.MODEL_NAME}", flush=True)
    print(f"  Папка:   {DEFAULT_WORKDIR}", flush=True)
    print(f"  Чаты:    {CHAT_DIR}", flush=True)
    print("", flush=True)
    print("  Ctrl+C — остановить", flush=True)
    print("", flush=True)

def _force_utf8_output():
    """Переключает stdout/stderr на UTF-8 (для Windows-консоли и русского текста)."""
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace",
            line_buffering=True, write_through=True,
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace",
            line_buffering=True, write_through=True,
        )
    except Exception:
        pass


def main():
    _force_utf8_output()
    banner()
    if "--cli" in sys.argv:
        cli_main(); return

    load_all_chats()
    print(f"  Загружено чатов: {len(CHATS)}", flush=True)
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
