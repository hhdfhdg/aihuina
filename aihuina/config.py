"""Настройки приложения: провайдеры, пути, лимиты и режимы усилия."""
import os
import json
from pathlib import Path


PROVIDERS = {
    "lmstudio": {
        "label": "LM Studio",
        "default_url": "http://127.0.0.1:1234/v1/chat/completions",
        "default_model": "qwen3.5-4b-super-coder",
    },
    "ollama": {
        "label": "Ollama",
        "default_url": "http://127.0.0.1:11434/v1/chat/completions",
        "default_model": "llama3.2",
    },
    "openai": {
        "label": "OpenAI",
        "default_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o-mini",
    },
    "xai": {
        "label": "xAI (Grok)",
        "default_url": "https://api.x.ai/v1/chat/completions",
        "default_model": "grok-4.5",
    },
    "custom": {
        "label": "Другой (OpenAI-compatible)",
        "default_url": "http://127.0.0.1:8000/v1/chat/completions",
        "default_model": "default",
    },
}


def _detect_provider(url: str) -> str:
    u = (url or "").lower()
    if "api.x.ai" in u:
        return "xai"
    if ":11434" in u or "ollama" in u:
        return "ollama"
    if ":1234" in u or "lmstudio" in u or "lm.studio" in u:
        return "lmstudio"
    if "api.openai.com" in u:
        return "openai"
    return "custom"


def _default_api_url() -> str:
    if os.getenv("XAI_API_KEY") and not os.getenv("AGENT_API_URL"):
        return PROVIDERS["xai"]["default_url"]
    prov = (os.getenv("AGENT_PROVIDER") or "").strip().lower()
    if prov in PROVIDERS:
        return PROVIDERS[prov]["default_url"]
    return PROVIDERS["lmstudio"]["default_url"]


def _default_model() -> str:
    if os.getenv("XAI_API_KEY") and not os.getenv("AGENT_API_URL"):
        return PROVIDERS["xai"]["default_model"]
    prov = (os.getenv("AGENT_PROVIDER") or "").strip().lower()
    if prov in PROVIDERS:
        return PROVIDERS[prov]["default_model"]
    return PROVIDERS["lmstudio"]["default_model"]


API_URL = os.getenv("AGENT_API_URL", _default_api_url())
MODEL_NAME = os.getenv("AGENT_MODEL", _default_model())
PROVIDER = (os.getenv("AGENT_PROVIDER") or _detect_provider(API_URL)).strip().lower()
if PROVIDER not in PROVIDERS:
    PROVIDER = _detect_provider(API_URL)
WEB_HOST = os.getenv("AGENT_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("AGENT_PORT", "8080"))
DATA_DIR = Path(os.getenv("AGENT_DATA", "./agent_data")).resolve()
CHAT_DIR = DATA_DIR / "chats"
WORKSPACE_DIR = DATA_DIR / "workspaces"

ACTIVE_WORKSPACE = Path(os.getenv("AGENT_ACTIVE_WORKSPACE", str(DATA_DIR / "Workspace"))).resolve()

CHAT_FILES_DIR = DATA_DIR / "chat_files"
CHAT_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
ACTIVE_WORKSPACE.mkdir(parents=True, exist_ok=True)
CHAT_FILES_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA_DIR / "settings.json"
ROOTS_FILE    = DATA_DIR / "roots.json"
CMD_TIMEOUT       = int(os.getenv("AGENT_CMD_TIMEOUT", "30"))
MAX_OUTPUT_CHARS  = int(os.getenv("AGENT_MAX_OUTPUT",  "8000"))
MODEL_TIMEOUT     = int(os.getenv("AGENT_MODEL_TIMEOUT", "600"))
MAX_ITERATIONS    = int(os.getenv("AGENT_MAX_ITERS",   "15"))
TEMPERATURE       = float(os.getenv("AGENT_TEMP",      "0.2"))


EFFORT_PRESETS = {
    "weak": {
        "label": "Слабо",
        "max_iterations": 8,
        "force_think": False,
        "preload": [],
        "note": "СИСТЕМА [усилие: слабо]: отвечай коротко и по существу, без лишних пояснений. Инструменты вызывай только тогда, когда без них ответ невозможен или заведомо неверен. Остановись, как только задача решена.",
    },
    "normal": {
        "label": "Нормально",
        "max_iterations": 20,
        "force_think": True,
        "preload": [],
        "note": """РЕЖИМ НОРМАЛЬНЫЙ — работай самостоятельно, проверяй факты и не выдумывай детали.
Для задач с файлами сначала оцени список файлов и выбери нужные, затем читай их инструментом read_file.
Используй инструменты, когда они дают проверяемую пользу: list_files для структуры, read_file для содержимого, bash для запуска и диагностики, edit/write для изменений. Не вызывай инструменты ради количества.
После изменения проверь результат хотя бы одной подходящей проверкой. Не заявляй о выполнении команды или теста, если инструмент этого не подтвердил.""",
    },
    "strong": {
        "label": "Сильно",
        "max_iterations": 36,
        "force_think": True,
        "preload": [],
        "note": """РЕЖИМ СИЛЬНЫЙ — приоритет качеству и реальному выполнению, а не быстрому текстовому ответу.
Сначала составь короткий план из нескольких шагов. Затем активно используй инструменты: осмотри структуру проекта, прочитай нужные исходники и конфиги, найди зависимости и точки входа, внеси изменения, запусти тесты, линтер или команду проверки и изучи ошибки.
После ошибки исправляй причину, а не симптом, и повторяй проверку. Не перечитывай один и тот же файл без причины. Предпочитай точечные изменения.
Перед финалом сверь результат с исходной задачей и сообщи, что именно реально проверено.""",
    },
    "very": {
        "label": "Очень сильно",
        "max_iterations": 32,
        "force_think": True,
        "preload": [],
        "note": """РЕЖИМ ОЧЕНЬ СИЛЬНЫЙ — глубокий анализ перед любым действием.
(1) Разбери задачу: цель, ограничения, крайние случаи и то, что может сломаться.
(2) Сравни 2–3 варианта решения, выбери лучший и коротко обоснуй выбор.
(3) Выполни изменения точечно, инструментами, не переписывая лишнего.
(4) Проверь результат тестом или запуском. Если проверка упала, найди первопричину и исправь её.
(5) Перед ответом перечитай исходную просьбу и убедись, что выполнено всё, а не только основная часть.
Итерации расходуй экономно: не повторяй чтение одного и того же без причины.""",
    },
    "max": {
        "label": "Максимально",
        "max_iterations": 64,
        "force_think": True,
        "preload": [],
        "note": """РЕЖИМ МАКСИМАЛЬНЫЙ — используй доступные инструменты максимально полно и последовательно, чтобы получить проверенный результат. Не ограничивайся советом или примером, если пользователь просит выполнить работу.
ОБЯЗАТЕЛЬНЫЙ ЦИКЛ: (1) уточни цель по сообщению и вложениям; (2) изучи рабочую папку через list_files; (3) составь план и выбери релевантные файлы; (4) прочитай нужные исходники и конфиги через read_file; (5) найди связанные места инструментами поиска; (6) внеси изменения через edit/write/append/insert или выполни команды через bash; (7) проверь изменения и запусти подходящие тесты, сборку или программу; (8) прочитай вывод и исправь ошибки; (9) повторяй проверку, пока задача не решена или не появится реальный блокер; (10) проверь итоговое состояние и дай краткий отчёт.
ИСПОЛЬЗОВАНИЕ ИНСТРУМЕНТОВ: перед изменениями изучи стиль и архитектуру проекта. Используй list_files и read_file для понимания, bash для диагностики и запуска, edit и write для правок, move_file для переименования, send_file для передачи результата. Если инструмент вернул ошибку, прочитай её, измени подход и попробуй снова, когда это безопасно.
НЕ ИМИТИРУЙ РАБОТУ: не выдумывай содержимое файлов, результаты команд, успешные тесты, пути или артефакты. Не говори «готово», пока инструменты не подтвердили результат. Если задача не требует инструментов (обычный вопрос), отвечай прямо.
БЕЗОПАСНОСТЬ: не выполняй разрушительные команды, массовую перезапись, изменение прав или системных настроек без явной необходимости и разрешения. Проверяй пути и границы рабочей папки. Перед рискованными массовыми изменениями делай резервные копии. Если прав или инструментов не хватает, честно назови блокер и сделай всё возможное.
ОСТАНОВКА: продолжай инструментальный цикл до проверенного результата, без бесполезных повторных вызовов. Лимит итераций — запас, а не цель. Если лимит достигнут, кратко перечисли незавершённые пункты и причину.""",
    },
}

def resolve_effort(level: str) -> dict:
    key = str(level or "normal").strip().lower()
    aliases = {
        "слабо": "weak", "weak": "weak", "low": "weak",
        "нормально": "normal", "normal": "normal", "medium": "normal",
        "сильно": "strong", "strong": "strong", "high": "strong",
        "очень сильно": "very", "very": "very", "very_strong": "very",
        "максимально": "max", "max": "max", "maximum": "max",
    }
    return EFFORT_PRESETS.get(aliases.get(key, key), EFFORT_PRESETS["normal"])
MAX_CTX_CHARS     = int(os.getenv("AGENT_MAX_CTX",     "48000"))
MAX_UPLOAD        = int(os.getenv("AGENT_MAX_UPLOAD",  str(200 * 1024 * 1024)))

BLOCKED_PATTERNS = [
    "rm -rf /", ":(){", "mkfs", "dd if=", "> /dev/sd",
    "shutdown", "reboot", "halt", "poweroff",
    "chmod 777 /", "chown -R /", "fork bomb",
    "> /dev/null &", ":(){:|:&};:",
]

THINK_NOTE = (
    "СИСТЕМА [режим Think включён]: прежде чем действовать, глубоко проанализируй задачу "
    "внутри <reasoning>…</reasoning> — разбери шаги, возможные ошибки и альтернативы, "
    "проверь свой план. Затем выполни действие (<write> или <tool>) и в конце дай "
    "короткий итоговый ответ на русском."
)

TEXT_EXTS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".csv", ".html", ".css",
    ".xml", ".yml", ".yaml", ".log", ".ini", ".toml", ".sql", ".sh", ".bat", ".c", ".cpp",
    ".h", ".java", ".go", ".rs", ".cs", ".php", ".rb",
}
