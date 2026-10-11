#!/usr/bin/env bash
# Установка и запуск ChatAihuina на Linux/macOS.
# Использование:  ./install.sh          — установить и запустить веб-интерфейс
#                 ./install.sh --cli    — установить и запустить консольный режим
set -e
cd "$(dirname "$0")"

PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then
  echo "Не найден Python 3.9+. Установите его с https://www.python.org/downloads/" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo ">> Создаю виртуальное окружение..."
  "$PY" -m venv .venv
fi

echo ">> Устанавливаю зависимости..."
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -e . -q

if [ ! -f .env ] && [ -f .env.example ]; then
  cp .env.example .env
  echo ">> Создан файл .env (настройки по умолчанию). При необходимости отредактируйте его."
fi

echo ">> Запуск..."
exec .venv/bin/chatai "$@"
