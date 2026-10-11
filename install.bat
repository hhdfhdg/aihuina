@echo off
REM Установка и запуск ChatAihuina на Windows.
REM Использование:  install.bat          — веб-интерфейс
REM                 install.bat --cli    — консольный режим
cd /d "%~dp0"

where py >nul 2>nul && set PY=py -3
if not defined PY (where python >nul 2>nul && set PY=python)
if not defined PY (
  echo Не найден Python 3.9+. Установите его с https://www.python.org/downloads/
  pause & exit /b 1
)

if not exist .venv (
  echo ^>^> Создаю виртуальное окружение...
  %PY% -m venv .venv || (pause & exit /b 1)
)

echo ^>^> Устанавливаю зависимости...
.venv\Scripts\python -m pip install --upgrade pip -q
.venv\Scripts\python -m pip install -e . -q || (pause & exit /b 1)

if not exist .env if exist .env.example (
  copy .env.example .env >nul
  echo ^>^> Создан файл .env с настройками по умолчанию.
)

echo ^>^> Запуск...
.venv\Scripts\chatai.exe %*
pause
