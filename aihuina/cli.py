"""Консольный режим (--cli)."""
import json
import queue
import threading
from . import config
from .chat import Chat
from .file_tools import _ctx, tool_list_files
from .settings import DEFAULT_WORKDIR


def cli_main():
    print("=== AI Agent (CLI) ===")
    print(f"API:     {config.API_URL}")
    print(f"Модель:  {config.MODEL_NAME}")
    print(f"Каталог: {DEFAULT_WORKDIR}")
    print("Команды: /exit, /clear, /files, /think (вкл/выкл режим Think)\n")

    chat = Chat()
    q: "queue.Queue" = queue.Queue()
    think = False

    while True:
        try:
            user = input("Вы: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!"); break
        if not user: continue
        if user == "/exit": print("Пока!"); break
        if user == "/clear":
            chat = Chat(workdir=chat.workdir); print("[История очищена]\n"); continue
        if user == "/files":
            _ctx.workdir = chat.workdir
            print(tool_list_files()); print(); continue
        if user == "/think":
            think = not think
            print(f"[Think: {'включён' if think else 'выключен'}]\n"); continue

        t = threading.Thread(target=chat.agent.run_turn, args=(user, q),
                             kwargs={"think": think}, daemon=True)
        t.start()
        in_reasoning = False
        while t.is_alive() or not q.empty():
            try: ev = q.get(timeout=0.1)
            except queue.Empty: continue
            k = ev[0]
            if k == "reasoning_delta":
                if think:
                    if not in_reasoning:
                        print("\n🧠 ", end="", flush=True); in_reasoning = True
                    print(ev[1], end="", flush=True)
            elif k == "reasoning_end":
                if in_reasoning: print(); in_reasoning = False
            elif k == "thought":        print(f"\n💭 {ev[1]}")
            elif k == "assistant_delta":print(ev[1], end="", flush=True)
            elif k == "assistant_end":  print()
            elif k == "status":         print(f"  ✓ {ev[1]}")
            elif k == "file":           print(f"  📎 {ev[1].get('action','')} {ev[1]['name']} ({ev[1]['size']} Б) → {ev[1]['path']}")
            elif k == "diff":           print(f"  ± {ev[1].get('summary','')} {ev[1].get('path','')}")
            elif k == "log":
                if ev[1] == "tool_call":
                    print(f"    🛠 {ev[2]['name']}({json.dumps(ev[2]['args'], ensure_ascii=False)[:200]})")
                else:
                    print(f"    → {ev[2]['text'][:200]}")
            elif k == "error":          print(f"\n[Ошибка] {ev[1]}")
        print()
