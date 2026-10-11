"""Разбор тегов ответа модели: reasoning, write, edit, tool."""
import re
import json
from .edit import _attrs, _guess_write_path


_TAG_PAIRS = [
    ("<reasoning>", "</reasoning>", "reasoning"),
    ("<thinking>",  "</thinking>",  "reasoning"),
    ("<think>",     "</think>",     "reasoning"),
    ("<analysis>",  "</analysis>",  "reasoning"),
    ("<reflection>", "</reflection>", "reasoning"),
    ("<thought>",   "</thought>",   "thought"),
    ("<thinking_out_loud>", "</thinking_out_loud>", "thought"),
    ("<status>",    "</status>",    "status"),
    ("<tool>",      "</tool>",      "tool"),
]
_OPEN_TAGS  = {t[0]: t[2] for t in _TAG_PAIRS}
_CLOSE_TAGS = {t[1]: t[2] for t in _TAG_PAIRS}
_MAX_TAG    = max(max(len(t[0]), len(t[1])) for t in _TAG_PAIRS)

_REASON_OPEN_FLEX_RE = re.compile(
    r"<(reasoning|thinking|think|analysis|reflection)\b[^>]*>",
    re.IGNORECASE,
)
_REASON_CLOSE_FLEX_RE = re.compile(
    r"</(reasoning|thinking|think|analysis|reflection)\s*>",
    re.IGNORECASE,
)

_WRITE_OPEN_RE = re.compile(r'<write\b([^>]*)>', re.IGNORECASE)
_WRITE_ATTR_RE = re.compile(r'<write\b[^>]*?\bpath\s*=\s*"([^"\n]+)"[^>]*?\bcontent\s*=\s*"(.*?)(?:"\s*/?>|</write\s*>)', re.IGNORECASE | re.DOTALL)
_WRITE_CLOSE   = "</write>"
_BLOCK_OPEN_RE = re.compile(r"<(edit|append|insert)\b([^>]*)>", re.IGNORECASE)
_BLOCK_CLOSE = {"edit": "</edit>", "append": "</append>", "insert": "</insert>"}


class StreamSplitter:
    """
    Разбивает поток модели на события (kind, text):
      assistant        — текст ответа (стримится)
      reasoning_delta  — куски рассуждения (стримятся)
      reasoning_end    — рассуждение закончилось
      thought / status — целиком, одним блоком
      write_done       — (path, content) блока <write>
      содержимое <edit>/<append>/<insert> наружу не отдаётся
    """
    def __init__(self):
        self.buf = ""
        self.mode = "assistant"
        self.tool_buf = self.status_buf = self.thought_buf = ""
        self.reasoning_open = False
        self.write_buf = ""
        self.write_path = ""
        self.block_buf = ""

    def feed(self, chunk: str):
        events = []
        self.buf += chunk
        while True:
            idx, tag, wm = self._find_next_tag()
            if idx == -1:
                cut = len(self.buf) - _MAX_TAG
                li = self.buf.rfind("<")
                if li != -1 and ">" not in self.buf[li:] and len(self.buf) - li < 500:
                    cut = min(cut, li)
                # незавершённый <write ...> удерживаем целиком, а не только с последнего «<»
                wi = self.buf.lower().rfind("<write")
                if (self.mode == "assistant" and wi != -1 and len(self.buf) - wi < 2000
                        and ">" not in self.buf[wi:]):
                    cut = min(cut, wi)
                if cut > 0:
                    safe, self.buf = self.buf[:cut], self.buf[cut:]
                    self._push(events, self.mode, safe)
                break
            if idx > 0:
                self._push(events, self.mode, self.buf[:idx])
            self.buf = self.buf[idx + len(tag):]

            if isinstance(wm, tuple) and wm[0] == "write_attr":
                am = wm[1]
                events.append(("write_done", (am.group(1).strip(), am.group(2))))
                continue
            if isinstance(wm, tuple) and wm[0] == "write":
                self.write_path = _guess_write_path(wm[1].group(1) if wm[1].lastindex else wm[1].group(0), "")
                self.write_buf = ""
                self.mode = "write"

                continue
            if isinstance(wm, tuple) and wm[0] == "block":
                b = wm[1]
                self.mode = b.group(1).lower()
                self.block_buf = ""

                continue
            if tag == _WRITE_CLOSE or (isinstance(tag, str) and tag.lower() == _WRITE_CLOSE and self.mode == "write"):
                path = self.write_path or _guess_write_path("", self.write_buf)
                if path in ("", "file.txt"):
                    path = _guess_write_path("", self.write_buf)
                events.append(("write_done", (path, self.write_buf)))
                self.write_buf = ""
                self.mode = "assistant"
                continue
            if self.mode in _BLOCK_CLOSE and tag and tag.lower() == _BLOCK_CLOSE[self.mode]:
                self.block_buf = ""
                self.mode = "assistant"
                continue

            tag_lower = tag.lower() if isinstance(tag, str) else ""

            if tag_lower in _OPEN_TAGS:
                m = _OPEN_TAGS[tag_lower]
                if m == "tool":
                    self.tool_buf = ""
                elif m == "status":
                    self.status_buf = ""
                elif m == "thought":
                    self.thought_buf = ""
                elif m == "reasoning":
                    self.reasoning_open = True
                self.mode = m

            elif tag_lower in _CLOSE_TAGS:
                m = _CLOSE_TAGS[tag_lower]
                if m == "tool" and self.mode == "tool":
                    events.append(("tool_done", self.tool_buf))
                    self.tool_buf = ""
                elif m == "status" and self.mode == "status":
                    s = self.status_buf.strip()
                    if s:
                        events.append(("status", s))
                    self.status_buf = ""
                elif m == "thought" and self.mode == "thought":
                    s = self.thought_buf.strip()
                    if s:
                        events.append(("thought", s))
                    self.thought_buf = ""
                elif m == "reasoning" and self.mode == "reasoning":
                    if self.reasoning_open:
                        events.append(("reasoning_end", ""))
                    self.reasoning_open = False
                self.mode = "assistant"

            elif _REASON_OPEN_FLEX_RE.fullmatch(tag or ""):
                self.reasoning_open = True
                self.mode = "reasoning"

            elif _REASON_CLOSE_FLEX_RE.fullmatch(tag or ""):
                if self.mode == "reasoning" and self.reasoning_open:
                    events.append(("reasoning_end", ""))
                self.reasoning_open = False
                self.mode = "assistant"
        return events

    def flush(self):
        events = []
        if self.buf:
            self._push(events, self.mode, self.buf)
            self.buf = ""
        if self.mode == "status" and self.status_buf.strip():
            events.append(("status", self.status_buf.strip()))
        elif self.mode == "thought" and self.thought_buf.strip():
            events.append(("thought", self.thought_buf.strip()))
        if self.reasoning_open:
            events.append(("reasoning_end", ""))
            self.reasoning_open = False
        self.status_buf = self.thought_buf = ""
        return events

    def _write_open_valid(self, m):
        """<write> — это тег записи, если указан path или тело начинается сразу после тега. Иначе это просто текст."""
        attrs = m.group(1)
        if re.search(r"\bpath\s*=", attrs):
            return True
        if re.search(r"</write|\bcontent\s*=", attrs):
            return False
        tail = self.buf[m.end():m.end() + 16]
        if not tail.strip():
            return False
        return bool(re.match(r"\s*(?:\n|<|```)", tail))

    def _find_next_tag(self):
        if self.mode == "write":
            i = self.buf.lower().find(_WRITE_CLOSE.lower())
            return (i, self.buf[i:i + len(_WRITE_CLOSE)], None) if i != -1 else (-1, None, None)

        if self.mode in _BLOCK_CLOSE:
            close = _BLOCK_CLOSE[self.mode]
            i = self.buf.lower().find(close.lower())
            if i == -1:
                return -1, None, None
            return i, self.buf[i:i + len(close)], None

        best_idx, best_tag = -1, None

        for t in list(_OPEN_TAGS) + list(_CLOSE_TAGS):
            i = self.buf.lower().find(t.lower())
            if i != -1 and (best_idx == -1 or i < best_idx):
                best_idx = i
                best_tag = self.buf[i:i + len(t)]

        for rx in (_REASON_OPEN_FLEX_RE, _REASON_CLOSE_FLEX_RE):
            m = rx.search(self.buf)
            if m and (best_idx == -1 or m.start() < best_idx):
                best_idx = m.start()
                best_tag = m.group(0)

        specials = []
        ma = _WRITE_ATTR_RE.search(self.buf)
        if ma:
            specials.append((ma.start(), ma.group(0), ("write_attr", ma)))
        m = _WRITE_OPEN_RE.search(self.buf)
        if m and not (ma and ma.start() == m.start()) and self._write_open_valid(m):
            specials.append((m.start(), m.group(0), ("write", m)))

        b = _BLOCK_OPEN_RE.search(self.buf)
        if b:
            specials.append((b.start(), b.group(0), ("block", b)))

        if specials:
            specials.sort(key=lambda x: x[0])
            s_idx, s_tag, s_meta = specials[0]
            if best_idx == -1 or s_idx < best_idx:
                return s_idx, s_tag, s_meta

        return best_idx, best_tag, None

    def _push(self, events, mode, text):
        if not text:
            return
        if mode == "tool":        self.tool_buf += text
        elif mode == "status":    self.status_buf += text
        elif mode == "thought":   self.thought_buf += text
        elif mode == "write":     self.write_buf += text
        elif mode in _BLOCK_CLOSE: self.block_buf += text
        elif mode == "reasoning": events.append(("reasoning_delta", text))
        else:                     events.append((mode, text))


TOOL_RE     = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL | re.IGNORECASE)
WRITE_RE    = re.compile(r'<write\b([^>]*)>(.*?)</write>',
                         re.DOTALL | re.IGNORECASE)
EDIT_RE     = re.compile(r"<edit\b([^>]*)>(.*?)</edit>", re.DOTALL | re.IGNORECASE)
APPEND_RE   = re.compile(r"<append\b([^>]*)>(.*?)</append>", re.DOTALL | re.IGNORECASE)
INSERT_RE   = re.compile(r"<insert\b([^>]*)>(.*?)</insert>", re.DOTALL | re.IGNORECASE)
SENDFILE_RE = re.compile(r"<sendfile\s+path=[\"']([^\"']+)[\"']\s*/?>", re.IGNORECASE)

_REASON_TAGS = (
    "reasoning",
    "thinking",
    "think",
    "analysis",
    "reflection",
    "thought",
)
_REASON_NAMES = "|".join(map(re.escape, _REASON_TAGS))
_REASON_RE = re.compile(
    rf"<(?:{_REASON_NAMES})\b[^>]*>.*?</(?:{_REASON_NAMES})\s*>",
    re.DOTALL | re.IGNORECASE,
)
_REASON_OPEN_RE = re.compile(
    rf"<(?:{_REASON_NAMES})\b[^>]*>.*$",
    re.DOTALL | re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Удаляет reasoning из текста перед показом/сохранением как обычного ответа."""
    if not text:
        return ""
    text = _REASON_RE.sub("", str(text))
    text = _REASON_OPEN_RE.sub("", text)
    return text


def history_text(text: str) -> str:
    """Что сохраняем в историю модели: без длинных рассуждений."""
    t = strip_reasoning(text).strip()
    return t or "(пусто)"


def _loose_json(s: str):
    depth = 0; start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def _iter_write_matches(text: str):
    """Теги <write>: принимаем, если указан path или тело начинается сразу после тега (перенос строки, <, ```)."""
    pos = 0
    while True:
        m = WRITE_RE.search(text, pos)
        if not m:
            return
        attrs, body = m.group(1), m.group(2)
        valid = (bool(re.search(r"\bpath\s*=", attrs)) or bool(re.match(r"\s*(?:\n|<|```)", body))) \
            and not re.search(r"</write|\bcontent\s*=", attrs)
        if valid:
            yield m
            pos = m.end()
        else:
            pos = m.start() + 1


def _clean_write_body(body: str) -> str:
    body = body.replace("\r\n", "\n")
    if "</reasoning>" in body:
        body = body.split("</reasoning>")[-1]
    body = re.sub(r"<write\b[^>]*>", "", body, flags=re.IGNORECASE)
    if body.startswith("\n"):
        body = body[1:]
    m = re.match(r"^\s*```[\w+#.-]*\n(.*?)\n?```\s*$", body, re.DOTALL)
    if m:
        body = m.group(1)
    return body.rstrip("\n") + "\n"


def extract_tool_call(text: str):
    cands = []
    m = TOOL_RE.search(text)
    if m: cands.append((m.start(), "tool", m))
    w = next(_iter_write_matches(text), None)
    if w: cands.append((w.start(), "write", w))
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    _, kind, m = cands[0]
    if kind == "write":
        body = _clean_write_body(m.group(2))
        return "write_file", {"path": _guess_write_path(m.group(1), body), "content": body}
    raw = m.group(1)
    try:
        call = json.loads(raw)
    except json.JSONDecodeError:
        call = _loose_json(raw)
        if call is None: return None
    name = call.get("name")
    args = call.get("args", {})
    if not isinstance(name, str): return None
    if not isinstance(args, dict): args = {}
    return name, args


def extract_sendfiles(text: str):
    return [m.group(1) for m in SENDFILE_RE.finditer(text)]


def extract_actions(text: str):
    """Все write/edit/append/insert/tool по порядку, без вложенных совпадений."""
    found = []
    for rx, kind in (
        (WRITE_RE, "write"),
        (_WRITE_ATTR_RE, "wattr"),
        (EDIT_RE, "edit"),
        (APPEND_RE, "append"),
        (INSERT_RE, "insert"),
        (TOOL_RE, "tool"),
    ):
        for m in (_iter_write_matches(text or "") if rx is WRITE_RE else rx.finditer(text or "")):
            found.append((m.start(), m.end(), kind, m))
    found.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    actions = []
    last = -1
    for start, end, kind, m in found:
        if start < last:
            continue
        last = end
        if kind == "write":
            body = _clean_write_body(m.group(2))
            actions.append(("write", {
                "path": _guess_write_path(m.group(1), body),
                "content": body,
            }))
        elif kind == "wattr":
            actions.append(("write", {"path": m.group(1).strip(), "content": _clean_write_body(m.group(2))}))
        elif kind in ("edit", "append", "insert"):
            attrs = _attrs(m.group(1))
            actions.append((kind, {
                "path": (attrs.get("path") or "").strip(),
                "body": m.group(2),
                "attrs": attrs,
            }))
        else:
            raw = m.group(1)
            try:
                call = json.loads(raw)
            except json.JSONDecodeError:
                call = _loose_json(raw)
            if not isinstance(call, dict):
                actions.append(("tool", {"name": "", "args": {}, "bad": raw[:200]}))
                continue
            name = call.get("name")
            args = call.get("args", {})
            if not isinstance(name, str):
                continue
            if not isinstance(args, dict):
                args = {}
            actions.append(("tool", {"name": name, "args": args}))
    return actions
