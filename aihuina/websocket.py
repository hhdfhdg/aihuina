"""Минимальная реализация WebSocket поверх http.server."""
import base64
import hashlib
import struct


GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def ws_handshake(handler):
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key:
        handler.send_error(400, "Missing Sec-WebSocket-Key")
        return False
    accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    return True


def ws_send_text(sock, text: str):
    data = text.encode("utf-8")
    header = bytearray([0x81])
    n = len(data)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126); header += struct.pack(">H", n)
    else:
        header.append(127); header += struct.pack(">Q", n)
    sock.sendall(bytes(header) + data)


def ws_send_close(sock):
    try: sock.sendall(b"\x88\x00")
    except Exception: pass


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        try:
            c = sock.recv(n - len(buf))
        except (ConnectionResetError, OSError):
            return None
        if not c:
            return None
        buf += c
    return buf


def ws_recv_frame(sock):
    """Читает сообщение целиком (с учётом фрагментации). Возвращает (opcode, payload)."""
    full_op = None
    data = b""
    while True:
        header = _recv_exact(sock, 2)
        if not header or len(header) < 2:
            return None, None
        b1, b2 = header[0], header[1]
        fin = (b1 & 0x80) != 0
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        n = b2 & 0x7F
        if n == 126:
            ext = _recv_exact(sock, 2)
            if not ext: return None, None
            n = struct.unpack(">H", ext)[0]
        elif n == 127:
            ext = _recv_exact(sock, 8)
            if not ext: return None, None
            n = struct.unpack(">Q", ext)[0]
        mask = b"\x00\x00\x00\x00"
        if masked:
            mask = _recv_exact(sock, 4)
            if not mask: return None, None
        payload = b""
        if n:
            payload = _recv_exact(sock, n)
            if payload is None: return None, None
        if masked:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode >= 0x8:
            return opcode, payload
        if opcode != 0x0:
            full_op = opcode
        data += payload
        if fin:
            return full_op if full_op is not None else opcode, data
