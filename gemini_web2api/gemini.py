"""Gemini StreamGenerate protocol implementation with httpx streaming."""
import json
import time
import uuid
import re
import urllib.request
import urllib.parse
import urllib.error
import ssl
import os
import hashlib
import threading

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from .config import CONFIG

_ssl_ctx = None
_cookie_cache = {"str": "", "sapisid": None, "mtime": 0}
_cookie_lock = threading.Lock()
_refresh_lock = threading.Lock()
_httpx_client = None


class ChatWindow:
    """Thread-safe multi-turn conversation state (c_xxx / r_xxx / rc_xxx)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.conv_id = ""
        self.resp_id = ""
        self.rcid = ""
        self.sent_count = 0
        self.prefix_hash = ""
        self.model_id = None
        self.think_mode = None

    @staticmethod
    def _hash_messages(messages: list) -> str:
        h = hashlib.md5()
        for m in messages:
            role = m.get("role") or ""
            raw_content = m.get("content")
            content = "" if raw_content is None else str(raw_content)[:200]
            tc_part = ""
            if m.get("tool_calls"):
                tc_names = [
                    (tc.get("function") or {}).get("name", "")
                    for tc in m["tool_calls"]
                    if isinstance(tc, dict)
                ]
                tc_part = ",".join(tc_names)
            h.update(f"{role}:{content}:{tc_part}|".encode("utf-8", errors="ignore"))
        return h.hexdigest()[:16]

    def get_continuation(self, messages: list, model_id: int, think_mode: int):
        """Return (conv_meta_list, start_idx) if messages continue the active window."""
        if not self.conv_id or len(messages) <= self.sent_count or self.sent_count == 0:
            return None, 0
        if model_id != self.model_id or think_mode != self.think_mode:
            return None, 0
        if self._hash_messages(messages[:self.sent_count]) != self.prefix_hash:
            return None, 0
        meta = [self.conv_id, self.resp_id, self.rcid, None, None, None, None, None, None, ""]
        return meta, self.sent_count

    def update(self, conv_id: str, resp_id: str, rcid: str, messages: list, assistant_msg, model_id: int, think_mode: int):
        if not conv_id:
            return
        self.conv_id = conv_id
        if resp_id:
            self.resp_id = resp_id
        if rcid:
            self.rcid = rcid
        if isinstance(assistant_msg, str):
            assistant_msg = {"role": "assistant", "content": assistant_msg}
        full_msgs = list(messages) + [assistant_msg]
        self.sent_count = len(full_msgs)
        self.prefix_hash = self._hash_messages(full_msgs)
        self.model_id = model_id
        self.think_mode = think_mode


_chat_window = ChatWindow()
_conv_tls = threading.local()


def get_last_conv_ids() -> tuple:
    return getattr(_conv_tls, "ids", ("", "", ""))


def set_last_conv_ids(ids: tuple) -> None:
    _conv_tls.ids = ids


def extract_conv_ids_from_line(line: str) -> tuple:
    """Extract (conv_id, resp_id, rcid) from a wrb.fr response line."""
    if '"wrb.fr"' not in line or len(line) < 100:
        return "", "", ""
    conv_id = resp_id = rcid = ""
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str:
            return "", "", ""
        inner = json.loads(inner_str)
        if isinstance(inner, list) and len(inner) > 1 and isinstance(inner[1], list):
            ids = inner[1]
            if len(ids) >= 2:
                if isinstance(ids[0], str) and ids[0].startswith("c_"):
                    conv_id = ids[0]
                if isinstance(ids[1], str) and ids[1].startswith("r_"):
                    resp_id = ids[1]
        if isinstance(inner, list) and len(inner) > 4 and inner[4]:
            for part in inner[4]:
                if isinstance(part, list) and len(part) > 0 and isinstance(part[0], str) and part[0].startswith("rc_"):
                    rcid = part[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        pass
    return conv_id, resp_id, rcid


def log(msg: str):
    if CONFIG["log_requests"]:
        import sys
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        sys.stderr.flush()


def _get_ssl_ctx():
    global _ssl_ctx
    if _ssl_ctx is None:
        _ssl_ctx = ssl.create_default_context()
    return _ssl_ctx


def _get_httpx_client():
    global _httpx_client
    if _httpx_client is None and HAS_HTTPX:
        proxy = CONFIG.get("proxy")
        transport = httpx.HTTPTransport(proxy=proxy) if proxy else None
        _httpx_client = httpx.Client(
            transport=transport,
            timeout=CONFIG["request_timeout_sec"],
            verify=True,
            headers={"Connection": "close"},
        )
    return _httpx_client


def _parse_netscape_cookies(raw: str) -> str:
    pairs = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            domain = parts[0].lstrip(".").lower()
            if domain == "google.com" or domain.endswith(".google.com"):
                pairs[parts[5]] = parts[6]
    return "; ".join(f"{k}={v}" for k, v in pairs.items())


def update_cookie_payload(data: dict) -> bool:
    """Update in-memory and on-disk cookie state from a pushed extension payload."""
    if not isinstance(data, dict) or not data.get("cookie"):
        return False
    with _cookie_lock:
        cookie_str = data["cookie"]
        sapisid = data.get("sapisid", "")
        if not sapisid and cookie_str:
            m = re.search(r"SAPISID=([^;\s]+)", cookie_str)
            sapisid = m.group(1) if m else ""
        if data.get("xsrf_token"):
            CONFIG["xsrf_token"] = data["xsrf_token"]
        if data.get("gemini_bl"):
            CONFIG["gemini_bl"] = data["gemini_bl"]
        if data.get("auth_user") not in (None, "") and CONFIG.get("auth_user") in (None, ""):
            CONFIG["auth_user"] = str(data["auth_user"])
        if isinstance(data.get("model_tickets"), dict):
            tickets = dict(CONFIG.get("model_tickets") or {})
            tickets.update(data["model_tickets"])
            CONFIG["model_tickets"] = tickets
        cf = CONFIG.get("cookie_file")
        if cf:
            try:
                with open(cf, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                mtime = os.path.getmtime(cf)
            except Exception:
                mtime = time.time()
        else:
            mtime = time.time()
        _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
        log("Updated cookies in memory and on disk via /v1/cookies")
        return True


def load_cookie() -> tuple:
    """Load cookie from file with mtime-based caching."""
    cookie_file = CONFIG.get("cookie_file")
    if not cookie_file or not os.path.exists(cookie_file):
        return "", None
    with _cookie_lock:
        try:
            mtime = os.path.getmtime(cookie_file)
            if mtime == _cookie_cache["mtime"] and _cookie_cache["str"]:
                return _cookie_cache["str"], _cookie_cache["sapisid"]
            with open(cookie_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.startswith("{"):
                data = json.loads(content)
                cookie_str = data.get("cookie", "")
                sapisid = data.get("sapisid", "")
                if data.get("xsrf_token"):
                    CONFIG["xsrf_token"] = data["xsrf_token"]
                if data.get("gemini_bl"):
                    CONFIG["gemini_bl"] = data["gemini_bl"]
                if data.get("auth_user") not in (None, "") and CONFIG.get("auth_user") in (None, ""):
                    CONFIG["auth_user"] = str(data["auth_user"])
                if isinstance(data.get("model_tickets"), dict):
                    tickets = dict(CONFIG.get("model_tickets") or {})
                    tickets.update(data["model_tickets"])
                    CONFIG["model_tickets"] = tickets
            elif "\t" in content:
                cookie_str = _parse_netscape_cookies(content)
                sapisid = ""
            else:
                cookie_str = content
                pairs = dict(p.split("=", 1) for p in cookie_str.split("; ") if "=" in p)
                sapisid = pairs.get("SAPISID", "")
            if not sapisid and cookie_str:
                m = re.search(r"SAPISID=([^;\s]+)", cookie_str)
                sapisid = m.group(1) if m else ""
            _cookie_cache.update({"str": cookie_str, "sapisid": sapisid or None, "mtime": mtime})
            return cookie_str, sapisid if sapisid else None
        except Exception as e:
            log(f"Cookie load error: {e}")
            return _cookie_cache["str"], _cookie_cache["sapisid"]


def make_sapisidhash(sapisid: str) -> str:
    ts = int(time.time())
    h = hashlib.sha1(f"{ts} {sapisid} https://gemini.google.com".encode()).hexdigest()
    return f"SAPISIDHASH {ts}_{h}"


def _account_prefix() -> str:
    auth_user = CONFIG.get("auth_user")
    if auth_user is None or auth_user == "":
        return ""
    return f"/u/{auth_user}"


def _build_headers(extra_fields: dict = None) -> dict:
    cookie_str, sapisid = load_cookie()
    account_prefix = _account_prefix()
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "https://gemini.google.com",
        "Referer": f"https://gemini.google.com{account_prefix}/app",
        "X-Same-Domain": "1",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if account_prefix:
        headers["X-Goog-AuthUser"] = str(CONFIG["auth_user"])
    if extra_fields and "_ticket" in extra_fields and CONFIG.get("send_model_ticket", True):
        headers["X-Goog-Ext-525001261-Jspb"] = extra_fields["_ticket"]
    if cookie_str:
        headers["Cookie"] = cookie_str
    if sapisid:
        headers["Authorization"] = make_sapisidhash(sapisid)
    return headers


def refresh_bl_and_xsrf() -> bool:
    """Refresh gemini_bl and xsrf_token from gemini.google.com."""
    with _refresh_lock:
        try:
            cookie_str, sapisid = load_cookie()
            account_prefix = _account_prefix()
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            if cookie_str:
                headers["Cookie"] = cookie_str
            if sapisid:
                headers["Authorization"] = make_sapisidhash(sapisid)
            req = urllib.request.Request(f"https://gemini.google.com{account_prefix}/app", headers=headers)
            ctx = _get_ssl_ctx()
            proxy = CONFIG.get("proxy")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx),
                )
                resp = opener.open(req, timeout=15)
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=15)
            html = resp.read().decode("utf-8", errors="replace")
            updated = False
            bl_m = re.search(r'"cfb2h"\s*:\s*"([^"]+)"', html) or re.search(
                r"(boq_assistant-bard-web-server_\d+\.\d+_p\d+)", html
            )
            if bl_m:
                CONFIG["gemini_bl"] = bl_m.group(1)
                updated = True
            at_m = re.search(r'"SNlM0e"\s*:\s*"([^"]+)"', html)
            if at_m:
                CONFIG["xsrf_token"] = at_m.group(1)
                updated = True
            return updated
        except Exception as e:
            log(f"BL/XSRF auto-refresh failed: {e}")
            return False


def _apply_chat_persistence_flags(inner: list) -> None:
    if CONFIG.get("temporary_chats", False):
        inner[41] = [1]
        inner[45] = 1
    else:
        inner[41] = [2]


def _build_payload(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    load_cookie()
    inner = [None] * 102
    if file_refs:
        refs = [[None, None, ref] for ref in file_refs]
        inner[0] = [prompt, 0, None, refs, None, None, 0]
    else:
        inner[0] = [prompt, 0, None, None, None, None, 0]
    inner[1] = ["en"]
    inner[2] = ["", "", "", None, None, None, None, None, None, ""]
    inner[6] = [0]
    inner[7] = 1
    inner[10] = 1
    inner[11] = 0
    inner[17] = [[think_mode]]
    inner[18] = 0
    inner[27] = 1
    inner[30] = [4]
    _apply_chat_persistence_flags(inner)
    inner[53] = 0
    inner[59] = str(uuid.uuid4())
    inner[61] = []
    inner[68] = 1
    inner[79] = model_id
    if extra_fields:
        for k, v in extra_fields.items():
            if isinstance(k, int):
                inner[k] = v
    outer = [None, json.dumps(inner)]
    params = {"f.req": json.dumps(outer)}
    if CONFIG.get("xsrf_token"):
        params["at"] = CONFIG["xsrf_token"]
    return urllib.parse.urlencode(params)


def _get_url() -> str:
    load_cookie()
    reqid = int(time.time()) % 1000000
    account_prefix = _account_prefix()
    return (
        f"https://gemini.google.com{account_prefix}/_/BardChatUi/data/"
        "assistant.lamda.BardFrontendService/StreamGenerate"
        f"?bl={CONFIG['gemini_bl']}&hl=en&_reqid={reqid}&rt=c"
    )


def clean_text(text: str, strip: bool = True) -> str:
    text = re.sub(
        r'```(?:python|javascript|text)\?code_(?:reference|stdout)&code_event_index=\d+\n.*?```\n?',
        '', text, flags=re.DOTALL
    )
    text = re.sub(r'http://googleusercontent\.com/card_content/\d+\n?', '', text)
    return text.strip() if strip else text


def extract_served_model_info(raw: str) -> tuple:
    """Extract (served_hex_id, served_label) from wrb.fr response frames if present."""
    for line in raw.split("\n"):
        if '"wrb.fr"' not in line or len(line) < 100:
            continue
        try:
            arr = json.loads(line)
            inner_str = arr[0][2]
            if not inner_str:
                continue
            inner = json.loads(inner_str)
            if isinstance(inner, list) and len(inner) > 39:
                hex_id = inner[39] if isinstance(inner[39], str) else None
                label = inner[42] if len(inner) > 42 and isinstance(inner[42], str) else None
                if hex_id or label:
                    return hex_id, label
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    return None, None


def _extract_texts_from_line(line: str) -> list:
    if '"wrb.fr"' not in line or len(line) < 200:
        return []
    try:
        arr = json.loads(line)
        inner_str = arr[0][2]
        if not inner_str or len(inner_str) < 50:
            return []
        inner = json.loads(inner_str)
        if not (isinstance(inner, list) and len(inner) > 4 and inner[4]):
            return []
        texts = []
        for part in inner[4]:
            if isinstance(part, list) and len(part) > 1 and part[1] and isinstance(part[1], list):
                for t in part[1]:
                    if isinstance(t, str) and t:
                        texts.append(t)
        return texts
    except (json.JSONDecodeError, IndexError, TypeError):
        return []


def extract_response_text(raw: str) -> str:
    bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', raw)
    if bard_err:
        raise RuntimeError(f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]")
    hex_id, label = extract_served_model_info(raw)
    if hex_id or label:
        log(f"[model-verify] Served model: {label or 'unknown'} [{hex_id or 'no-hex'}]")
    last_text = ""
    conv_id = resp_id = rcid = ""
    for line in raw.split("\n"):
        c, r, rc = extract_conv_ids_from_line(line)
        if c:
            conv_id = c
        if r:
            resp_id = r
        if rc:
            rcid = rc
        for t in _extract_texts_from_line(line):
            cleaned = clean_text(t)
            if cleaned:
                last_text = cleaned
    set_last_conv_ids((conv_id, resp_id, rcid))
    return last_text


def generate(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None) -> str:
    load_cookie()
    ctx = _get_ssl_ctx()
    proxy = CONFIG.get("proxy")
    refreshed = False

    last_err = None
    attempts = max(1, int(CONFIG.get("retry_attempts", 3)))
    attempt = 0
    while attempt < attempts:
        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields).encode()
        url = _get_url()
        headers = _build_headers(extra_fields)
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            if proxy:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy, "https": proxy}),
                    urllib.request.HTTPSHandler(context=ctx)
                )
                resp = opener.open(req, timeout=CONFIG["request_timeout_sec"])
            else:
                resp = urllib.request.urlopen(req, context=ctx, timeout=CONFIG["request_timeout_sec"])
            raw = resp.read().decode("utf-8", errors="replace")
            return extract_response_text(raw)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (400, 405) and not refreshed and refresh_bl_and_xsrf():
                refreshed = True
                log(f"HTTP {e.code}: refreshed BL/XSRF, retrying immediately...")
                continue
            attempt += 1
            if attempt < attempts:
                log(f"Retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
        except Exception as e:
            last_err = e
            attempt += 1
            if attempt < attempts:
                log(f"Retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err


def generate_stream(prompt: str, model_id: int, think_mode: int, file_refs: list = None, extra_fields: dict = None):
    if not HAS_HTTPX:
        text = generate(prompt, model_id, think_mode, file_refs, extra_fields)
        if text:
            yield text
        return

    load_cookie()
    client = _get_httpx_client()
    refreshed = False

    last_err = None
    emitted_raw_text = ""
    attempts = max(1, int(CONFIG.get("retry_attempts", 3)))
    attempt = 0
    while attempt < attempts:
        body = _build_payload(prompt, model_id, think_mode, file_refs, extra_fields)
        url = _get_url()
        headers = _build_headers(extra_fields)
        try:
            with client.stream("POST", url, content=body, headers=headers) as resp:
                if resp.status_code in (400, 405) and not refreshed and not emitted_raw_text:
                    refreshed = True
                    if refresh_bl_and_xsrf():
                        log(f"Stream HTTP {resp.status_code}: refreshed BL/XSRF, retrying immediately...")
                        continue
                resp.raise_for_status()
                buf = ""
                for chunk in resp.iter_text():
                    buf += chunk
                    if "BardErrorInfo" in buf:
                        bard_err = re.search(r'BardErrorInfo\s*\[(\d+)\]', buf)
                        if bard_err:
                            raise RuntimeError(
                                f"Gemini upstream rejected request: BardErrorInfo [{bard_err.group(1)}]"
                            )
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        c, r, rc = extract_conv_ids_from_line(line)
                        if c or r or rc:
                            prev_c, prev_r, prev_rc = get_last_conv_ids()
                            set_last_conv_ids((c or prev_c, r or prev_r, rc or prev_rc))
                        for t in _extract_texts_from_line(line):
                            if t == emitted_raw_text or emitted_raw_text.startswith(t):
                                continue
                            if not t.startswith(emitted_raw_text):
                                raise RuntimeError("Gemini stream content changed during retry")
                            delta = clean_text(t[len(emitted_raw_text):], strip=False)
                            emitted_raw_text = t
                            if delta:
                                yield delta
            return
        except Exception as e:
            last_err = e
            if emitted_raw_text:
                raise
            attempt += 1
            if attempt < attempts:
                log(f"Stream retry {attempt}/{attempts}: {e}")
                time.sleep(CONFIG["retry_delay_sec"])
    raise last_err
