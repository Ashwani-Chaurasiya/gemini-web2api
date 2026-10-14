"""Model definitions and mapping from Gemini frontend JS source."""
import json
import uuid

# MODE_CATEGORY enum from Gemini Web frontend JS (_.dF):
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING,
#   6=FLASH_LITE, 7=FLASH_PLUS, 8=ULTRA (4 Argon)
#
# THINKING_LEVEL enum in X-Goog-Ext-525001261-Jspb slot 15 (_.Fod):
#   1=THINKING_LEVEL_STANDARD (Low)
#   4=THINKING_LEVEL_MEDIUM   (Medium)
#   2=THINKING_LEVEL_EXTENDED (High)
#   3=THINKING_LEVEL_DEEP_THINK

REASONING_EFFORT_MAP = {
    "minimal": (4, 1),
    "low": (4, 1),
    "medium": (2, 4),
    "high": (0, 2),
    "xhigh": (0, 2),
    "max": (0, 2),
}

THINK_TO_VARIANT = {
    0: 2,
    1: 2,
    2: 4,
    3: 1,
    4: 1,
}

MODELS = {
    "gemini-4-argon": {
        "mode": 8, "think": 2, "variant": 4,
        "ticket": "95a4c0abcba24364",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (Ultra web routing, medium thinking default)",
    },
    "gemini-4-argon-low": {
        "mode": 8, "think": 4, "variant": 1,
        "ticket": "95a4c0abcba24364",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (low thinking effort)",
    },
    "gemini-4-argon-medium": {
        "mode": 8, "think": 2, "variant": 4,
        "ticket": "95a4c0abcba24364",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (medium thinking effort)",
    },
    "gemini-4-argon-high": {
        "mode": 8, "think": 0, "variant": 2,
        "ticket": "95a4c0abcba24364",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (high / extended thinking effort)",
    },
    "gemini-3.8-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Latest all-around model (Gemini 3.8 Flash)",
    },
    "gemini-3.8-flash-thinking": {
        "mode": 2, "think": 0, "variant": 2,
        "ticket": "e051ce1aa80aa576",
        "ticket_key": "thinking",
        "desc": "Gemini 3.8 Flash with deep thinking (~20k chars)",
    },
    "gemini-3.8-flash-thinking-lite": {
        "mode": 5, "think": 0, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Gemini 3.8 Flash with adaptive thinking depth",
    },
    "gemini-3.7-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "All-around model (Gemini 3.7 Flash)",
    },
    "gemini-3.6-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "All-around model (Gemini 3.6 Flash)",
    },
    "gemini-3.5-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Alias for gemini-3.6-flash (backend upgraded)",
    },
    "gemini-3.5-flash-thinking": {
        "mode": 2, "think": 0, "variant": 1,
        "ticket": "e051ce1aa80aa576",
        "ticket_key": "thinking",
        "desc": "Deep thinking mode, longest output (~20k chars)",
    },
    "gemini-3.1-pro": {
        "mode": 3, "think": 4, "variant": 1,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "desc": "Pro model (requires cookie for real routing)",
    },
    "gemini-3.1-pro-low": {
        "mode": 3, "think": 4, "variant": 1,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "desc": "Pro model (low thinking effort)",
    },
    "gemini-3.1-pro-medium": {
        "mode": 3, "think": 2, "variant": 4,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "desc": "Pro model (medium thinking effort)",
    },
    "gemini-3.1-pro-high": {
        "mode": 3, "think": 0, "variant": 2,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "desc": "Pro model (high thinking effort)",
    },
    "gemini-3.1-pro-enhanced": {
        "mode": 3, "think": 4, "variant": 3,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "extra": {31: 2, 80: 3},
        "desc": "Pro with enhanced output (experimental)",
    },
    "gemini-auto": {
        "mode": 4, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Auto model selection",
    },
    "gemini-3.5-flash-thinking-lite": {
        "mode": 5, "think": 0, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Dynamic thinking with adaptive depth",
    },
    "gemini-flash-lite": {
        "mode": 6, "think": 4, "variant": 1,
        "ticket": "8c46e95b1a07cecc",
        "ticket_key": "flash_lite",
        "desc": "Lightweight fast model",
    },
}


def normalize_model_name(model_name: str) -> str:
    """Strip optional provider prefix (e.g. 'google/gemini-4-argon') when matched."""
    if not isinstance(model_name, str):
        return ""
    name = model_name.strip()
    if "/" in name and name not in MODELS:
        candidate = name.rsplit("/", 1)[-1]
        if candidate.rsplit("@think=", 1)[0] in MODELS:
            return candidate
    return name


def resolve_ticket_hex(cfg: dict, model_name: str = None, model_tickets: dict = None) -> str:
    """Resolve the 16-hex model ticket ID, preferring runtime/config overrides."""
    tickets = model_tickets or {}
    if model_name and tickets.get(model_name):
        return tickets[model_name]
    ticket_key = cfg.get("ticket_key")
    if ticket_key and tickets.get(ticket_key):
        return tickets[ticket_key]
    return cfg.get("ticket")


def build_model_ticket(
    cfg: dict,
    model_name: str = None,
    variant_override: int = None,
    model_tickets: dict = None,
) -> str:
    """Build the X-Goog-Ext-525001261-Jspb header value for StreamGenerate."""
    ticket = resolve_ticket_hex(cfg, model_name=model_name, model_tickets=model_tickets)
    if not ticket:
        return None
    variant = variant_override if variant_override is not None else cfg.get("variant", 1)
    arr = [
        1, None, None, None, ticket, None, None, 0,
        [4, 5, 6, 8, 4, 5, 6, 8],
        None, None, 2, None, None,
        cfg["mode"], variant, str(uuid.uuid4()),
    ]
    return json.dumps(arr, separators=(",", ":"))


def resolve_model(
    model_name: str,
    default: str = None,
    reasoning_effort: str = None,
    model_tickets: dict = None,
):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields)."""
    from .config import CONFIG
    from .gemini import load_cookie, log

    load_cookie()

    if default is None or default not in MODELS:
        cfg_default = CONFIG.get("default_model", "gemini-3.6-flash")
        default = cfg_default if cfg_default in MODELS else "gemini-3.6-flash"

    model_name = normalize_model_name(model_name or default)
    think_override = None
    variant_override = None

    if reasoning_effort:
        effort_key = str(reasoning_effort).strip().lower()
        if effort_key in REASONING_EFFORT_MAP:
            think_override, variant_override = REASONING_EFFORT_MAP[effort_key]

    if "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
            if think_override in THINK_TO_VARIANT:
                variant_override = THINK_TO_VARIANT[think_override]
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None

    cfg = MODELS.get(model_name)
    if not cfg:
        log(f"Unknown model '{model_name}', falling back to '{default}'")
        model_name = default
        cfg = MODELS[default]

    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = dict(cfg.get("extra") or {})
    effective_variant = variant_override if variant_override is not None else cfg.get("variant", 1)

    if CONFIG.get("send_model_ticket", True):
        if model_tickets is None:
            model_tickets = CONFIG.get("model_tickets") or {}
        ticket_header = build_model_ticket(
            cfg,
            model_name=model_name,
            variant_override=effective_variant,
            model_tickets=model_tickets,
        )
        if ticket_header:
            extra["_ticket"] = ticket_header

    return model_name, mode_id, think_mode, None, extra
