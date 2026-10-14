"""Model definitions and mapping from Gemini frontend JS source."""
import json
import uuid

# MODE_CATEGORY enum from Gemini Web frontend JS:
#   1=FAST, 2=THINKING, 3=PRO, 4=AUTO, 5=FAST_DYNAMIC_THINKING, 6=FLASH_LITE
#
# Model routing on current Gemini Web requires BOTH:
#   - f.req inner[79] = mode (family), inner[80] = variant, inner[17] = [[think_mode]]
#   - HTTP header X-Goog-Ext-525001261-Jspb containing the 16-hex model ticket ID

REASONING_EFFORT_MAP = {
    "minimal": (4, 1),
    "low": (4, 1),
    "medium": (2, 2),
    "high": (0, 3),
    "xhigh": (0, 3),
    "max": (0, 3),
}

MODELS = {
    "gemini-4-argon": {
        "mode": 3, "think": 2, "variant": 2,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (Pro/Ultra web routing, medium thinking default)",
    },
    "gemini-4-argon-low": {
        "mode": 3, "think": 4, "variant": 1,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (low thinking effort)",
    },
    "gemini-4-argon-medium": {
        "mode": 3, "think": 2, "variant": 2,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (medium thinking effort)",
    },
    "gemini-4-argon-high": {
        "mode": 3, "think": 0, "variant": 3,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "argon",
        "desc": "Gemini 4 Argon (high / deepest thinking effort)",
    },
    "gemini-3.8-flash": {
        "mode": 1, "think": 4, "variant": 1,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash",
        "desc": "Latest all-around model (Gemini 3.8 Flash)",
    },
    "gemini-3.8-flash-thinking": {
        "mode": 2, "think": 0, "variant": 1,
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
        "mode": 3, "think": 2, "variant": 2,
        "ticket": "e6fa609c3fa255c0",
        "ticket_key": "pro",
        "desc": "Pro model (medium thinking effort)",
    },
    "gemini-3.1-pro-high": {
        "mode": 3, "think": 0, "variant": 3,
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
        "mode": 6, "think": 4, "variant": 0,
        "ticket": "56fdd199312815e2",
        "ticket_key": "flash_lite",
        "desc": "Lightweight fast model",
    },
}


def resolve_ticket_hex(cfg: dict, model_name: str = None, model_tickets: dict = None) -> str:
    """Resolve the 16-hex model ticket ID, preferring runtime/config overrides."""
    tickets = model_tickets or {}
    if model_name and model_name in tickets and tickets[model_name]:
        return tickets[model_name]
    ticket_key = cfg.get("ticket_key")
    if ticket_key and ticket_key in tickets and tickets[ticket_key]:
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
    default: str = "gemini-3.6-flash",
    reasoning_effort: str = None,
    model_tickets: dict = None,
):
    """Resolve model name to (name, mode_id, think_mode, error, extra_fields).

    Supports:
      - @think=N suffix (e.g. gemini-4-argon@think=0)
      - reasoning_effort ("low", "medium", "high")
    Unknown model names fall back to default rather than erroring,
    since upstream clients may request arbitrary model identifiers.
    """
    think_override = None
    variant_override = None
    if model_name and "@think=" in model_name:
        model_name, think_str = model_name.rsplit("@think=", 1)
        try:
            think_override = int(think_str)
        except ValueError:
            return None, None, None, f"Invalid think level: {think_str}", None

    if reasoning_effort:
        effort_key = str(reasoning_effort).strip().lower()
        if effort_key in REASONING_EFFORT_MAP:
            effort_think, effort_variant = REASONING_EFFORT_MAP[effort_key]
            if think_override is None:
                think_override = effort_think
            variant_override = effort_variant

    cfg = MODELS.get(model_name)
    if not cfg:
        from .gemini import log
        log(f"Unknown model '{model_name}', falling back to '{default}'")
        model_name = default
        cfg = MODELS[default]

    mode_id = cfg["mode"]
    think_mode = think_override if think_override is not None else cfg["think"]
    extra = dict(cfg.get("extra") or {})
    effective_variant = variant_override if variant_override is not None else cfg.get("variant")
    if effective_variant is not None and (80 not in extra or variant_override is not None):
        extra[80] = effective_variant

    if model_tickets is None:
        from .config import CONFIG
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
