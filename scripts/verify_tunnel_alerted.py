#!/usr/bin/env python3
"""Daily canary: run verify_tunnel.py, alert on red.

Designed for Windows Task Scheduler (PF-VerifyTunnel). Always exits 0 so
Task Scheduler doesn't classify the run as failed and trigger retries —
the alerts (Telegram + critical_errors.jsonl fallback) ARE the failure
signal.

Two intentional escapes from project conventions:

1. Bypasses `telegram.mute_all` and `telegram.layer1_messages` gates by
   posting directly to the Telegram Bot API rather than going through
   portfolio.telegram_notifications.send_telegram(). Reasoning: the user
   mutes their main Telegram feed to silence trading alerts. The tunnel
   canary fires only on actual problems (security regression, tunnel
   down, origin down), all of which the user wants to know about even
   under mute_all=true. If you want the canary muted too, disable the
   PF-VerifyTunnel scheduled task instead of touching mute_all.

2. Falls back to data/critical_errors.jsonl if Telegram itself fails.
   That feeds the existing PF-FixAgentDispatcher (CLAUDE.md) which spawns
   a Claude fix agent.

Failure modes detected:
  - exit 1: verify_tunnel.py asserted false on a green-deployment claim
            (cookie not set, gate not firing, alias broken, etc.)
  - exit 2: config.json's dashboard_token missing — broken deployment
  - subprocess.TimeoutExpired: verifier hung; tunnel likely unreachable
  - any other exception: meta-failure of the canary itself
"""

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT / ".venv" / "Scripts" / "python.exe"
VERIFIER = PROJECT / "scripts" / "verify_tunnel.py"
TIMEOUT = 60


def send_telegram_direct(msg: str, token: str, chat_id: str) -> bool:
    """Direct Bot API POST — bypasses mute_all / layer1_messages gates.

    Mirrors the logic in portfolio.telegram_notifications.send_telegram:
    try Markdown first, fall back to plain text on parse error so the
    message still arrives (unformatted is better than lost).
    """
    import requests
    api = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(
            api,
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=30,
        )
        if r.ok:
            return True
        # Markdown parse failure → retry plain text
        if r.status_code == 400:
            r2 = requests.post(
                api,
                json={"chat_id": chat_id, "text": msg},
                timeout=30,
            )
            return r2.ok
        return False
    except requests.RequestException:
        return False


def write_critical_error(category: str, message: str, context: dict) -> None:
    sys.path.insert(0, str(PROJECT))
    from portfolio.file_utils import atomic_append_jsonl
    atomic_append_jsonl(
        str(PROJECT / "data" / "critical_errors.jsonl"),
        {
            "ts": datetime.now(UTC).isoformat(),
            "level": "critical",
            "category": category,
            "caller": "verify_tunnel_alerted",
            "message": message,
            "context": context,
        },
    )


def alert(headline: str, output: str, exit_code: str) -> None:
    sys.path.insert(0, str(PROJECT))
    from portfolio.file_utils import load_json
    cfg = load_json(PROJECT / "config.json", default={}) or {}
    tg = cfg.get("telegram", {})
    token = tg.get("token", "")
    chat_id = tg.get("chat_id", "")

    msg = f"🚨 *Tunnel canary*: {headline}\n```\n{output}\n```"
    sent = False
    if token and chat_id:
        sent = send_telegram_direct(msg, token, chat_id)

    if not sent:
        write_critical_error(
            category="tunnel_canary_alert_failed",
            message=f"Tunnel canary fired but Telegram alert not sent: {headline}",
            context={
                "verifier_exit": exit_code,
                "verifier_output_tail": output,
                "telegram_configured": bool(token and chat_id),
            },
        )


def main() -> int:
    try:
        result = subprocess.run(
            [str(PYTHON), str(VERIFIER)],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=str(PROJECT),
        )
        if result.returncode == 0:
            return 0  # All checks passed. Silent success.

        output_tail = (result.stdout + result.stderr)[-1500:].strip()
        alert(
            headline=f"verify_tunnel.py exited {result.returncode}",
            output=output_tail,
            exit_code=str(result.returncode),
        )
    except subprocess.TimeoutExpired:
        alert(
            headline=f"verify_tunnel.py timed out after {TIMEOUT}s",
            output="Tunnel likely unreachable (cloudflared service down or DNS broken).",
            exit_code="timeout",
        )
    except Exception as exc:  # noqa: BLE001 — last-ditch alert path
        alert(
            headline="canary itself failed to run",
            output=f"{type(exc).__name__}: {exc}",
            exit_code="exception",
        )

    # Always exit 0. The alert IS the failure signal; we don't want Task
    # Scheduler retry policies thrashing.
    return 0


if __name__ == "__main__":
    sys.exit(main())
