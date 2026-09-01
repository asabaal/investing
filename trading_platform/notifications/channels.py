"""Notification channels with severity levels (plan §28).

Severity: INFORMATION | ACTION | WARNING | CRITICAL.
v1 channels: log (always), file (alert inbox), desktop (best-effort), and a
generic webhook (Slack/Discord-compatible) enabled via environment variable
ALERT_WEBHOOK_URL.  One channel is enough to start - the router makes adding
more trivial.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SEVERITY_ORDER = {"INFORMATION": 0, "ACTION": 1, "WARNING": 2, "CRITICAL": 3}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


class LogChannel:
    name = "log"

    def send(self, severity: str, subject: str, body: str, signal_id: str | None = None) -> bool:
        line = f"[{severity}] {subject}"
        if severity == "CRITICAL":
            import logging
            logging.getLogger("trading_platform").error(line)
        print(line)
        return True


class FileChannel:
    name = "file"

    def __init__(self, alert_dir: str | Path):
        self.alert_dir = Path(alert_dir)
        self.alert_dir.mkdir(parents=True, exist_ok=True)

    def send(self, severity: str, subject: str, body: str, signal_id: str | None = None) -> bool:
        stamp = _now_iso().replace(":", "")
        fname = self.alert_dir / f"{severity.lower()}_{stamp}.txt"
        fname.write_text(f"{subject}\n\n{body}\n", encoding="utf-8")
        return True


class DesktopChannel:
    """Best-effort desktop notification via notify-send (Linux)."""
    name = "desktop"

    def send(self, severity: str, subject: str, body: str, signal_id: str | None = None) -> bool:
        try:
            subprocess.Popen(
                ["notify-send", f"-u", {"CRITICAL": "critical"}.get(severity, "normal"),
                 subject, body[:200]],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False


class WebhookChannel:
    """Generic JSON webhook (Slack/Discord-compatible payload)."""
    name = "webhook"

    def __init__(self, url: str):
        self.url = url

    def send(self, severity: str, subject: str, body: str, signal_id: str | None = None) -> bool:
        import requests

        try:
            requests.post(self.url, json={"content": f"[{severity}] {subject}\n{body}"},
                          timeout=5)
            return True
        except Exception:
            return False


class NotificationRouter:
    def __init__(self, alert_dir: str | Path = "alerts", min_severity: str = "INFORMATION",
                 channels: list | None = None):
        self.channels = channels if channels is not None else [
            LogChannel(), FileChannel(alert_dir),
        ]
        webhook = os.environ.get("ALERT_WEBHOOK_URL")
        if webhook:
            self.channels.append(WebhookChannel(webhook))
        self.min_severity = SEVERITY_ORDER.get(min_severity, 0)
        self._db = None

    def attach_db(self, db) -> None:
        self._db = db

    def send(self, severity: str, subject: str, body: str,
             signal_id: str | None = None) -> None:
        if SEVERITY_ORDER.get(severity, 0) < self.min_severity:
            return
        for ch in self.channels:
            try:
                ok = ch.send(severity, subject, body, signal_id)
            except Exception:
                ok = False
            if self._db is not None:
                self._db.record_alert(severity, ch.name, subject, body, signal_id)
