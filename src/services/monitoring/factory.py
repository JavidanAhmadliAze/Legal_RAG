from __future__ import annotations

from .client import MonitoringClient

_client: MonitoringClient | None = None


def get_monitoring_client() -> MonitoringClient:
    global _client
    if _client is None:
        _client = MonitoringClient()
    return _client
