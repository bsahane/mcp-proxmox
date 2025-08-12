from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse


@dataclass
class ProxmoxEnv:
    base_url: str
    token_id: str
    token_secret: str
    verify: bool
    default_node: Optional[str] = None
    default_storage: Optional[str] = None
    default_bridge: Optional[str] = None


def strtobool(value: Optional[str], default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def read_env() -> ProxmoxEnv:
    base_url = os.environ.get("PROXMOX_API_URL", "").strip()
    token_id = os.environ.get("PROXMOX_TOKEN_ID", "").strip()
    token_secret = os.environ.get("PROXMOX_TOKEN_SECRET", "").strip()
    verify = strtobool(os.environ.get("PROXMOX_VERIFY"), True)

    default_node = os.environ.get("PROXMOX_DEFAULT_NODE") or None
    default_storage = os.environ.get("PROXMOX_DEFAULT_STORAGE") or None
    default_bridge = os.environ.get("PROXMOX_DEFAULT_BRIDGE") or None

    if not base_url:
        raise ValueError("Missing PROXMOX_API_URL")
    if not token_id:
        raise ValueError("Missing PROXMOX_TOKEN_ID (format: user@realm!tokenname)")
    if not token_secret:
        raise ValueError("Missing PROXMOX_TOKEN_SECRET")

    return ProxmoxEnv(
        base_url=base_url,
        token_id=token_id,
        token_secret=token_secret,
        verify=verify,
        default_node=default_node,
        default_storage=default_storage,
        default_bridge=default_bridge,
    )


def parse_api_url(base_url: str) -> Dict[str, Any]:
    """Parse API URL into components suitable for proxmoxer.ProxmoxAPI.

    Accepts forms like:
      - https://host:8006
      - https://host:8006/api2/json
      - https://host
    """
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError(f"Invalid PROXMOX_API_URL: {base_url}")
    port = parsed.port or 8006
    return {
        "host": parsed.hostname,
        "port": port,
        "scheme": parsed.scheme,
    }


def split_token_id(token_id: str) -> Dict[str, str]:
    """Split token_id of the form 'user@realm!tokenname' into components."""
    if "!" not in token_id:
        raise ValueError("PROXMOX_TOKEN_ID must include '!' separating user and token name, e.g. root@pam!mcp")
    user, token_name = token_id.split("!", 1)
    if "@" not in user:
        raise ValueError("PROXMOX_TOKEN_ID user part must include '@realm', e.g. root@pam!mcp")
    return {"user": user, "token_name": token_name}


def now_ms() -> int:
    return int(time.time() * 1000)
