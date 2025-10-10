from __future__ import annotations

import os
import time
import subprocess
import asyncio
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


def require_confirm(confirm: Optional[bool]) -> None:
    """Require confirmation for destructive operations."""
    if not confirm:
        raise ValueError("This operation is destructive. Pass confirm=true to proceed.")


def format_size(size_bytes: int) -> str:
    """Format byte size into human readable format."""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    if i == 0:
        return f"{int(size_bytes)} {size_names[i]}"
    else:
        return f"{size_bytes:.1f} {size_names[i]}"


async def run_command(
    cmd: list[str], 
    input_data: Optional[str] = None,
    shell: bool = False,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Run a command asynchronously and return the result."""
    try:
        if shell and isinstance(cmd, list):
            cmd = " ".join(cmd)
        
        process = await asyncio.create_subprocess_shell(
            cmd if shell else " ".join(cmd),
            stdin=subprocess.PIPE if input_data else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env
        )
        
        stdout, stderr = await process.communicate(
            input=input_data.encode() if input_data else None
        )
        
        return {
            "return_code": process.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
            "command": cmd
        }
        
    except Exception as e:
        return {
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "command": cmd,
            "error": str(e)
        }


def format_error(message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Format an error response with context."""
    error_response = {
        "error": True,
        "message": message,
        "timestamp": time.time()
    }
    
    if context:
        error_response["context"] = context
        
    return error_response
