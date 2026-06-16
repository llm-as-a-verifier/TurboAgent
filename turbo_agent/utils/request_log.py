import json
import os
import random
import string
from datetime import datetime, timezone
from typing import Any, Dict

from .logging_utils import create_logger

_logger = create_logger("request_log")


def _short_id() -> str:
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=6))


def create_request_log(api: str, config: Any) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    ts = now.isoformat().replace(":", "-").replace(".", "-")
    return {
        "id": f"{ts}_{_short_id()}",
        "timestamp": now.isoformat(),
        "api": api,
        "config": config,
        "request": None,
        "contextRefinement": {"enabled": False},
        "responses": [],
        "reflection": {"enabled": False},
        "verifier": {"enabled": False},
        "progressMonitor": {"enabled": False},
        "finalResponse": None,
        "elapsedMs": 0,
    }


def save_request_log(log: Dict[str, Any], log_dir: str) -> None:
    try:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, f"{log['id']}.json")
        with open(file_path, "w") as f:
            json.dump(log, f, indent=2, default=str)
        _logger.info(f"Saved request log: {file_path}")
    except Exception as e:
        _logger.error(f"Failed to save request log: {e}")
