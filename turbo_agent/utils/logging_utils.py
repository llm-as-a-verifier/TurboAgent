import json
import os
import sys
from datetime import datetime
from typing import Union


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Logger:
    def __init__(self, name: str):
        self.name = name

    def info(self, msg: str) -> None:
        print(
            f"\033[2m[{_timestamp()}]\033[0m \033[36m{self.name}\033[0m {msg}",
            flush=True,
        )

    def warn(self, msg: str) -> None:
        print(
            f"\033[2m[{_timestamp()}]\033[0m \033[33m{self.name}\033[0m {msg}",
            file=sys.stderr,
            flush=True,
        )

    def error(self, msg: str) -> None:
        print(
            f"\033[2m[{_timestamp()}]\033[0m \033[31m{self.name}\033[0m {msg}",
            file=sys.stderr,
            flush=True,
        )

    def debug(self, msg: str) -> None:
        if os.environ.get("DEBUG"):
            print(
                f"\033[2m[{_timestamp()}]\033[0m \033[90m{self.name}\033[0m {msg}",
                flush=True,
            )


logger = Logger("turbo_agent")


def create_logger(name: str) -> Logger:
    return Logger(name)


def summarize_request_body(body: Union[bytes, str]) -> str:
    try:
        data = json.loads(body if isinstance(body, str) else body.decode())
        parts = []
        if data.get("model"):
            parts.append(f"model={data['model']}")
        if data.get("messages"):
            parts.append(f"messages={len(data['messages'])}")
        if "stream" in data:
            parts.append(f"stream={data['stream']}")
        if data.get("max_tokens"):
            parts.append(f"max_tokens={data['max_tokens']}")
        system = data.get("system")
        if system:
            if isinstance(system, str):
                parts.append(f"system={len(system)} chars")
            elif isinstance(system, list):
                parts.append(f"system={len(system)} blocks")
        if data.get("tools"):
            parts.append(f"tools={len(data['tools'])}")
        if data.get("thinking"):
            parts.append(f"thinking={data['thinking']}")
        return " | ".join(parts) if parts else "(empty JSON)"
    except Exception:
        length = len(body) if isinstance(body, (str, bytes)) else 0
        return f"({length} bytes, not JSON)"


def log_response_summary(body: Union[bytes, str], status_code: int) -> None:
    try:
        data = json.loads(body if isinstance(body, str) else body.decode())
        parts = [f"status={status_code}"]
        if data.get("stop_reason"):
            parts.append(f"stop_reason={data['stop_reason']}")
        usage = data.get("usage")
        if usage:
            parts.append(
                f"tokens(in={usage.get('input_tokens', '?')}, out={usage.get('output_tokens', '?')})"
            )
        if data.get("type"):
            parts.append(f"type={data['type']}")
        logger.info(f"\033[1;32mRESP\033[0m {' | '.join(parts)}")
    except Exception:
        length = len(body) if isinstance(body, (str, bytes)) else 0
        logger.info(f"RESP status={status_code} ({length} bytes, not JSON)")
