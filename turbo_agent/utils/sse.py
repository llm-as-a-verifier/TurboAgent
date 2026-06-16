import json
import uuid
from typing import Optional


class SSEFormatter:
    @staticmethod
    def event(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def message_start(model_name: str, msg_id: Optional[str] = None) -> str:
        if not msg_id:
            msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        return SSEFormatter.event("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

    @staticmethod
    def content_block_start(
        index: int,
        block_type: str = "text",
        tool_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> str:
        if block_type == "text":
            content_block = {"type": "text", "text": ""}
        elif block_type == "tool_use":
            content_block = {
                "type": "tool_use",
                "id": tool_id or "",
                "name": tool_name or "",
                "input": {},
            }
        else:
            content_block = {"type": block_type}

        return SSEFormatter.event("content_block_start", {
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        })

    @staticmethod
    def content_block_delta(index: int, delta: dict) -> str:
        return SSEFormatter.event("content_block_delta", {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        })

    @staticmethod
    def text_delta(index: int, text: str) -> str:
        return SSEFormatter.content_block_delta(index, {
            "type": "text_delta",
            "text": text,
        })

    @staticmethod
    def input_json_delta(index: int, partial_json: str) -> str:
        return SSEFormatter.content_block_delta(index, {
            "type": "input_json_delta",
            "partial_json": partial_json,
        })

    @staticmethod
    def content_block_stop(index: int) -> str:
        return SSEFormatter.event("content_block_stop", {
            "type": "content_block_stop",
            "index": index,
        })

    @staticmethod
    def message_delta(stop_reason: str, output_tokens: int = 0) -> str:
        return SSEFormatter.event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        })

    @staticmethod
    def message_stop() -> str:
        return SSEFormatter.event("message_stop", {"type": "message_stop"})

    @staticmethod
    def error(message: str) -> str:
        return SSEFormatter.event("error", {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        })
