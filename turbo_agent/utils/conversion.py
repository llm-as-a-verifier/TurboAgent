import json
import uuid
from typing import Any


STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


class AnthropicToOpenAI:
    @staticmethod
    def content_block(block: Any) -> Any:
        if isinstance(block, str):
            return block

        t = block.get("type")
        if t == "text":
            result: dict = {"type": "text", "text": block.get("text", "")}
            if block.get("cache_control") is not None:
                result["cache_control"] = block["cache_control"]
            return result
        if t == "image":
            source = block.get("source", {})
            result = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}",
                },
            }
            if block.get("cache_control") is not None:
                result["cache_control"] = block["cache_control"]
            return result
        if t in ("tool_use", "tool_result"):
            return None
        if block.get("text"):
            return {"type": "text", "text": block["text"]}
        return None

    @staticmethod
    def messages(anthropic_body: dict, text_only: bool = False) -> list:
        """Convert Anthropic messages to OpenAI format.

        Args:
            text_only: If True, inline tool calls/results as text instead of
                using OpenAI tool_calls/tool role. Use this when forwarding to
                models that reject foreign tool call history (e.g. Gemini
                thought_signature requirement).
        """
        openai_messages: list = []

        # System prompt
        system = anthropic_body.get("system")
        if system:
            if isinstance(system, str):
                openai_messages.append({"role": "system", "content": system})
            elif isinstance(system, list):
                has_cache_control = any(
                    isinstance(block, dict) and block.get("cache_control") is not None
                    for block in system
                )
                if has_cache_control:
                    # Preserve block boundaries so each block's cache_control
                    # breakpoint survives the conversion.
                    blocks = []
                    for block in system:
                        if isinstance(block, dict) and block.get("type") == "text":
                            b = {"type": "text", "text": block.get("text", "")}
                            if block.get("cache_control") is not None:
                                b["cache_control"] = block["cache_control"]
                            blocks.append(b)
                        elif isinstance(block, str):
                            blocks.append({"type": "text", "text": block})
                    if blocks:
                        openai_messages.append({"role": "system", "content": blocks})
                else:
                    text_parts = []
                    for block in system:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block["text"])
                        elif isinstance(block, str):
                            text_parts.append(block)
                    if text_parts:
                        openai_messages.append(
                            {"role": "system", "content": "\n".join(text_parts)}
                        )

        for msg in anthropic_body.get("messages", []):
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue

            if not isinstance(content, list):
                openai_messages.append({"role": role, "content": content})
                continue

            if role == "assistant":
                if text_only:
                    openai_messages.append(
                        AnthropicToOpenAI._convert_assistant_message_text_only(content)
                    )
                else:
                    openai_messages.append(
                        AnthropicToOpenAI._convert_assistant_message(content)
                    )
            elif role == "user":
                if text_only:
                    openai_messages.extend(
                        AnthropicToOpenAI._convert_user_message_text_only(content)
                    )
                else:
                    openai_messages.extend(
                        AnthropicToOpenAI._convert_user_message(content)
                    )
            else:
                openai_messages.append({"role": role, "content": content})

        system_parts: list = []
        conversation: list = []
        for msg in openai_messages:
            if msg.get("role") != "system":
                conversation.append(msg)
                continue
            msg_content = msg.get("content")
            if isinstance(msg_content, str):
                system_parts.append(msg_content)
            elif isinstance(msg_content, list):
                for block in msg_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        system_parts.append(block)

        system_parts = [part for part in system_parts if part]
        if system_parts:
            return [
                {"role": "system", "content": "\n\n".join(system_parts)}
            ] + conversation
        return conversation

    @staticmethod
    def _convert_assistant_message(content: list) -> dict:
        tool_calls = []
        text_parts = []
        text_cache_control = None

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_input = block.get("input", {})
                tool_call: dict = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": (
                            json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input)
                        ),
                    },
                }
                if block.get("cache_control") is not None:
                    tool_call["cache_control"] = block["cache_control"]
                tool_calls.append(tool_call)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
                if block.get("cache_control") is not None:
                    text_cache_control = block["cache_control"]
            elif isinstance(block, str):
                text_parts.append(block)
            # Skip thinking, thinking_delta, and other Claude-specific blocks

        oai_msg: dict = {"role": "assistant"}
        joined_text = "\n".join(text_parts) if text_parts else None
        if text_cache_control is not None and joined_text is not None:
            oai_msg["content"] = [
                {"type": "text", "text": joined_text, "cache_control": text_cache_control}
            ]
        else:
            oai_msg["content"] = joined_text
        if tool_calls:
            oai_msg["tool_calls"] = tool_calls
        return oai_msg

    @staticmethod
    def _convert_assistant_message_text_only(content: list) -> dict:
        """Convert assistant message, inlining tool calls as text.

        Used when forwarding to models that don't support thought signatures
        from other providers (e.g. Gemini rejecting Claude's tool call history).
        """
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_input = block.get("input", {})
                args = (
                    json.dumps(tool_input)
                    if isinstance(tool_input, dict)
                    else str(tool_input)
                )
                text_parts.append(
                    f"[tool_use: {block.get('name', '')}({args})]"
                )
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        return {
            "role": "assistant",
            "content": "\n".join(text_parts) if text_parts else None,
        }

    @staticmethod
    def _convert_user_message(content: list) -> list:
        results = []
        regular_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    texts = []
                    for b in tr_content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            texts.append(b["text"])
                        elif isinstance(b, str):
                            texts.append(b)
                    tr_content = "\n".join(texts)
                tool_msg: dict = {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(tr_content) if tr_content else "",
                }
                if block.get("cache_control") is not None:
                    tool_msg["cache_control"] = block["cache_control"]
                results.append(tool_msg)
            else:
                converted = AnthropicToOpenAI.content_block(block)
                if converted is not None:
                    regular_parts.append(converted)

        msgs = list(results)

        if regular_parts:
            if len(regular_parts) == 1 and isinstance(regular_parts[0], str):
                msgs.append({"role": "user", "content": regular_parts[0]})
            elif (
                len(regular_parts) == 1
                and isinstance(regular_parts[0], dict)
                and regular_parts[0].get("type") == "text"
                and "cache_control" not in regular_parts[0]
            ):
                msgs.append({"role": "user", "content": regular_parts[0]["text"]})
            else:
                msgs.append({"role": "user", "content": regular_parts})

        return msgs

    @staticmethod
    def _convert_user_message_text_only(content: list) -> list:
        """Convert user message, inlining tool results as text."""
        text_parts = []

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tr_content = block.get("content", "")
                if isinstance(tr_content, list):
                    texts = []
                    for b in tr_content:
                        if isinstance(b, dict) and b.get("type") == "text":
                            texts.append(b["text"])
                        elif isinstance(b, str):
                            texts.append(b)
                    tr_content = "\n".join(texts)
                text_parts.append(
                    f"[tool_result: {tr_content}]"
                )
            else:
                converted = AnthropicToOpenAI.content_block(block)
                if converted is not None:
                    if isinstance(converted, str):
                        text_parts.append(converted)
                    elif isinstance(converted, dict) and converted.get("type") == "text":
                        text_parts.append(converted["text"])

        if not text_parts:
            return []
        return [{"role": "user", "content": "\n".join(text_parts)}]

    @staticmethod
    def tools(anthropic_tools: list) -> list:
        result = []
        for tool in anthropic_tools:
            entry: dict = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            if tool.get("cache_control") is not None:
                entry["cache_control"] = tool["cache_control"]
            result.append(entry)
        return result

    @staticmethod
    def tool_choice(anthropic_tool_choice: dict) -> Any:
        tc_type = anthropic_tool_choice.get("type")
        if tc_type == "auto":
            return "auto"
        if tc_type == "any":
            return "required"
        if tc_type == "tool":
            return {
                "type": "function",
                "function": {"name": anthropic_tool_choice.get("name")},
            }
        return "auto"


class OpenAIToAnthropic:
    @staticmethod
    def response(oai_response: dict, model_name: str) -> dict:
        choice = oai_response["choices"][0]
        message = choice["message"]

        content: list = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                try:
                    tool_input = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, KeyError):
                    tool_input = {}
                content.append({
                    "type": "tool_use",
                    "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": tc["function"]["name"],
                    "input": tool_input,
                })

        finish_reason = choice.get("finish_reason", "stop")
        stop_reason = STOP_REASON_MAP.get(finish_reason, "end_turn")

        usage = oai_response.get("usage", {})
        usage_dict = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": oai_response.get("model", model_name),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": usage_dict,
        }
