from typing import Any, Dict, List

from ..utils import ContextConfig, llm_completion, create_logger

_logger = create_logger("context")


class ContextRefiner:
    def __init__(self, config: ContextConfig):
        self.config = config

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            role = (msg.get("role") or "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "image_url":
                            texts.append("[image]")
                    else:
                        texts.append(str(block))
                content = "\n".join(texts)
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    async def refine(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        context_str = self._format_messages(messages)
        prompt = self.config.refinement_prompt.replace("{context}", context_str)

        _logger.info(f"Refining context with {self.config.model_name}")

        try:
            response = await llm_completion(
                model=self.config.model_name,
                api_key=self.config.api_key,
                messages=[{"role": "user", "content": prompt}],
            )
            refined = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception as e:
            _logger.error(f"Context refinement error: {e} — skipping refinement")
            return messages

        _logger.info(f"Refined context ({len(refined)} chars)")

        new_messages = list(messages)
        if new_messages and new_messages[0].get("role") == "system":
            new_messages[0] = {
                **new_messages[0],
                "content": refined + "\n\n" + (new_messages[0].get("content") or ""),
            }
        else:
            new_messages.insert(0, {"role": "system", "content": refined})

        return new_messages
