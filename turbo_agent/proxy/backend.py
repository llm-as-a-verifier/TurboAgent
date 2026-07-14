import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from ..utils import (
    Config,
    AnthropicToOpenAI,
    OpenAIToAnthropic,
    STOP_REASON_MAP,
    SSEFormatter,
    llm_completion,
    llm_stream_completion,
    create_logger,
    create_request_log,
    save_request_log,
)
from ..context import ContextRefiner
from ..progress_monitor import ProgressMonitor
from ..verifier import Verifier

logger = create_logger("backend")


class Backend:
    """Request pipeline: (optional) context refinement -> concurrent inference
    -> pivot-tournament verification -> best response."""

    def __init__(self, config: Config):
        self.config = config
        self._setup_env()

        self.refiner: Optional[ContextRefiner] = None
        self.verifier: Optional[Verifier] = None
        self.progress_monitor: Optional[ProgressMonitor] = None
        self._bg_tasks: set = set()  # strong refs so background tasks aren't GC'd

        ctx_cfg = config.context_config
        if ctx_cfg:
            self.refiner = ContextRefiner(ctx_cfg)
            logger.info(f"Context refinement enabled (model={ctx_cfg.model_name})")

        ver_cfg = config.verifier_config
        if ver_cfg and config.total_candidates > 1:
            self.verifier = Verifier(ver_cfg)
            logger.info(
                f"Verifier enabled (total_candidates={config.total_candidates})"
            )

        pm_cfg = config.progress_monitor_config
        if pm_cfg:
            self.progress_monitor = ProgressMonitor(pm_cfg)
            logger.info(f"Progress monitor enabled (model={pm_cfg.model.name})")

    def _setup_env(self) -> None:
        import os

        for model in self.config.models:
            api_key = model.get("api_key", "")
            if not api_key:
                continue
            if model.get("name", "").startswith("gemini/"):
                os.environ["GEMINI_API_KEY"] = api_key

    @property
    def model_name(self) -> str:
        return self.config.default_model["name"]

    @property
    def api_key(self) -> str:
        return self.config.default_model.get("api_key", "")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model_params(model: dict) -> dict:
        params: dict = {}
        if model.get("temperature") is not None:
            params["temperature"] = model["temperature"]
        if model.get("max_tokens") is not None:
            params["max_tokens"] = model["max_tokens"]
        thinking = model.get("thinking")
        if thinking is not None:
            if isinstance(thinking, (int, float)):
                params["thinking_budget"] = int(thinking)
            elif isinstance(thinking, str):
                params["reasoning_effort"] = thinking
        return params

    def _base_params(self) -> dict:
        return {
            "model": self.model_name,
            "api_key": self.api_key,
            **self._parse_model_params(self.config.default_model),
        }

    def _model_entries(self) -> List[dict]:
        """One entry per candidate to generate (num_candidates per model)."""
        entries: List[dict] = []
        for model in self.config.models:
            num = model.get("num_candidates", 1)
            entry = {
                "name": model["name"],
                "api_key": model.get("api_key", ""),
                **self._parse_model_params(model),
            }
            for _ in range(num):
                entries.append(entry)
        return entries

    def _sanitized_config(self) -> dict:
        raw = dict(self.config.raw_config)
        if raw.get("backend", {}).get("models"):
            raw["backend"] = {
                **raw["backend"],
                "models": [
                    {**m, "api_key": "***"} for m in raw["backend"]["models"]
                ],
            }
        if raw.get("context", {}).get("refinement_model"):
            raw["context"] = {
                **raw["context"],
                "refinement_model": {
                    **raw["context"]["refinement_model"],
                    "api_key": "***",
                },
            }
        if raw.get("verifier", {}).get("model"):
            raw["verifier"] = {
                **raw["verifier"],
                "model": {**raw["verifier"]["model"], "api_key": "***"},
            }
        if raw.get("progress_monitor", {}).get("model"):
            raw["progress_monitor"] = {
                **raw["progress_monitor"],
                "model": {**raw["progress_monitor"]["model"], "api_key": "***"},
            }
        return raw

    async def _refine_messages(
        self, params: dict, req_log: Optional[dict] = None,
    ) -> dict:
        if self.refiner:
            original_messages = params["messages"]
            refined = await self.refiner.refine(params["messages"])
            if req_log:
                req_log["contextRefinement"] = {
                    "enabled": True,
                    "originalMessages": original_messages,
                    "refinedMessages": refined,
                }
            return {**params, "messages": refined}
        return params

    async def _gather_completions(
        self, params_base: dict,
    ) -> List[Tuple[dict, str]]:
        entries = self._model_entries()

        async def call_model(entry: dict) -> Tuple[dict, str]:
            name = entry["name"]
            api_key = entry.get("api_key", "")
            model_params = {
                k: v for k, v in entry.items() if k not in ("name", "api_key")
            }
            p = {**params_base, "model": name, "api_key": api_key, **model_params}
            p.pop("stream", None)
            resp = await llm_completion(**p)
            return resp, name

        tasks = [call_model(entry) for entry in entries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Tuple[dict, str]] = []
        errors: List[Exception] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"CONCURRENT REQUEST FAILED: {type(r).__name__}: {r}")
                errors.append(r)
            else:
                successes.append(r)

        if not successes:
            raise RuntimeError(
                f"All {len(errors)} concurrent requests failed. "
                f"First error: {type(errors[0]).__name__}: {errors[0]}"
            ) from errors[0]
        return successes

    async def _pick_best(
        self,
        responses: List[Tuple[dict, str]],
        messages: list,
        req_log: Optional[dict] = None,
    ) -> Tuple[dict, str]:
        if len(responses) == 1 or not self.verifier:
            return responses[0]

        history_str = Backend.format_history(messages)

        # Drop responses with empty choices.
        valid_responses: List[Tuple[dict, str]] = []
        for resp, model_name in responses:
            if resp.get("choices"):
                valid_responses.append((resp, model_name))
            else:
                logger.warn(
                    f"RESPONSE model={model_name} returned empty choices, skipping"
                )

        if not valid_responses:
            logger.error("All responses had empty choices, falling back to first")
            return responses[0]
        if len(valid_responses) == 1:
            return valid_responses[0]

        actions = [Backend.format_action(resp) for resp, _ in valid_responses]
        for (_, model_name), action in zip(valid_responses, actions):
            logger.info(f"RESPONSE model={model_name} text='{action[:50]}'")

        try:
            result = await self.verifier.select_best(history_str, actions)
        except Exception as e:
            logger.error(
                f"Verifier failed ({type(e).__name__}: {e}); "
                f"falling back to first response"
            )
            return valid_responses[0]

        best_idx = result.best_index
        verifier_scores = [
            {
                "index": i,
                "model": valid_responses[i][1],
                "score": result.scores[i] if i < len(result.scores) else 0.0,
                "details": {
                    "score": result.scores[i] if i < len(result.scores) else 0.0,
                    "criterionScores": [],
                },
            }
            for i in range(len(valid_responses))
        ]
        for s in verifier_scores:
            logger.info(f"VERIFY model={s['model']} score={s['score']:.3f}")

        best_resp, best_model = valid_responses[best_idx]
        best_score = result.scores[best_idx] if best_idx < len(result.scores) else 0.0
        logger.info(f"BEST model={best_model} score={best_score:.3f}")

        if req_log:
            req_log["verifier"] = {
                "enabled": True,
                "logprobsObserved": result.logprob_calls > 0,
                "logprobCalls": result.logprob_calls,
                "scores": verifier_scores,
                "comparisons": [c.to_dict() for c in result.comparisons],
                "bestIndex": best_idx,
                "bestModel": best_model,
                "bestScore": best_score,
            }

        return best_resp, best_model

    def _spawn_progress(
        self, messages: list, final_response: Optional[dict], req_log: dict,
    ) -> None:
        """Kick off progress evaluation in the background so it never delays the
        client's response. When it finishes it updates req_log and re-saves the
        log file (already written once without progress)."""
        if not self.progress_monitor:
            return
        log_dir = self.config.log_dir

        async def _run() -> None:
            await self._evaluate_progress(messages, final_response, req_log)
            save_request_log(req_log, log_dir)

        task = asyncio.create_task(_run())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _evaluate_progress(
        self, messages: list, final_response: Optional[dict],
        req_log: Optional[dict] = None,
    ) -> None:
        """Post-hoc progress estimate. Runs after the response is selected and
        never changes it — observability only. The score lands in the request
        log and the visualizer's progress node."""
        if not self.progress_monitor:
            return
        problem = Backend.format_history(messages)
        response_text = (
            Backend.format_action(final_response)
            if final_response and final_response.get("choices")
            else "(empty response)"
        )
        try:
            result = await self.progress_monitor.evaluate(problem, response_text)
            if req_log is not None:
                req_log["progressMonitor"] = {
                    "enabled": True,
                    "score": result.score,
                    "details": result.to_dict(),
                }
        except Exception as e:
            logger.error(f"Progress monitor failed: {type(e).__name__}: {e}")
            if req_log is not None:
                req_log["progressMonitor"] = {"enabled": True, "error": str(e)}

    @staticmethod
    def format_history(messages: list) -> str:
        parts: List[str] = []
        for msg in messages:
            role = (msg.get("role") or "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                texts: List[str] = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            texts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            texts.append(
                                f"[tool_result: {block.get('content', '')}]"
                            )
                        elif block.get("type") == "tool_use":
                            texts.append(
                                f"[tool_use: {block.get('name', '')}"
                                f"({json.dumps(block.get('input', {}))})]"
                            )
                    else:
                        texts.append(str(block))
                content = "\n".join(texts)
            if content:
                parts.append(f"{role}: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def format_action(response: dict) -> str:
        message = response["choices"][0]["message"]
        parts: List[str] = []
        if message.get("content"):
            parts.append(message["content"])
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                parts.append(
                    f"[tool_call: {tc['function']['name']}"
                    f"({tc['function']['arguments']})]"
                )
        return "\n".join(parts) if parts else "(empty response)"

    # ------------------------------------------------------------------
    # Anthropic-format API
    # ------------------------------------------------------------------

    def _build_anthropic_params(self, anthropic_body: dict) -> dict:
        params: dict = {
            **self._base_params(),
            "messages": AnthropicToOpenAI.messages(anthropic_body),
        }

        for key in ("max_tokens", "temperature", "top_p"):
            if key in anthropic_body:
                params[key] = anthropic_body[key]

        if anthropic_body.get("stop_sequences"):
            params["stop"] = anthropic_body["stop_sequences"]
        if "stream" in anthropic_body:
            params["stream"] = anthropic_body["stream"]
        if anthropic_body.get("tools"):
            params["tools"] = AnthropicToOpenAI.tools(anthropic_body["tools"])
        if anthropic_body.get("tool_choice"):
            params["tool_choice"] = AnthropicToOpenAI.tool_choice(
                anthropic_body["tool_choice"]
            )

        return params

    async def complete_anthropic(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            anthropic_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("anthropic", self._sanitized_config())
        req_log["request"] = anthropic_body

        params = self._build_anthropic_params(anthropic_body)
        params = await self._refine_messages(params, req_log)
        params.pop("stream", None)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (anthropic)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, model_name = await self._pick_best(
                responses, params["messages"], req_log,
            )
            final_result = OpenAIToAnthropic.response(response, model_name)
        else:
            logger.info(f"BACKEND calling {self.model_name} (anthropic)")
            response = await llm_completion(**params)
            req_log["responses"] = [
                {"model": self.model_name, "response": response}
            ]
            final_result = OpenAIToAnthropic.response(response, self.model_name)

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(params["messages"], response, req_log)
        return final_result, None

    async def stream_anthropic(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        anthropic_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log(
            "anthropic_stream", self._sanitized_config(),
        )
        req_log["request"] = anthropic_body

        params = self._build_anthropic_params(anthropic_body)
        params = await self._refine_messages(params, req_log)

        # When the verifier is active, collect all responses, verify, replay.
        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (anthropic stream)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, best_model = await self._pick_best(
                responses, params["messages"], req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(params["messages"], best_resp, req_log)
            async for event in self._replay_anthropic_sse(best_resp, best_model):
                yield event
            return

        params["stream"] = True
        logger.info(f"BACKEND streaming {self.model_name} (anthropic)")

        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(self.model_name, msg_id)

        stream = await llm_stream_completion(**params)

        block_index = 0
        text_block_open = False
        tool_blocks: Dict[str, dict] = {}
        current_tool_id: Optional[str] = None
        output_tokens = 0

        async for chunk in stream:
            chunk_dict = (
                chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            )
            choices = chunk_dict.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {})

            usage = chunk_dict.get("usage")
            if usage and usage.get("completion_tokens"):
                output_tokens = usage["completion_tokens"]

            if delta.get("content"):
                if not text_block_open:
                    yield SSEFormatter.content_block_start(block_index, "text")
                    text_block_open = True
                yield SSEFormatter.text_delta(block_index, delta["content"])

            if delta.get("tool_calls"):
                for tc_delta in delta["tool_calls"]:
                    tc_id = tc_delta.get("id")

                    if tc_id and tc_id not in tool_blocks:
                        if text_block_open:
                            yield SSEFormatter.content_block_stop(block_index)
                            block_index += 1
                            text_block_open = False

                        tool_blocks[tc_id] = {
                            "index": block_index,
                            "name": tc_delta.get("function", {}).get("name", ""),
                        }
                        current_tool_id = tc_id

                        yield SSEFormatter.content_block_start(
                            block_index,
                            "tool_use",
                            tool_id=tc_id,
                            tool_name=tool_blocks[tc_id]["name"],
                        )

                    target_id = tc_id or current_tool_id
                    if target_id and target_id in tool_blocks:
                        func = tc_delta.get("function", {})
                        if func.get("arguments"):
                            yield SSEFormatter.input_json_delta(
                                tool_blocks[target_id]["index"],
                                func["arguments"],
                            )

            if choice.get("finish_reason"):
                if text_block_open:
                    yield SSEFormatter.content_block_stop(block_index)
                    block_index += 1
                    text_block_open = False

                for tinfo in tool_blocks.values():
                    yield SSEFormatter.content_block_stop(tinfo["index"])
                    block_index += 1

                stop_reason = STOP_REASON_MAP.get(
                    choice["finish_reason"], "end_turn"
                )
                yield SSEFormatter.message_delta(stop_reason, output_tokens)
                yield SSEFormatter.message_stop()

    async def _replay_anthropic_sse(
        self, response: dict, model_name: str,
    ) -> AsyncIterator[str]:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        yield SSEFormatter.message_start(model_name, msg_id)

        choice = response["choices"][0]
        message = choice["message"]
        block_index = 0

        if message.get("content"):
            yield SSEFormatter.content_block_start(block_index, "text")
            yield SSEFormatter.text_delta(block_index, message["content"])
            yield SSEFormatter.content_block_stop(block_index)
            block_index += 1

        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                yield SSEFormatter.content_block_start(
                    block_index,
                    "tool_use",
                    tool_id=tc.get("id", ""),
                    tool_name=tc["function"]["name"],
                )
                if tc["function"].get("arguments"):
                    yield SSEFormatter.input_json_delta(
                        block_index, tc["function"]["arguments"],
                    )
                yield SSEFormatter.content_block_stop(block_index)
                block_index += 1

        stop_reason = STOP_REASON_MAP.get(
            choice.get("finish_reason", "stop"), "end_turn",
        )
        output_tokens = response.get("usage", {}).get("completion_tokens", 0)
        yield SSEFormatter.message_delta(stop_reason, output_tokens)
        yield SSEFormatter.message_stop()

    # ------------------------------------------------------------------
    # OpenAI-format API
    # ------------------------------------------------------------------

    def _build_openai_params(self, openai_body: dict) -> dict:
        params: dict = {
            **self._base_params(),
            "messages": openai_body.get("messages", []),
        }

        direct_keys = [
            "temperature", "top_p", "stop", "tools", "tool_choice",
            "response_format", "seed", "n", "presence_penalty",
            "frequency_penalty", "logit_bias",
        ]
        for key in direct_keys:
            if key in openai_body:
                params[key] = openai_body[key]

        if openai_body.get("max_tokens"):
            params["max_tokens"] = openai_body["max_tokens"]
        if openai_body.get("max_completion_tokens"):
            params["max_completion_tokens"] = openai_body["max_completion_tokens"]
        if "stream" in openai_body:
            params["stream"] = openai_body["stream"]
        if openai_body.get("stream_options"):
            params["stream_options"] = openai_body["stream_options"]

        return params

    async def complete_openai(
        self, body: bytes | str,
    ) -> Tuple[Optional[dict], Optional[str]]:
        start = time.monotonic()
        try:
            openai_body = json.loads(
                body if isinstance(body, str) else body.decode()
            )
        except json.JSONDecodeError as e:
            return None, f"Invalid JSON: {e}"

        req_log = create_request_log("openai", self._sanitized_config())
        req_log["request"] = openai_body

        params = self._build_openai_params(openai_body)
        params.pop("stream", None)
        params = await self._refine_messages(params, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} "
                f"concurrent requests (openai)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            response, _ = await self._pick_best(
                responses, params["messages"], req_log,
            )
            final_result = response
        else:
            logger.info(f"BACKEND calling {self.model_name} (openai)")
            final_result = await llm_completion(**params)
            req_log["responses"] = [
                {"model": self.model_name, "response": final_result}
            ]

        req_log["finalResponse"] = final_result
        req_log["elapsedMs"] = (time.monotonic() - start) * 1000
        save_request_log(req_log, self.config.log_dir)
        self._spawn_progress(params["messages"], final_result, req_log)
        return final_result, None

    async def stream_openai(
        self, body: bytes | str,
    ) -> AsyncIterator[str]:
        start = time.monotonic()
        openai_body = json.loads(
            body if isinstance(body, str) else body.decode()
        )
        req_log = create_request_log("openai_stream", self._sanitized_config())
        req_log["request"] = openai_body

        params = self._build_openai_params(openai_body)
        params = await self._refine_messages(params, req_log)

        if self.verifier:
            logger.info(
                f"BACKEND sending {self.config.total_candidates} concurrent "
                f"requests for verification (openai stream)"
            )
            responses = await self._gather_completions(params)
            req_log["responses"] = [
                {"model": m, "response": r} for r, m in responses
            ]
            best_resp, _ = await self._pick_best(
                responses, params["messages"], req_log,
            )
            req_log["finalResponse"] = best_resp
            req_log["elapsedMs"] = (time.monotonic() - start) * 1000
            save_request_log(req_log, self.config.log_dir)
            self._spawn_progress(params["messages"], best_resp, req_log)
            async for event in self._replay_openai_sse(best_resp):
                yield event
            return

        params["stream"] = True
        logger.info(f"BACKEND streaming {self.model_name} (openai)")

        stream = await llm_stream_completion(**params)
        async for chunk in stream:
            chunk_dict = (
                chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            )
            yield f"data: {json.dumps(chunk_dict, default=str)}\n\n"

        yield "data: [DONE]\n\n"

    async def _replay_openai_sse(
        self, response: dict,
    ) -> AsyncIterator[str]:
        choices = response.get("choices", [])
        choice = choices[0] if choices else {}

        chunk = {
            "id": response.get("id", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": response.get("created", 0),
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": choice.get("message", {}),
                    "finish_reason": choice.get("finish_reason"),
                },
            ],
        }
        yield f"data: {json.dumps(chunk, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    def get_models_response(self) -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": model["name"],
                    "object": "model",
                    "created": 0,
                    "owned_by": "turbo-agent",
                }
                for model in self.config.models
            ],
        }
