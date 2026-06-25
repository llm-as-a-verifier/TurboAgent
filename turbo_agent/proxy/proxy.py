import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.responses import StreamingResponse

from .backend import Backend
from ..utils import (
    Config,
    SSEFormatter,
    create_logger,
    log_response_summary,
    summarize_request_body,
)
from ..visualizer import register_visualizer_routes

logger = create_logger("proxy")

UPSTREAM = "https://api.anthropic.com"


class ProxyServer:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self._backend = Backend(self.config)
        self.app = FastAPI()
        register_visualizer_routes(self.app, self.config.log_dir)
        self._register_routes()

    @property
    def backend(self) -> Backend:
        return self._backend

    def _register_routes(self) -> None:
        app = self.app

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            if not request.url.path.startswith("/visualizer"):
                logger.info(f"REQ incoming {request.method} {request.url.path}")
            response = await call_next(request)
            return response

        # Catch-all proxy route (must be after visualizer routes)
        @app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
        )
        async def proxy_all(request: Request, path: str):
            return await self._proxy(request)

    async def _proxy(self, request: Request) -> Response:
        path = request.url.path
        method = request.method
        body = await request.body()

        clean_path = path.split("?")[0].rstrip("/")

        # --- OpenAI: GET /v1/models ---
        if method == "GET" and (
            clean_path.endswith("/models") or clean_path == "models"
        ):
            logger.info(f"REQ {method} {path} (openai models)")
            return JSONResponse(content=self._backend.get_models_response())

        # --- OpenAI: POST /v1/chat/completions ---
        if method == "POST" and clean_path.endswith("/chat/completions"):
            return await self._handle_openai(request, path, body)

        # --- Anthropic: POST /v1/messages ---
        if method == "POST" and clean_path.endswith("/messages"):
            return await self._handle_anthropic(request, path, body)

        # --- Upstream passthrough ---
        return await self._handle_upstream(request, path, body)

    # ------------------------------------------------------------------
    # Anthropic path
    # ------------------------------------------------------------------

    async def _handle_anthropic(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        logger.info(f"REQ {request.method} {path} (anthropic)")
        for h in ("x-api-key", "anthropic-version", "content-type", "anthropic-beta"):
            val = request.headers.get(h)
            if val:
                display = val[:12] + "..." if h == "x-api-key" and len(val) > 12 else val
                logger.debug(f"HDR {h}: {display}")
        if body:
            logger.info(f"BODY {summarize_request_body(body)}")

        is_streaming = self._body_is_streaming(body)
        start = time.monotonic()

        if is_streaming:
            return await self._anthropic_streaming(body, start)
        else:
            return await self._anthropic_non_streaming(body, start)

    async def _anthropic_non_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        result, error = await self._backend.complete_anthropic(body)
        elapsed = time.monotonic() - start

        if error:
            logger.error(f"BACKEND ERROR {error}")
            return JSONResponse(
                status_code=500,
                content={
                    "type": "error",
                    "error": {"type": "api_error", "message": error},
                },
            )

        resp_body = json.dumps(result, default=str)
        log_response_summary(resp_body, 200)
        logger.info(f"TIME {elapsed:.2f}s")
        return Response(
            content=resp_body,
            media_type="application/json",
        )

    async def _anthropic_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        async def generate():
            try:
                async for event in self._backend.stream_anthropic(body):
                    yield event
            except Exception as e:
                logger.error(f"BACKEND STREAM ERROR {e}")
                yield SSEFormatter.error(str(e))
            finally:
                elapsed = time.monotonic() - start
                logger.info(f"TIME {elapsed:.2f}s")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # OpenAI path
    # ------------------------------------------------------------------

    async def _handle_openai(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        logger.info(f"REQ {request.method} {path} (openai)")
        for h in ("authorization", "content-type"):
            val = request.headers.get(h)
            if val:
                display = val[:20] + "..." if h == "authorization" and len(val) > 20 else val
                logger.debug(f"HDR {h}: {display}")
        if body:
            logger.info(f"BODY {summarize_request_body(body)}")

        is_streaming = self._body_is_streaming(body)
        start = time.monotonic()

        if is_streaming:
            return await self._openai_streaming(body, start)
        else:
            return await self._openai_non_streaming(body, start)

    async def _openai_non_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        result, error = await self._backend.complete_openai(body)
        elapsed = time.monotonic() - start

        if error:
            logger.error(f"BACKEND ERROR {error}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": error,
                        "type": "invalid_request_error",
                        "code": None,
                    },
                },
            )

        resp_body = json.dumps(result, default=str)
        logger.info(f"RESP status=200 | model={result.get('model', '?')}")
        logger.info(f"TIME {elapsed:.2f}s")
        return Response(
            content=resp_body,
            media_type="application/json",
        )

    async def _openai_streaming(
        self, body: bytes, start: float,
    ) -> Response:
        async def generate():
            try:
                async for event in self._backend.stream_openai(body):
                    yield event
            except Exception as e:
                logger.error(f"BACKEND STREAM ERROR {e}")
                yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'server_error'}})}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                elapsed = time.monotonic() - start
                logger.info(f"TIME {elapsed:.2f}s")

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ------------------------------------------------------------------
    # Upstream passthrough
    # ------------------------------------------------------------------

    async def _handle_upstream(
        self, request: Request, path: str, body: bytes,
    ) -> Response:
        upstream_url = f"{UPSTREAM}{path}"
        headers = {}
        for k, v in request.headers.items():
            if k.lower() not in (
                "host", "content-length", "transfer-encoding",
            ):
                headers[k] = v

        async with httpx.AsyncClient() as client:
            resp = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body if body else None,
            )

        resp_headers = {}
        for k, v in resp.headers.items():
            if k.lower() not in (
                "transfer-encoding",
                "content-encoding",
                "content-length",
                "connection",
            ):
                resp_headers[k] = v

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _body_is_streaming(body: bytes) -> bool:
        if not body:
            return False
        try:
            return json.loads(body).get("stream") is True
        except Exception:
            return False
