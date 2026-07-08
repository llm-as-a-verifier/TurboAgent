import argparse
import sys

import uvicorn

from .utils import logger
from .proxy import ProxyServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Turbo Agent Proxy")
    parser.add_argument("-p", "--port", type=int, default=8888)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("check", help="Check provider API keys")
    args = parser.parse_args()

    if args.command == "check":
        from . import check_api_key

        sys.exit(check_api_key.main())

    server = ProxyServer()

    logger.info("=== Turbo Agent Proxy ===")
    logger.info(f"Listening on http://localhost:{args.port}")
    logger.info(f"Backend model: {server.backend.model_name}")
    logger.info("Anthropic: POST /v1/messages")
    logger.info("OpenAI:    POST /v1/chat/completions | GET /v1/models")
    logger.info(f"Visualizer: http://localhost:{args.port}/visualizer")

    uvicorn.run(
        server.app,
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
