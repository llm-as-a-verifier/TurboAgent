import json
import os
from pathlib import Path

from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

router = APIRouter()

# Set at startup by register_visualizer_routes()
_log_dir: str = ""
_static_dir: str = ""


def register_visualizer_routes(app, log_dir: str) -> None:
    global _log_dir, _static_dir
    _log_dir = log_dir
    _static_dir = str(Path(__file__).parent / "visualizer-dist")
    app.include_router(router)


# API: list log entries
@router.get("/visualizer/api/entries")
async def list_entries():
    try:
        files = sorted(
            [f for f in os.listdir(_log_dir) if f.endswith(".json")],
            reverse=True,
        )
        return [
            {"id": f.replace(".json", ""), "filename": f} for f in files
        ]
    except Exception:
        return []


# API: get single entry
@router.get("/visualizer/api/entries/{entry_id}")
async def get_entry(entry_id: str):
    file_path = os.path.join(_log_dir, entry_id + ".json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception:
        return JSONResponse(status_code=404, content={"error": "Not found"})


# Redirect /visualizer to /visualizer/
@router.get("/visualizer")
async def redirect_visualizer():
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/visualizer/")


# Serve static frontend files (SPA fallback)
@router.get("/visualizer/{path:path}")
async def serve_visualizer(path: str):
    # Skip API routes (already handled above)
    if path.startswith("api/"):
        return

    file_path = os.path.join(_static_dir, path)

    # Serve the file if it exists
    if path and os.path.isfile(file_path):
        ext = os.path.splitext(file_path)[1]
        mime_types = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            ".json": "application/json",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }
        content_type = mime_types.get(ext, "application/octet-stream")
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type=content_type)

    # SPA fallback: serve index.html
    index_path = os.path.join(_static_dir, "index.html")
    if os.path.isfile(index_path):
        with open(index_path, "rb") as f:
            return Response(content=f.read(), media_type="text/html")

    return Response(
        content="Visualizer not built. Run: cd frontend && npm run build",
        status_code=404,
    )
