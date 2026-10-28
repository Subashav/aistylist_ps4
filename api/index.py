"""Vercel serverless entrypoint: exposes the FastAPI app from ./backend as an ASGI function."""
import os
import sys
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

# Only /tmp is writable on Vercel: keep Hugging Face / Gradio caches and temp files there.
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")
os.makedirs("/tmp/hf", exist_ok=True)
os.makedirs("/tmp/gradio", exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.main import app as _fastapi_app  # noqa: E402


class RestoreOriginalPath:
    """
    vercel.json rewrites every API route to /api/index?__path=<original path>, so the function
    receives the rewrite destination. Put the real path back before FastAPI routes the request.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and scope.get("path", "").startswith("/api/"):
            pairs = parse_qsl(scope.get("query_string", b"").decode(), keep_blank_values=True)
            original = next((v for k, v in pairs if k == "__path"), None)
            if original is not None:
                path = "/" + original.lstrip("/")
                rest = [(k, v) for k, v in pairs if k != "__path"]
                scope = dict(scope, path=path, raw_path=path.encode(), query_string=urlencode(rest).encode())
        await self.inner(scope, receive, send)


app = RestoreOriginalPath(_fastapi_app)
