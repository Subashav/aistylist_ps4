"""
Durable image storage.

* With ``BLOB_READ_WRITE_TOKEN`` (Vercel Blob) set, generated images are uploaded and an absolute
  https URL is returned, so they survive serverless cold starts and are visible to every instance.
* Otherwise files stay under ``/uploads`` on local disk (the original local-dev behaviour).
"""
import logging
import mimetypes
from pathlib import Path
from typing import Optional

from app.config import settings, BASE_DIR, DATA_DIR

logger = logging.getLogger(__name__)


def blob_enabled() -> bool:
    return bool(settings.BLOB_READ_WRITE_TOKEN)


def persist_bytes(data: bytes, pathname: str, content_type: Optional[str] = None) -> Optional[str]:
    """Upload bytes to Vercel Blob; returns the public URL, or None if Blob isn't configured/failed."""
    if not blob_enabled():
        return None
    try:
        import os
        from vercel.blob import sync as blob

        # The SDK reads credentials from the environment (a .env file only feeds `settings`).
        os.environ.setdefault("BLOB_READ_WRITE_TOKEN", settings.BLOB_READ_WRITE_TOKEN)

        res = blob.put_blob(
            pathname,
            data,
            access=settings.BLOB_ACCESS,
            content_type=content_type or mimetypes.guess_type(pathname)[0] or "application/octet-stream",
            add_random_suffix=True,
        )
        return res.url
    except Exception as exc:  # noqa: BLE001
        logger.error("Blob upload failed for %s: %s", pathname, exc)
        return None


def persist_local_url(url: Optional[str]) -> Optional[str]:
    """
    Turn a local ``/uploads/...`` URL into a durable Blob URL (when Blob is configured).
    Already-absolute URLs and unknown files are returned unchanged.
    """
    if not url or not url.startswith("/uploads/") or not blob_enabled():
        return url
    rel = url[len("/uploads/"):]
    for root in (DATA_DIR / "uploads", BASE_DIR / "uploads"):
        path = root / rel
        if path.is_file():
            durable = persist_bytes(path.read_bytes(), f"uploads/{rel}")
            return durable or url
    return url
