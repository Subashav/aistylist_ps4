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


def s3_enabled() -> bool:
    return bool(settings.AWS_ENDPOINT_URL_S3 and settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY)


def blob_enabled() -> bool:
    """True when some durable store (Neon Object Storage or Vercel Blob) is configured."""
    return s3_enabled() or bool(settings.BLOB_READ_WRITE_TOKEN)


_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        from botocore.config import Config

        _s3_client = boto3.client(
            "s3",
            endpoint_url=settings.AWS_ENDPOINT_URL_S3,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
    return _s3_client


def s3_get(key: str):
    """Fetch an object from the private Neon bucket (used by the /media proxy route)."""
    return _s3().get_object(Bucket=settings.NEON_BUCKET, Key=key)


def persist_bytes(data: bytes, pathname: str, content_type: Optional[str] = None) -> Optional[str]:
    """
    Store bytes durably and return the URL to use in API responses, or None when nothing is configured.
    Neon Object Storage (private bucket) is served back through the app's /media/<key> route;
    Vercel Blob returns its own public URL.
    """
    if s3_enabled():
        import uuid

        key = f"{pathname.rsplit('.', 1)[0]}_{uuid.uuid4().hex[:8]}.{pathname.rsplit('.', 1)[-1]}" if "." in pathname else pathname
        try:
            _s3().put_object(
                Bucket=settings.NEON_BUCKET,
                Key=key,
                Body=data,
                ContentType=content_type or mimetypes.guess_type(pathname)[0] or "application/octet-stream",
            )
            return f"/media/{key}"
        except Exception as exc:  # noqa: BLE001
            logger.error("Neon Object Storage upload failed for %s: %s", pathname, exc)
            return None
    if not settings.BLOB_READ_WRITE_TOKEN:
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
