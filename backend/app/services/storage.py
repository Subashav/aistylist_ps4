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


def _sigv4_request(method: str, key: str, body: bytes = b"", content_type: str = ""):
    """Minimal AWS Signature V4 request to the S3-compatible Neon endpoint (path-style). Avoids boto3's size."""
    import datetime
    import hashlib
    import hmac
    from urllib.parse import quote, urlparse

    import httpx

    endpoint = settings.AWS_ENDPOINT_URL_S3.rstrip("/")
    host = urlparse(endpoint).netloc
    uri = "/" + quote(f"{settings.NEON_BUCKET}/{key}", safe="/")
    now = datetime.datetime.now(datetime.timezone.utc)
    amz_date, date = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(body).hexdigest()
    headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
    if content_type:
        headers["content-type"] = content_type
    signed = ";".join(sorted(headers))
    nl = chr(10)
    canonical_headers = "".join(f"{k}:{headers[k]}{nl}" for k in sorted(headers))
    canonical = nl.join([method, uri, "", canonical_headers, signed, payload_hash])
    scope = f"{date}/{settings.AWS_REGION}/s3/aws4_request"
    to_sign = nl.join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])

    def _h(k, m):
        return hmac.new(k, m.encode(), hashlib.sha256).digest()

    k = _h(("AWS4" + settings.AWS_SECRET_ACCESS_KEY).encode(), date)
    for part in (settings.AWS_REGION, "s3", "aws4_request"):
        k = _h(k, part)
    signature = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    headers["authorization"] = (
        f"AWS4-HMAC-SHA256 Credential={settings.AWS_ACCESS_KEY_ID}/{scope}, SignedHeaders={signed}, Signature={signature}"
    )
    headers.pop("host")
    r = httpx.request(method, endpoint + uri, headers=headers, content=body or None, timeout=60)
    r.raise_for_status()
    return r


def s3_get(key: str):
    """Fetch an object from the private Neon bucket -> (bytes, content_type). Used by the /media route."""
    r = _sigv4_request("GET", key)
    return r.content, r.headers.get("content-type")


def get_fixed(key: str) -> Optional[bytes]:
    """Read a cache object by exact key from the Neon bucket; None if absent or storage is not configured."""
    if not s3_enabled():
        return None
    try:
        return _sigv4_request("GET", key).content
    except Exception:  # noqa: BLE001 - missing key (404) or transient error: treat as a cache miss
        return None


def put_fixed(key: str, data: bytes, content_type: str = "image/png") -> None:
    """Write a cache object under an exact key (no random suffix). Best effort."""
    if not s3_enabled():
        return
    try:
        _sigv4_request("PUT", key, data, content_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cache write failed for %s: %s", key, exc)


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
            _sigv4_request("PUT", key, data, content_type or mimetypes.guess_type(pathname)[0] or "application/octet-stream")
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
