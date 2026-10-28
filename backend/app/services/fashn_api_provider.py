"""
Hosted FASHN virtual try-on (https://docs.fashn.ai) - no GPU, no model weights, works on Vercel.

Flow: POST {FASHN_API_URL}/run  ->  poll GET {FASHN_API_URL}/status/{id}  ->  download output image.
Exposes the same interface as FASHNVTONLocalProvider so it is a drop-in replacement.
"""
import asyncio
import base64
import logging
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx
from PIL import Image

from app.config import settings, ON_VERCEL
from app.services.fashn_vton_local_provider import normalize_clothing_category
from app.services.storage import persist_bytes

logger = logging.getLogger(__name__)

MAX_SIDE = 1296  # FASHN processes inputs at 864x1296 anyway; smaller payloads upload faster


def _to_data_uri(img_input: Union[str, Path, bytes, Image.Image]) -> str:
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, (bytes, bytearray)):
        img = Image.open(BytesIO(img_input))
    else:
        img = Image.open(str(img_input))
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class FASHNAPIProvider:
    """Calls the hosted FASHN API. `device` is "api" so callers never apply CPU-only guards."""

    device = "api"

    def __init__(self):
        self._load_error: Optional[str] = None if settings.FASHN_API_KEY else "FASHN_API_KEY is not set"

    @property
    def is_ready(self) -> bool:
        return bool(settings.FASHN_API_KEY)

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(self):  # interface parity with the local provider
        return

    async def _run(self, client: httpx.AsyncClient, person_uri: str, garment_uri: str, category: str, seed: int) -> str:
        headers = {"Authorization": f"Bearer {settings.FASHN_API_KEY}"}
        payload = {
            "model_name": settings.FASHN_MODEL_NAME,
            "inputs": {
                "model_image": person_uri,
                "garment_image": garment_uri,
                "category": category,
                "mode": settings.FASHN_MODE,
                "seed": seed,
                "num_samples": 1,
                "output_format": "jpeg",
            },
        }
        r = await client.post(f"{settings.FASHN_API_URL}/run", json=payload, headers=headers)
        r.raise_for_status()
        body = r.json()
        if body.get("error"):
            raise RuntimeError(f"FASHN run error: {body['error']}")
        job_id = body["id"]

        deadline = time.monotonic() + settings.FASHN_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(2)
            sr = await client.get(f"{settings.FASHN_API_URL}/status/{job_id}", headers=headers)
            sr.raise_for_status()
            st = sr.json()
            status = st.get("status")
            if status == "completed":
                out = st.get("output") or []
                if not out:
                    raise RuntimeError("FASHN completed without output")
                return out[0]
            if status in ("failed", "canceled", "time_out"):
                err = st.get("error")
                raise RuntimeError(f"FASHN {status}: {err.get('message') if isinstance(err, dict) else err}")
        raise TimeoutError(f"FASHN try-on did not finish within {settings.FASHN_TIMEOUT}s")

    async def generate_try_on(
        self,
        person_image: Union[str, Path, bytes, Image.Image],
        garment_image: Union[str, Path, bytes, Image.Image],
        category: str,
        styling_context: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.is_ready:
            return {
                "success": False,
                "result_type": "unavailable",
                "error_code": "VTON_UNAVAILABLE",
                "message": "FASHN_API_KEY is not configured.",
            }
        fashn_cat = normalize_clothing_category(category)
        t0 = time.perf_counter()
        try:
            person_uri, garment_uri = await asyncio.gather(
                asyncio.to_thread(_to_data_uri, person_image),
                asyncio.to_thread(_to_data_uri, garment_image),
            )
            async with httpx.AsyncClient(timeout=60) as client:
                out_url = await self._run(client, person_uri, garment_uri, fashn_cat, seed)
                # FASHN CDN links expire (~3 days): copy the result into durable storage.
                img_bytes = (await client.get(out_url)).content

            filename = f"tryon_{uuid.uuid4().hex}.jpg"
            image_url = await asyncio.to_thread(persist_bytes, img_bytes, f"virtual_tryon/{filename}", "image/jpeg")
            if not image_url:
                # No Blob store: keep a local copy (dev) or fall back to FASHN's CDN link (serverless).
                if not ON_VERCEL and settings.VIRTUAL_TRYON_DIR.exists():
                    (settings.VIRTUAL_TRYON_DIR / filename).write_bytes(img_bytes)
                    image_url = f"/uploads/virtual_tryon/{filename}"
                else:
                    image_url = out_url
            elapsed = time.perf_counter() - t0
            logger.info("FASHN API try-on done in %.1fs", elapsed)
            return {
                "success": True,
                "result_type": "actual_try_on",
                "image_url": image_url,
                "provider": "fashn_api",
                "category": fashn_cat,
                "execution_time_seconds": round(elapsed, 2),
                "error": None,
            }
        except Exception as err:  # noqa: BLE001
            logger.error("FASHN API try-on failed: %s", err, exc_info=True)
            return {
                "success": False,
                "result_type": "unavailable",
                "error_code": "VTON_API_ERROR",
                "message": f"FASHN API try-on failed: {err}",
            }
