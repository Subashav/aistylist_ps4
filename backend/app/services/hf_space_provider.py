"""
Free virtual try-on through a public Hugging Face Space (IDM-VTON, ZeroGPU) - no API key, no GPU.

Uses the Space's Gradio API (`/tryon`). An optional free HF_TOKEN raises the ZeroGPU quota.
Note: IDM-VTON is licensed CC BY-NC-SA 4.0 (non-commercial use only).
Same interface as the other providers, so it is a drop-in.
"""
import asyncio
import logging
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

from PIL import Image

from app.config import settings, ON_VERCEL
from app.services.fashn_vton_local_provider import normalize_clothing_category
from app.services.storage import persist_bytes

logger = logging.getLogger(__name__)

MAX_SIDE = 1024


def _to_jpeg_file(img_input: Union[str, Path, bytes, Image.Image], directory: Path, stem: str) -> str:
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, (bytes, bytearray)):
        img = Image.open(BytesIO(img_input))
    else:
        img = Image.open(str(img_input))
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    path = directory / f"{stem}.jpg"
    img.save(path, format="JPEG", quality=92)
    return str(path)


class HFSpaceTryOnProvider:
    device = "api"

    def __init__(self):
        self._client = None
        self._load_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return settings.HF_TRYON_ENABLED

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(self):  # interface parity
        return

    def _get_client(self):
        if self._client is None:
            from gradio_client import Client

            import inspect

            kwargs = {"verbose": False}
            if settings.HF_TOKEN:
                # gradio_client 1.x calls this `hf_token`, 2.x renamed it to `token`.
                params = inspect.signature(Client.__init__).parameters
                kwargs["token" if "token" in params else "hf_token"] = settings.HF_TOKEN
            self._client = Client(settings.HF_TRYON_SPACE, **kwargs)
        return self._client

    def _predict_blocking(self, person_path: str, garment_path: str, description: str, seed: int) -> bytes:
        from gradio_client import handle_file

        client = self._get_client()
        out = client.predict(
            {"background": handle_file(person_path), "layers": [], "composite": None},
            handle_file(garment_path),
            description,
            True,   # auto-generated person mask
            False,  # no crop/resize
            settings.HF_TRYON_STEPS,
            float(seed),
            api_name="/tryon",
        )
        result_path = out[0] if isinstance(out, (list, tuple)) else out
        return Path(result_path).read_bytes()

    async def generate_try_on(
        self,
        person_image: Union[str, Path, bytes, Image.Image],
        garment_image: Union[str, Path, bytes, Image.Image],
        category: str,
        styling_context: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        detected_colour_shade: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        fashn_cat = normalize_clothing_category(category)
        if fashn_cat != "tops":
            # IDM-VTON only dresses the upper body; let the caller use its outfit-preview fallback.
            return {
                "success": False,
                "result_type": "unavailable",
                "error_code": "VTON_UNSUPPORTED_CATEGORY",
                "message": f"Free try-on supports tops only (got '{category}').",
            }
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                person_path = await asyncio.to_thread(_to_jpeg_file, person_image, tmpdir, "person")
                garment_path = await asyncio.to_thread(_to_jpeg_file, garment_image, tmpdir, "garment")
                description = f"{detected_colour_shade} {category}".strip() or "a garment"
                img_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self._predict_blocking, person_path, garment_path, description, seed),
                    timeout=settings.HF_TRYON_TIMEOUT,
                )

            filename = f"tryon_{uuid.uuid4().hex}.png"
            image_url = await asyncio.to_thread(persist_bytes, img_bytes, f"virtual_tryon/{filename}", "image/png")
            if not image_url:
                if ON_VERCEL:
                    raise RuntimeError("No durable storage configured to hold the try-on image")
                (settings.VIRTUAL_TRYON_DIR / filename).write_bytes(img_bytes)
                image_url = f"/uploads/virtual_tryon/{filename}"
            elapsed = time.perf_counter() - t0
            logger.info("HF Space try-on done in %.1fs", elapsed)
            return {
                "success": True,
                "result_type": "actual_try_on",
                "image_url": image_url,
                "provider": "hf_space_idm_vton",
                "category": fashn_cat,
                "execution_time_seconds": round(elapsed, 2),
                "error": None,
            }
        except Exception as err:  # noqa: BLE001 - quota/queue/network errors must not break analysis
            self._client = None  # reconnect next time
            self._load_error = str(err)
            msg = str(err)
            if "quota" in msg.lower() or msg.strip("'\" ") == "RuntimeError":  # bare RuntimeError = ZeroGPU refused (no quota left for this IP)
                msg = ("Free GPU quota for the try-on model is used up for now. Try again later, "
                       "or set a free HF_TOKEN (huggingface.co/settings/tokens) to get your own daily quota.")
            logger.error("HF Space try-on failed: %s", err)
            return {
                "success": False,
                "result_type": "unavailable",
                "error_code": "VTON_HF_ERROR",
                "message": f"Free try-on unavailable right now: {msg[:240]}",
            }
