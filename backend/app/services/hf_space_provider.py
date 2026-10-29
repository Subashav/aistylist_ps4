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
from app.services.storage import persist_bytes, get_fixed, put_fixed

logger = logging.getLogger(__name__)

MAX_SIDE = 1024


def _to_jpeg_file(img_input: Union[str, Path, bytes, Image.Image], directory: Path, stem: str, fmt: str = "JPEG") -> str:
    if isinstance(img_input, Image.Image):
        img = img_input
    elif isinstance(img_input, (bytes, bytearray)):
        img = Image.open(BytesIO(img_input))
    else:
        img = Image.open(str(img_input))
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    if fmt == "PNG":
        path = directory / f"{stem}.png"
        img.save(path, format="PNG")
    else:
        path = directory / f"{stem}.jpg"
        img.save(path, format="JPEG", quality=92)
    return str(path)


def _torso_changed(person_path: str, result: bytes, threshold: float = 0.10) -> bool:
    """True when the try-on actually repainted the upper body (not just returned the input photo)."""
    try:
        import numpy as np

        a = Image.open(person_path).convert("RGB")
        b = Image.open(BytesIO(result)).convert("RGB").resize(a.size)
        size = (128, max(1, int(128 * a.height / a.width)))
        x = np.asarray(a.resize(size), dtype=np.int16)
        y = np.asarray(b.resize(size), dtype=np.int16)
        h = x.shape[0]
        band = slice(int(h * 0.18), int(h * 0.50))  # shoulders to hips
        diff = np.abs(x[band] - y[band]).sum(axis=2) > 40
        return float(diff.mean()) >= threshold
    except Exception:  # noqa: BLE001 - never block a result on the check itself
        return True


def _build_body_mask(person_path: str, out_path: str) -> Optional[str]:
    """
    Mask of the whole body below the chin (clothes, shawl, trousers, shoes), so the try-on repaints the
    complete outfit while the face, hair, pose and background stay as photographed.
    Uses a free CPU background-remover Space for the person cut-out; None if it is unavailable.
    """
    try:
        import cv2
        import numpy as np
        from gradio_client import Client, handle_file

        res = Client("briaai/BRIA-RMBG-1.4").predict(handle_file(person_path), api_name="/run")
        alpha = np.array(Image.open(res[1]).convert("RGBA"))[:, :, 3]
        w, h = Image.open(person_path).size
        person = (cv2.resize(alpha, (w, h)) > 128).astype(np.uint8) * 255
        rows = np.where(person.any(axis=1))[0]
        if len(rows) < h * 0.4:
            return None  # not a usable full-body cut-out
        top, bottom = int(rows.min()), int(rows.max())
        mask = person.copy()
        mask[: int(top + 0.135 * (bottom - top))] = 0  # keep head and hair
        mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
        layer = np.zeros((h, w, 4), np.uint8)
        layer[..., :3] = mask[..., None]
        layer[..., 3] = 255
        Image.fromarray(layer, "RGBA").save(out_path)
        return out_path
    except Exception as err:  # noqa: BLE001
        logger.warning("Body mask unavailable (%s); using the standard upper-body mask", err)
        return None


def _trim_garment(path: str) -> str:
    """Crop a product photo to the garment itself (uniform light border -> tight box on white)."""
    try:
        import numpy as np

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.int16)
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
        bg = np.median(border, axis=0)
        if bg.min() < 200 or border.std(axis=0).max() > 12:
            return path  # busy or dark background: leave as is
        mask = np.abs(arr - bg).sum(axis=2) > 60
        ys, xs = np.where(mask)
        if len(ys) < 0.02 * mask.size:
            return path
        pad = int(0.04 * max(img.size))
        box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, img.width), min(ys.max() + pad, img.height))
        img.crop(box).save(path, format="PNG")
    except Exception:  # noqa: BLE001
        pass
    return path


def _configured_tokens():
    """All distinct HF tokens (HF_TOKENS comma list + HF_TOKEN); [""] means anonymous."""
    raw = (settings.HF_TOKENS or "").split(",") + [settings.HF_TOKEN or ""]
    seen, out = set(), []
    for t in (x.strip() for x in raw):
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out or [""]


def _is_quota_error(msg: str) -> bool:
    low = msg.lower()
    # A bare RuntimeError is how ZeroGPU refuses a request when the account has no quota left.
    return "quota" in low or "unauthorized" in low or "401" in low or msg.strip("'\" ") == "RuntimeError"


class HFSpaceTryOnProvider:
    device = "api"
    COOLDOWN_SECONDS = 30 * 60  # skip an exhausted token for a while before trying it again

    def __init__(self):
        self._clients = {}
        self._exhausted_until = {}
        self._rr = 0
        self._load_error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return settings.HF_TRYON_ENABLED

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def load_model(self):  # interface parity
        return

    def _get_client(self, token: str = "", space: Optional[str] = None):
        space = space or settings.HF_TRYON_SPACE
        key = (space, token)
        if key not in self._clients:
            import inspect

            from gradio_client import Client

            kwargs = {"verbose": False}
            if token:
                # gradio_client 1.x calls this `hf_token`, 2.x renamed it to `token`.
                params = inspect.signature(Client.__init__).parameters
                kwargs["token" if "token" in params else "hf_token"] = token
            self._clients[key] = Client(space, **kwargs)
        return self._clients[key]

    def _ordered_tokens(self):
        """Round-robin start (spreads load across accounts); tokens in cooldown go last."""
        tokens = _configured_tokens()
        n = len(tokens)
        start, self._rr = self._rr % n, self._rr + 1
        rotated = tokens[start:] + tokens[:start]
        now = time.monotonic()
        fresh = [t for t in rotated if self._exhausted_until.get(t, 0) <= now]
        return fresh + [t for t in rotated if t not in fresh]

    def _predict_with(self, token: str, person_path: str, garment_path: str, description: str, seed: int,
                      mask_path: Optional[str] = None) -> bytes:
        from gradio_client import handle_file

        if mask_path:
            # Our own mask: everything below the chin is repainted, face/hair/background are kept.
            editor = {"background": handle_file(person_path), "layers": [handle_file(mask_path)],
                      "composite": handle_file(person_path)}
        else:
            editor = {"background": handle_file(person_path), "layers": [], "composite": None}
        out = self._get_client(token).predict(
            editor,
            handle_file(garment_path),
            description,
            not mask_path,  # auto-generated person mask only when we supply none
            False,  # no crop/resize
            int(settings.HF_TRYON_STEPS),
            int(seed),
            api_name="/tryon",
        )
        result_path = out[0] if isinstance(out, (list, tuple)) else out
        return Path(result_path).read_bytes()

    def _predict_blocking(self, person_path: str, garment_path: str, description: str, seed: int,
                          mask_path: Optional[str] = None) -> bytes:
        last_err: Optional[Exception] = None
        for token in self._ordered_tokens():
            label = f"token ...{token[-4:]}" if token else "anonymous"
            try:
                return self._predict_with(token, person_path, garment_path, description, seed, mask_path)
            except Exception as err:  # noqa: BLE001
                last_err = err
                self._clients.pop((settings.HF_TRYON_SPACE, token), None)  # reconnect fresh next time
                if _is_quota_error(str(err)):
                    self._exhausted_until[token] = time.monotonic() + self.COOLDOWN_SECONDS
                    logger.warning("HF try-on: %s has no quota left, trying next account", label)
                    continue
                raise
        raise last_err if last_err else RuntimeError("no Hugging Face account available")

    # ------------------------------------------------------------------
    # Complete look on the user's real photo: add suggested trousers (Leffa, lower body)
    # ------------------------------------------------------------------
    def _rotate(self, fn):
        """Run fn(token) against each configured HF account in turn until one has quota."""
        last_err: Optional[Exception] = None
        for token in self._ordered_tokens():
            try:
                return fn(token)
            except Exception as err:  # noqa: BLE001
                last_err = err
                if _is_quota_error(str(err)):
                    self._exhausted_until[token] = time.monotonic() + self.COOLDOWN_SECONDS
                    continue
                raise
        raise last_err if last_err else RuntimeError("no Hugging Face account available")

    def _generate_bottom_blocking(self, description: str) -> bytes:
        """Text-to-image (FLUX.1-schnell, MIT) for a clean product photo of the trousers."""
        prompt = (
            f"product photo of {description}, laid flat, front view, centered, plain white background, "
            "studio lighting, no model, no mannequin, high detail fabric texture"
        )

        def call(token: str) -> bytes:
            client = self._get_client(token, settings.HF_BOTTOM_SPACE)
            out = client.predict(prompt, 11, False, 768, 768, 4, api_name="/infer")
            first = out[0]
            path = first["path"] if isinstance(first, dict) else first
            return Path(path).read_bytes()

        return self._rotate(call)

    def _leffa_blocking(self, person_path: str, garment_path: str, seed: int,
                        control_type: str = "dress_code", cloth_type: str = "lower_body") -> bytes:
        from gradio_client import handle_file

        def call(token: str) -> bytes:
            client = self._get_client(token, settings.HF_LOWER_SPACE)
            out = client.predict(
                handle_file(person_path),
                handle_file(garment_path),
                False, 30, 2.5, int(seed), control_type, cloth_type, False,
                api_name="/leffa_predict_vt",
            )
            first = out[0]
            path = first["path"] if isinstance(first, dict) else first
            return Path(path).read_bytes()

        return self._rotate(call)

    def _lower_body_blocking(self, person_path: str, garment_path: str, seed: int) -> bytes:
        return self._leffa_blocking(person_path, garment_path, seed, "dress_code", "lower_body")

    def _upper_body_blocking(self, person_path: str, garment_path: str, seed: int) -> bytes:
        """Second opinion for tops when IDM-VTON leaves the person's own clothes untouched."""
        return self._leffa_blocking(person_path, garment_path, seed, "viton_hd", "upper_body")

    def _bottom_garment_bytes(self, description: str) -> bytes:
        """Product image of the suggested trousers; generated once, then cached in the bucket."""
        import re as _re

        key = "garment-cache/bottom_" + _re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")[:60] + ".png"
        cached = get_fixed(key)
        if cached:
            return cached
        data = self._generate_bottom_blocking(description)
        buf = BytesIO()
        Image.open(BytesIO(data)).convert("RGB").save(buf, format="PNG")
        put_fixed(key, buf.getvalue())
        return buf.getvalue()

    async def generate_complete_look(
        self,
        base_image: bytes,
        look: Dict[str, Any],
        seed: int = 142,
    ) -> Dict[str, Any]:
        """
        Take the user's try-on result (garment already on) and dress the lower body in the look's
        suggested trousers, so the user sees a complete outfit. Needs a full-body photo.
        """
        bottom = look.get("bottom")
        if isinstance(bottom, dict):
            description = f"{bottom.get('color', '')} {bottom.get('item', '')}".strip()
        else:
            description = str(bottom or "")
        description = description or "charcoal tailored trousers"
        t0 = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                person_path = await asyncio.to_thread(_to_jpeg_file, base_image, tmpdir, "base", "PNG")
                garment_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self._bottom_garment_bytes, description), timeout=settings.HF_TRYON_TIMEOUT
                )
                garment_path = await asyncio.to_thread(_to_jpeg_file, garment_bytes, tmpdir, "bottom", "PNG")
                img_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self._lower_body_blocking, person_path, garment_path, seed),
                    timeout=settings.HF_TRYON_TIMEOUT,
                )
            filename = f"look2_{uuid.uuid4().hex}.png"
            image_url = await asyncio.to_thread(persist_bytes, img_bytes, f"virtual_tryon/{filename}", "image/png")
            if not image_url:
                if ON_VERCEL:
                    raise RuntimeError("No durable storage configured to hold the image")
                (settings.VIRTUAL_TRYON_DIR / filename).write_bytes(img_bytes)
                image_url = f"/uploads/virtual_tryon/{filename}"
            return {
                "success": True,
                "result_type": "actual_try_on",
                "image_url": image_url,
                "provider": "hf_space_complete_look",
                "execution_time_seconds": round(time.perf_counter() - t0, 2),
            }
        except Exception as err:  # noqa: BLE001
            msg = str(err)
            if _is_quota_error(msg):
                msg = "Free GPU quota is used up for now. Try again later or add more Hugging Face tokens."
            logger.error("Complete-look generation failed: %s", err)
            return {
                "success": False,
                "result_type": "unavailable",
                "message": f"Complete look unavailable: {msg[:240]}",
            }

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
                garment_path = await asyncio.to_thread(_to_jpeg_file, garment_image, tmpdir, "garment", "PNG")
                description = f"a {detected_colour_shade} {category}".strip().lower() if detected_colour_shade else f"a {category}".lower()
                garment_path = await asyncio.to_thread(_trim_garment, garment_path)
                mask_path = None
                if kwargs.get("full_body"):
                    mask_path = await asyncio.to_thread(_build_body_mask, person_path, str(tmpdir / "mask.png"))
                img_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self._predict_blocking, person_path, garment_path, description, seed, mask_path),
                    timeout=settings.HF_TRYON_TIMEOUT,
                )
                if not _torso_changed(person_path, img_bytes):
                    logger.warning("IDM-VTON left the torso unchanged; retrying with Leffa upper-body")
                    try:
                        alt = await asyncio.wait_for(
                            asyncio.to_thread(self._upper_body_blocking, person_path, garment_path, seed),
                            timeout=settings.HF_TRYON_TIMEOUT,
                        )
                    except Exception as alt_err:  # noqa: BLE001
                        logger.warning("Leffa upper-body fallback failed: %s", alt_err)
                        alt = None
                    if alt is None or not _torso_changed(person_path, alt):
                        raise RuntimeError(
                            "The garment could not be fitted onto this photo. Use a clear, front-facing photo "
                            "and a flat product-style photo of the garment."
                        )
                    img_bytes = alt

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
                "image_bytes": img_bytes,  # internal: look 2 builds on this (never sent to clients)
                "category": fashn_cat,
                "execution_time_seconds": round(elapsed, 2),
                "error": None,
            }
        except Exception as err:  # noqa: BLE001 - quota/queue/network errors must not break analysis
            self._load_error = str(err)
            msg = str(err)
            if _is_quota_error(msg):
                msg = ("Free GPU quota for the try-on model is used up for now. Try again later, "
                       "or set a free HF_TOKEN (huggingface.co/settings/tokens) to get your own daily quota.")
            logger.error("HF Space try-on failed: %s", err)
            return {
                "success": False,
                "result_type": "unavailable",
                "error_code": "VTON_HF_ERROR",
                "message": f"Free try-on unavailable right now: {msg[:240]}",
            }
