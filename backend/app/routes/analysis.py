import os
from pathlib import Path
import time
import uuid
import asyncio
import logging
import cv2
import numpy as np
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings, BASE_DIR, ON_VERCEL
from app.database import get_db
from app.models.user import User
from app.utils.security import get_current_user
from app.schemas.analysis import (
    AnalysisResponse,
    ColourSwatch,
    OutfitSuggestion,
    PersonalizedStyleAnalysis,
    VirtualTryOnResponse,
)
from app.services.clothing_detection import detect_clothing_region
from app.services.colour_detection import extract_actual_clothing_colour
from app.services.recommendation_engine import generate_fashion_recommendations
from app.services.visual_provider import get_visual_provider
from app.services.gemini_service import (
    analyze_garment_semantics,
    analyze_personalized_fit,
    generate_four_distinct_looks,
)
from app.services.user_analysis_service import analyze_user_portrait, UserStylingProfile
from app.services.virtual_tryon_provider import get_virtual_tryon_provider
from app.services.duplicate_detector import check_for_duplicates
from app.services.storage import blob_enabled, persist_local_url


logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Analysis"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_image_file(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type ({file.filename}). Please upload a JPEG, PNG, or WebP image."
        )
    return ext if ext in ALLOWED_EXTENSIONS else ".jpg"

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_clothing(
    clothing_image: UploadFile = File(...),
    user_image: Optional[UploadFile] = File(None),
    clothing_category_hint: Optional[str] = Form(None),
    model_style: Optional[str] = Form(None),
    full_body: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Validate & Save Clothing Image
    c_ext = validate_image_file(clothing_image)
    c_filename = f"clothing_{uuid.uuid4().hex}{c_ext}"
    c_path = settings.UPLOAD_DIR / c_filename
    
    c_content = await clothing_image.read()
    if len(c_content) == 0:
        raise HTTPException(status_code=400, detail="Clothing image file is empty.")
        
    with open(c_path, "wb") as f:
        f.write(c_content)
        
    # 2. Process User Portrait Photo if provided
    u_url = None
    u_path = None
    if user_image and user_image.filename:
        u_ext = validate_image_file(user_image)
        u_filename = f"user_{uuid.uuid4().hex}{u_ext}"
        u_path = settings.UPLOAD_DIR / u_filename
        u_content = await user_image.read()
        if len(u_content) > 0:
            with open(u_path, "wb") as f:
                f.write(u_content)
            u_url = f"/uploads/{u_filename}"

    # 3. Decode clothing image for OpenCV
    t_start = time.perf_counter()
    nparr = np.frombuffer(c_content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode clothing image. Please upload a clear photo.")
        
    # 4. Computer Vision Step: Clothing Segmentation & Skin Suppression
    t_cv_start = time.perf_counter()
    fabric_pixels, masked_vis, detected_cat = detect_clothing_region(img_bgr)
    
    # 5. Computer Vision Step: LAB K-Means Clustering & CIEDE2000 Shade Detection
    color_info = extract_actual_clothing_colour(fabric_pixels)
    t_cv_end = time.perf_counter()
    logger.info("CV Pipeline Time: %.3fs (Color: %s %s, Confidence: %.2f)",
                t_cv_end - t_cv_start, color_info["colour_shade"], color_info["hex_value"], color_info["confidence"])
    
    # 6. Optional Gemini Semantic Understanding Step for Garment & User Portrait
    t_gem_start = time.perf_counter()
    semantic_data = None
    user_profile = None

    async def _fetch_garment_semantics():
        if settings.GEMINI_ENABLED and settings.GEMINI_API_KEY:
            return await analyze_garment_semantics(c_path)
        return None

    async def _fetch_user_analysis():
        if u_path and u_path.exists():
            return await analyze_user_portrait(u_path)
        return None

    semantic_data, user_profile = await asyncio.gather(
        _fetch_garment_semantics(),
        _fetch_user_analysis(),
    )
    t_gem_end = time.perf_counter()
    logger.info("Semantic / Profile Analysis Time: %.3fs", t_gem_end - t_gem_start)

    # Resolve garment category: Manual user hint > Gemini Subcategory > CV Inferred
    if clothing_category_hint and clothing_category_hint.strip():
        category = clothing_category_hint.strip()
    elif semantic_data and semantic_data.subcategory:
        category = semantic_data.subcategory
    else:
        category = detected_cat

    # 7. Personalized Color Harmony Engine & Enriched Outfit Suggestions
    t_rec_start = time.perf_counter()
    pers_fit_data = None
    gemini_looks_plan = None
    if user_profile and getattr(user_profile, "is_valid_person", False):
        pers_fit_data, gemini_looks_plan = await asyncio.gather(
            analyze_personalized_fit(
                colour_shade=color_info["colour_shade"],
                hex_value=color_info["hex_value"],
                clothing_type=category,
                garment_semantics=semantic_data,
                user_profile=user_profile,
            ),
            generate_four_distinct_looks(
                colour_shade=color_info["colour_shade"],
                hex_value=color_info["hex_value"],
                clothing_type=category,
                garment_semantics=semantic_data,
                user_profile=user_profile,
            ),
        )

    recommendations = generate_fashion_recommendations(
        clothing_type=category,
        detected_colour=color_info["base_colour"],
        colour_shade=color_info["colour_shade"],
        hex_value=color_info["hex_value"],
        semantic_data=semantic_data,
        user_profile=user_profile,
        personalized_fit_override=pers_fit_data,
        gemini_looks_plan=gemini_looks_plan,
    )
    t_rec_end = time.perf_counter()
    logger.info("Recommendation Engine Time: %.3fs", t_rec_end - t_rec_start)

    # 8. Local FASHN VTON Virtual Try-On Generation (Sequential GPU Inference for exactly 2 looks)
    t_img_start = time.perf_counter()
    vton_provider = get_virtual_tryon_provider()
    max_looks_to_generate = getattr(settings, "TRYON_MAX_LOOKS", 2)
    _local = getattr(vton_provider, "local_provider", None)
    if _local.__class__.__name__ == "HFSpaceTryOnProvider":
        max_looks_to_generate = min(max_looks_to_generate, settings.HF_TRYON_MAX_LOOKS)
    if _local is not None and getattr(_local, "device", "cpu") == "cpu" and not settings.VTON_ALLOW_CPU:
        logger.info("No CUDA GPU detected: skipping local virtual try-on (set VTON_ALLOW_CPU=true to force it).")
        max_looks_to_generate = 0

    outfit_suggestions_with_visuals = []
    has_actual_tryon = False
    try_on_message = None

    # Look 1 = the user's own portrait wearing the garment. Look 2 = the same garment on a full-body
    # editorial model who is already in a complete outfit (trousers, shoes...), i.e. a styled head-to-toe look.
    # Without a portrait, both looks use editorial models.
    models_root = BASE_DIR / "app" / "assets" / "models"
    wants_women = (model_style or "").strip().lower() == "women"
    # Men's editorial photos live in models/ (original set); women's in models/women/.
    models_dir = models_root / "women" if wants_women else models_root
    has_portrait = bool(u_path and u_path.exists())
    photo_is_full_body = (full_body or "").strip().lower() in ("1", "true", "yes", "on")

    def _pick_model(look: Dict[str, Any], slot: int) -> Optional[Path]:
        available = sorted(models_dir.glob("*.jpg")) + sorted(models_dir.glob("*.png"))
        if not available:
            return None
        text = f"{look.get('title', '')} {look.get('style', '')}".lower()
        if "evening" in text:
            wanted = "evening_minimalist"
        elif "layer" in text or "elevated" in text:
            wanted = "elevated_casual"
        elif "smart" in text or "elegan" in text or slot == 1:
            wanted = "smart_casual_elegance"
        else:
            wanted = "casual_classic"
        for path in available:
            if path.stem.startswith(wanted):
                return path
        return available[slot % len(available)]

    def _person_for(idx: int, look: Dict[str, Any]):
        if idx == 0 and has_portrait:
            return u_path, "you"
        return _pick_model(look, idx), "editorial_model"

    async def _run_look(idx: int, look: Dict[str, Any]):
        person, source = _person_for(idx, look)
        if person is None:
            return {
                "success": False,
                "result_type": "unavailable",
                "message": "No styled model photos for this selection yet, so look 2 is shown as an outfit guide.",
            }, source
        res = await vton_provider.generate_try_on(
            person_image=person,
            garment_image=c_path,
            category=category,
            styling_context=look,
            detected_colour_shade=color_info["colour_shade"],
            detected_colour_hex=color_info["hex_value"],
            semantic_data=semantic_data,
            seed=42 + idx * 101,
            full_body=has_portrait and idx == 0 and photo_is_full_body,
        )
        return res, source

    looks = recommendations["outfit_suggestions"]
    can_complete_look = _local.__class__.__name__ == "HFSpaceTryOnProvider" and hasattr(_local, "generate_complete_look")

    async def _portrait_flow():
        """Both looks use the user's real photo: (1) garment on the user, (2) that result + suggested trousers."""
        first = await _run_look(0, looks[0])
        results = [first]
        if max_looks_to_generate > 1 and len(looks) > 1:
            res1 = first[0]
            if not photo_is_full_body:
                res2 = {
                    "success": False,
                    "result_type": "unavailable",
                    "message": "Upload a full-body photo (head to toe) to see a complete outfit on you.",
                }
            elif res1.get("success") and res1.get("image_bytes"):
                res2 = await _local.generate_complete_look(res1["image_bytes"], looks[1])
            else:
                res2 = {
                    "success": False,
                    "result_type": "unavailable",
                    "message": res1.get("message") or "The first try-on failed, so look 2 could not be built.",
                }
            results.append((res2, "you_complete"))
        return results

    if has_portrait and can_complete_look:
        job_results = await _portrait_flow()
    else:
        jobs = [_run_look(i, l) for i, l in enumerate(looks) if i < max_looks_to_generate]
        job_results = await asyncio.gather(*jobs, return_exceptions=True)  # parallel; different HF accounts

    for idx, outfit_data in enumerate(looks):
        try:
            if idx < max_looks_to_generate:
                outcome = job_results[idx]
                if isinstance(outcome, Exception):
                    raise outcome
                res, person_source = outcome
                img_url = res.get("image_url")
                res_type = res.get("result_type", "actual_try_on")
                if res_type != "actual_try_on" and not try_on_message:
                    try_on_message = res.get("try_on_error") or res.get("message")
                if res_type == "actual_try_on":
                    has_actual_tryon = True

                outfit_suggestions_with_visuals.append(
                    OutfitSuggestion(
                        **outfit_data,
                        image_url=img_url,
                        try_on_image_url=img_url,
                        result_type=res_type,
                        try_on_type=res_type,
                        provider=res.get("provider", "fashn_vton_local"),
                        person_source=person_source,
                    )
                )
            else:
                # Looks 3 & 4 (beyond the generated looks):
                # Complete Outfit & Material Guide with fabric coordination & palette
                outfit_suggestions_with_visuals.append(
                    OutfitSuggestion(
                        **outfit_data,
                        image_url=None,
                        try_on_image_url=None,
                        result_type="outfit_guide",
                        try_on_type="outfit_guide",
                        provider="outfit_material_guide",
                    )
                )
        except Exception as look_err:
            logger.warning("Error generating visual for look %d ('%s'): %s", idx + 1, outfit_data.get("title"), look_err)
            outfit_suggestions_with_visuals.append(
                OutfitSuggestion(
                    **outfit_data,
                    image_url=None,
                    try_on_image_url=None,
                    result_type="unavailable",
                    try_on_type="unavailable",
                    provider="none",
                )
            )

    # 8b. Duplicate Detection
    gen_paths = []
    for s in outfit_suggestions_with_visuals:
        if s.image_url:
            gen_paths.append(BASE_DIR / s.image_url.lstrip("/"))
        else:
            gen_paths.append(None)
    duplicates = check_for_duplicates(gen_paths)
    if duplicates:
        logger.warning(
            "Duplicate check detected %d pair(s) of identical/near-identical try-on images: %s",
            len(duplicates),
            duplicates,
        )

    t_img_end = time.perf_counter()
    t_total = time.perf_counter() - t_start
    logger.info("Virtual Try-On Generation Time: %.3fs, Total Request Time: %.3fs",
                t_img_end - t_img_start, t_total)

    # 9. Construct Personalized Response
    pers_analysis = None
    if "personalized_analysis" in recommendations:
        pers_analysis = PersonalizedStyleAnalysis(**recommendations["personalized_analysis"])
        if has_actual_tryon:
            pers_analysis.try_on_status = "actual_try_on"
        else:
            pers_analysis.try_on_status = "outfit_preview"

    # 10. Make images durable: on serverless, /tmp files vanish, so push them to Vercel Blob.
    clothing_url = f"/uploads/{c_filename}"
    if blob_enabled():
        for sug in outfit_suggestions_with_visuals:
            sug.image_url = await asyncio.to_thread(persist_local_url, sug.image_url)
            sug.try_on_image_url = sug.image_url
        clothing_url = await asyncio.to_thread(persist_local_url, clothing_url)
    if blob_enabled() or ON_VERCEL:
        u_url = None  # never publish the portrait; on serverless its local file is gone anyway

    return AnalysisResponse(
        clothing_type=category,
        detected_colour=color_info["base_colour"],
        colour_shade=color_info["colour_shade"],
        hex_value=color_info["hex_value"],
        rgb_value=color_info["rgb_value"],
        confidence=color_info["confidence"],
        is_low_confidence=color_info["is_low_confidence"],
        clothing_image_url=clothing_url,
        user_image_url=u_url,
        try_on_message=try_on_message,
        recommended_colours=[ColourSwatch(**s) for s in recommendations["recommended_colours"]],
        outfit_suggestions=outfit_suggestions_with_visuals,
        overall_advice=recommendations["overall_advice"],
        personalized_analysis=pers_analysis,
        user_attributes=user_profile.model_dump() if user_profile else None,
        try_on_available=True,
    )


@router.post("/virtual-tryon", response_model=VirtualTryOnResponse)
async def direct_virtual_tryon(
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("T-shirt"),
    outfit_description: Optional[str] = Form(None),
    styling_instructions: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    Direct Virtual Try-On endpoint:
    Takes person_image and garment_image with styling instructions,
    and returns an actual try-on image generated via local FASHN VTON.
    """
    p_ext = validate_image_file(person_image)
    p_filename = f"person_{uuid.uuid4().hex}{p_ext}"
    p_path = settings.UPLOAD_DIR / p_filename
    p_content = await person_image.read()
    if len(p_content) == 0:
        raise HTTPException(status_code=400, detail="Person image is empty.")
    with open(p_path, "wb") as f:
        f.write(p_content)

    g_ext = validate_image_file(garment_image)
    g_filename = f"garment_{uuid.uuid4().hex}{g_ext}"
    g_path = settings.UPLOAD_DIR / g_filename
    g_content = await garment_image.read()
    if len(g_content) == 0:
        raise HTTPException(status_code=400, detail="Garment image is empty.")
    with open(g_path, "wb") as f:
        f.write(g_content)

    # Extract garment colour for styling record
    nparr = np.frombuffer(g_content, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    detected_shade = "Classic"
    detected_hex = "#000000"
    if img_bgr is not None:
        try:
            fabric_pixels, _, _ = detect_clothing_region(img_bgr)
            color_info = extract_actual_clothing_colour(fabric_pixels)
            detected_shade = color_info.get("colour_shade", "Classic")
            detected_hex = color_info.get("hex_value", "#000000")
        except Exception as e:
            logger.warning("Color extraction exception during direct tryon: %s", e)

    look_data = {
        "title": "Clean Casual",
        "style_direction": styling_instructions or "Relaxed contemporary styling",
        "bottom": "Tailored trousers or clean denim",
        "shoes": "Classic low-top sneakers",
        "accessories": ["Minimalist watch"],
    }

    vton_provider = get_virtual_tryon_provider()
    res = await vton_provider.generate_try_on(
        person_image=p_path,
        garment_image=g_path,
        category=category,
        styling_context=look_data,
        detected_colour_shade=detected_shade,
        detected_colour_hex=detected_hex,
    )

    return VirtualTryOnResponse(
        success=res.get("success", False),
        result_type=res.get("result_type", "actual_try_on"),
        image_url=res.get("image_url"),
        provider=res.get("provider", "fashn_vton_local"),
        category=category,
        look=look_data,
        error_code=res.get("error_code"),
        message=res.get("message"),
    )


@router.get("/diagnostics/tryon")
async def tryon_diagnostics():
    """Reports which try-on provider is active and whether it can reach its backend (uses no GPU quota)."""
    import asyncio

    prov = get_virtual_tryon_provider().local_provider
    info = {"provider": prov.__class__.__name__, "ready": bool(getattr(prov, "is_ready", False))}
    if prov.__class__.__name__ == "HFSpaceTryOnProvider":
        # Never reveal the token itself - only whether one is configured and looks valid.
        from app.services.hf_space_provider import _configured_tokens

        toks = [t for t in _configured_tokens() if t]
        info["hf_accounts_configured"] = len(toks)
        info["hf_tokens_look_valid"] = all(t.startswith("hf_") and len(t) > 20 for t in toks) if toks else False
    if prov.__class__.__name__ == "HFSpaceTryOnProvider":
        from app.services.hf_space_provider import _configured_tokens
        try:
            await asyncio.wait_for(asyncio.to_thread(prov._get_client, (_configured_tokens() or [""])[0]), timeout=60)
            info["space_reachable"] = True
        except Exception as exc:  # noqa: BLE001
            info["space_reachable"] = False
            info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    info["last_error"] = getattr(prov, "load_error", None)
    return info


@router.get("/diagnostics/system")
def get_system_diagnostics():
    """
    Returns system diagnostic telemetry including GPU, VRAM, Gemini, and VTON status.
    Never exposes API keys.
    """
    try:
        import torch
    except Exception:
        torch = None

    gpu_available = bool(torch is not None and torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    vram_total = None
    vram_free = None

    if gpu_available:
        try:
            total_b = torch.cuda.get_device_properties(0).total_memory
            reserved_b = torch.cuda.memory_reserved(0)
            allocated_b = torch.cuda.memory_allocated(0)
            vram_total = int(total_b / (1024 * 1024))
            vram_free = int((total_b - reserved_b) / (1024 * 1024))
        except Exception:
            pass

    gemini_configured = bool(settings.GEMINI_ENABLED and settings.GEMINI_API_KEY)
    vton_weights_exist = (
        settings.VTON_WEIGHTS_DIR.exists()
        and (settings.VTON_WEIGHTS_DIR / "model.safetensors").exists()
    )

    vton_provider = get_virtual_tryon_provider()
    vton_ready = getattr(vton_provider.local_provider, "is_ready", False) or vton_weights_exist

    return {
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total,
        "vram_available_mb": vram_free,
        "gemini_configured": gemini_configured,
        "vton_configured": settings.VTON_ENABLED,
        "vton_model": settings.VTON_MODEL_NAME,
        "vton_ready": vton_ready,
    }



