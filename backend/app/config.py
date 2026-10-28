import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

# On Vercel the deployment filesystem is read-only; only /tmp is writable (and not shared between
# instances), so scratch files go there and durable data lives in Postgres + Vercel Blob.
ON_VERCEL = bool(os.environ.get("VERCEL"))
DATA_DIR = Path("/tmp/ai_stylist") if ON_VERCEL else BASE_DIR

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(BASE_DIR / ".env"), str(BASE_DIR.parent / ".env.local")),
        env_file_encoding="utf-8",
        extra="allow",
    )

    SECRET_KEY: str = "ai_stylist_super_secret_jwt_key_2026_fashion_harmony"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'ai_stylist.db'}"
    UPLOAD_DIR: Path = (DATA_DIR / "uploads").resolve()
    # Sub-directory for AI-composed outfit recommendation images
    OUTFIT_IMAGE_DIR: Path = (DATA_DIR / "uploads" / "outfit_recommendations").resolve()
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    # Visual Provider Settings: 'ai' or 'pillow'
    OUTFIT_VISUAL_PROVIDER: str = "ai"
    # AI Backend: 'pollinations' (free/instant photorealistic flux), 'openai' (DALL-E 3), 'stability', 'custom'
    AI_IMAGE_PROVIDER: str = "pollinations"
    # Optional API key for OpenAI / Stability / Replicate / custom provider
    IMAGE_GENERATION_API_KEY: str = ""
    # Model name
    AI_IMAGE_MODEL: str = "flux"
    # Timeout in seconds for AI image generation requests
    AI_IMAGE_TIMEOUT: int = 35
    # Gemini Vision Semantic Understanding (Reasoning & Styling)
    GEMINI_ENABLED: bool = True
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_TIMEOUT: int = 15

    # Local FASHN VTON v1.5 Settings
    VTON_ENABLED: bool = True
    VTON_MODEL_NAME: str = "FASHN VTON v1.5"
    VTON_WEIGHTS_DIR: Path = (DATA_DIR / "weights").resolve()
    # Local try-on diffusion is far too slow on CPU-only laptops (minutes per look). It is skipped there
    # unless VTON_ALLOW_CPU=true is set; colour analysis + outfit guides still work everywhere.
    VTON_ALLOW_CPU: bool = False
    # Hosted FASHN API (https://docs.fashn.ai): used instead of the local GPU model when a key is set.
    # This is what makes virtual try-on work on Vercel / any machine without a GPU.
    FASHN_API_KEY: str = ""
    FASHN_API_URL: str = "https://api.fashn.ai/v1"
    FASHN_MODEL_NAME: str = "tryon-v1.6"
    FASHN_MODE: str = "balanced"  # performance | balanced | quality
    FASHN_TIMEOUT: int = 120  # seconds to wait for one try-on result
    # Free hosted try-on (default): public Hugging Face Space running IDM-VTON on ZeroGPU. No key needed;
    # a free HF_TOKEN gives a bigger daily GPU quota. Licence: CC BY-NC-SA 4.0 (non-commercial).
    HF_TRYON_ENABLED: bool = True
    HF_TRYON_SPACE: str = "yisol/IDM-VTON"
    HF_TRYON_STEPS: int = 30
    # Free ZeroGPU quota is limited (~tens of seconds per try-on): render at most this many looks per analysis.
    HF_TRYON_MAX_LOOKS: int = 2
    HF_TRYON_TIMEOUT: int = 150
    HF_TOKEN: str = ""
    # Several team members' tokens, comma-separated (each account has its own daily GPU quota).
    # The app rotates to the next token when one runs out.
    HF_TOKENS: str = ""
    # Vercel Blob (durable image storage). Set automatically when a Blob store is connected.
    BLOB_READ_WRITE_TOKEN: str = ""
    # Neon Object Storage (S3-compatible, private bucket). Injected by `neon deploy` / `neon env pull`.
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_ENDPOINT_URL_S3: str = ""
    AWS_REGION: str = "us-east-2"
    NEON_BUCKET: str = "uploads"
    BLOB_ACCESS: str = "public"  # must match the Blob store's access mode
    TRYON_MAX_LOOKS: int = 2  # 2 personalized try-on images; remaining are complete outfit & material guides
    VIRTUAL_TRYON_DIR: Path = (DATA_DIR / "uploads" / "virtual_tryon").resolve()

settings = Settings()

# Neon / Heroku style URLs ("postgres://", "postgresql://") -> SQLAlchemy's psycopg2 driver.
if settings.DATABASE_URL.startswith("postgres://"):
    settings.DATABASE_URL = "postgresql+psycopg2://" + settings.DATABASE_URL[len("postgres://"):]
elif settings.DATABASE_URL.startswith("postgresql://"):
    settings.DATABASE_URL = "postgresql+psycopg2://" + settings.DATABASE_URL[len("postgresql://"):]

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTFIT_IMAGE_DIR, exist_ok=True)
os.makedirs(settings.VIRTUAL_TRYON_DIR, exist_ok=True)
os.makedirs(settings.VTON_WEIGHTS_DIR, exist_ok=True)


