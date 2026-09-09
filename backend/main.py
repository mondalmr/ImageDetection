from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from schemas.verification import VerificationResponse
from services.verification import verify_image
from utils.image import MAX_FILE_SIZE_BYTES, read_and_validate_image

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_photo(photo: UploadFile = File(...)) -> dict[str, str | bool]:
    content, _ = await read_and_validate_image(photo)

    original_name = photo.filename or "photo"
    extension = Path(original_name).suffix
    safe_name = f"{uuid4().hex}{extension}"
    destination = UPLOAD_DIR / safe_name

    destination.write_bytes(content)

    size_kb = len(content) / 1024
    return {
        "success": True,
        "message": f"Uploaded {original_name} ({size_kb:.1f} KB)",
    }


@app.post("/api/verify", response_model=VerificationResponse)
async def verify_photo(photo: UploadFile = File(...)) -> VerificationResponse:
    content, image = await read_and_validate_image(photo)
    return verify_image(content, image)
