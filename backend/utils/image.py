from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


async def read_and_validate_image(photo: UploadFile) -> tuple[bytes, Image.Image]:
    if not photo.content_type or not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    content = await photo.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size must not exceed 5 MB.")

    try:
        image = Image.open(BytesIO(content))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to decode image file.") from exc

    return content, image
