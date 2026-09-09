from __future__ import annotations

from typing import Any

import piexif
from PIL import Image

EDITING_SOFTWARE_KEYWORDS = (
    "photoshop",
    "gimp",
    "lightroom",
    "snapseed",
    "affinity",
    "pixelmator",
    "canva",
    "paint.net",
    "corel",
    "capture one",
    "darktable",
    "fotor",
    "picsart",
)

FLAG_WEIGHTS: dict[str, float] = {
    "editing_software_detected": 0.35,
    "missing_exif_on_jpeg": 0.25,
    "datetime_mismatch": 0.20,
    "thumbnail_dimension_mismatch": 0.30,
    "orientation_dimension_mismatch": 0.25,
    "exif_orientation_tag_present": 0.10,
}


def _decode_exif_value(value: Any) -> str | int | float | None:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="ignore").strip("\x00")
        except Exception:
            return None
    if isinstance(value, tuple):
        if len(value) == 2 and all(isinstance(v, int) for v in value):
            return value[0] / value[1] if value[1] else float(value[0])
        return ", ".join(str(_decode_exif_value(v)) for v in value)
    return value


def _get_tag(exif_dict: dict, ifd: str, tag: str) -> Any:
    ifd_data = exif_dict.get(ifd, {})
    for key, name in piexif.TAGS.get(ifd, {}).items():
        if name == tag and key in ifd_data:
            return _decode_exif_value(ifd_data[key])
    return None


def analyze_metadata(image: Image.Image, raw_bytes: bytes) -> tuple[float, list[str], dict[str, Any]]:
    flags: list[str] = []
    details: dict[str, Any] = {
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "has_exif": False,
    }

    exif_dict: dict = {}
    try:
        exif_bytes = image.info.get("exif") or piexif.load(raw_bytes)
        if isinstance(exif_bytes, dict):
            exif_dict = exif_bytes
        else:
            exif_dict = piexif.load(exif_bytes)
        details["has_exif"] = bool(exif_dict)
    except Exception:
        exif_dict = {}

    software = _get_tag(exif_dict, "0th", "Software")
    make = _get_tag(exif_dict, "0th", "Make")
    model = _get_tag(exif_dict, "0th", "Model")
    date_time = _get_tag(exif_dict, "0th", "DateTime")
    date_time_original = _get_tag(exif_dict, "Exif", "DateTimeOriginal")
    orientation = _get_tag(exif_dict, "0th", "Orientation")
    gps_info = exif_dict.get("GPS", {})

    details.update(
        {
            "software": software,
            "make": make,
            "model": model,
            "date_time": date_time,
            "date_time_original": date_time_original,
            "orientation": orientation,
            "has_gps": bool(gps_info),
        }
    )

    if software and isinstance(software, str):
        software_lower = software.lower()
        if any(keyword in software_lower for keyword in EDITING_SOFTWARE_KEYWORDS):
            flags.append("editing_software_detected")
            details["detected_software"] = software

    if image.format in ("JPEG", "JPG") and not exif_dict:
        if image.width >= 800 and image.height >= 600:
            flags.append("missing_exif_on_jpeg")

    if date_time and date_time_original and date_time != date_time_original:
        flags.append("datetime_mismatch")

    thumbnail = exif_dict.get("thumbnail")
    if thumbnail and isinstance(thumbnail, tuple) and len(thumbnail) >= 2:
        try:
            from io import BytesIO

            thumb_image = Image.open(BytesIO(thumbnail[1]))
            if thumb_image.width != image.width or thumb_image.height != image.height:
                flags.append("thumbnail_dimension_mismatch")
                details["thumbnail_size"] = [thumb_image.width, thumb_image.height]
        except Exception:
            pass

    if orientation and orientation != 1:
        flags.append("exif_orientation_tag_present")
        details["orientation_tag"] = orientation

    raw_score = sum(FLAG_WEIGHTS.get(flag, 0.15) for flag in flags)
    score = min(1.0, raw_score)

    return score, flags, details
