from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from api.config import get_settings

try:
    from google.cloud import vision  # type: ignore
except Exception:  # noqa: BLE001
    vision = None


def load_google_ocr_credentials() -> Path:
    settings = get_settings()
    credentials_path = settings.google_ocr_credentials_path
    if credentials_path.is_dir():
        json_candidates = sorted(credentials_path.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"Google OCR credentials JSON not found in directory: {credentials_path}")
        credentials_path = json_candidates[0]
    if not credentials_path.exists():
        raise FileNotFoundError(f"Google OCR credentials file not found: {credentials_path}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    return credentials_path


def _build_vision_client():
    settings = get_settings()
    if not settings.google_ocr_enabled:
        raise RuntimeError("Google OCR is disabled")
    if vision is None:
        raise RuntimeError("google-cloud-vision is not installed")
    load_google_ocr_credentials()
    return vision.ImageAnnotatorClient()


def _extract_lines_and_blocks(response) -> Dict[str, Any]:
    raw_text = ""
    if getattr(response, "text_annotations", None):
        raw_text = str(response.text_annotations[0].description or "")

    lines: List[str] = []
    blocks: List[dict] = []
    confidences: List[float] = []

    for page in getattr(getattr(response, "full_text_annotation", None), "pages", []):
        for block in getattr(page, "blocks", []):
            block_lines: List[str] = []
            for paragraph in getattr(block, "paragraphs", []):
                words = []
                paragraph_conf = getattr(paragraph, "confidence", None)
                if paragraph_conf is not None:
                    confidences.append(float(paragraph_conf))
                for word in getattr(paragraph, "words", []):
                    token = "".join(getattr(symbol, "text", "") for symbol in getattr(word, "symbols", []))
                    if token:
                        words.append(token)
                    word_conf = getattr(word, "confidence", None)
                    if word_conf is not None:
                        confidences.append(float(word_conf))
                line = " ".join(words).strip()
                if line:
                    lines.append(line)
                    block_lines.append(line)
            if block_lines:
                blocks.append({"text": "\n".join(block_lines)})

    if not lines and raw_text:
        lines = [item.strip() for item in raw_text.splitlines() if item.strip()]

    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    return {
        "raw_text": raw_text,
        "lines": lines,
        "blocks": blocks,
        "confidence": confidence,
        "source": "google_ocr",
    }


def extract_text_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    try:
        client = _build_vision_client()
        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image)
        if response.error.message:
            return {"error": response.error.message, "source": "google_ocr"}
        return _extract_lines_and_blocks(response)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "source": "google_ocr"}


def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    path = Path(image_path)
    if not path.exists():
        return {"error": f"image not found: {image_path}", "source": "google_ocr"}
    try:
        with path.open("rb") as file:
            image_bytes = file.read()
        return extract_text_from_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "source": "google_ocr"}


def save_temp_upload(image_bytes: bytes, suffix: str) -> Path:
    settings = get_settings()
    settings.upload_temp_dir.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="ocr_upload_", suffix=suffix, dir=str(settings.upload_temp_dir))
    os.close(fd)
    path = Path(temp_path)
    path.write_bytes(image_bytes)
    return path
