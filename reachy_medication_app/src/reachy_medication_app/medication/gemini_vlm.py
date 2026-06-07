"""Thin wrapper around Gemini's image-analysis API.

Used by the medication tools (scan_bottle, verify_dose) so they don't each
re-implement the API call. Mirrors the call shape used in the tray-inventory
app, so test_vlm.py and the conversation app produce comparable results.

Reads:
  GEMINI_API_KEY   — required
  GEMINI_MODEL     — defaults to gemini-3.5-flash
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Optional

import requests


logger = logging.getLogger(__name__)


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env or environment "
            "to enable bottle scanning and dose verification."
        )
    return key


def _model() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip()


def call_gemini(
    prompt: str,
    jpeg_images: list[bytes],
    *,
    max_output_tokens: int = 2048,
    temperature: float = 0.2,
    timeout_s: float = 45.0,
) -> dict[str, Any]:
    """Call Gemini with one or more JPEG images and a text prompt.

    Returns a dict with keys:
      raw_text         : the raw model output as a string
      parsed_json      : dict if the raw_text was valid JSON, else None
      model            : model name used
      status_code      : HTTP status
      error            : error message if the call failed, else None
    """
    api_key = _api_key()
    model = _model()

    parts: list[dict[str, Any]] = []
    for img_bytes in jpeg_images:
        if not img_bytes:
            continue
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_bytes).decode(),
                }
            }
        )
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        },
    }
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
    except requests.RequestException as e:
        logger.exception("Gemini request failed")
        return {"raw_text": "", "parsed_json": None, "model": model, "status_code": 0, "error": str(e)}

    if r.status_code != 200:
        snippet = r.text[:500] if r.text else ""
        logger.error("Gemini HTTP %s: %s", r.status_code, snippet)
        return {
            "raw_text": "",
            "parsed_json": None,
            "model": model,
            "status_code": r.status_code,
            "error": f"HTTP {r.status_code}: {snippet}",
        }

    try:
        body = r.json()
        raw_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        logger.exception("Unexpected Gemini response shape")
        return {
            "raw_text": "",
            "parsed_json": None,
            "model": model,
            "status_code": r.status_code,
            "error": f"unexpected response: {e}",
        }

    parsed = _try_parse_json(raw_text)
    return {
        "raw_text": raw_text,
        "parsed_json": parsed,
        "model": model,
        "status_code": r.status_code,
        "error": None,
    }


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    """Try to parse a JSON object from a model response.

    Models often wrap JSON in ```json ... ``` markdown fences even when
    asked not to. This strips those before parsing.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None
