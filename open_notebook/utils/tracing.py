import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

CONTENT_STATE_KEYS = (
    "url",
    "file_path",
    "content",
    "delete_source",
    "url_engine",
    "document_engine",
    "output_format",
    "audio_provider",
    "audio_model",
)


def new_trace_id(prefix: str = "trace") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {
        key: getattr(value, key)
        for key in CONTENT_STATE_KEYS
        if hasattr(value, key)
    }


def _safe_url(value: Any) -> Optional[str]:
    if not value:
        return None

    raw_url = str(value)
    parts = urlsplit(raw_url)
    if not parts.scheme or not parts.netloc:
        return raw_url[:200]

    path = parts.path
    if len(path) > 120:
        path = f"{path[:117]}..."
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def summarize_file_path(file_path: Any) -> dict[str, Any]:
    if not file_path:
        return {}

    path = Path(str(file_path))
    summary: dict[str, Any] = {
        "file_name": path.name,
        "file_ext": path.suffix.lower(),
        "file_path": str(path),
    }

    try:
        summary["file_exists"] = path.exists()
        if path.exists():
            summary["file_size_bytes"] = path.stat().st_size
    except OSError as exc:
        summary["file_stat_error"] = str(exc)

    return summary


def summarize_content_state(value: Any) -> dict[str, Any]:
    state = _to_mapping(value)
    summary: dict[str, Any] = {
        "keys": sorted(state.keys()),
    }

    if state.get("url"):
        summary["url"] = _safe_url(state.get("url"))

    if state.get("file_path"):
        summary.update(summarize_file_path(state.get("file_path")))

    if state.get("content") is not None:
        summary["content_chars"] = len(str(state.get("content") or ""))

    for key in (
        "delete_source",
        "url_engine",
        "document_engine",
        "output_format",
        "audio_provider",
        "audio_model",
    ):
        if key in state and state.get(key) is not None:
            summary[key] = state.get(key)

    return summary


def summarize_upload(filename: Optional[str], content_type: Optional[str]) -> dict[str, Any]:
    safe_filename = os.path.basename(filename or "") or None
    return {
        "upload_filename": safe_filename,
        "upload_ext": Path(safe_filename).suffix.lower() if safe_filename else None,
        "upload_content_type": content_type,
    }
