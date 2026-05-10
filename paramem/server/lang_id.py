"""fastText lid.176 detector for the text /chat path.

STT carries Whisper's language detection. Text-only /chat requests have
no equivalent signal, so the cloud path falls through to English regardless
of the actual query language. This module fills that gap with an offline
fastText classifier whose 126 MB model is fetched once via
``scripts/setup/download-langid-model.sh`` and lives at
``~/.cache/paramem/lang_id/lid.176.bin`` by default.

Loaded eagerly from the server's lifespan (``load_at_startup``) when
``text_lang_detection.enabled`` is set, so the first /chat call does not
pay the ~200 ms model read. The handle is cached as a module-level
singleton; a one-shot guard prevents retry storms when the file is
missing or corrupt. ``detect`` will still load on demand if startup
loading is bypassed (tests, callers without a config block).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _DetectorHandle:
    model: Any  # fasttext.FastText._FastText
    model_path: Path


_singleton: _DetectorHandle | None = None
_load_lock = Lock()
_load_attempted = False


def _default_model_path() -> Path:
    return Path(os.path.expanduser("~/.cache/paramem/lang_id/lid.176.bin"))


def get_detector(model_path: Path | None = None) -> _DetectorHandle | None:
    """Return the loaded detector or ``None`` when the model is unavailable.

    Server lifespan calls ``load_at_startup`` so the first call here is
    O(1). Direct callers (tests, ad-hoc scripts) trigger the load on first
    invocation. A failed load latches the disabled state — we do not retry
    on every request, since the failure mode is "operator hasn't run the
    setup script yet" and a single warning is enough.
    """
    global _singleton, _load_attempted
    if _singleton is not None:
        return _singleton
    with _load_lock:
        if _singleton is not None:
            return _singleton
        if _load_attempted:
            return None
        _load_attempted = True
        path = model_path or _default_model_path()
        if not path.exists():
            logger.warning(
                "lang_id: model file not found at %s — text-side detection disabled "
                "(run scripts/setup/download-langid-model.sh)",
                path,
            )
            return None
        try:
            import fasttext

            model = fasttext.load_model(str(path))
        except Exception:
            logger.exception("lang_id: failed to load %s — text-side detection disabled", path)
            return None
        _singleton = _DetectorHandle(model=model, model_path=path)
        logger.info("lang_id: loaded fastText lid.176 from %s", path)
        return _singleton


def detect(text: str, model_path: Path | None = None) -> tuple[str | None, float]:
    """Identify the language of ``text``.

    Returns
    -------
    tuple of (iso639-1 code or ``None``, probability in [0, 1])
        ``(None, 0.0)`` when the model is unavailable, the input is empty,
        or fastText returns no labels.
    """
    handle = get_detector(model_path)
    if handle is None or not text:
        return None, 0.0
    sanitized = text.replace("\n", " ").strip()
    if not sanitized:
        return None, 0.0
    labels, probs = handle.model.predict(sanitized, k=1)
    if not labels:
        return None, 0.0
    label = labels[0]
    if label.startswith("__label__"):
        label = label[len("__label__") :]
    return label, float(probs[0])


def load_at_startup(model_path_str: str = "") -> _DetectorHandle | None:
    """Eagerly load the detector during server lifespan.

    Invoked when ``text_lang_detection.enabled`` is true so the first
    /chat request does not pay the cold-load cost. Returns the handle
    on success, or ``None`` if the model file is missing — same
    fail-closed semantics as ``get_detector`` and ``detect``.
    """
    path = Path(model_path_str) if model_path_str else None
    return get_detector(path)


def reset_for_tests() -> None:
    """Reset the singleton state so unit tests can swap implementations."""
    global _singleton, _load_attempted
    with _load_lock:
        _singleton = None
        _load_attempted = False
