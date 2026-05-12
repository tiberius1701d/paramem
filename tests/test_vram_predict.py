"""Unit tests for paramem.server.vram_predict.

CPU-only — no GPU required. Uses tmp_path fake cache dirs and
monkey-patched huggingface_hub.try_to_load_from_cache.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers — build fake HF cache layouts
# ---------------------------------------------------------------------------


def _make_fake_snap(tmp_path: Path, files: dict[str, int]) -> Path:
    """Create fake snapshot dir with named files of given byte sizes.

    Files are created *sparse* (``truncate`` extends an empty file to the
    target length without allocating disk blocks) — ``vram_predict`` only
    reads ``stat().st_size``, so we get the right reported size for ~zero
    I/O.  This matters: the multi-GB shard sizes used below would otherwise
    write/delete tens of GB per run and blow the ``pytest-timeout`` budget
    on a slow CI runner.
    """
    snap = tmp_path / "snap"
    snap.mkdir(parents=True, exist_ok=True)
    for name, size in files.items():
        f = snap / name
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            fh.truncate(size)
    return snap


class _FakeModelConfig:
    def __init__(self, model_id: str, quantization: str = "nf4", compute_dtype: str = "bfloat16"):
        self.model_id = model_id
        self.quantization = quantization
        self.compute_dtype = compute_dtype


class _FakeSTTConfig:
    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "cuda",
        model: str = "distil-large-v3",
    ):
        self.enabled = enabled
        self.device = device
        self.model = model


class _FakeTTSVoice:
    def __init__(self, *, engine: str, model: str = "", device: str | None = None):
        self.engine = engine
        self.model = model
        self.device = device


class _FakeTTSConfig:
    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "cuda",
        voices: dict | None = None,
        model_dir: str = "",
    ):
        self.enabled = enabled
        self.device = device
        self.voices = voices or {}
        self.model_dir = model_dir


# ---------------------------------------------------------------------------
# _hf_cache_dir tests
# ---------------------------------------------------------------------------


def test_hf_cache_dir_returns_none_on_cache_miss():
    from paramem.server.vram_predict import _hf_cache_dir

    with patch("paramem.server.vram_predict.try_to_load_from_cache", return_value=None):
        result = _hf_cache_dir("some/model")
    assert result is None


def test_hf_cache_dir_returns_parent_dir(tmp_path):
    from paramem.server.vram_predict import _hf_cache_dir

    config_file = tmp_path / "config.json"
    config_file.touch()

    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = _hf_cache_dir("some/model")
    assert result == tmp_path


# ---------------------------------------------------------------------------
# predict_base_bytes tests
# ---------------------------------------------------------------------------


def test_predict_base_bytes_nf4_factor_applied(tmp_path):
    """NF4 factor (0.55) must be applied to safetensors disk bytes."""
    from paramem.server.vram_predict import _NF4_FACTOR, predict_base_bytes

    shard_size = 8_000_000_000  # 8 GiB raw
    snap = _make_fake_snap(tmp_path, {"model.safetensors": shard_size})
    config_file = snap / "config.json"
    config_file.touch()

    model_cfg = _FakeModelConfig("test/model", quantization="nf4")
    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = predict_base_bytes(model_cfg)

    assert result == int(shard_size * _NF4_FACTOR)


def test_predict_base_bytes_fp16_factor_applied(tmp_path):
    """FP16 factor (1.0) means predicted == raw disk bytes."""
    from paramem.server.vram_predict import _FP16_FACTOR, predict_base_bytes

    shard_size = 4_000_000_000
    snap = _make_fake_snap(tmp_path, {"model.safetensors": shard_size})
    config_file = snap / "config.json"
    config_file.touch()

    model_cfg = _FakeModelConfig("test/model", quantization="fp16")
    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = predict_base_bytes(model_cfg)

    assert result == int(shard_size * _FP16_FACTOR)


def test_predict_base_bytes_returns_none_on_cache_miss():
    from paramem.server.vram_predict import predict_base_bytes

    model_cfg = _FakeModelConfig("uncached/model")
    with patch("paramem.server.vram_predict.try_to_load_from_cache", return_value=None):
        result = predict_base_bytes(model_cfg)
    assert result is None


def test_predict_base_bytes_sums_multiple_shards(tmp_path):
    """Multi-shard models: all .safetensors shards are summed."""
    from paramem.server.vram_predict import _NF4_FACTOR, predict_base_bytes

    snap = _make_fake_snap(
        tmp_path,
        {
            "model-00001-of-00003.safetensors": 3_000_000_000,
            "model-00002-of-00003.safetensors": 3_000_000_000,
            "model-00003-of-00003.safetensors": 3_000_000_000,
        },
    )
    config_file = snap / "config.json"
    config_file.touch()

    model_cfg = _FakeModelConfig("test/model", quantization="nf4")
    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = predict_base_bytes(model_cfg)

    assert result == int(9_000_000_000 * _NF4_FACTOR)


# ---------------------------------------------------------------------------
# predict_stt_bytes tests
# ---------------------------------------------------------------------------


def test_predict_stt_bytes_returns_zero_when_disabled():
    from paramem.server.vram_predict import predict_stt_bytes

    cfg = _FakeSTTConfig(enabled=False)
    assert predict_stt_bytes(cfg) == 0


def test_predict_stt_bytes_returns_zero_when_cpu():
    from paramem.server.vram_predict import predict_stt_bytes

    cfg = _FakeSTTConfig(device="cpu")
    assert predict_stt_bytes(cfg) == 0


def test_predict_stt_bytes_returns_zero_when_permanent_cloud_only():
    from paramem.server.vram_predict import predict_stt_bytes

    cfg = _FakeSTTConfig(device="cuda")
    assert predict_stt_bytes(cfg, permanent_cloud_only=True) == 0


def test_predict_stt_bytes_ct2_skips_quant_factor(tmp_path):
    """CT2 path: disk bytes × _CT2_WORKSPACE_FACTOR only — no quant factor."""
    from paramem.server.vram_predict import _CT2_WORKSPACE_FACTOR, predict_stt_bytes

    disk_bytes = 1_000_000_000
    snap = _make_fake_snap(tmp_path, {"model.bin": disk_bytes, "config.json": 100})
    config_file = snap / "config.json"

    cfg = _FakeSTTConfig(device="cuda", model="distil-large-v3")
    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = predict_stt_bytes(cfg)

    # Must be disk_bytes × 1.2, NOT disk_bytes × 0.55 × 1.2
    expected = int((disk_bytes + 100) * _CT2_WORKSPACE_FACTOR)
    assert result == expected


def test_predict_stt_bytes_returns_none_on_cache_miss():
    from paramem.server.vram_predict import predict_stt_bytes

    cfg = _FakeSTTConfig(device="cuda", model="distil-large-v3")
    with patch("paramem.server.vram_predict.try_to_load_from_cache", return_value=None):
        result = predict_stt_bytes(cfg)
    assert result is None


# ---------------------------------------------------------------------------
# predict_tts_bytes tests
# ---------------------------------------------------------------------------


def test_predict_tts_bytes_returns_zero_when_disabled():
    from paramem.server.vram_predict import predict_tts_bytes

    cfg = _FakeTTSConfig(enabled=False)
    assert predict_tts_bytes(cfg) == 0


def test_predict_tts_bytes_returns_zero_when_permanent_cloud_only():
    from paramem.server.vram_predict import predict_tts_bytes

    cfg = _FakeTTSConfig(
        device="cuda", voices={"en": _FakeTTSVoice(engine="piper", model="en_US-test")}
    )
    assert predict_tts_bytes(cfg, permanent_cloud_only=True) == 0


def test_predict_tts_bytes_piper_sums_onnx_plus_ort_context_once(tmp_path):
    """Multiple Piper voices on GPU: each voice's .onnx size + single ORT context."""
    from paramem.server.vram_predict import _TTS_PIPER_ORT_CONTEXT_BYTES, predict_tts_bytes

    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    onnx_a = piper_dir / "en_US-lessac-high.onnx"
    onnx_b = piper_dir / "de_DE-thorsten-high.onnx"
    onnx_a.write_bytes(b"\x00" * 80_000_000)
    onnx_b.write_bytes(b"\x00" * 90_000_000)

    cfg = _FakeTTSConfig(
        device="cuda",
        model_dir=str(piper_dir),
        voices={
            "en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high"),
            "de": _FakeTTSVoice(engine="piper", model="de_DE-thorsten-high"),
        },
    )
    result = predict_tts_bytes(cfg)
    expected = 80_000_000 + 90_000_000 + _TTS_PIPER_ORT_CONTEXT_BYTES
    assert result == expected


def test_predict_tts_bytes_ort_context_counted_once(tmp_path):
    """ORT context is added once regardless of number of Piper voices."""
    from paramem.server.vram_predict import _TTS_PIPER_ORT_CONTEXT_BYTES, predict_tts_bytes

    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    for name in ["a.onnx", "b.onnx", "c.onnx"]:
        (piper_dir / name).write_bytes(b"\x00" * 10_000_000)

    cfg = _FakeTTSConfig(
        device="cuda",
        model_dir=str(piper_dir),
        voices={
            "a": _FakeTTSVoice(engine="piper", model="a"),
            "b": _FakeTTSVoice(engine="piper", model="b"),
            "c": _FakeTTSVoice(engine="piper", model="c"),
        },
    )
    result = predict_tts_bytes(cfg)
    assert result == 3 * 10_000_000 + _TTS_PIPER_ORT_CONTEXT_BYTES


def test_predict_tts_bytes_cpu_voice_excluded(tmp_path):
    """Voice with device='cpu' must not contribute to GPU budget."""
    from paramem.server.vram_predict import _TTS_PIPER_ORT_CONTEXT_BYTES, predict_tts_bytes

    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    (piper_dir / "en_US-gpu.onnx").write_bytes(b"\x00" * 50_000_000)
    (piper_dir / "de_DE-cpu.onnx").write_bytes(b"\x00" * 70_000_000)

    cfg = _FakeTTSConfig(
        device="cuda",
        model_dir=str(piper_dir),
        voices={
            "en": _FakeTTSVoice(engine="piper", model="en_US-gpu"),
            "de": _FakeTTSVoice(engine="piper", model="de_DE-cpu", device="cpu"),
        },
    )
    result = predict_tts_bytes(cfg)
    # Only "en" GPU voice counts
    assert result == 50_000_000 + _TTS_PIPER_ORT_CONTEXT_BYTES


def test_predict_tts_bytes_returns_none_on_piper_cache_miss(tmp_path):
    """If any GPU Piper .onnx is missing, returns None."""
    from paramem.server.vram_predict import predict_tts_bytes

    piper_dir = tmp_path / "piper"
    piper_dir.mkdir()
    # "en" present, "de" missing
    (piper_dir / "en_US-test.onnx").write_bytes(b"\x00" * 60_000_000)

    cfg = _FakeTTSConfig(
        device="cuda",
        model_dir=str(piper_dir),
        voices={
            "en": _FakeTTSVoice(engine="piper", model="en_US-test"),
            "de": _FakeTTSVoice(engine="piper", model="de_DE-missing"),
        },
    )
    result = predict_tts_bytes(cfg)
    assert result is None


def test_predict_tts_bytes_mms_uses_cached_safetensors(tmp_path):
    """MMS voices: sum cached safetensors (no ORT context)."""
    from paramem.server.vram_predict import predict_tts_bytes

    snap = _make_fake_snap(tmp_path, {"model.safetensors": 200_000_000, "config.json": 100})
    config_file = snap / "config.json"

    cfg = _FakeTTSConfig(
        device="cuda",
        voices={"tl": _FakeTTSVoice(engine="mms", model="facebook/mms-tts-tgl")},
    )
    with patch(
        "paramem.server.vram_predict.try_to_load_from_cache",
        return_value=str(config_file),
    ):
        result = predict_tts_bytes(cfg)

    assert result == 200_000_000
