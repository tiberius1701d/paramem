"""Tests for the calibration-tool inference-param plumbing in
``paramem.evaluation.recall.generate_answer``.

These tests verify that ``top_p`` / ``top_k`` / ``seed`` are forwarded
to ``model.generate(...)`` correctly, and that ``seed`` is implemented
via ``torch.manual_seed`` immediately before generation (global torch RNG),
making sampling reproducible at temperature > 0 (no-op at temperature 0).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from paramem.evaluation.recall import generate_answer


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt"):
        out = MagicMock()
        out.to = lambda device: {
            "input_ids": torch.zeros((1, 4), dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
        return out

    def encode(self, token_name, add_special_tokens=False):
        return [99]  # any single id; recall code dedups

    def decode(self, ids, skip_special_tokens=True):
        return "ok"


def _fake_model() -> MagicMock:
    m = MagicMock()
    m.device = torch.device("cpu")
    m.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
    return m


def test_top_p_top_k_forwarded_to_generate():
    model = _fake_model()
    tok = _FakeTokenizer()
    generate_answer(model, tok, "p", max_new_tokens=4, top_p=0.95, top_k=40)
    kwargs = model.generate.call_args.kwargs
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 40
    assert "generator" not in kwargs  # seed not provided → no generator kwarg


def test_seed_sets_global_rng_and_no_generator_kwarg():
    """seed= must call torch.manual_seed (global RNG) immediately before
    generation and must NOT pass a ``generator=`` kwarg to model.generate.
    The global RNG is mutated; that is the intended contract for the
    serialized calibration use case."""
    model = _fake_model()
    tok = _FakeTokenizer()

    # Capture global RNG state; assert it IS changed after the call (seed was applied).
    before = torch.random.get_rng_state()
    generate_answer(model, tok, "p", max_new_tokens=4, seed=12345)
    after = torch.random.get_rng_state()
    assert not torch.equal(before, after), (
        "generate_answer(seed=...) must set the global torch RNG via torch.manual_seed"
    )

    kwargs = model.generate.call_args.kwargs
    assert "generator" not in kwargs, (
        "generate_answer(seed=...) must not pass a generator= kwarg to model.generate"
    )


def test_no_overrides_no_extra_kwargs():
    model = _fake_model()
    tok = _FakeTokenizer()
    generate_answer(model, tok, "p", max_new_tokens=4)
    kwargs = model.generate.call_args.kwargs
    for k in ("top_p", "top_k", "generator"):
        assert k not in kwargs, f"unexpected kwarg {k!r} forwarded when no override given"


def test_filter_anthropic_top_p_top_k_forwarded(monkeypatch):
    """``_filter_anthropic`` must thread top_p / top_k into
    ``client.messages.create``.  Anthropic has no seed parameter — the
    calibration tool documents this; here we just verify the two
    sampling knobs reach the SDK."""
    from paramem.graph import extractor as ex_mod

    captured = {}

    class _FakeMsg:
        def __init__(self, text):
            self.text = text

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            response = MagicMock()
            response.content = [_FakeMsg("hello")]
            return response

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()

    fake_module = MagicMock()
    fake_module.Anthropic = _FakeAnthropic
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)

    out = ex_mod._filter_anthropic(
        prompt="p",
        api_key="k",
        filter_model="claude",
        top_p=0.9,
        top_k=20,
    )
    assert out == "hello"
    assert captured.get("top_p") == 0.9
    assert captured.get("top_k") == 20


def test_filter_anthropic_no_overrides(monkeypatch):
    from paramem.graph import extractor as ex_mod

    captured = {}

    class _FakeMsg:
        def __init__(self, text):
            self.text = text

    class _FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            response = MagicMock()
            response.content = [_FakeMsg("hello")]
            return response

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            self.messages = _FakeMessages()

    fake_module = MagicMock()
    fake_module.Anthropic = _FakeAnthropic
    monkeypatch.setitem(__import__("sys").modules, "anthropic", fake_module)

    ex_mod._filter_anthropic(prompt="p", api_key="k", filter_model="claude")
    assert "top_p" not in captured
    assert "top_k" not in captured
