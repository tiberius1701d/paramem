"""``paramem.server.app._check_scan_skeleton_drift`` — the boot-time check
that the live prefix table's rendered SCAN skeleton still fits the pinned
token budget, measured on the CHAT-WRAPPED render (system + user), never
the user section alone.
"""

from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import pytest

import paramem.server.app as app_module
from paramem.cloud.anonymize import AnonymizerPrompts
from paramem.server.app import _check_scan_skeleton_drift


class _WrappedRenderTokenizer:
    """A tokenizer stub whose ``apply_chat_template`` joins every message's
    content (so the caller can tell a system-wrapped render from a
    user-only one) and whose ``__call__`` counts one token per character —
    a fully deterministic, controllable token count with no real model."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "||".join(m["content"] for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text)}


def _config(prompts_dir=None) -> SimpleNamespace:
    return SimpleNamespace(prompts_dir=prompts_dir)


def _fake_prompts(scan_system: str, scan: str) -> AnonymizerPrompts:
    return AnonymizerPrompts(scan_system=scan_system, scan=scan, anchor_system="", anchor="")


class TestCheckScanSkeletonDrift:
    def test_measures_the_chat_wrapped_render_not_the_user_section_alone(self, monkeypatch):
        """The wrapped render (``"SYSCONTENT||SCANBODY"``, 20 chars) is
        longer than the user section alone (``"SCANBODY"``, 8 chars). A
        threshold between the two (15) raises only if the measurement is
        taken on the WRAPPED text — a regression that measured the user
        section alone would never see 20 and would wrongly pass."""
        monkeypatch.setattr(
            "paramem.graph.anonymizer_prompts.load_anonymizer_prompts",
            lambda *, prompts_dir=None: _fake_prompts("SYSCONTENT", "SCANBODY"),
        )
        monkeypatch.setattr("paramem.utils.tokens.ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS", 15)

        with pytest.raises(RuntimeError, match="SCAN prompt skeleton drift"):
            _check_scan_skeleton_drift(_config(), _WrappedRenderTokenizer())

    def test_passes_when_the_wrapped_render_is_at_or_under_the_pinned_constant(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.graph.anonymizer_prompts.load_anonymizer_prompts",
            lambda *, prompts_dir=None: _fake_prompts("SYSCONTENT", "SCANBODY"),
        )
        monkeypatch.setattr("paramem.utils.tokens.ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS", 20)

        _check_scan_skeleton_drift(_config(), _WrappedRenderTokenizer())  # must not raise

    def test_the_error_names_the_constant_and_the_measured_size(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.graph.anonymizer_prompts.load_anonymizer_prompts",
            lambda *, prompts_dir=None: _fake_prompts("SYSCONTENT", "SCANBODY"),
        )
        monkeypatch.setattr("paramem.utils.tokens.ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS", 1)

        with pytest.raises(RuntimeError) as exc_info:
            _check_scan_skeleton_drift(_config(), _WrappedRenderTokenizer())

        message = str(exc_info.value)
        assert "1" in message  # the pinned constant
        assert "20" in message  # the measured wrapped size


class TestBuildRuntimeComponentsSetsThreadsFirst:
    """``torch.set_num_threads(config.cpu.threads)`` is the FIRST statement
    of ``_build_runtime_components`` — a structural check on the source,
    since exercising the routine behaviourally would load every runtime
    component (speaker embedding, TTS/STT, cloud agent, HA client)."""

    def test_torch_set_num_threads_is_the_first_statement_in_the_source(self):
        source = inspect.getsource(app_module._build_runtime_components)
        tree = ast.parse(source)
        (func_def,) = tree.body
        assert isinstance(func_def, ast.FunctionDef)

        body = func_def.body
        # body[0] is the docstring (an Expr wrapping a str Constant) when
        # present; the first EXECUTABLE statement is the one right after it.
        first_statement = body[1] if _is_docstring(body[0]) else body[0]

        assert isinstance(first_statement, ast.Expr)
        call = first_statement.value
        assert isinstance(call, ast.Call)
        assert ast.unparse(call.func) == "torch.set_num_threads"
        assert ast.unparse(call.args[0]) == "config.cpu.threads"


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
