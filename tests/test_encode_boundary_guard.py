"""Structural guard: pin the single encode/render chokepoint for chat-template text.

A rendered chat template (:func:`paramem.models.loader.render_chat_prompt`)
carries its own literal BOS (and, for training targets, a literal trailing
EOS). Two independent ways to reintroduce a double-BOS / lost-EOS bug exist:

* A bare ``tokenizer(text)`` / ``_tokenizer(text)`` call — or a
  ``tokenizer.encode(text)`` / ``_tokenizer.encode(text)`` method call —
  with no explicit ``add_special_tokens`` keyword — the tokenizer's own
  default (usually ``True``) re-adds a second BOS on top of the template's
  own. :func:`paramem.utils.tokens.encode_rendered` is THE tensorizer that
  fixes ``add_special_tokens=False`` for rendered text; every call site
  that skips it and calls the tokenizer directly (either shape) must say
  so explicitly.
* A direct ``apply_chat_template(...)`` call outside
  ``paramem/models/loader.py`` — :func:`paramem.models.loader.render_chat_prompt`
  is the one production renderer (it also applies ``adapt_messages``
  internally); a second renderer can drift from it silently.

This test scans the tracked codebase (``paramem/``, ``experiments/``,
``scripts/``) and fails on any new hit outside the allowlists below. Pair
with ``tests/test_prompts_present.py::TestEncodeBoundaryWeightCoupledPin``
(the weight-coupling pin for the real tokenizer) and
``tests/test_tokens.py::TestEncodeCountIdentity`` (the count/encode
identity ``encode_rendered`` and ``estimate_tokens`` share).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests._guard_utils import tracked_python_files

_SCAN_PREFIXES = ("paramem/", "experiments/", "scripts/")

_TOKENIZER_CALL_NAMES = frozenset({"tokenizer", "_tokenizer"})

_LOADER_FILE = "paramem/models/loader.py"

# Predicate A allowlist — bare tokenizer(...)/_tokenizer(...) calls that
# never touch chat-template-rendered text, so add_special_tokens=False is
# not a correctness requirement at that call site. Each entry states why;
# TestAllowlistEntriesAreLiveHits below fails if an entry stops matching a
# real hit (a dead allowlist entry is a silent gap in the guard).
_ENCODE_GUARD_ALLOWLIST: dict[str, str] = {
    "paramem/server/tts.py": (
        "MMS-TTS's own tokenizer (self._tokenizer), called on raw synthesis "
        "text. Never chat-template-rendered — no BOS/EOS policy shared with "
        "the LLM encode path this guard protects."
    ),
    "experiments/smoke_procedural_mlp.py": (
        "Encodes a raw fixture sentence for a memorization smoke check — "
        "never rendered through render_chat_prompt, so there is no "
        "template-literal BOS to double."
    ),
    "experiments/test_prompt_engineering.py": (
        "Mixed provenance: the try/except above this call falls back to the "
        "raw, un-templated prompt string when apply_chat_template raises "
        "(see the apply_chat_template hit in the same file), so the "
        "downstream tokenize call cannot uniformly assume rendered text "
        "carrying its own BOS. Standalone prompt-engineering probe, not a "
        "production encode path."
    ),
    "experiments/test11_adapter_extraction.py": (
        "tokenizer.encode(raw_output) counts tokens in the model's own raw "
        "generated output (for a truncation diagnostic), never chat-"
        "template-rendered text — there is no template-literal BOS to "
        "double at this call site."
    ),
}

# Predicate B allowlist — files outside paramem/models/loader.py permitted
# to call apply_chat_template directly.
_RENDER_GUARD_ALLOWLIST: dict[str, str] = {
    "experiments/test_prompt_engineering.py": (
        "Standalone prompt-engineering probe with its own local "
        "apply_chat_template + tokenizer(...) fallback pair, predating "
        "render_chat_prompt/encode_rendered. Not part of the production "
        "encode path this guard protects."
    ),
}

_ENCODE_GUARD_MESSAGE = (
    "Bare tokenizer(text) / _tokenizer(text) call with no explicit "
    "add_special_tokens keyword. Chat-template-rendered text already "
    "carries its own literal BOS (and, for training targets, trailing EOS) "
    "— a naive re-encode with the tokenizer's default add_special_tokens "
    "re-adds a second one. Route through "
    "paramem.utils.tokens.encode_rendered (always add_special_tokens=False), "
    "or add a justified entry to _ENCODE_GUARD_ALLOWLIST in "
    "tests/test_encode_boundary_guard.py."
)

_RENDER_GUARD_MESSAGE = (
    "apply_chat_template called outside paramem/models/loader.py. "
    "paramem.models.loader.render_chat_prompt is the ONE production chat-"
    "template renderer (it also applies adapt_messages internally). Route "
    "through it, or add a justified entry to _RENDER_GUARD_ALLOWLIST in "
    "tests/test_encode_boundary_guard.py."
)


def _callee_final_name(func: ast.expr) -> str | None:
    """Return the final dotted component of a call's callee.

    ``tokenizer(...)`` -> ``"tokenizer"`` (an ``ast.Name``).
    ``self._tokenizer(...)`` -> ``"_tokenizer"`` (an ``ast.Attribute``).
    Anything else (a subscript, a call result, ...) -> ``None``.
    """
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_tokenizer_receiver(expr: ast.expr) -> bool:
    """True when *expr* is a ``tokenizer``/``_tokenizer``-named receiver.

    ``tokenizer`` -> True (an ``ast.Name``). ``self._tokenizer`` -> True
    (an ``ast.Attribute`` whose own final component is ``_tokenizer``).
    Used to scope the ``.encode(...)`` method-call predicate to the
    tokenizer's own encode method — never a same-named ``.encode(...)`` on
    an unrelated receiver (bytes ``str.encode(...)``, a sentence-transformer
    ``model.encode(...)``), which is not a stop-token call this guard
    protects.
    """
    if isinstance(expr, ast.Name):
        return expr.id in _TOKENIZER_CALL_NAMES
    if isinstance(expr, ast.Attribute):
        return expr.attr in _TOKENIZER_CALL_NAMES
    return False


def find_bare_tokenizer_calls(tree: ast.AST) -> list[int]:
    """Return line numbers of tokenizer stop-token calls missing an
    explicit ``add_special_tokens`` keyword.

    Two call shapes:

    * ``tokenizer(...)`` / ``_tokenizer(...)`` — the tokenizer's own
      ``__call__``.
    * ``tokenizer.encode(...)`` / ``_tokenizer.encode(...)`` (or an
      attribute-chained receiver ending in ``_tokenizer``, e.g.
      ``self._tokenizer.encode(...)``) — the tokenizer's ``.encode()``
      method, scoped to a tokenizer-named receiver via
      :func:`_is_tokenizer_receiver` so an unrelated ``.encode(...)`` (bytes
      encoding, a sentence-transformer embedding call) never matches.

    A ``**kwargs`` forward that itself carries the keyword (e.g.
    :func:`~paramem.utils.tokens.encode_rendered`'s own body,
    ``tokenizer(text, add_special_tokens=False, **tokenizer_kwargs)``)
    self-passes — the explicit keyword is present in the AST regardless of
    what else is forwarded.
    """
    out: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if any(kw.arg == "add_special_tokens" for kw in node.keywords):
            continue
        func = node.func
        name = _callee_final_name(func)
        is_bare_call = name in _TOKENIZER_CALL_NAMES
        is_encode_method_call = (
            isinstance(func, ast.Attribute)
            and func.attr == "encode"
            and _is_tokenizer_receiver(func.value)
        )
        if is_bare_call or is_encode_method_call:
            out.append(node.lineno)
    return out


def find_apply_chat_template_calls(tree: ast.AST) -> list[int]:
    """Return line numbers of ``<x>.apply_chat_template(...)`` calls."""
    out: list[int] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "apply_chat_template"
        ):
            out.append(node.lineno)
    return out


def _scan_files(repo_root: Path) -> list[tuple[str, ast.AST]]:
    out: list[tuple[str, ast.AST]] = []
    for py_file in tracked_python_files(repo_root):
        rel = py_file.relative_to(repo_root).as_posix()
        if not rel.startswith(_SCAN_PREFIXES):
            continue
        try:
            text = py_file.read_text()
        except UnicodeDecodeError:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        out.append((rel, tree))
    return out


def test_no_bare_tokenizer_calls_outside_allowlist():
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for rel, tree in _scan_files(repo_root):
        if rel in _ENCODE_GUARD_ALLOWLIST:
            continue
        for lineno in find_bare_tokenizer_calls(tree):
            offenders.append(f"{rel}:{lineno}")

    assert not offenders, _ENCODE_GUARD_MESSAGE + "\n" + "\n".join(f"  {o}" for o in offenders)


def test_no_apply_chat_template_calls_outside_loader():
    repo_root = Path(__file__).resolve().parent.parent
    offenders: list[str] = []
    for rel, tree in _scan_files(repo_root):
        if rel == _LOADER_FILE:
            continue
        if rel in _RENDER_GUARD_ALLOWLIST:
            continue
        for lineno in find_apply_chat_template_calls(tree):
            offenders.append(f"{rel}:{lineno}")

    assert not offenders, _RENDER_GUARD_MESSAGE + "\n" + "\n".join(f"  {o}" for o in offenders)


class TestAllowlistEntriesAreLiveHits:
    """Every allowlist entry must correspond to an ACTUAL predicate hit in
    the current tree. An entry for code that no longer trips the predicate
    (the call site was rewritten or deleted) is a stale exemption masking
    nothing — it should be removed, not carried forward."""

    def test_encode_guard_allowlist_entries_are_live(self):
        repo_root = Path(__file__).resolve().parent.parent
        for rel in _ENCODE_GUARD_ALLOWLIST:
            tree = ast.parse((repo_root / rel).read_text())
            assert find_bare_tokenizer_calls(tree), (
                f"{rel} is in _ENCODE_GUARD_ALLOWLIST but no bare tokenizer() "
                "call was found there — remove the stale entry."
            )

    def test_render_guard_allowlist_entries_are_live(self):
        repo_root = Path(__file__).resolve().parent.parent
        for rel in _RENDER_GUARD_ALLOWLIST:
            tree = ast.parse((repo_root / rel).read_text())
            assert find_apply_chat_template_calls(tree), (
                f"{rel} is in _RENDER_GUARD_ALLOWLIST but no apply_chat_template "
                "call was found there — remove the stale entry."
            )


class TestPredicateSelfTest:
    """Self-test the two AST predicates against inline synthetic snippets —
    proof the guard actually catches what it claims to, independent of
    whatever the current tree happens to contain."""

    def test_bare_name_form_tokenizer_call_is_flagged(self):
        tree = ast.parse("tokenizer(text)")
        assert find_bare_tokenizer_calls(tree) == [1]

    def test_bare_attribute_form_tokenizer_call_is_flagged(self):
        tree = ast.parse("self._tokenizer(text)")
        assert find_bare_tokenizer_calls(tree) == [1]

    def test_tokenizer_call_with_explicit_add_special_tokens_is_not_flagged(self):
        tree = ast.parse("tokenizer(text, add_special_tokens=False)")
        assert find_bare_tokenizer_calls(tree) == []

    def test_encode_rendered_own_internal_call_shape_self_passes(self):
        """Mirrors encode_rendered's own body — the explicit keyword plus a
        forwarded **kwargs must still count as passing the flag."""
        tree = ast.parse("tokenizer(text, add_special_tokens=False, **tokenizer_kwargs)")
        assert find_bare_tokenizer_calls(tree) == []

    def test_unrelated_call_name_is_not_flagged(self):
        tree = ast.parse("something.other_method(text)")
        assert find_bare_tokenizer_calls(tree) == []

    def test_bare_name_encode_method_call_is_flagged(self):
        tree = ast.parse("tokenizer.encode(text)")
        assert find_bare_tokenizer_calls(tree) == [1]

    def test_attribute_chained_encode_method_call_is_flagged(self):
        tree = ast.parse("self._tokenizer.encode(text)")
        assert find_bare_tokenizer_calls(tree) == [1]

    def test_encode_method_call_with_explicit_add_special_tokens_is_not_flagged(self):
        tree = ast.parse("tokenizer.encode(text, add_special_tokens=False)")
        assert find_bare_tokenizer_calls(tree) == []

    def test_encode_method_call_on_unrelated_receiver_is_not_flagged(self):
        """``.encode(...)`` is also bytes-encoding and sentence-transformer
        embedding shorthand — only a tokenizer-named receiver is a stop-
        token call this guard protects."""
        tree = ast.parse('data.encode("utf-8")')
        assert find_bare_tokenizer_calls(tree) == []

    def test_unrelated_encode_call_with_no_receiver_is_not_flagged(self):
        tree = ast.parse("encode(text)")
        assert find_bare_tokenizer_calls(tree) == []

    def test_apply_chat_template_call_is_flagged(self):
        tree = ast.parse("tokenizer.apply_chat_template(messages, tokenize=False)")
        assert find_apply_chat_template_calls(tree) == [1]

    def test_bare_name_call_is_not_flagged_as_apply_chat_template(self):
        """apply_chat_template is always an attribute call (`<x>.apply_chat_template`)
        — a same-named bare function call is a different, hypothetical shape
        and must not be flagged."""
        tree = ast.parse("apply_chat_template(messages)")
        assert find_apply_chat_template_calls(tree) == []
