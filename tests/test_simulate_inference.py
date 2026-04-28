"""End-to-end simulate-mode /chat coverage.

Exercises ``handle_chat`` → ``_probe_and_reason`` → ``probe_keys_from_disk``
in simulate mode using an on-disk simulate store, a stubbed model, and a
stubbed router.  No real LLM is invoked — ``generate_answer`` is patched at
the inference module boundary.

Pattern reference: ``tests/test_abstention_integration.py``.

Six tests are grouped in ``TestSimulateInferenceEndToEnd``:

* T1  — PA match with canonical episodic keyed_pairs in ``paths.simulate``
        returns the disk answer.
* T2  — PA match with no keyed_pairs.json walks HA → SOTA → fallback.
* T2b — PA match, HA fails + SOTA unavailable → base-model helper reached.
* T3  — All three tiers (episodic, semantic, procedural) read from
        ``<paths.simulate>/<tier>/keyed_pairs.json``.
* T4  — simulate-mode chat ignores stale keyed_pairs under ``paths.adapters``.
* T4b — train-mode chat never calls ``probe_keys_from_disk``.

Tokenizer setup note
--------------------
In these tests the tokenizer is a ``MagicMock``.  The
``tokenizer.apply_chat_template(messages, ...)`` call inside
``_probe_and_reason`` produces the prompt string that ``generate_answer``
receives as its third positional argument.  To make the prompt a real string
that carries the disk-read facts, every test configures::

    tokenizer.apply_chat_template.side_effect = lambda msgs, **_: json.dumps(msgs)

This causes ``apply_chat_template`` to return a JSON serialisation of the
messages list, which contains the ``augmented_text`` (and therefore the
disk-read answer token) as the user-turn content.  Without this configuration
the call returns a ``MagicMock`` object, not a string, and ``"Heilbronn." in
str(mock_obj)`` is vacuously false.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.config import PathsConfig, load_server_config
from paramem.server.inference import ChatResult, handle_chat
from paramem.server.router import Intent, RoutingPlan, RoutingStep

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_keyed_pairs(path: Path, pairs: list[dict]) -> None:
    """Write *pairs* to *path* as JSON, creating parent directories.

    Mirrors the ``_write_pairs`` helper in
    ``tests/test_simulate_train_parity.py:224-226``.

    Args:
        path: Absolute path to the target ``keyed_pairs.json`` file.
        pairs: List of keyed-pair dicts, each with at least ``key``,
            ``question``, and ``answer`` fields.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs))


def _simulate_config(tmp_path: Path):
    """Return a ``ServerConfig`` with simulate mode and tmp_path stores.

    Loads ``tests/fixtures/server.yaml`` (the CI test fixture), then
    overrides:

    * ``consolidation.mode`` → ``"simulate"``
    * ``paths.simulate``      → ``tmp_path / "simulate"``
    * ``paths.data``          → ``tmp_path / "data"``  (drives ``paths.adapters``)

    All other fields inherit from the fixture so the sanitizer,
    abstention, and voice configs are real but stable across machines.

    Per CLAUDE.md: tests must use the fixture, not ``configs/server.yaml``
    (operator-local, gitignored, drifts per machine) or
    ``configs/server.yaml.example`` (deployment-shape, drifts as
    deployment patterns evolve).

    Args:
        tmp_path: pytest ``tmp_path`` fixture value — unique per test.

    Returns:
        A ``ServerConfig`` configured for simulate-mode inference.
    """
    config = load_server_config("tests/fixtures/server.yaml")
    config.consolidation.mode = "simulate"
    config.paths = PathsConfig(
        data=tmp_path / "data",
        sessions=tmp_path / "data" / "sessions",
        debug=tmp_path / "data" / "debug",
        simulate=tmp_path / "simulate",
        prompts=config.paths.prompts,  # keep real prompts path
    )
    return config


def _pa_router_stub(adapter_name: str, keys: list[str]) -> MagicMock:
    """Return a router stub that emits a PA-match plan for *adapter_name*.

    The stub mirrors the minimal interface ``handle_chat`` reads from the
    router:

    * ``router.route(...)`` → ``RoutingPlan``
    * ``router._speaker_key_index`` — used by the abstention gate;
      setting it to a non-empty dict prevents the cold-start response.
    * ``router._all_entities`` — used by the sanitizer gate; empty set is
      safe for non-personal queries.

    Args:
        adapter_name: Tier name (``"episodic"``, ``"semantic"``, or
            ``"procedural"``).
        keys: List of key strings to probe (e.g. ``["graph1"]``).

    Returns:
        A ``MagicMock`` router that returns a PA routing plan.
    """
    plan = RoutingPlan(
        strategy="direct",
        steps=[RoutingStep(adapter_name=adapter_name, keys_to_probe=keys)],
        intent=Intent.PERSONAL,
    )
    router = MagicMock()
    router.route = lambda text, speaker=None, speaker_id=None: plan
    # Non-empty index prevents the abstention cold-start branch.
    router._speaker_key_index = {"spk-test": ["graph1"]}
    router._all_entities = set()
    return router


def _stub_model() -> MagicMock:
    """Return a minimal model stub for non-GPU simulate-mode tests.

    Uses a plain ``MagicMock()`` (no spec restriction) so attribute access
    works normally.  A plain ``MagicMock`` is not an instance of
    ``peft.PeftModel``, so the ``isinstance(model, PeftModel)`` guard in
    ``_probe_and_reason`` and ``_base_model_answer`` correctly takes the
    ``else`` branch — no ``with model.disable_adapter():`` wrapping occurs.

    ``gradient_checkpointing_disable`` is a no-op mock.
    ``generate`` raises ``RuntimeError`` — proves the test does NOT
    exercise real model inference (``generate_answer`` is always patched
    upstream in the success-path tests).
    """
    model = MagicMock()
    model.gradient_checkpointing_disable = MagicMock()
    model.generate.side_effect = RuntimeError(
        "model.generate() must not be called — simulate-mode disk recall failed"
    )
    return model


def _chat_template_tokenizer() -> MagicMock:
    """Return a tokenizer stub whose apply_chat_template yields a real string.

    ``apply_chat_template(messages, ...)`` is configured to return a JSON
    serialisation of the messages list.  This makes the prompt argument
    passed to ``generate_answer`` a real string that contains the
    user-turn content (i.e. the disk-read answer token) so that
    ``assert "Heilbronn." in prompt`` works as expected.

    Without this configuration the call returns a ``MagicMock`` object;
    ``"Heilbronn." in str(mock_obj)`` is always false.

    Returns:
        A ``MagicMock`` tokenizer with a string-producing
        ``apply_chat_template`` side-effect.
    """
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda msgs, **_: json.dumps(msgs)
    return tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulateInferenceEndToEnd:
    """End-to-end /chat tests for simulate-mode inference.

    All tests use a real ``handle_chat`` call (no monkey-patching of the
    function itself), a real ``configs/server.yaml`` loaded config, a
    ``MagicMock`` model, and a ``MagicMock`` router.  ``generate_answer``
    is always patched so no LLM is invoked.
    """

    # ------------------------------------------------------------------
    # T1
    # ------------------------------------------------------------------

    def test_pa_match_reads_canonical_episodic_from_simulate_store(self, tmp_path):
        """Disk-read answer from paths.simulate reaches the generate_answer prompt.

        Setup:
        - ``paths.simulate/episodic/keyed_pairs.json`` contains one pair with
          ``answer="Heilbronn."``.
        - ``paths.adapters`` (paths.data/adapters) is empty.
        - Config is simulate mode.

        Assertions:
        - ``generate_answer`` called once.
        - Prompt arg contains ``"Heilbronn."`` — proves disk-read fact
          reached the layered context.
        - ``result.text == "Alex lives in Heilbronn."`` (echoed from mock).
        - ``result.escalated is False``.
        - ``"graph1" in result.probed_keys``.

        Why this catches a regression: before the canonicalization, simulate mode
        read from ``paths.adapters``; this test enforces the post-canonicalization
        contract that simulate reads from ``paths.simulate``.
        """
        config = _simulate_config(tmp_path)

        _write_keyed_pairs(
            config.simulate_dir / "episodic" / "keyed_pairs.json",
            [
                {
                    "key": "graph1",
                    "question": "Where does Alex live?",
                    "answer": "Heilbronn.",
                    "source_subject": "Alex",
                }
            ],
        )
        # adapters directory intentionally left absent — simulate mode must NOT
        # read from it.

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = _chat_template_tokenizer()

        with patch(
            "paramem.server.inference.generate_answer",
            return_value="Alex lives in Heilbronn.",
        ) as mock_ga:
            result = handle_chat(
                text="Where does Alex live?",
                conversation_id="t1-simulate-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=None,
                ha_client=None,
                speaker_id="spk-test",
            )

        assert result.text == "Alex lives in Heilbronn."
        assert result.escalated is False
        assert "graph1" in result.probed_keys

        mock_ga.assert_called_once()
        call_args = mock_ga.call_args
        # The prompt is the third positional arg (model, tokenizer, prompt, …).
        prompt_arg = call_args.args[2] if call_args.args else call_args.kwargs.get("prompt", "")
        assert "Heilbronn." in str(prompt_arg), (
            f"Disk-read fact 'Heilbronn.' not found in generate_answer prompt arg: {prompt_arg!r}"
        )

    # ------------------------------------------------------------------
    # T2
    # ------------------------------------------------------------------

    def test_pa_match_kp_missing_falls_through_to_ha_then_base(self, tmp_path):
        """No keyed_pairs.json → fallback chain HA → base-model.

        Under the intent-keyed dispatch, ``intent=PERSONAL`` (set by the
        ``_pa_router_stub``) blocks SOTA at every internal escalation site.
        When PA disk-recall fails for a personal-class query the chain is:

            HA tool fallback (tools may help even for personal queries)
            ↓ (returns None)
            SOTA blocked by privacy invariant (would have run pre-Phase-2)
            ↓
            base-model fallthrough

        Setup:
        - ``paths.simulate`` exists but is empty (no keyed_pairs).
        - HA returns ``None``.
        - SOTA mock present but must NOT be invoked.
        - ``_base_model_answer`` patched to return the terminal answer.

        Assertions:
        - HA was tried once (tool fallback allowed for PERSONAL).
        - SOTA was NOT called (privacy invariant for PERSONAL).
        - ``_base_model_answer`` was reached as the terminal step.

        Why this catches a regression: any future change that re-enables
        the SOTA fallback for PERSONAL queries on disk-recall failure
        breaks this test.
        """
        config = _simulate_config(tmp_path)
        # The privacy invariant tested here ("PERSONAL never reaches
        # SOTA via direct escalation") only holds under cloud_mode=block
        # — under cloud_mode=anonymize PERSONAL queries DO reach SOTA
        # via the anonymization+deanonymization round-trip.  Pin
        # cloud_mode here so the test's stated invariant is the one
        # being asserted, regardless of the deployed default.
        config.sanitization.cloud_mode = "block"
        # Create the simulate_dir but write no keyed_pairs.json.
        config.simulate_dir.mkdir(parents=True, exist_ok=True)

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = MagicMock()

        base_result = ChatResult(text="<base>")

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("Where does Alex live?", []),
            ),
            patch(
                "paramem.server.inference._escalate_to_ha_agent",
                return_value=None,
            ) as mock_ha,
            patch(
                "paramem.server.inference._escalate_to_sota",
                return_value=ChatResult(text="<from sota>", escalated=True),
            ) as mock_sota,
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=base_result,
            ) as mock_base,
        ):
            result = handle_chat(
                text="Where does Alex live?",
                conversation_id="t2-simulate-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=MagicMock(),  # non-None to prove invariant gates it
                ha_client=MagicMock(),
                speaker_id="spk-test",
            )

        assert result.text == "<base>"
        mock_ha.assert_called_once()
        mock_sota.assert_not_called()
        mock_base.assert_called_once()
        model.generate.assert_not_called()

    # ------------------------------------------------------------------
    # T2b
    # ------------------------------------------------------------------

    def test_pa_match_kp_missing_all_cloud_failed_falls_to_base(self, tmp_path):
        """No keyed_pairs + HA fails + SOTA unavailable → base-model helper reached.

        Setup:
        - ``paths.simulate`` exists but empty.
        - ``sota_agent=None`` and HA returns ``None``.
        - ``_base_model_answer`` patched to return ``ChatResult(text="<base>")``.

        Code path: inside ``_probe_and_reason``, when ``layers`` is empty and
        ``sanitize_for_cloud`` allows escalation:

            result = _escalate_to_ha_agent(...)  # → None
            if result is not None: return result  # skipped
            if sota_agent is not None: ...        # skipped (sota_agent is None)
            return _base_model_answer(...)        # reached

        Assertion:
        - ``_base_model_answer`` was called once.

        Why this catches a regression: when all cloud paths are exhausted
        the caller must fall through to ``_base_model_answer``.  This
        verifies the fallback is not short-circuited.
        """
        config = _simulate_config(tmp_path)
        config.simulate_dir.mkdir(parents=True, exist_ok=True)

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = MagicMock()

        base_result = ChatResult(text="<base>")

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("Where does Alex live?", []),
            ),
            patch(
                "paramem.server.inference._escalate_to_ha_agent",
                return_value=None,
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=base_result,
            ) as mock_base,
        ):
            result = handle_chat(
                text="Where does Alex live?",
                conversation_id="t2b-simulate-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=None,  # None so SOTA branch is skipped; falls to _base_model_answer
                ha_client=MagicMock(),
                speaker_id="spk-test",
            )

        mock_base.assert_called_once()
        assert result.text == "<base>"

    # ------------------------------------------------------------------
    # T3
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("tier", ["episodic", "semantic", "procedural"])
    def test_canonical_layout_per_tier(self, tmp_path, tier):
        """Per-tier canonical layout: each tier's disk answer reaches generate_answer.

        Parametrized over ``("episodic", "semantic", "procedural")``.  For each
        tier, writes a single keyed pair under
        ``paths.simulate/<tier>/keyed_pairs.json`` and asserts the pair's answer
        appears in the ``generate_answer`` prompt argument.

        Why this catches a regression: validates the canonicalization is
        correct for all three tiers, not just episodic — the special case was
        on episodic, but the fix touches the path expression for all tiers
        in ``probe_keys_from_disk``.
        """
        config = _simulate_config(tmp_path)

        answer_token = f"ANSWER-FOR-{tier.upper()}"
        key = f"{tier[:3]}1"
        _write_keyed_pairs(
            config.simulate_dir / tier / "keyed_pairs.json",
            [
                {
                    "key": key,
                    "question": f"Question for {tier}?",
                    "answer": answer_token,
                    "source_subject": "Alex",
                }
            ],
        )

        router = _pa_router_stub(tier, [key])
        model = _stub_model()
        tokenizer = _chat_template_tokenizer()

        with patch(
            "paramem.server.inference.generate_answer",
            return_value=f"Answer from {tier}.",
        ) as mock_ga:
            handle_chat(
                text=f"Question for {tier}?",
                conversation_id=f"t3-{tier}-simulate-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=None,
                ha_client=None,
                speaker_id="spk-test",
            )

        mock_ga.assert_called_once()
        prompt_arg = mock_ga.call_args.args[2]
        assert answer_token in str(prompt_arg), (
            f"Tier '{tier}' disk answer {answer_token!r} not found in "
            f"generate_answer prompt: {prompt_arg!r}"
        )

    # ------------------------------------------------------------------
    # T4
    # ------------------------------------------------------------------

    def test_simulate_mode_ignores_adapters_store(self, tmp_path):
        """Simulate mode reads paths.simulate, ignores stale paths.adapters data.

        Setup:
        - Decoy keyed_pairs under ``paths.adapters/episodic/keyed_pairs.json``
          with ``answer="WRONG-from-adapters"``.
        - Correct keyed_pairs under ``paths.simulate/episodic/keyed_pairs.json``
          with ``answer="CORRECT-from-simulate"``.

        Assertions:
        - ``generate_answer`` prompt contains ``"CORRECT-from-simulate"``.
        - ``generate_answer`` prompt does NOT contain ``"WRONG-from-adapters"``.

        Why this catches a regression: guards locked decision #2 boundary.
        Without this test, a future refactor could make the reader scan both
        stores (e.g. via an rglob over ``paths.data``) and silently produce
        wrong answers when both stores have data.
        """
        config = _simulate_config(tmp_path)

        # Decoy in adapters store — must NOT be read in simulate mode.
        _write_keyed_pairs(
            config.adapter_dir / "episodic" / "keyed_pairs.json",
            [
                {
                    "key": "graph1",
                    "question": "Where does Alex live?",
                    "answer": "WRONG-from-adapters",
                    "source_subject": "Alex",
                }
            ],
        )
        # Real data in simulate store — MUST be read in simulate mode.
        _write_keyed_pairs(
            config.simulate_dir / "episodic" / "keyed_pairs.json",
            [
                {
                    "key": "graph1",
                    "question": "Where does Alex live?",
                    "answer": "CORRECT-from-simulate",
                    "source_subject": "Alex",
                }
            ],
        )

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = _chat_template_tokenizer()

        with patch(
            "paramem.server.inference.generate_answer",
            return_value="Alex lives somewhere nice.",
        ) as mock_ga:
            handle_chat(
                text="Where does Alex live?",
                conversation_id="t4-simulate-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=None,
                ha_client=None,
                speaker_id="spk-test",
            )

        mock_ga.assert_called_once()
        prompt_arg = str(mock_ga.call_args.args[2])
        assert "CORRECT-from-simulate" in prompt_arg, (
            f"Expected 'CORRECT-from-simulate' in prompt, got: {prompt_arg!r}"
        )
        assert "WRONG-from-adapters" not in prompt_arg, (
            f"'WRONG-from-adapters' (from paths.adapters decoy) must not appear "
            f"in simulate-mode prompt, but found it in: {prompt_arg!r}"
        )

    # ------------------------------------------------------------------
    # T4b
    # ------------------------------------------------------------------

    def test_train_mode_ignores_simulate_store(self, tmp_path):
        """Train mode never calls probe_keys_from_disk — simulate store is unused.

        Setup:
        - Config set to ``mode="train"``.
        - Decoy keyed_pairs under ``paths.simulate/episodic/keyed_pairs.json``.
        - ``probe_keys_grouped_by_adapter`` patched to return a successful result
          (simulates a trained adapter with the answer ``"TRAIN-from-adapter"``).

        Assertions:
        - ``probe_keys_from_disk`` is never called.
        - ``generate_answer`` prompt contains ``"TRAIN-from-adapter"`` (the
          in-memory/adapter probe result, not the simulate-store decoy).

        Why this catches a regression: guards the mode branch at
        ``paramem/server/inference.py:547-548``.  Without this test, a future
        refactor could activate the disk-read path in train mode, silently
        returning stale simulate-store data when a fresh adapter is available.
        """
        config = _simulate_config(tmp_path)
        config.consolidation.mode = "train"

        # Decoy in simulate store — must NOT be read in train mode.
        _write_keyed_pairs(
            config.simulate_dir / "episodic" / "keyed_pairs.json",
            [
                {
                    "key": "graph1",
                    "question": "Where does Alex live?",
                    "answer": "DECOY-from-simulate",
                    "source_subject": "Alex",
                }
            ],
        )

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = _chat_template_tokenizer()

        # Simulate a successful adapter probe result for train mode.
        adapter_probe_result = {
            "graph1": {
                "question": "Where does Alex live?",
                "answer": "TRAIN-from-adapter",
                "confidence": 0.95,
            }
        }

        with (
            patch(
                "paramem.training.indexed_memory.probe_keys_from_disk",
            ) as mock_disk_read,
            patch(
                "paramem.server.inference.generate_answer",
                return_value="Alex lives somewhere nice.",
            ) as mock_ga,
            patch(
                "paramem.training.indexed_memory.probe_keys_grouped_by_adapter",
                return_value=adapter_probe_result,
            ),
        ):
            handle_chat(
                text="Where does Alex live?",
                conversation_id="t4b-train-test",
                speaker="Alex",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                sota_agent=None,
                ha_client=None,
                speaker_id="spk-test",
            )

        mock_disk_read.assert_not_called()
        mock_ga.assert_called_once()
        prompt_arg = str(mock_ga.call_args.args[2])
        assert "TRAIN-from-adapter" in prompt_arg, (
            f"Expected 'TRAIN-from-adapter' in generate_answer prompt, got: {prompt_arg!r}"
        )
        assert "DECOY-from-simulate" not in prompt_arg, (
            f"'DECOY-from-simulate' must not appear in train-mode prompt, "
            f"but found it in: {prompt_arg!r}"
        )
