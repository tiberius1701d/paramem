"""End-to-end simulate-mode /chat coverage.

Exercises ``handle_chat`` → ``_probe_and_reason`` → ``probe_keys_from_graph``
in simulate mode using an on-disk simulate store, a stubbed model, and a
stubbed router.  No real LLM is invoked — ``generate_answer`` is patched at
the inference module boundary.

Pattern reference: ``tests/test_abstention_integration.py``.

Six tests are grouped in ``TestSimulateInferenceEndToEnd``:

* T1  — PA match with canonical episodic graph.json in ``paths.simulate``
        returns the graph answer.
* T2  — PA match with no graph.json walks HA → SOTA → fallback.
* T2b — PA match, HA fails + SOTA unavailable → base-model helper reached.
* T3  — All three tiers (episodic, semantic, procedural) read from
        ``<paths.simulate>/<tier>/graph.json``.
* T4  — simulate-mode chat ignores stale keyed_pairs under ``paths.adapters``.
* T4b — train-mode chat never calls ``probe_keys_from_graph``.

Tokenizer setup note
--------------------
In these tests the tokenizer is a ``MagicMock``.  The
``tokenizer.apply_chat_template(messages, ...)`` call inside
``_probe_and_reason`` produces the prompt string that ``generate_answer``
receives as its third positional argument.  To make the prompt a real string
that carries the graph-read facts, every test configures::

    tokenizer.apply_chat_template.side_effect = lambda msgs, **_: json.dumps(msgs)

This causes ``apply_chat_template`` to return a JSON serialisation of the
messages list, which contains the ``augmented_text`` (and therefore the
graph-read answer token) as the user-turn content.  Without this configuration
the call returns a ``MagicMock`` object, not a string, and ``"Heilbronn." in
str(mock_obj)`` is vacuously false.

Simulate mode requires ``indexed_format='quad'`` (enforced by
``ConsolidationScheduleConfig.__post_init__``).  Every test that enables
simulate mode also sets ``indexed_format='quad'`` on the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from paramem.server.config import PathsConfig, load_server_config
from paramem.server.inference import ChatResult, handle_chat
from paramem.server.router import Intent, RoutingPlan, RoutingStep
from paramem.server.simulate_store import _IK_KEY_ATTR, save_simulate_graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_simulate_graph(path: Path, quads: list[dict]) -> None:
    """Write *quads* as a simulate-mode ``graph.json`` at *path*.

    Builds a ``MultiDiGraph`` with one edge per quad, stores the indexed-memory
    key in the ``_IK_KEY_ATTR`` edge attribute (``"ik_key"``), and persists via
    :func:`paramem.server.simulate_store.save_simulate_graph`.

    Args:
        path: Absolute path to the target ``graph.json`` file.
        quads: List of six-field quad dicts with at minimum ``key``,
            ``subject``, ``predicate``, ``object``, ``speaker_id``,
            and ``first_seen_cycle``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for quad in quads:
        graph.add_edge(
            quad["subject"],
            quad["object"],
            **{
                _IK_KEY_ATTR: quad["key"],
                "predicate": quad.get("predicate", ""),
                "speaker_id": quad.get("speaker_id", ""),
                "first_seen_cycle": quad.get("first_seen_cycle", 0),
            },
        )
    save_simulate_graph(graph, path, encrypted=False)


def _full_quad(
    key: str,
    subject: str = "Alex",
    predicate: str = "lives_in",
    object_: str = "Heilbronn",
    *,
    speaker_id: str = "spk-test",
    first_seen_cycle: int = 1,
) -> dict:
    """Build a canonical six-field quad dict for simulate-store fixtures."""
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": object_,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _simulate_config(tmp_path: Path):
    """Return a ``ServerConfig`` with simulate mode and tmp_path stores.

    Loads ``tests/fixtures/server.yaml`` (the CI test fixture), then
    overrides:

    * ``consolidation.mode`` → ``"simulate"``
    * ``consolidation.indexed_format`` → ``"quad"`` (required by the validator)
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
    config.consolidation.indexed_format = "quad"
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
        "model.generate() must not be called — simulate-mode graph recall failed"
    )
    return model


def _chat_template_tokenizer() -> MagicMock:
    """Return a tokenizer stub whose apply_chat_template yields a real string.

    ``apply_chat_template(messages, ...)`` is configured to return a JSON
    serialisation of the messages list.  This makes the prompt argument
    passed to ``generate_answer`` a real string that contains the
    user-turn content (i.e. the graph-read answer token) so that
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

    Simulate mode requires ``indexed_format='quad'``; every test's config
    has both set via ``_simulate_config``.
    """

    # ------------------------------------------------------------------
    # T1
    # ------------------------------------------------------------------

    def test_pa_match_reads_canonical_episodic_from_simulate_store(self, tmp_path):
        """Graph-read answer from paths.simulate reaches the generate_answer prompt.

        Setup:
        - ``paths.simulate/episodic/graph.json`` contains one quad with
          ``object="Heilbronn"`` (fact_text: "Alex lives in Heilbronn").
        - ``paths.adapters`` (paths.data/adapters) is empty.
        - Config is simulate mode + quad format.

        Assertions:
        - ``generate_answer`` called once.
        - Prompt arg contains ``"Alex lives in Heilbronn"`` — proves graph-read
          fact reached the layered context.
        - ``result.text == "Alex lives in Heilbronn."`` (echoed from mock).
        - ``result.escalated is False``.
        - ``"graph1" in result.probed_keys``.

        Why this catches a regression: before the canonicalization, simulate mode
        read from ``paths.adapters``; this test enforces the post-canonicalization
        contract that simulate reads from ``paths.simulate``.
        """
        config = _simulate_config(tmp_path)

        _write_simulate_graph(
            config.simulate_dir / "episodic" / "graph.json",
            [_full_quad("graph1", "Alex", "lives_in", "Heilbronn")],
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
        assert "Alex lives in Heilbronn" in str(prompt_arg), (
            f"Graph-read fact 'Alex lives in Heilbronn' not found in "
            f"generate_answer prompt arg: {prompt_arg!r}"
        )

    # ------------------------------------------------------------------
    # T2
    # ------------------------------------------------------------------

    def test_pa_match_graph_missing_falls_through_to_ha_then_abstains(self, tmp_path):
        """No graph.json → fallback chain HA → abstention.

        Under the intent-keyed dispatch, ``intent=PERSONAL`` (set by the
        ``_pa_router_stub``) blocks SOTA at every internal escalation site.
        When PA graph-recall fails for a personal-class query the chain is:

            HA tool fallback (tools may help even for personal queries)
            ↓ (returns None)
            SOTA blocked by privacy invariant (would have run pre-Phase-2)
            ↓
            abstention canned response (NOT base-model — base model would
            confabulate personal facts; AbstentionBench-grounded short-
            circuit prevents that)

        Setup:
        - ``paths.simulate`` exists but is empty (no graph.json).
        - HA returns ``None``.
        - SOTA mock present but must NOT be invoked.
        - ``_base_model_answer`` patched but must NOT be invoked either —
          abstention short-circuits before reaching it.

        Assertions:
        - HA was tried once (tool fallback allowed for PERSONAL).
        - SOTA was NOT called (privacy invariant for PERSONAL).
        - ``_base_model_answer`` was NOT called (abstention pre-empts it).
        - The returned text is the configured abstention response.
        """
        config = _simulate_config(tmp_path)
        # The privacy invariant tested here ("PERSONAL never reaches
        # SOTA via direct escalation") only holds under cloud_mode=block
        config.sanitization.cloud_mode = "block"
        # Create the simulate_dir but write no graph.json.
        config.simulate_dir.mkdir(parents=True, exist_ok=True)

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = MagicMock()

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
                return_value=ChatResult(text="<base>"),
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

        assert result.text == config.abstention.load_response()
        mock_ha.assert_called_once()
        mock_sota.assert_not_called()
        mock_base.assert_not_called()
        model.generate.assert_not_called()

    # ------------------------------------------------------------------
    # T2b
    # ------------------------------------------------------------------

    def test_pa_match_graph_missing_all_cloud_failed_abstains(self, tmp_path):
        """No graph.json + HA fails + SOTA unavailable → abstention.

        Setup:
        - ``paths.simulate`` exists but empty.
        - ``sota_agent=None`` and HA returns ``None``.
        - ``_base_model_answer`` patched but must NOT fire — abstention
          pre-empts it.

        Assertion:
        - The returned text is the abstention canned response.
        - ``_base_model_answer`` was NOT called.
        """
        config = _simulate_config(tmp_path)
        config.simulate_dir.mkdir(parents=True, exist_ok=True)

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = MagicMock()

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
                return_value=ChatResult(text="<base>"),
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
                sota_agent=None,  # None so SOTA branch is skipped
                ha_client=MagicMock(),
                speaker_id="spk-test",
            )

        assert result.text == config.abstention.load_response()
        mock_base.assert_not_called()

    # ------------------------------------------------------------------
    # T3
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("tier", ["episodic", "semantic", "procedural"])
    def test_canonical_layout_per_tier(self, tmp_path, tier):
        """Per-tier canonical layout: each tier's graph answer reaches generate_answer.

        Parametrized over ``("episodic", "semantic", "procedural")``.  For each
        tier, writes a single quad under
        ``paths.simulate/<tier>/graph.json`` and asserts the quad's object
        appears in the ``generate_answer`` prompt argument.

        Why this catches a regression: validates the canonicalization is
        correct for all three tiers, not just episodic.
        """
        config = _simulate_config(tmp_path)

        answer_token = f"ANSWER-FOR-{tier.upper()}"
        key = f"{tier[:3]}1"
        _write_simulate_graph(
            config.simulate_dir / tier / "graph.json",
            [_full_quad(key, "Alex", "object_is", answer_token)],
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
            f"Tier '{tier}' graph answer {answer_token!r} not found in "
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
        - Correct graph.json under ``paths.simulate/episodic/graph.json``
          with ``object="CORRECT-from-simulate"``.

        Assertions:
        - ``generate_answer`` prompt contains ``"CORRECT-from-simulate"``.
        - ``generate_answer`` prompt does NOT contain ``"WRONG-from-adapters"``.
        """
        config = _simulate_config(tmp_path)

        # Decoy in adapters store — must NOT be read in simulate mode.
        decoy_kp = config.adapter_dir / "episodic" / "keyed_pairs.json"
        decoy_kp.parent.mkdir(parents=True, exist_ok=True)
        decoy_kp.write_text(
            json.dumps([{"key": "graph1", "question": "Q?", "answer": "WRONG-from-adapters"}])
        )
        # Real data in simulate store — MUST be read in simulate mode.
        _write_simulate_graph(
            config.simulate_dir / "episodic" / "graph.json",
            [_full_quad("graph1", "Alex", "object_is", "CORRECT-from-simulate")],
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
        """Train mode never calls probe_keys_from_graph — simulate store is unused.

        Setup:
        - Config set to ``mode="train"``.
        - Decoy graph.json under ``paths.simulate/episodic/graph.json``.
        - ``probe_keys_grouped_by_adapter`` patched to return a successful result
          (simulates a trained adapter with the answer ``"TRAIN-from-adapter"``).

        Assertions:
        - ``probe_keys_from_graph`` is never called.
        - ``generate_answer`` prompt contains ``"TRAIN-from-adapter"`` (the
          in-memory/adapter probe result, not the simulate-store decoy).
        """
        config = _simulate_config(tmp_path)
        config.consolidation.mode = "train"
        config.consolidation.indexed_format = "qa"  # train mode can use qa

        # Decoy in simulate store — must NOT be read in train mode.
        _write_simulate_graph(
            config.simulate_dir / "episodic" / "graph.json",
            [_full_quad("graph1", "Alex", "lives_in", "DECOY-from-simulate")],
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
                "format": "qa",
                "fact_text": "TRAIN-from-adapter",
            }
        }

        with (
            patch(
                "paramem.training.indexed_memory.probe_keys_from_graph",
            ) as mock_graph_read,
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

        mock_graph_read.assert_not_called()
        mock_ga.assert_called_once()
        prompt_arg = str(mock_ga.call_args.args[2])
        assert "TRAIN-from-adapter" in prompt_arg, (
            f"Expected 'TRAIN-from-adapter' in generate_answer prompt, got: {prompt_arg!r}"
        )
        assert "DECOY-from-simulate" not in prompt_arg, (
            f"'DECOY-from-simulate' must not appear in train-mode prompt, "
            f"but found it in: {prompt_arg!r}"
        )

    # ------------------------------------------------------------------
    # T-quad: quad simulate store — de-underscored bullet reaches prompt
    # ------------------------------------------------------------------

    def test_quad_simulate_store_fact_text_in_prompt(self, tmp_path):
        """Quad graph.json in simulate store → de-underscored bullet in prompt.

        This is the A2.1 win (the subject is no longer dropped): the fact
        ``Alex lives_in Heilbronn`` becomes the bullet
        ``"Alex lives in Heilbronn"`` (predicate de-underscored) in the
        context assembled by ``_probe_and_reason``.

        Setup:
        - ``paths.simulate/episodic/graph.json`` written via
          ``_write_simulate_graph`` with one quad entry.
        - Config is simulate mode + quad format.

        Assertions:
        - ``generate_answer`` called once.
        - Prompt contains ``"Alex lives in Heilbronn"`` (de-underscored triple).
        """
        config = _simulate_config(tmp_path)

        _write_simulate_graph(
            config.simulate_dir / "episodic" / "graph.json",
            [_full_quad("graph1", "Alex", "lives_in", "Heilbronn")],
        )

        router = _pa_router_stub("episodic", ["graph1"])
        model = _stub_model()
        tokenizer = _chat_template_tokenizer()

        with patch(
            "paramem.server.inference.generate_answer",
            return_value="Alex lives in Heilbronn.",
        ) as mock_ga:
            result = handle_chat(
                text="Where does Alex live?",
                conversation_id="t-quad-simulate-test",
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

        assert result.escalated is False
        assert "graph1" in result.probed_keys

        mock_ga.assert_called_once()
        prompt_arg = str(mock_ga.call_args.args[2])
        # The de-underscored triple must appear as a bullet in the prompt.
        assert "Alex lives in Heilbronn" in prompt_arg, (
            f"De-underscored quad fact not found in prompt: {prompt_arg!r}"
        )
