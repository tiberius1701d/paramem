# Test fixtures

This directory holds tracked test inputs — config files, hand-curated
fixtures, golden outputs.

## `server.yaml` — canonical config for GPU/integration tests

`tests/fixtures/server.yaml` is the **stable test fixture** that mirrors
`configs/server.yaml.example` option-for-option but pins values for
deterministic test runs:

| Pinned | Why |
|---|---|
| `model: mistral` | Mistral 7B is the validated production model. All contract-test thresholds are calibrated against it. |
| All optional services off (`agents.sota.enabled: false`, `speaker.enabled: false`, `stt.enabled: false`, `tts.enabled: false`, etc.) | Tests run without API keys, model downloads, daemons. |
| Ports shifted by 10000/9000 (`server.port: 18420`, `stt.port: 19300`, `tts.port: 19301`) | A test fixture can coexist with a live server during development. |

### Loading from a test

```python
from paramem.server.config import load_server_config
from paramem.models.loader import load_base_model

cfg = load_server_config("tests/fixtures/server.yaml")
model, tokenizer = load_base_model(cfg.model_config)
```

### Why not load `configs/server.yaml.example` directly?

The shipped operator template is **drift-prone**. As deployment patterns
evolve (new model defaults, different cloud-egress posture, new optional
services), the example shifts. Tests anchored to it would be a perpetual
rebase.

Tests anchor to **this fixture** instead. The fixture's option set tracks
the example via the structural parity test
(`tests/server/test_config_parity.py`); the fixture's *values* are pinned
against drift.

Lint guard `tests/test_test_config_loader_usage.py` forbids tests from
loading `configs/server.yaml.example` outside an explicit allowlist, and
forbids the legacy `paramem.utils.config.load_config()` path entirely.

### Override per-test

When a single test needs a different value, override at construction time
rather than editing this file:

```python
cfg = load_server_config("tests/fixtures/server.yaml")
cfg.consolidation.mode = "simulate"           # this test only
cfg.sanitization.cloud_mode = "anonymize"     # this test only
```

Editing `tests/fixtures/server.yaml` ripples to every test that loads it.

## Other fixtures

* `plausibility_contract.json` — labelled fact set for the plausibility
  contract test (16 facts derived lexically from the 6 DROP rules + 4 KEEP
  positives). Calibrated against Mistral 7B.
* `provenance_gate_failures.json` — assistant-into-graph hallucination
  cases extracted from the PerLTQA probe (10 of 206 = ~5%).
* `longmemeval_oracle_*.json` — LongMemEval evaluation samples.
* `perltqa_probe_sample.json` — PerLTQA dataset probe sample.
* `text_pdf_sample.pdf`, `scanned_pdf_sample.pdf` — document-extraction
  fixtures.
