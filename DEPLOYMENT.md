# ParaMem — Deployment Guide

Operator reference for installing, configuring, and running the ParaMem server.
For the research context, experiment results, and architecture overview see
[README.md](README.md). For encryption, authentication, and threat model see
[SECURITY.md](SECURITY.md).

## Table of Contents

1. [Installation](#installation)
2. [Models & Adapters](#models--adapters)
3. [Server Deployment](#server-deployment)
   - [Quick Start](#quick-start)
   - [Configuration](#configuration)
   - [Routing](#routing)
   - [Consolidation & Crash Safety](#consolidation--crash-safety)
   - [Backup & Migration](#backup--migration)
   - [Encryption & recovery operations](#encryption--recovery-operations)
   - [Per-user token management](#per-user-token-management)
   - [Enabling Web Push](#enabling-web-push)
   - [Troubleshooting](#troubleshooting)
   - [GPU Lifecycle](#gpu-lifecycle)
   - [API](#api)
   - [Home Assistant Integration](#home-assistant-integration)
   - [Voice Pipeline](#voice-pipeline)
4. [Prompt Engineering](#prompt-engineering)
   - [Principles](#principles)
   - [Calibration loop](#calibration-loop)
   - [Editing checklist](#editing-checklist)

---

## Installation

### Prerequisites

- **Python 3.11+**
- **NVIDIA GPU with 8 GB+ VRAM** (tested on RTX 5070). All supported base
  models run with NF4 4-bit quantization; loading without a CUDA-capable GPU
  is not supported.
- **CUDA toolkit via conda** — on WSL2 do not use system CUDA packages;
  install through conda (the `environment.yml` path handles this automatically).
- **RTX 50-series (Blackwell sm_120)?** The stable `bitsandbytes` release
  lacks native sm_120 kernels and will crash on models ≥ 3 B parameters.
  Install the pre-release build before continuing:
  ```bash
  pip install bitsandbytes --upgrade --pre
  # or from source:
  pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git
  ```
  This is a build-infrastructure gap (native sm_120 kernels vs PTX JIT), not a correctness issue; the standard `pip install` works once bitsandbytes 0.50.0 ships.

### Install paths

**Option A — conda (recommended for full GPU stack):**

```bash
git clone https://github.com/tiberius1701d/paramem.git
cd paramem
conda env create -f environment.yml
conda activate paramem
pip install -e ".[voice,dev]"   # add voice stack + dev tools on top of conda base
```

`environment.yml` pins Python 3.11, PyTorch, and CUDA via the pytorch/nvidia
conda channels. The `pip install -e` step adds the server extras not included
in the conda recipe.

**Option B — pip only:**

```bash
git clone https://github.com/tiberius1701d/paramem.git
cd paramem
pip install -e ".[voice,dev]"
```

> **Run from the repo checkout.** ParaMem loads runtime assets — the prompt
> templates (`configs/prompts/`), the graph schema (`configs/schema.yaml`), and
> the config template (`configs/server.yaml.example`) — from the project
> directory, resolved relative to the repository root rather than the process's
> working directory. These are **not** shipped as Python package data, so the
> editable install (`pip install -e`) above is required: it keeps the package
> inside the checkout where the assets live. A non-editable install into
> `site-packages` will not find them, and the server fails fast at startup with
> a clear error if the prompt assets are absent.

### Extras

| Extra | Installs | When to use |
|-------|----------|-------------|
| `anthropic` | `anthropic` SDK | Anthropic cloud provider |
| `google` | `google-genai` SDK | Google cloud provider |
| `all-agents` | Both SDKs | All cloud providers |
| `speaker` | `pyannote-audio>=4.0` | Voice speaker identification |
| `wyoming` | `wyoming>=1.8.0` | Wyoming STT/TTS protocol |
| `stt` | wyoming + faster-whisper | Local Whisper STT |
| `tts` | wyoming + piper-tts + onnxruntime | Local Piper TTS (CPU) |
| `tts-gpu` | wyoming + piper-tts + onnxruntime-gpu | Local Piper TTS (GPU) |
| `kokoro` | wyoming + kokoro | Optional higher-quality TTS (en/fr/es/it/pt/hi/ja/zh) |
| `voice` | stt + tts + speaker | Full local voice stack — recommended for HA deployments |
| `dev` | pytest + ruff | Development and CI tools — recommended for contributors |

Recommended installs:
- Full local voice deployment: `pip install -e ".[voice]"`
- Development / contributing: `pip install -e ".[voice,dev]"`
- Minimal server (cloud agents only, no local voice): `pip install -e "."`

### WSL2 notes

- **Keep all data on the Linux filesystem.** Never store model weights or
  training data under `/mnt/c/` — WSL2 filesystem bridges impose severe I/O
  overhead that degrades training throughput.
- **HF_DEACTIVATE_ASYNC_LOAD workaround.** Transformers 5.3+ uses a
  `ThreadPoolExecutor` for parallel weight loading. On some WSL2 configurations
  this races the dxg memory mapper. If you encounter `CUDA driver error: device
  not ready` on cold model loads, add to `.env`:
  ```bash
  HF_DEACTIVATE_ASYNC_LOAD=1
  ```
  As of NVIDIA driver 596.36 + Windows 11 KB5088467 (2026-04), this race no
  longer reproduced on our test host — try without it first, and only enable if
  you see the error.
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `.env` to reduce
  CUDA allocator fragmentation under QLoRA training.
- **Modern Standby (laptop GPUs).** Windows Modern Standby can power-cycle the
  GPU during idle, causing a TDR BSOD (bugcheck 0x116) if a CUDA workload is
  active. The `acquire_gpu()` context manager (`experiments/utils/gpu_guard.py`)
  holds `ES_CONTINUOUS | ES_SYSTEM_REQUIRED` via a background process for the
  duration of GPU work — automatic for any experiment that uses it. A cooling
  pad helps thermal recovery between runs (TGP is the binding constraint under
  sustained load).

### One-time assets

- **Language-ID model** (only if `text_lang_detection.enabled: true` in
  `server.yaml`): downloads a 126 MB fastText model to
  `~/.cache/paramem/lang_id/lid.176.bin`.
  ```bash
  bash scripts/setup/download-langid-model.sh
  ```
- **HF model cache.** Base models are downloaded from HuggingFace on first
  run and cached in `~/.cache/huggingface/`. No manual download step needed
  for public models. For gated models (e.g. Llama, Gemma) set `HF_TOKEN`
  in `.env` and accept the model licence on the HuggingFace website first.

### Config bootstrap

```bash
# Copy the tracked example to your local override (gitignored)
cp configs/server.yaml.example configs/server.yaml

# Create your .env from the template
cp .env.example .env
# Edit .env and fill in required values (see comments in the file)
```

`configs/server.yaml` is gitignored. The server falls back to
`configs/server.yaml.example` when no local override exists, so the copy is
only needed when you want to diverge from the ship-safe defaults.

### Verification

```bash
# Black-box post-install check — drives the running server over REST
python examples/quick_start.py
```

`quick_start.py` injects facts via `POST /chat`, runs the real pipeline via
`POST /consolidate/interim` (extraction → indexed-key training → recall, all per
`server.yaml`), then asserts recall via `POST /debug/probe`, exiting non-zero
on failure. Prerequisites: server running, `debug: true` in the active
`server.yaml`, and `PARAMEM_API_TOKEN` set (env or `.env`).

### systemd service

A systemd user service file is provided at
`scripts/server/paramem-server.service`. To enable the server to start with
your user session (or at boot — see `headless_boot` in `server.yaml`):

```bash
# Install and start
systemctl --user enable --now paramem-server

# Status
systemctl --user status paramem-server
journalctl --user -u paramem-server -f
```

---

## Models & Adapters

### Switching models (YAML-only)

Set `model:` in `configs/server.yaml` to one of the eight registry keys below.
No code change is needed — the loader (`paramem/models/loader.py`) reads the
registry entry and handles quantization, device mapping, and chat-template
detection automatically.

| Key | HF model id | VRAM notes |
|-----|------------|------------|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | nf4, no cpu_offload — **deployment default; current live-server model** |
| `gemma` | `google/gemma-2-9b-it` | nf4, `cpu_offload=True`, `max_memory: {GPU: 7 GiB, CPU: 20 GiB}` — requires `llm_int8_enable_fp32_cpu_offload` |
| `qwen3b` | `Qwen/Qwen2.5-3B-Instruct` | nf4, no cpu_offload — smallest model, fastest loads |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` | nf4, no cpu_offload |
| `ministral` | `mistralai/Ministral-8B-Instruct-2410` | nf4, no cpu_offload |
| `llama` | `meta-llama/Llama-3.1-8B-Instruct` | nf4, no cpu_offload — gated model, requires `HF_TOKEN` |
| `gemma4` | `principled-intelligence/gemma-4-E4B-it-text-only` | nf4, no cpu_offload |
| `qwen3-4b` | `Qwen/Qwen3-4B-Instruct-2507` | nf4, no cpu_offload |

All entries verified against `MODEL_REGISTRY` in
`paramem/server/config.py` lines 52–111.

**8 GB VRAM implication.** Models with a working set larger than ~7 GB must
use `cpu_offload=True` with explicit `max_memory` limits (`gemma` is the only
such entry in the current registry). Models without offload fit on the 8 GB
device in NF4 with room for the adapter stack, KV cache, and STT/TTS residency.
Do not add a model with a >7 GB working set without setting `cpu_offload=True`
and verifying the startup VRAM topology check passes.

### Adding a new model

Add one `ModelConfig` entry to `MODEL_REGISTRY` in
`paramem/server/config.py`. The required fields:

```python
"my-model": ModelConfig(
    model_id="org/Model-Name",       # HuggingFace model id
    quantization="nf4",              # always nf4 for 8 GB devices
    compute_dtype="bfloat16",        # bfloat16 is the validated default
    trust_remote_code=True,          # required for most community models
    cpu_offload=False,               # set True + max_memory_* for >7 GB models
    # max_memory_gpu="7GiB",         # only when cpu_offload=True
    # max_memory_cpu="20GiB",        # only when cpu_offload=True
),
```

**Chat-template and system-role detection is automatic.** The loader calls
`tokenizer.apply_chat_template()` and introspects the tokenizer's chat
template for system-role support — no per-model code change is needed.

**`target_modules` override.** The adapter stack defaults to the attention
projection layers `["q_proj", "v_proj", "k_proj", "o_proj"]` (attention-only
routing). This covers the standard Llama / Mistral / Qwen attention naming. If
a model uses non-standard layer names (e.g. `["query_key_value"]` for some
Falcon variants), override `target_modules` in the `adapters:` block of
`server.yaml` for the affected adapter tier:

```yaml
adapters:
  episodic:
    target_modules: ["query_key_value"]   # model-specific override
```

The procedural adapter extends this to MLP layers for representational
imprinting of behavioral patterns:
```
["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Tuning LoRA adapters

The `adapters:` block in `server.yaml` controls each tier independently:

```yaml
adapters:
  episodic:
    enabled: true
    rank: 8
    alpha: 16                  # rule: alpha = 2 × rank
    learning_rate: 1.0e-4
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  semantic:
    enabled: true
    rank: 8
    alpha: 16
    learning_rate: 1.0e-5      # lower LR: promoted, well-reinforced knowledge
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  procedural:
    enabled: true
    rank: 8
    alpha: 16
    learning_rate: 5.0e-5
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
```

**Invariants:**
- `alpha = 2 × rank` — validated across the Test 1–8 campaign. Deviating
  from this ratio degrades indexed-key recall.
- **Minimum 30 epochs for indexed keys.** The validated training budget from
  Test 1–8 for large-N folds (`loss convergence ≠ fact encoding` — loss
  plateaus at ~15 epochs; 30 epochs are required for 100% indexed-key
  recall). The training funnel derives this automatically per fold from the
  key-triple count (`paramem.utils.config._BUDGET_TABLE`: N≥128 → 30 epochs;
  16–127 → 50; <16 → 80 — smaller folds get a LARGER derived budget, not a
  smaller one), unconditionally and unclamped — there is no operator
  ceiling. `hit_cap` telemetry and recall-based early stopping are the
  wall-time feedback channels: early stopping usually terminates training
  before the bucket cap is reached, and `hit_cap` telemetry records when it
  does not.
- **Procedural is disabled by default** in new deployments that haven't
  yet collected behavioral data. Enable once the episodic/semantic tiers
  are stable.
- Per-tier learning rates from the example config: episodic 1e-4, procedural
  5e-5, semantic 1e-5 (slowest — consolidated knowledge).

---

## Server Deployment

ParaMem includes a REST server for persistent deployment. The server keeps the model loaded in VRAM, serves chat inference, runs scheduled consolidation, and escalates non-memory queries to Home Assistant's conversation agent.

### Quick Start

```bash
# Start the server
bash scripts/server/start-server.sh

# Or with systemd (recommended)
systemctl --user enable --now paramem-server

# Verify
curl http://localhost:8420/status
```

The server listens on port 8420. On startup it auto-detects GPU availability — if another process holds the GPU (e.g., a training run), it starts in cloud-only mode and auto-reclaims once the GPU is free.

Set `headless_boot: true` in `configs/server.yaml` to have the server come up before any interactive login. On every start, `scripts/setup/headless-boot.sh` reconciles OS-level state with the flag: it enables/disables systemd user linger, and on WSL hosts registers/removes a Windows scheduled task (`ParaMem-Start-WSL-Boot`) that launches the WSL VM at system startup. The reconciler is idempotent and non-fatal — if elevation is unavailable it WARNs with the exact manual command. When invoked without a TTY (systemd path), it pops a WSL console window so sudo can be approved interactively.

### Configuration

The shipped template is `configs/server.yaml.example` (tracked, disabled-by-default
for optional services). Operators copy it to `configs/server.yaml` (gitignored)
to add local overrides such as API keys or enabled services. The server falls
back to the template on a fresh checkout, so `cp configs/server.yaml.example
configs/server.yaml` is only required when you actually want to diverge from the
ship-safe defaults.

`configs/server.yaml.example` is fully commented — every option has inline docs
explaining its effect, privacy implications, and interaction with other
options. A short map of the top-level sections:

| Section | Purpose |
|---------|---------|
| `cloud_only` | Opt-out of local PM — route every query to the configured cloud agent. Security-critical. |
| `headless_boot` | Auto-start the server before any interactive login. Reconciles systemd linger + (WSL) a Windows startup task on every start via `scripts/setup/headless-boot.sh`. |
| `server` | Host, port, auto-reclaim polling, restart policy. |
| `vram` | Per-process cap fraction (`process_cap_fraction`); KV cache + activation headroom (`vram_cache_headroom_gib`, code default 1.0 GiB, shipped yaml 1.5 GiB). |
| `model` | Base model (`mistral`, `gemma`, `qwen3b`, `gemma4`). |
| `debug` | Diagnostics mode — `true` FORCES retention of consolidated session transcripts regardless of `consolidation.retain_sessions`, and additionally writes per-cycle debug snapshot artifacts. `false` does NOT disable transcript retention: `consolidation.retain_sessions` governs that independently (default `true`; `false` deletes transcripts after consolidation instead of archiving them). Session snapshots still write either way (envelope-encrypted under Security-ON, plaintext under Security-OFF) so mid-turn state survives graceful restarts. |
| `paths` | Data, sessions, debug, prompts directories. |
| `adapters` | Per-adapter `enabled` / `rank` / `alpha` / `learning_rate` / `target_modules`. |
| `inference` | Serving-path options. `preload_cache` (default `true`): keep the recalled-fact cache warm at boot so a query costs one generate instead of one probe per key. `max_response_tokens` (default `512`): ceiling on how long a single locally-generated reply may get. Sized so a worst-case reply still returns inside the 30s Home Assistant conversation timeout on the measured decode rate; a reply that reaches the ceiling is cut back to its last complete sentence rather than delivered mid-word. Raise only if your HA client timeout is raised to match and you want longer summaries; lower to tighten worst-case turn latency. |
| `session` | Per-conversation session lifecycle. `idle_timeout_minutes` (default `10`): the gap since a conversation's last turn after which its next turn opens a fresh session; a conversation also opens a fresh session once it grows past what one consolidation pass can take in. |
| `consolidation` | **`refresh_cadence` is the only scheduling knob** (default `"12h"`). Full-cycle period is derived: `refresh_cadence × max_interim_count` (default 12h × 7 = 84h). `retain_sessions` (default `true`) governs whether a consolidated session's transcript is archived under the retention root or deleted — independent of `debug` (see the `debug` row above). Also gates the extraction pipeline stages (plausibility, anonymization) and the thermal-throttle quiet-hours policy (`quiet_hours_mode` = `always_on`/`always_off`/`auto` with `start`/`end`). The two full-fold-only graph-refinement passes (`refinement_enrichment`, `refinement_normalization`) and their sizing knobs are covered separately below under [Graph refinement](#graph-refinement). |
| `cloud` | **The single master switch for all cloud egress.** `cloud.enabled` (default `false`) gates every site alike — the conversation agent, per-session extraction enrichment, graph-tier enrichment and `/calibrate/enrich`. Necessary but not sufficient: each call is admitted by `evaluate_cloud_egress`, which also requires a supported provider, a model, a resolvable API key and (for OpenAI-compatible providers) an endpoint. With no provider and no key there is no cloud mode. `cloud.allow_degraded_serving` (default `false`) decides whether the cloud leg stays open when the local model becomes unavailable for a reason the operator did not choose (GPU held elsewhere, insufficient VRAM, a failed reload/apply, a persistent CUDA fault); the HA leg stays open either way, and a degraded conversation is announced once. Graph-tier enrichment (full-fold only, `consolidation.refinement_enrichment`) is one of the sites this switch gates — see [Graph refinement](#graph-refinement) below. |
| `agents` | Cloud provider, model and credentials (`agents.cloud` + `agents.cloud_providers`) — these carry **no on-off of their own**; use `cloud.enabled`. Also `agents.ha_agent_id`, which **must point at a LOCAL HA conversation agent** — a cloud-backed one turns the hop ParaMem treats as on-premise into unsanitized cloud egress. See `SECURITY.md §3`. |
| `tools.ha` | HA URL, token, language filter, entity allowlist, tool timeout. |
| `sanitization` | Cloud-egress policy. `cloud_mode` (`block` / `anonymize` / `both`, default `block`) decides what may leave: `block` drops personal queries and sends the rest verbatim, `anonymize` placeholders outbound text via the local LLM and de-anonymizes the reply, `both` does both. `scrub` is the PII vocabulary the anonymizer works from. A query counts as personal when either the intent classifier or the known-entity / first-person detector says so — one verdict, computed once. The first-person check is encoder-based and multilingual when the encoder is loaded; falls back to an English token-set. See `personal_referent` below. |
| `intent` | Intent classifier — HA fast-path + content-driven residual. `intent.mode: llm` (default) uses the loaded local LLM with a focused classifier prompt; robust to paraphrase and novel phrasings, no exemplar maintenance. `intent.mode: embeddings` uses the multilingual sentence encoder (`intfloat/multilingual-e5-small`) vs per-class exemplar bank under `configs/intents/<class>.<lang>.txt`; cheaper per query but brittle on shapes the operator hasn't anticipated. In local mode the encoder is loaded regardless of `intent.mode` and reused by `sentence_type` and `personal_referent`, and `llm` mode auto-falls back to `embeddings` when no local model is registered. In **cloud-only** mode none of these encoders is loaded at all, so `classify_intent` returns `UNKNOWN` — cloud-only mode holds no reachable stored knowledge and applies no egress policy, by design (`architecture.md` AD-21). |
| `sentence_type` | Encoder-based interrogative-vs-non-interrogative classifier with exemplars under `configs/sentence_types/<class>.<lang>.txt`. Adding a language is one new file pair, no code change. Falls back to terminal-punctuation + English first-word lexicon when the encoder isn't available. |
| `personal_referent` | Encoder-based about-speaker-vs-not-about-speaker classifier with exemplars under `configs/personal_referent/<class>.<lang>.txt`. Closes the multilingual hole in the sanitizer: German / Mandarin / Spanish / etc. self-referential queries are blocked at the cloud-egress gate even though the legacy English token-set wouldn't match. Falls back to that token-set when the encoder isn't available. |
| `abstention` | Deterministic canned-response guard against confabulation on a personal interrogative parametric memory cannot answer. `enabled` (default `true`). Three messages, each `*_file` (default under `configs/prompts/`) with a `*_override` that pins the text inline instead: `response_file` (a known speaker, coverage gap — this query missed their facts), `cold_start_response_file` (a known speaker with no facts yet), `no_identity_response_file` (no speaker resolved at all — fired on the relay path; NOT gated on `enabled`, since refusing there is a structural impossibility — no identity, no store — not a feature toggle). |
| `text_lang_detection` | fastText `lid.176` detector for the text-only `/chat` path. STT carries Whisper's language signal on audio; pure-text requests had no equivalent and fell through to English regardless of input language. Eager-loaded at server startup when `enabled` is true (CPU-only, ~126 MB resident, zero VRAM cost). One-time setup: `bash scripts/setup/download-langid-model.sh`. Disabled by default so deployments without the model file do not warn. |
| `mobile_pwa` | Progressive Web App configuration. `enabled` (default `false`): serve the static PWA shell at `/app` and activate per-user cookie/bearer-token auth (see `SECURITY.md §5`). `static_dir` (default: bundled `paramem/web/static`): filesystem path to the compiled static bundle. `cookie_name` (default: `paramem_token`): name of the cookie the middleware will accept if the client presents one; the server does not issue this cookie — tokens are carried via the `Authorization: Bearer` header in practice. `push_enabled` (default `false`): enable Web Push lock-screen notifications — set to `true` together with `enabled` to activate the `/push/subscribe` endpoint; the VAPID keypair is auto-generated and persisted (see `SECURITY.md §5`). `vapid_contact` (default `mailto:admin@localhost`): operator contact URI in the VAPID JWT; set to your own `mailto:` address. |
| `voice` | Voice prompt file, per-speaker greeting cadence, per-language greeting text (`voice.greetings`). |
| `speaker` | pyannote thresholds, enrollment flow, embedding caps. |
| `stt`, `tts` | Whisper model + Wyoming port; Piper/MMS voices per language. |

#### Memory tiers

Two knobs under `consolidation:` govern how a fact moves between the episodic
and semantic adapters over its lifetime. Neither ever deletes anything.

| Parameter | Default | Effect | When to adjust |
|---|---|---|---|
| `promotion_threshold` | `3` | How much standing a fact needs before it moves from the episodic adapter to the semantic one. Standing comes from being said again in a *later* conversation — repetition inside one conversation does not count — and a fact also keeps the standing of any duplicate merged into it, so consolidating two records of the same fact never costs it its place. | Raise to keep the semantic tier smaller and more selective, so only facts confirmed across several conversations settle there. Lower to promote sooner, at the cost of promoting things that turned out to be passing remarks. |
| `decay_window` | `10` | How many consolidation cycles a fact may go unmentioned before it is logged as a decay candidate. Advisory only — nothing is deleted, and the fact stays recallable; unimportant facts fade on their own as the adapter is retrained around them. | Lower to see fading candidates sooner in the logs; raise to quieten them. Purely diagnostic — changing it does not change what the server keeps. |

#### Graph refinement

Two post-merge passes over the consolidation fold's cumulative graph, both under `consolidation:`. Both are **full-fold only** — an interim cycle never runs either, regardless of these settings; interim cycles still get session-tier cloud enrichment over each transcript (see `cloud` above).

| Parameter | Default | Effect | When to adjust |
|---|---|---|---|
| `refinement_enrichment` | `off` | Cross-session second-order relations + `same_as` entity-coreference discovery via a cloud call over the merged graph. Requires `cloud.enabled: true` to fire at all. | Enable when interim-cycle session-tier enrichment isn't surfacing cross-session inferences you expect; cost is cloud egress of graph content (see `SECURITY.md`). |
| `refinement_normalization` | `on` | Collapses synonym predicates (e.g. "likes"/"enjoys") sharing a subject/object pair. Runs **after** enrichment so a cloud-coined predicate synonym is collapsed before the fold mints keys from the graph. | Disable only if a deployment relies on predicate surface being preserved verbatim; the default is safe to leave on. |
| `graph_enrichment_neighborhood_hops` | `2` | Ego-graph radius the enrichment pass chunks around each focal entity. | Raise for more cross-entity context per cloud call (higher cost); lower to reduce cost/latency on a smaller deployment. |
| `graph_enrichment_max_entities_per_pass` | `50` | Bounds how many high-recurrence entities — and therefore how many cloud calls — one enrichment pass considers. | Lower to bound cost/latency on a large graph; raise for more thorough cross-session coverage per fold. |

Config loading is strict: an unknown key anywhere in `configs/server.yaml`
raises `TypeError` at boot rather than being silently ignored. This means a
config field removed in a later release (e.g. the `extraction_verify_anonymization`
/ `extraction_ner_check` / `extraction_ner_model` consolidation knobs, retired
in favour of the unified anonymization table — spaCy NER is gone from the
pipeline entirely, not an optional extra) will fail to boot an existing
deployment's `configs/server.yaml` until the stale keys are deleted from it.
When upgrading, diff your local `configs/server.yaml` against the current
`configs/server.yaml.example` and remove any key the template no longer
documents.

The `process.restart` block controls the systemd restart policy baked into
`~/.config/systemd/user/paramem-server.service.d/restart.conf` on each server
start. Key knobs: `on_failure` (retry on crash vs. never), `max_attempts` /
`window_seconds` (rate-limit gate), and `permanent_failure_exit_codes` (exit
codes that are never retried — defaults to `[3]`, the `FatalConfigError` code
raised by the encryption consistency gate). See `configs/server.yaml.example`
for the full field reference.

Operational invariant: consolidation has exactly one user-facing scheduling
knob (`consolidation.refresh_cadence`). Everything else derives from it.
Scheduling is owned by a systemd user timer (`paramem-consolidate.timer`)
with `Persistent=true`. A tick missed while suspended, powered off, or while
the server itself was still starting is caught up once the server finishes
starting — for every cadence form (daily, weekly, `HH:MM`, and interval
cadences alike), not only uneven intervals. A duplicate or repeated tick
inside the same schedule window is a no-op; the very first scheduled tick on
a fresh deployment establishes the schedule without folding anything.

### Routing

```
Voice Satellite → Wyoming STT (Whisper + pyannote) → HA → ParaMem /chat
  ├─ Speaker match (centroid) → attach identity
  ├─ Entity match in knowledge graph? → adapter recall → reason → respond
  ├─ HA entity match? → HA conversation agent (tools, device control)
  └─ Neither → cloud agent (reasoning, search)
Response → HA TTS → Sonos (announce)
```

ParaMem owns memory (speaker identification, entity routing, adapter recall, consolidation). Home Assistant owns everything else (device control, search, weather, music, prompt engineering, model selection). Non-memory queries are forwarded to HA's configured conversation agent, which handles tool execution, entity resolution, and room-aware context internally.

### Consolidation & Crash Safety

- **Two adapter tiers:** committed main adapters (`episodic` / `semantic` / `procedural`) plus short-lived **interim adapters** minted at each `refresh_cadence` tick. Interim adapters absorb new facts so recall works inside a refresh window without waiting for the full cycle. They accumulate up to `max_interim_count` (default 7), capped by VRAM.
- **Bounded ring overflow (`interim_overflow_slack`, default 0):** when the interim ring is full (`max_interim_count` slots) up to `interim_overflow_slack` additional later-stamped overflow slots may be minted — each its own adapter, preserving temporal order — instead of immediately keeping sessions pending; the slack is included in the boot-time VRAM budget so the extension is proven to fit before the server starts. When the ring *and* its overflow are exhausted, new sessions are kept pending (lossless), not dropped. Two incidents surface the state: `interim_cap_reached` (warning) when an overflow slot was minted, `interim_overflow_pending` (failed) when capacity is fully exhausted and sessions are held pending. Both auto-resolve on the next successful full fold, which purges the ring — or on an operator-invoked `POST /interim/discard` (see [Discarding the interim ring](#discarding-the-interim-ring) below), which purges it without folding.
- **Full-fold-only mode (`max_interim_count: 0`):** no interim adapters are minted; every cycle's sessions stay pending in the session buffer and the scheduled full fold extracts and trains them directly into the main tiers. The full cycle runs every `refresh_cadence` itself — the derived period is `refresh_cadence`, not `refresh_cadence × 0`. A non-empty `refresh_cadence` is required: the server refuses to start without one, because at count 0 the full fold is the only training venue and pending sessions would otherwise accumulate unboundedly. It also **requires `consolidation.mode: train`** — the pairing `max_interim_count: 0` + `mode: simulate` is rejected at config load, because in simulate mode the full fold does not consume pending sessions and there is no interim venue either, so ingestion would stall silently.
- **`consolidation.mode` is a closed vocabulary:** `train` (persist to LoRA weights) or `simulate` (persist `graph.json` per tier). Any other value is rejected at config load rather than being silently treated as `simulate`.
- **Nothing new → no cycle:** a full or interim consolidation is skipped when its own action has nothing to consume, in both `train` and `simulate` mode and whether the cycle was reached by the schedule or requested directly. For a full cycle that means no payload-bearing interim slot on disk (checked regardless of the current `max_interim_count`, so a slot minted before the count was lowered to 0 still counts) and, only at `max_interim_count: 0`, no pending NAMED session either; for an interim cycle it means no pending NAMED session. `POST /consolidate` and `POST /consolidate/interim` are held to the identical check their scheduled counterpart would apply — a deliberate request drops only the TIME condition (is a cycle due right now), never the CONTENT condition — so the response is `noop_*` rather than an empty GPU cycle. `POST /reconsolidate` is the one exception — its input is the main tiers' own stored keys, which always exist, so it always runs.
- **Two session-rotation triggers, one visible to serving:** a conversation opens a fresh session after an idle gap (`session.idle_timeout_minutes`) or once its accumulated size would exceed what one consolidation pass can take in. An idle rotation starts a fresh conversation context; a size rotation is invisible to what gets served — the conversation's context still follows it across the rotated sessions. Document-ingest chunk sessions never rotate.
- **Enrichment-incident arbitration:** a failed local-anonymization pass and a degraded cloud-enrichment pass surface as distinct `enrichment_degraded` incidents, so an operator watching `/status` can tell which stage needs attention. A clean or opted-out anonymization pass resolves its own incident and lets the enrichment outcome govern the other; with cloud egress disabled entirely, any still-open incident of either kind resolves the next time a session is processed, recording a reason that distinguishes a clean run from cloud being off.

**Triggering consolidation.** Four endpoints, none of which takes a request body; each returns **HTTP 200** with a `status` and an `action` field:

| Endpoint | What it does |
|---|---|
| `POST /consolidate` | Collapse the interim slots into main memory now — the same content check the schedule uses to decide a full cycle is due, minus the deadline math: any payload-bearing interim slot on disk (or, at `max_interim_count: 0`, a pending NAMED session) is enough, whatever the schedule would say. |
| `POST /consolidate/interim` | Absorb recent conversations into memory now. Main memory is untouched. |
| `POST /reconsolidate` | Rebuild main memory from **its own** stored knowledge. The interim slots are neither folded in nor reaped and the pending conversations stay pending, so nothing is lost by running it. Use after changing the base model, the extraction prompts, or the extraction config. |
| `POST /scheduled-tick` | The systemd user-timer entrypoint — one of two doors (the other is the boot-completion catch-up, run in-process after server startup) that let the schedule's deadline math (`_is_full_cycle_due`) decide between a full fold and an interim absorb, and carry the catch-up gate and the cadence stamp. Skipped when there is nothing new, the same way `POST /consolidate` and `POST /consolidate/interim` are. |

Typical operator flow: `POST /consolidate/interim` → poll `GET /status` until `consolidating` is false → `POST /consolidate` when the interim slots should be folded into the mains, or `POST /reconsolidate` when the mains should be rebuilt from what they already hold.

**A refusal is not an error.** These endpoints are non-blocking: they submit the run and return immediately. When the server is busy (another fold running, someone chatting, the GPU held, cloud-only mode) the response is still **200** with `status: "deferred_*"`; when there is nothing to do it is **200** with `status: "noop_*"`. `curl --fail` therefore does **not** exit non-zero on a deferral — read `status`. The only 4xx from this family is **409 `trial_active`** while a migration TRIAL is in progress.

- **Atomic full-cycle finalize:** at the full-consolidation boundary, all interim adapters are rebuilt into the mains via replay on `all_active_keys ∪ all_interim_keys` (facts are regenerated from the merged graph each cycle, not loaded from a stored file), recall-sanity-checked, and purged. On sanity-check failure the cycle rolls back to the pre-finalize snapshot — mains and interim state are preserved.
- **Staging slot:** a reserved `in_training` adapter slot isolates inference from model reload during consolidation — `/chat` never blocks on training.
- **Epoch-level resume:** `BackgroundTrainer` writes `staging_resume.json` + keeps the two most recent HF Trainer checkpoints in `bg_checkpoint_epoch/` at each epoch boundary. A crash mid-cycle resumes at the last completed epoch after SHA-256 fingerprint validation of the regenerated training dataset + training config. Stale state is discarded.
- **Systemd user timer:** `paramem-consolidate.timer` drives scheduling with `Persistent=true`. A trigger missed while the laptop is suspended, powered off, or still starting up is caught up once the server has finished starting, for every cadence form. A catch-up that would fall during an active migration TRIAL defers instead of running and is retried at the next real tick; the timer's own live tick (`POST /scheduled-tick`) still returns 409 `trial_active` as usual while a TRIAL is in progress.
- **VRAM topology check + live gate:** `paramem/server/vram_validator.py` reads cache-derived predictions from `paramem/server/vram_predict.py` (HF cache size × quant factor) to assess whether base model + main adapters + `max_interim_count` + staging slot + STT + TTS + KV cache headroom fits the device pre-load. On cache miss the assessment is skipped; the live gate (`vram_guard.vram_measure` records `mem_get_info` deltas around each load + `enforce_post_load_budget` post-load) is authoritative and `sys.exit(1)`s on overrun rather than OOM mid-request.
- **Extraction anonymize budget (`consolidation.extraction_anonymize_token_envelope`, default `8192`):** ceiling on how much work one local-model anonymize call inside the extraction pipeline (session-tier extraction, graph-tier enrichment, chat egress) may take on before the call is split into more than one. Sized to match the `vram` section's KV-cache headroom (`vram_cache_headroom_gib`) above — the shipped defaults for the two keys are matched to each other. This is a ceiling, not a guarantee: each call is additionally clamped to live free VRAM at call time, so an oversized call splitting further under a tighter-than-usual free-VRAM moment is expected behavior, not a bug. Raise it only alongside a matching increase in `vram.vram_cache_headroom_gib` on a host with more free VRAM to spare; there is little reason to lower it below the shipped default.
- **Word-to-token estimate ratio (`consolidation.extraction_token_estimate_ratio`, default `3.7`):** fallback ratio used to size a payload only when no live tokenizer is available (in practice, only the CLI document ingester takes this path). It must match the value the codebase measures against internally — config load raises an error and the server refuses to start if the two disagree, so the key can never silently stop governing anything. After swapping the base model, re-measure this ratio against the new tokenizer and update both together; the server separately re-checks the live tokenizer's own ratio at boot and raises a `/status` attention item if it has drifted past the configured value, so a stale ratio doesn't go unnoticed between deliberate updates.

#### Discarding the interim ring

`POST /interim/discard` is a separate admin door, outside the four consolidation endpoints above: it takes a request body, runs synchronously, and can refuse with **409**. It removes the entire interim ring — every not-yet-folded interim slot — without folding its content into main memory first. The facts held only in those slots are the only copy; once this call returns `"discarded"` they are gone. Main memory, pending conversations, and a backup bundle captured before the call are all untouched — restoring that bundle brings the discarded slots back.

Call it with `{"confirm": true}` to actually discard. Without `confirm`, it returns the blast radius (per-slot key counts) and mutates nothing. It refuses with **409** while a fold, background training, or a migration TRIAL is in progress, or while the server is in cloud-only mode (reacquire the GPU with `POST /gpu/acquire` first, then discard).

### Backup & Migration

The `paramem` management CLI talks to the running server over HTTP (default `http://127.0.0.1:8420`; override per-command with `--server-url`). Exit codes: `0` success, `1` HTTP error, `2` server unreachable.

**Backups are self-contained.** Each backup is a single timestamped *bundle* under `data/ha/backups/snapshot/<ts>/` holding everything needed to restore the system's recall:

- `server.yaml` (config)
- the key registry (`key_metadata.json`) and, per adapter tier, its `indexed_key_registry.json` (SimHash fingerprints live inside this file, not a separate one) — **without the registries the weights are useless** (you can't enumerate or verify recalled facts)
- each enabled adapter's live slot — `adapter_model.safetensors` + `adapter_config.json` + `meta.json` — resolved the same way the server mounts it (finalized main slot, or the live interim slot when no full cycle has run yet)
- `speaker_profiles.json` (voice enrollment)
- a top-level `bundle.meta.json` with the file inventory, per-adapter registry hashes, and the base-model identity
- `server.yaml.candidate` *(present only in pre-base-swap snapshots)* — the candidate config preserved for a later retry; hash-indexed in the manifest but never restored automatically

The transient knowledge graph is **not** included (it lives only in the running loop and is rebuilt each cycle — knowledge lives in the weights), nor is regenerable training scaffolding (checkpoints, in-training slots).

**Encryption is byte-faithful.** A bundle preserves each file's on-disk encryption state: under Security ON the sensitive artifacts (weights, registries, speaker profiles) stay age-encrypted and the operational carve-outs (`server.yaml`, `meta.json`) stay plaintext; under Security OFF everything is plaintext. Restore reproduces that exact posture, so the server boots cleanly in either mode (validated end-to-end: backup → restore → server start → adapters mounted → recall). See [`SECURITY.md`](SECURITY.md) for the encryption model.

**Server-mediated.** Capturing the live adapter set requires the daily key (to resolve which slot is live and read the registries), so backups run through the running server. The scheduled systemd timer (`paramem-backup.timer`) and the CLI both reach it; if the server is unreachable when the timer fires, the run is recorded as skipped rather than producing an incomplete backup.

**Scheduled backups catch up too.** A scheduled backup missed while the host was suspended, powered off, or the server was still starting is run once the server finishes starting, for every `backups.schedule` cadence form; a redundant or repeated wakeup inside the same window is a no-op. The very first scheduled backup on a fresh deployment runs immediately rather than waiting for the next window. A `security.backups.schedule` change promoted through `paramem migrate` reaches the backup timer immediately, the same way a `consolidation.refresh_cadence` change does — no server restart required.

```bash
# Take a self-contained backup now (via the running server)
paramem backup-create --label pre-upgrade

# List backups (newest first)
paramem backup-list

# Apply the retention policy (preview, then commit)
paramem backup-prune --dry-run
paramem backup-prune

# Restore a bundle (atomic; a server restart applies it). Add --restore-config
# to also overwrite server.yaml (off by default — a restore won't change your config).
paramem backup-restore 20260521-07385752
paramem backup-restore 20260521-07385752 --restore-config
```

Restore verifies every file's hash and decryptability **before** touching the live store, writes a safety bundle of the current state, then swaps the recovery set into place atomically with the registry written **last** (a crash leaves the old set live — never a half-restored one). It refuses while a migration trial or background-training run is active, and returns `restart_required` — the restored adapters mount on the next server start. Because the bundle is self-contained, it can also be copied off-host for disaster recovery (off-host replication itself is out of scope — see Non-goals).

Configuration under `security.backups` in `server.yaml`:

```yaml
security:
  backups:
    schedule: "daily 04:00"     # "off" disables scheduled backups
    adapter_scope: live         # "live" = main + live interim slots; "main" = finalized mains only
    max_total_disk_gb: 20       # global cap (must be > 0); oldest slots pruned first, writes refused at/over it
    retention:
      daily:   { keep: 7 }
      weekly:  { keep: 4 }
      monthly: { keep: 12 }
```

A backup — scheduled, manual, or a pre-migration snapshot — is refused once the store has reached `max_total_disk_gb`; `paramem backup-prune` or raising the cap clears the refusal. A restore or a migration rollback is never blocked by it — those are undo anchors, not new accumulation. `max_total_disk_gb` must be greater than zero; to stop taking backups altogether use `schedule: "off"`.

**Configuration migration.** For `server.yaml` changes that could affect memory quality — extraction prompts, adapter shape, consolidation cadence, base model — `paramem migrate` runs a guarded **trial**: it backs up the live state, applies the candidate, runs one consolidation cycle under the new config, and reports a before/after comparison so you can promote or roll back.

```bash
# Preview + trial a candidate config (absolute path required)
paramem migrate /home/you/configs/server-new.yaml
```

The interactive flow shows a unified diff with each change tier-classified (**Destructive** — `model`, `paths.*`, adapter `rank`/`alpha`: explicit confirm; **Pipeline-altering** — extraction/consolidation/routing flags: diff + confirm; **Operational** — host/port, STT/TTS, speaker: hot-apply), a `SHAPE CHANGE — DESTRUCTIVE` warning when adapter geometry changes, then `Proceed? [y/N]`. If an adapter's shape check could not be completed (e.g. its tier registry could not be read), the preview also prints a warning naming the affected adapter rather than silently omitting it. On `y` it confirms, polls the sanity gates, prints the comparison report, and prompts `accept / rollback / cancel`. The pre-migration backup is the rollback target. For non-interactive use, or to decide later (the trial keeps running server-side):

```bash
paramem migrate-status      # current trial state + gate results
paramem migrate-accept      # promote the candidate and apply it live
paramem migrate-rollback    # restore the previous config and apply it live
paramem migrate-cancel      # discard a staged candidate (before confirm)
```

**Accept and rollback apply the config in-process — no hard restart.** While the base model reloads under the new config the server switches to a brief **cloud-only window** (it keeps answering through the cloud agent), rebuilds its derived state, and returns to `local` only once the recall cache is rehydrated — a partial reload or preload stays cloud-only rather than serving from a half-built state. The same cloud-only-then-reclaim path covers boot: a server that comes up without enough free VRAM degrades to cloud-only and reclaims to `local` automatically once the GPU frees (`/gpu/acquire`). (Backups don't use this window — they read the live adapter set from disk and never reload the model.) Two changes are carve-outs the in-process path cannot cover:

> - **STT / TTS port change** — the Wyoming listener must rebind, so the CLI pre-flights the new port and, if it is bindable, asks you to consent to a one-shot restart; if the port is already in use it reports that instead of restarting.
> - **`paths.data` / `paths.sessions` change** — existing data is **not** moved automatically; the CLI prints a manual-restart hint and leaves the move to you.

> **Base-model swaps.** A `model:` change runs a dedicated base-swap migration (flagged Destructive in preview): each tier's graph is captured from the live adapters (Phase A), the base model is reloaded in-process to the candidate, and every adapter is retrained on the new base and gated at 100% recall before the swap commits (Phase B) — the candidate is exercised end to end. It is resumable across restarts and revertible from the pre-swap snapshot bundle (`POST /backup/restore` with `restore_config: true`; see [`SECURITY.md`](SECURITY.md)). The pre-swap bundle is **retention-immune** (same protection class as pre-migration snapshots — it survives pruning for 30 days even after a rollback clears the trial marker) and carries a `server.yaml.candidate` sidecar so the operator can pull the candidate config and retry later. The gate proves recall parity, not extraction/reasoning quality on the new base — validate those separately before adopting a new base permanently. Exercised Mistral 7B → Qwen3-4B.

**A config is validated before it can be promoted.** Every candidate is constructed into a full `ServerConfig` before anything is staged or swapped, so a config that would not boot is rejected while the live server is still untouched:

| Response | Endpoint | Meaning |
|---|---|---|
| 400 `candidate_invalid_config` | `POST /migration/preview` | The candidate does not construct. Nothing is staged; state stays LIVE. |
| 409 `candidate_invalid_config` | `POST /migration/confirm` | The staged candidate no longer constructs. The swap does not happen. |
| 409 `candidate_changed` | `POST /migration/confirm` | The candidate on disk changed after it was previewed — what you saw is not what would be promoted. Re-run the preview. |
| 400 `backup_unbootable` | `POST /backup/restore` (`kind=config`), `POST /migration/rollback` | The backed-up config cannot be constructed into a bootable config (e.g. it predates a validation rule). The live config is NOT modified. |

### Encryption & recovery operations

For the encryption design rationale, two-identity key model, and threat model see [`SECURITY.md`](SECURITY.md).

#### Quick start: enable encryption

A fresh install runs **Security-OFF** by default — infrastructure metadata is plaintext on disk and the server emits a loud startup warning. To enable:

```bash
paramem generate-key                 # mint daily + recovery; print recovery bech32
# Save the printed AGE-SECRET-KEY-1… offline (paper, metal plate, password
# manager note kept separately from your daily passphrase). It is NEVER
# stored on this device — lose it together with your daily passphrase and
# the data is unrecoverable.

# Put the passphrase you chose into .env (or your systemd drop-in):
#   PARAMEM_DAILY_PASSPHRASE=<your passphrase>

systemctl --user restart paramem-server
```

After this the startup log reads `SECURITY: ON (age daily identity loaded, recovery recipient available)` and `/status` reports `encryption: on`.

Day-to-day key operations:

| Command | When to use |
|---|---|
| `paramem change-passphrase` | Change the passphrase that wraps `daily_key.age`. Identity itself is unchanged — no re-encrypt of the data store. |
| `paramem rotate-daily` | Periodic hygiene, or suspected compromise of the daily identity (not just the passphrase). Mints a fresh X25519 identity and re-encrypts every envelope. Recovery recipient is preserved. |
| `paramem rotate-recovery` | Rotate the printed paper (new bech32, old one invalidated). Daily identity is unchanged. |
| `paramem restore --recovery-key-file PATH` | Hardware replacement — given the recovery bech32 from paper, mints a fresh daily identity on the new host and re-keys every envelope. |
| `paramem dump PATH` | Decrypt a single envelope for debugging. |
| `paramem encrypt-infra` | Migrate a plaintext data directory to age envelopes in-place (idempotent). Use `--dry-run` to preview. |

All lifecycle commands are per-file atomic + idempotent + resumable via a crash-safe rotation manifest (`~/.config/paramem/rotation.manifest.json`).

#### Startup-gate refusal reset runbook

When the server refuses to start due to a mode-consistency mismatch (plaintext files alongside a loaded key, or vice versa), the error message points at one of two paths.

**Migration path** — the store is sound but partially encrypted (typical when encryption was enabled after some files were already written plaintext, or a writer skipped the encrypted helper):

```
paramem encrypt-infra --dry-run     # preview what would change
paramem encrypt-infra                # migrate plaintext files in-place
```

The CLI walks the same `infra_paths` set the gate scans, skips files already in age form, and atomically encrypts the rest under the loaded daily identity. Idempotent — safe to re-run. Refuses without `PARAMEM_DAILY_PASSPHRASE` set unless `--dry-run`.

**Reset path** — the on-disk artifacts are stale (old training checkpoints, debug dumps from prior runs) and migrating them would only preserve clutter. Drops everything except voice enrollment and the keys themselves:

```bash
systemctl --user stop paramem-server.service
SAFETY=/tmp/paramem-fresh-backup-$(date +%Y%m%d-%H%M)
mkdir -p "$SAFETY"
cd "$PARAMEM_DATA_DIR"   # typically data/ha/
mv adapters debug sessions backups state observed_languages.json registry/key_metadata.json "$SAFETY/"
rmdir registry 2>/dev/null
systemctl --user start paramem-server.service
```

Preserved: `data/ha/speaker_profiles.json` (voice enrollment), `data/ha/tts/` (static models), `~/.config/paramem/{daily_key.age,recovery.pub}` (key material).

Rollback by `mv`-ing each item back from `$SAFETY/` into `data/ha/`. Once the new state has soaked for long enough that you trust it, `rm -rf "$SAFETY"`.

### Per-user token management

When `mobile_pwa.enabled: true` (or after the first `mint-user-token` run, which wires the same store), each device that should access the server must be issued its own bearer token — there is no separate shared-secret credential; every token is a `UserTokenStore` entry. Tokens are minted offline via the CLI, which prints a QR code for one-tap onboarding on mobile devices. For a **personal device**, mint a per-user token (bound to `speaker_id`) via `paramem mint-user-token` — identity is resolved from the token on every request. For a **shared device**, mint an unattributed token (`--unattributed`) instead — the server then identifies speakers by voice embedding and runs the enrollment flow automatically.

```bash
paramem mint-user-token [SPEAKER_ID] \
    [--label LABEL] \
    [--onboard-url URL] \
    [--config PATH] \
    [--png FILE] \
    [--scope {chat,admin}] \
    [--unattributed] \
    [--force-admin]
```

- `SPEAKER_ID` — the speaker this token authenticates (e.g. `speaker0`). Required unless `--unattributed` is given.
- `--scope` — token capability: `chat` (conversational endpoints `/chat`, `/voice`, `/push/*`, `/status` only — the secure default) or `admin` (all endpoints, including operational ones like `/gpu/*`, `/consolidate`, `/backup/*`). Default: `chat`.
- `--unattributed` — mint a token with no bound speaker (for shared devices that identify speakers by voice embedding). Cannot be combined with a positional `SPEAKER_ID`.
- `--force-admin` — required when combining `--scope admin` with `--unattributed`. Prints a warning: an unattributed admin token cannot be revoked by speaker; use `revoke-user-token --label` to revoke it.
- `--label` — human-readable device or purpose label stored with the token (e.g. `phone`).
- `--onboard-url` — base URL of the ParaMem server (e.g. `https://<your-host>.<your-tailnet>.ts.net`). Required to produce a native-camera-scannable QR deep-link; omitting it skips the QR and emits a warning.
- `--config` — server config path (default: `configs/server.yaml`), used to resolve the data directory.
- `--png` — also save the QR as a PNG file.

The command prints a terminal QR encoding a deep-link URL (`https://<host>/app#token=<t>&url=<encoded-onboard-url>`) plus a text fallback, then exits. The QR is scannable with the phone's native camera — no app is needed. The plaintext token is never written to any log file. Example:

```bash
paramem mint-user-token speaker0 \
    --label phone \
    --onboard-url "https://<your-host>.<your-tailnet>.ts.net"
```

**Encryption note.** If `PARAMEM_DAILY_PASSPHRASE` is set and the daily key is loaded (Security ON), `user_tokens.json` is age-encrypted; the passphrase must be available when running this command. Without a daily key the store is written in plaintext (Security OFF). See [`SECURITY.md §5`](SECURITY.md) for the full token-store encryption contract.

**Upgrade note (shared-token deployments).** An existing deployment that predates per-user tokens set `PARAMEM_API_TOKEN` as a single shared credential validated directly by the server. That validation path is retired: `PARAMEM_API_TOKEN` is now only a carrier env var that infrastructure callers read to source their own `Authorization` header — the server itself never validates its value. Upgrading such a deployment **without** running `mint-user-token` lands it OPEN (`AUTH: OFF`, every REST endpoint reachable without a credential) — the server logs a loud warning naming this exact condition at startup, but the old token no longer protects anything. Run `paramem mint-user-token --unattributed --scope admin --force-admin` once, then update `PARAMEM_API_TOKEN` (and any systemd drop-in / HA component config reading it) to the newly minted value, to restore the server's protected posture.

#### Shared (multi-user) device

A device used by more than one person — e.g. a kitchen tablet — is **not** given a per-user token (that would attribute every speaker to a single identity). Instead, mint it an **unattributed** token (no bound `speaker_id`) — there is no separate shared-secret credential; an unattributed token is a normal `UserTokenStore` entry like any other, just without a `speaker_id`:

```bash
paramem mint-user-token --unattributed --scope chat --label "Kitchen Tablet"
```

1. Open the PWA at `/app` and paste the printed token into Settings, or open the deep-link URL printed by the command on the device. The token is stored in `localStorage` and sent as `Authorization: Bearer` on every request — the same transport as a per-user token.
2. Because an unattributed token attaches no `speaker_id` (`auth_speaker_id` is absent), `POST /voice` computes a voice embedding and identifies each speaker by voice, running the enrollment / greeting / name-disclosure flow automatically. A fresh `conversation_id` per push-to-talk press keeps multiple speakers on one device correctly attributed.

This token reaches `/chat`, `/voice`, `/push/*`, and `/status` but gets 403 on every operational endpoint — the secure default for a shared device. Restrict access at the network layer (Tailscale / LAN — never the public internet) as the outer defence; token scope provides the inner defence. A shared device that genuinely needs admin reach (rare — prefer a personal admin token instead) can be minted `--unattributed --scope admin --force-admin`, which prints a warning that it cannot be revoked by speaker (use `revoke-user-token --label` for it).

**Infrastructure token (systemd / HA carrier).** Non-interactive consumers — the systemd consolidation-tick timer, the HA custom component — read their bearer-token value from the `PARAMEM_API_TOKEN` environment variable at execution time (see `render_service_unit` in `paramem/server/systemd_timer.py`). The server does not validate this env var itself; the value placed in it must be a token actually minted into the store. Mint an unattributed admin token for the timer (it calls `POST /scheduled-tick`, an admin-scoped endpoint) and put its plaintext value into `.env`:

```bash
paramem mint-user-token --unattributed --scope admin --force-admin --label "systemd infra"
# copy the printed token into .env: PARAMEM_API_TOKEN=<token>
```

The HA component only needs `/chat`, so an unattributed **chat**-scope token is sufficient and lower-privilege — mint one the same way with `--scope chat` (no `--force-admin` needed) if HA is the only consumer of that credential.

#### Household topology quick guide

| Scenario | Token to issue | Identity source |
|----------|---------------|-----------------|
| Personal phone / tablet (one person) | Per-user token (`mint-user-token <speaker_id> --scope chat`) | Token binding — cheap, no embedding |
| Shared device, restricted (chat only) | Unattributed chat token (`mint-user-token --unattributed --scope chat`) | Voice embedding — enrollment flow runs automatically |
| Shared device, full admin reach | `--unattributed --scope admin --force-admin` | Voice embedding |
| systemd / HA infrastructure carrier | `--unattributed --scope admin --force-admin` (timer) or `--scope chat` (HA) — value placed in `PARAMEM_API_TOKEN` | N/A (machine caller) |

When in doubt, issue a per-user token for every person who has their own device. For shared devices, prefer `--unattributed --scope chat` (least privilege) over an admin-scope token.

#### PWA installation

**iOS / iPadOS (Safari only)**

1. Open `https://<your-host>.<your-tailnet>.ts.net/app` in **Safari** (Chrome on iOS does not support PWA install or Web Push).
2. Tap the Share button (the box-with-arrow icon in the toolbar).
3. Scroll down and tap **Add to Home Screen**.
4. Accept the default name or rename it, then tap **Add**.
5. Launch the app from the Home Screen icon — it opens full-screen without the Safari chrome.

Web Push requires iOS/iPadOS 16.4 or later. The PWA must be launched from the Home Screen icon (not opened in Safari) to receive push notifications.

**Android (Chrome)**

1. Open `https://<your-host>.<your-tailnet>.ts.net/app` in Chrome.
2. Chrome shows a banner or a small install icon in the address bar; tap **Install** (or use the three-dot menu → **Add to Home Screen**).
3. Tap **Install** in the confirmation dialog.
4. Launch from the Home Screen icon.

#### Per-user onboarding walkthrough

This is the flow for adding a household member (e.g. "Alice's iPhone") to an existing deployment.

**Admin side (run once per device)**

```bash
paramem mint-user-token speaker1 \
    --label "Alice iPhone" \
    --onboard-url "https://<your-host>.<your-tailnet>.ts.net" \
    --png /tmp/alice-iphone-qr.png
```

The command prints a terminal QR code encoding a deep-link onboarding URL, a text `deeplink:` line (tap-able manual fallback), and a `token:` line. Hand the QR or the deep-link URL to the member — the plaintext token is never stored on disk.

**Member side (one-time setup on their device)**

1. Point the phone's **native camera** at the QR code (or tap the deep-link URL in a message). The camera opens the PWA URL automatically and the PWA stores the token in `localStorage` without any manual entry. Done — skip to step 4.
2. If native-camera onboarding is not available: open the PWA URL in Safari (iOS) or Chrome (Android) and tap the gear icon (top-right) to open Settings.
3. Enter the server URL in the **Server URL** field and paste the token into the **Bearer token** field. Tap **Save**.
4. Grant microphone permission when prompted (for voice), and notification permission when prompted (for Web Push, if enabled).
5. The app is now paired. Text and voice queries carry the member's `speaker_id` automatically.

#### Token revocation

Use `paramem revoke-user-token` to revoke tokens without manually editing the encrypted store:

```bash
# List current tokens (speaker_id, label, created, revoked):
paramem revoke-user-token --list --config configs/server.yaml

# Revoke all tokens for a speaker (e.g. lost device, access change):
paramem revoke-user-token --speaker speaker0 --config configs/server.yaml

# Revoke by device label (e.g. revoke a specific device only):
paramem revoke-user-token --label "phone" --config configs/server.yaml

# Skip the confirmation prompt in scripts:
paramem revoke-user-token --speaker speaker0 --yes --config configs/server.yaml
```

The command reads and writes `user_tokens.json` via the same encrypted-store path as `mint-user-token` — no manual decrypt/re-encrypt step is needed. If `PARAMEM_DAILY_PASSPHRASE` is set and the daily key is loaded (Security ON), the store is read and written as an age envelope automatically.

**Takes effect immediately.** Revocation (and scope changes via re-mint + revoke) takes effect on the next request — the running server re-reads `user_tokens.json` whenever the file's mtime changes (mtime-triggered live reload). No server restart is required.

A revoked token causes a `401 Unauthorized` response; the PWA reopens the Settings drawer automatically.

#### Forgetting a speaker (data erasure)

Revoking a token removes a member's **access**; `POST /speaker/forget` removes their **data**. Use it when a member leaves the household, or to clean up a synthetic/test speaker. The endpoint is admin-scoped — send your `PARAMEM_API_TOKEN` as a bearer token.

```bash
curl -sS -X POST http://localhost:8420/speaker/forget \
  -H "Authorization: Bearer $PARAMEM_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"speaker_id": "speaker0"}'
# → {"removed_speaker": true, "erased_keys": ["graph12", ...], "discarded_sessions": ["conv-abc"],
#    "reaped_tiers": [], "unloaded_adapters": [], "removed_dirs": []}
```

In one call it: (1) removes the speaker's indexed-memory keys — their cached fact content, per-tier registry entries (active and stale), SimHash fingerprints, and provenance — in memory (keyed recall stops serving them at once) and on disk, including the underlying fact content in every tier's stored memory (neither a restart nor a later storage-mode migration will resurrect them); (2) deletes the speaker profile from the speaker store; (3) discards any of the speaker's pending (not-yet-consolidated) sessions. The response reports exactly what was removed. `erased_keys` lists the keys erased. `reaped_tiers` lists any tier (interim or main) the erase left with zero known keys; `unloaded_adapters` and `removed_dirs` report what that reap unmounted and deleted — both empty when nothing was reaped.

The operation is a hard **erase**, not a soft stale transition: each key, its cached content, and its provenance are dropped entirely, in memory and on disk — safe to run on a live, serving store. It does **not** trigger retraining, and the key becomes unservable immediately (the SimHash gate can no longer verify it). What happens to a tier's adapter weights depends on the erase's outcome for that tier: a tier that still holds other keys keeps its registry-level posture — under warm-init consolidation, a routine fold keeps a resident tier's existing weights rather than retraining it from scratch, so the residual weight encoding is not automatically trimmed by ordinary training cycles, and the only door that rebuilds a tier's weights from its live keys is an operator-invoked `POST /reconsolidate`. A tier the erase reduces to zero known keys is reaped immediately instead, in the same request: its adapter is unmounted and its on-disk weight and registry artifacts are deleted on the spot. `speaker_id` is the only request field; there is no erasure variant to select.

The door refuses with **409** while a fold, background training, or a migration TRIAL is in progress, or while the server is in cloud-only mode — the same refusal band as [Discarding the interim ring](#discarding-the-interim-ring). It also serializes on the GPU lock, so it can wait behind an in-flight chat reply before starting.

What forgetting does not reach: a backup slot taken before the forget still holds the speaker's registry and key metadata, and restoring it restores them; transcripts already archived by a prior consolidation (when `retain_sessions` or `debug` is enabled) are untouched — only pending, not-yet-consolidated sessions are discarded here. A crash partway through leaves one of two partial states, both recoverable by re-running the request: a tier the erase had not yet reached still holds the key fully, as if the request had not run; a tier the erase had already reached is unservable (keyed recall will not return it) even if its underlying stored content briefly lags behind on disk. A reap that crashes mid-removal leaves its on-disk artifacts partially deleted; the remainder is removed at the next startup consistency check. Discarding an interim slot wholesale (rather than one speaker's keys within it) is a separate door — see [Discarding the interim ring](#discarding-the-interim-ring).

Responses: `200` with the removal report on success. `409` while consolidating, background training is active, a migration TRIAL is in progress, or the server is cloud-only — matching [Discarding the interim ring](#discarding-the-interim-ring)'s refusal band. `500` when one of the speaker's keys belongs to a malformed interim tier whose on-disk location cannot be resolved, or when a tier's registry exists on disk but cannot be read or decrypted (e.g. Security ON with no daily identity loaded) — both of those checks run before any key is erased, so the request aborts with the store left untouched. A reap that fails mid-operation *after* the erase has already run also surfaces as a `500`; see the crash-partway state described above for what that leaves behind. A `removed_speaker: false` with empty lists means the speaker ID was unknown to the store — the call is idempotent and safe to repeat.

#### Enabling Web Push

Web Push is opt-in and off by default. To enable it:

1. In `configs/server.yaml`, set `mobile_pwa.push_enabled: true` and update `mobile_pwa.vapid_contact` to a real `mailto:` address (the default `mailto:admin@localhost` is valid but browsers may reject it as non-canonical).
2. Restart the server. A VAPID EC P-256 keypair is auto-generated on the first start with push enabled and persisted to `<paths.data>/vapid_keys.json`.
3. Each PWA client subscribes automatically on the next launch after the token is saved — the client calls `GET /push/vapid-public-key`, then `POST /push/subscribe`, and the server stores the endpoint in `<paths.data>/push_subscriptions.json`.

The VAPID keypair must remain stable: rotating it invalidates all existing browser subscriptions (they will not receive notifications until they re-open the app and re-subscribe). The PWA must be installed to the Home Screen and launched from there — a browser tab does not receive push notifications on iOS.

Push payloads carry no personal content. The notification is a generic ping; the member opens the app to read the actual reply.

#### Troubleshooting

- **The PWA URL shows a bearer-token prompt / raw API JSON instead of the chat UI.** Either `mobile_pwa.enabled` is `false` in the server config (check `/status`), or you navigated to `/` instead of `/app`. The PWA shell is served at `/app`.
- **"Enter your bearer token in Settings to get started" appears on every launch.** The token was not saved — tap the gear icon, paste the token, and tap **Save**.
- **Push notifications do not arrive.**
  - The PWA must be installed to the Home Screen and launched from the icon, not opened as a browser tab.
  - On iOS, check Settings → Notifications → ParaMem and confirm notifications are allowed.
  - On Android, long-press the Home Screen icon → App info → Notifications.
  - Verify `mobile_pwa.push_enabled: true` and that the VAPID contact is a valid `mailto:` address.
- **Voice (mic button) has no effect or shows "unsupported".**
  - iOS: microphone access for the PWA must be granted in Settings → Privacy & Security → Microphone → Safari.
  - Android: allow microphone in the site permissions (tap the lock icon in the Chrome address bar).
  - The PWA must be served over HTTPS — `getUserMedia` is not available on plain HTTP.
- **401 on every request after revoking a token.** Open Settings in the PWA, clear the token field, and paste a newly minted token.

### GPU Lifecycle

The server shares the GPU with ML workloads.  Release is brokered by
`gpu_guard` (machine-level arbitration, config-driven consumers).
ParaMem registers as the ``paramem-server`` consumer in
``~/.config/gpu-guard/config.toml``:

- **`POST /gpu/release`** → in-process unload, switch to cloud-only.
  Default release primitive used by `gpu_guard` and other workloads.
  Synchronous; idempotent; returns 503 mid-consolidation, or 409 while a
  base-swap migration is actively running, so the caller can retry either
  way.
- **SIGUSR1** → graceful exit (snapshot + `os._exit(1)` → systemd
  restart).  Alternate release primitive, retained for callers that
  prefer signal semantics over HTTP.
- **Auto-reclaim** → periodically checks if GPU is free, reloads model.
- **Startup guard** → if GPU is occupied at startup, starts in cloud-only mode.
- **`--cloud-only`** → explicit flag to skip model loading.

```bash
# Release GPU for a training run (preferred)
curl -X POST http://localhost:8420/gpu/release

# Server auto-reclaims when GPU is free (default: 10min polling)
```

#### Deferred-mode hold and orphan recovery

ML workloads started through `experiments/utils/gpu_guard.py` (or the
`tresume` shell flow) set `PARAMEM_EXTRA_ARGS=--defer-model` in the
systemd user environment so the server stays cloud-only for the duration
of the run.  The holder also stamps `PARAMEM_HOLD_PID`,
`PARAMEM_HOLD_STARTED_AT`, and `PARAMEM_HOLD_CMD` so the server can tell
a legitimate mid-training hold apart from an orphaned env var left
behind by a `SIGKILL`ed test.

`/status` surfaces the hold as:

```json
{"hold": {"hold_active": true, "owner_pid": 12345, "owner_alive": true,
          "age_seconds": 240, "owner_hint": "python / experiments.test8_large_scale"}}
```

`pstatus` renders it inline on the PID row.  Three cases:

| State | PID-row annotation | Meaning |
|-------|--------------------|---------|
| Alive holder | `(held by [python / experiments.test8_large_scale] (age 4m))` | Legitimate mid-training hold — auto-reclaim respects it. |
| Orphaned (holder PID dead) | `(orphaned hold by [...] (age 15m) — pstatus --acquire)` (yellow) | `SIGKILL`ed test.  Auto-reclaim has emitted a single WARN and stopped looping. |
| Orphaned (no holder registered) | `(orphaned hold, no holder registered — pstatus --acquire)` (yellow) | `PARAMEM_EXTRA_ARGS` set by legacy caller / manual tinkering. |

Operator recovery is a single command:

```bash
pstatus --acquire
# → POST /gpu/acquire: clears PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_*
#   and, if the running server is in --defer-model, reloads the base
#   model in-process (no service restart needed).
```

Auto-reclaim **never auto-clears orphans** — by design, visibility over
silent self-healing.  The loop stops on orphan detection and waits for
the operator.

`pstatus --config` renders the effective `ServerConfig` (after yaml
load + env merge) as YAML — useful for verifying what the running server
actually sees, not what is on disk.  When an active-store migration is
pending (mode flip detected at startup, see *Background training*),
`pstatus` prints a `REHYDRATING` banner with per-tier completed/failed
state until the migration finishes.

### API

Complete REST endpoint reference. Auth scopes: **unauthenticated** — no token required; **chat** — any valid token minted with `--scope chat` (attributed or unattributed); **admin** — admin-scope token required (`require_admin` dependency, app.py:2615).

| Method | Path | Scope | Description |
|--------|------|-------|-------------|
| GET | `/health` | unauthenticated | Liveness probe — returns `{"status": "ok"}`. Exempt from bearer-token auth; safe to poll without a token (e.g. HA `binary_sensor`). |
| GET | `/` | unauthenticated | Redirect to the PWA shell at `/app/`. Exempt from auth. |
| GET | `/app/sw.js` | unauthenticated | Serve the PWA service-worker script with `Cache-Control: no-cache` to force update checks on every navigation. Under the `/app/` exempt prefix. |
| POST | `/chat` | chat | Send a conversation turn; returns the assistant reply. Speaker is resolved from the bearer token (attributed per-user token) or voice embedding (unattributed token). Conversation context is assembled server-side from the stored transcript — the request carries no history field, so a client never sends prior turns. See curl example below. |
| POST | `/voice` | chat | PWA push-to-talk: accepts a raw audio blob (`audio/mp4`, `audio/webm`, `audio/L16`; 25 MB cap), transcribes via Whisper, and returns `{transcript, reply, audio, audio_format, follow_up?}`. Same routing path as `/chat`, including server-assembled context. See [Voice Pipeline](#voice-pipeline). |
| GET | `/status` | chat | Full operational snapshot — server mode, model id + device, per-adapter specs (`rank`/`alpha`/`lr`/`target_kind`), interim adapter inventory + capacity, speaker embedding backend/model/device, STT/TTS engines, enrolled speakers, pending sessions + orphans + oldest age, consolidating flag + BG trainer state, last consolidation result, schedule + next-run ETA, deferred-mode `hold` block (owner PID + liveness + age + cmd hint), `attention` block (open operator incidents, e.g. `enrichment_degraded`) |
| GET | `/push/vapid-public-key` | chat | Return the VAPID EC P-256 application server public key for `PushManager.subscribe()`. 503 when push is disabled. See [Enabling Web Push](#enabling-web-push). |
| POST | `/push/subscribe` | chat | Register a browser push subscription for the authenticated speaker. Requires an attributed per-user token (unattributed tokens have no bound speaker_id). See [Enabling Web Push](#enabling-web-push). |
| POST | `/consolidate` | admin | Collapse the interim slots into main memory now — the content check the schedule uses to call a full cycle due, minus the deadline math; noops when there is no payload-bearing interim slot (and, at `max_interim_count: 0`, no pending NAMED session either). Non-blocking: returns immediately, poll `GET /status` → `consolidating`. |
| POST | `/consolidate/interim` | admin | Absorb recent conversations into memory now, without waiting for the schedule. With no pending attributable session the content gate reports a `noop_*` status and nothing is dispatched. |
| POST | `/reconsolidate` | admin | Rebuild main memory from its own stored knowledge — runs even when nothing is new, and leaves the interim slots and pending conversations untouched. Use after changing the base model, the extraction prompts, or the extraction config. |
| POST | `/scheduled-tick` | admin | Systemd user-timer entrypoint (`paramem-consolidate.timer`). The schedule's own door: it resolves the tick to an interim absorb or a full fold, honours the catch-up gate (so a redundant or repeated tick inside the same schedule window fires no cycle) and advances the cadence stamp on dispatch — the two things `POST /consolidate` does not do — and is skipped when there is nothing new to consume. 409 `trial_active` while a migration TRIAL is in progress. |
| POST | `/refresh-ha` | admin | Rebuild the HA entity graph from `/api/states` + `/api/services`. |
| POST | `/ingest-sessions` | admin | Enqueue pre-chunked document segments for the next consolidation cycle (operator CLI: `scripts/ingest_docs.py`). Idempotent — chunks already in the ingest registry are skipped. |
| POST | `/ingest-sessions/cancel` | admin | Discard queued ingest sessions by session ID without running consolidation. |
| POST | `/gpu/acquire` | admin | Clear any `PARAMEM_EXTRA_ARGS=--defer-model` hold and, if this process is in defer mode, reload the base model in-process.  Called by `pstatus --acquire`.  Idempotent. Refuses with **503** while a consolidation cycle is in flight, or **409** while a base-swap migration is actively running. |
| POST | `/gpu/release` | admin | Release the base model in-process and switch to cloud-only mode, freeing VRAM. Refuses with **503** while a consolidation cycle is in flight, or **409** while a base-swap migration is actively running. |
| POST | `/incidents/{incident_id}/ack` | admin | Acknowledge an active incident, silencing its attention row in `/status`. |
| GET | `/integrity` | admin | Run the infrastructure integrity check (registries, simhash, meta, graph files) and return a `{ok, checks, failures}` report. Cloud-only-safe — no GPU dependency. |
| POST | `/admin/assign-orphans` | admin | Operator-only corrective action: permanently attribute orphan sessions to a single speaker.  Rewrites session jsonls on disk when present; the bound speaker_id flows through the next consolidation cycle into permanent storage regardless of debug mode. |
| POST | `/speaker/forget` | admin | Remove a speaker's profile, erase their indexed-memory keys — cached content, per-tier registry and SimHash entries, provenance, and the underlying stored facts — and discard pending sessions. Does not trigger retraining; a surviving tier's weights are trimmed only by a later operator-invoked `POST /reconsolidate`, but a tier reduced to zero known keys is unmounted and its weight/registry artifacts deleted immediately. Refuses with **409** while busy, training, TRIAL, or cloud-only. See [Forgetting a speaker](#forgetting-a-speaker-data-erasure). |
| POST | `/interim/discard` | admin | Discard the entire interim ring without folding it into main memory — the only door that removes interim slots without absorbing them first. Synchronous; returns the blast radius without `{"confirm": true}`. Refuses with **409** while busy, training, TRIAL, or cloud-only. See [Discarding the interim ring](#discarding-the-interim-ring). |
| GET | `/backup/list` | admin | List all backup slots with metadata. See [Backup & Migration](#backup--migration). |
| POST | `/backup/create` | admin | Take an immediate backup (default `snapshot_bundle` — config, registry, adapter weights, speaker profiles). See [Backup & Migration](#backup--migration). |
| POST | `/backup/restore` | admin | Restore a backup slot onto the live store (`kind=config` or `kind=snapshot_bundle`). See [Backup & Migration](#backup--migration). |
| POST | `/backup/prune` | admin | Apply the 5-rule retention policy; returns counts of deleted and preserved slots. See [Backup & Migration](#backup--migration). |
| POST | `/migration/preview` | admin | Stage a candidate `server.yaml` and return a unified diff + tier-classified change list. No files written. See [Backup & Migration](#backup--migration). |
| GET | `/migration/diff` | admin | Return the diff for the currently-staged candidate (same shape as `/migration/preview`). Valid only in STAGING state. See [Backup & Migration](#backup--migration). |
| GET | `/migration/status` | admin | Return the current migration state (`LIVE` / `STAGING` / `TRIAL`), trial gate results, and comparison report when accept-eligible. See [Backup & Migration](#backup--migration). |
| POST | `/migration/confirm` | admin | Atomically transition STAGING → TRIAL: write pre-migration backup, swap config, kick off trial consolidation. See [Backup & Migration](#backup--migration). |
| POST | `/migration/accept` | admin | Promote trial config B to live, archive trial adapter, clear trial state. Valid only when gates finished with `pass` or `no_new_sessions`. See [Backup & Migration](#backup--migration). |
| POST | `/migration/cancel` | admin | Discard the staged candidate and return to LIVE state. Valid only in STAGING. See [Backup & Migration](#backup--migration). |
| POST | `/migration/rollback` | admin | Restore config A from backup, archive trial adapter, clear trial state. Valid from TRIAL at any time. See [Backup & Migration](#backup--migration). |
| POST | `/debug/probe` | admin + `config.debug=true` | Operator-only ephemeral probe of the chat handler with explicit `speaker_id` injection.  Bypasses `_resolve_speaker`; **no buffer mutation, no jsonl rewrite, no consolidation impact** — pure single-call probe in RAM only.  Body: `{text, speaker_id, conversation_id?}`. |
| POST | `/debug/recall` | admin + `config.debug=true` | Operator-only direct adapter recall probe.  Bypasses the router and reasoning step: activates `adapter` (or disables all when `adapter="none"`), runs `text` through the model, returns raw output + a `parse_recalled_entry` attempt + the active adapter + latency.  Use to measure direct natural-language recall from adapter weights as distinct from the cache-driven enumerate-then-reason path on `/chat`.  Body: `{text, adapter, system_prompt?, max_new_tokens?, temperature?}`. |
| GET | `/debug/dump` | admin + `config.debug=true` | Operator-only zero-GPU read of the in-memory `MemoryStore`.  Walks `iter_entries()` and returns every `(tier, key, entry)` as a flat list.  ~5 ms for typical operator-scale stores vs ~min for the equivalent per-key `/debug/recall` sweep.  Use for content inventory, cross-model A/B setup, or scoring against a probe-suite output. |
| POST | `/calibrate/extract` | admin + `calibrate_endpoint_enabled` | Run the extraction chain from a transcript. `stop_phase` selects which step's output comes back (default: run to the end); every other calibration route fixes its own. |
| POST | `/calibrate/procedural` | admin + `calibrate_endpoint_enabled` | Run the procedural extractor on a transcript. |
| POST | `/calibrate/anonymize` | admin + `calibrate_endpoint_enabled` | Enter the chain at `anonymize` with a supplied `graph` and return the anonymizer's output. |
| POST | `/calibrate/normalize` | admin + `calibrate_endpoint_enabled` | Run the graph tier's predicate-normalization pass over supplied `relations` or a `snapshot_path`. Uses the production engine selection (cloud when egress is permitted, local otherwise) and the production survivor rule. |
| POST | `/calibrate/plausibility` | admin + `calibrate_endpoint_enabled` + cloud egress | Enter the chain at `anonymize` with a supplied `graph` and return the de-anonymized plausibility judge's output. Reaching that judge runs the cloud enrichment it sits downstream of, so this route places a cloud call. |
| POST | `/calibrate/enrich` | admin + `calibrate_endpoint_enabled` + cloud egress | Enter the chain at `anonymize` with a supplied `graph` and return the `cloud_enrich` step's output. Places a cloud call. Does **not** reach the separate graph-level enrichment, which has no calibration route. |
| POST | `/calibrate/name` | admin + `calibrate_endpoint_enabled` | Run the enrollment name extractor over supplied `turns`. |

Every route above runs the **real production chain** — none re-invokes a
step's primitive on its own, so what a probe reports is what the
consolidation cycle would do on that input with that prompt. Each route
declares where the chain is entered, which artifact it accepts, and which
step's output it returns; routes entering past `local_extract` require the
`graph` that step would have produced (typically a prior
`/calibrate/extract` response, posted back verbatim). When the configured
chain cannot reach the step a route reports on — cloud egress refused, an
injected graph with no relations, a pass short-circuiting on its own floor
— the call returns HTTP 400 naming the steps that did run and the cloud
verdict, rather than a response whose provenance is silently empty. All
routes are gated by `consolidation.calibrate_endpoint_enabled=true`
(default off); none writes weights or production data.

**Chat request:**

```bash
curl -X POST http://localhost:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Where does Marcus work?", "conversation_id": "session1"}'
```

The server buffers conversation turns automatically. On the next consolidation cycle, it extracts knowledge from buffered sessions, merges into the knowledge graph, encodes the merged facts as indexed-key training data, and retrains the adapter.

### Home Assistant Integration

A custom conversation agent for Home Assistant is included in `custom_components/paramem/`. The component is a thin REST client — all intelligence runs on the ParaMem server.

**Setup:**
1. Copy `custom_components/paramem/` to your HA `custom_components/` directory
2. Restart HA
3. Add the integration via Settings → Devices → Add Integration → ParaMem
4. Configure the server URL (default: `http://localhost:8420`)

**How it works:**
- HA sends voice/text queries to ParaMem's `/chat` endpoint
- ParaMem handles memory-related queries locally (parametric recall from adapter weights)
- Non-memory queries are forwarded back to HA's configured conversation agent via WebSocket (`conversation.process`), which handles device control, search, weather, and other tools with full room awareness
- The HA conversation agent's prompt, model, and tools are configured entirely on the HA side — no duplication in ParaMem

### Voice Pipeline

ParaMem includes a local voice pipeline for privacy-first operation:

- **Local STT:** Whisper distil-large-v3 on GPU via Wyoming protocol (port 10300). CPU fallback: distil-small.en.
- **Speaker identification:** WeSpeaker (`pyannote/wespeaker-voxceleb-resnet34-LM`, 256-dim) voice embeddings via pyannote-audio. Multi-embedding profiles (up to 50 per speaker) with L2-normalized centroid matching for cross-device robustness. Auto-enrichment on confirmed matches. Deferred identity binding keeps anonymous utterances bound to a BPE-stable `speaker{N}` placeholder until the speaker discloses a name, after which the graph is retro-claimed without a rewrite at training time (the name is substituted for the placeholder only at the reply boundary).
- **Multilingual TTS:** Piper voices per language with MMS-TTS fallback; language detection on the response text, speaker binding so each speaker's preferred voice persists, routed to media players via HA.
- **Anti-confabulation voice prompt:** a separate system prompt at the voice turn tells the model not to invent facts about the speaker when the parametric memory has nothing to say, and to fall through to the cloud path cleanly.
- **Server-assembled context:** conversation history for every voice turn is read from the server's own session store — the same path `POST /chat` uses — not carried in the request; a client never sends prior turns.
- **Mobile PWA voice path:** The PWA (served at `/app` when `mobile_pwa.enabled: true`) supports push-to-talk voice in addition to text. The browser records audio and POSTs it to `POST /voice` (raw audio blob; `audio/mp4`, `audio/webm`, or `audio/L16`; 25 MB hard cap). The server decodes to 16 kHz int16 mono, transcribes via Whisper, and returns `{transcript, reply, audio, audio_format, follow_up?}` — where `audio` is a base64-encoded WAV of the synthesised reply voiced through the same per-language TTS voices as the HA satellites (e.g. Kokoro `af_heart` for English), or `""` when TTS is unavailable (the PWA falls back to text display). Routing goes through the same `_run_chat_turn` path as `POST /chat`. **Token-type selector:** an attributed per-user token resolves identity from the token (no embedding computed, cheap); an unattributed token triggers voice-embedding identification and the same enrollment/greeting/name-disclosure path as `POST /chat`, with a fresh per-utterance conversation_id on each push-to-talk press. Deployment: personal device → issue an attributed per-user token; shared device → issue an unattributed token with voice enrollment. Error statuses: `404` when `mobile_pwa.enabled` is false, `503` when STT is not loaded (cloud-only mode), `413` for an oversized payload, `400` for an undecodable audio body.

---

## Prompt Engineering

The extraction pipeline's behaviour is shaped almost entirely by the prompt
files under `configs/prompts/`. This section captures the principles that
govern those files and the calibration loop used to iterate on them. **Read
this before editing any prompt** — most of the principles below were learned
empirically and contradict natural intuition about how to write LLM prompts.

### Principles

**Few-shot examples carry the schema.** A prompt does not need to declare
the entity-type or relation-type taxonomy verbatim. Listing them via
template slots like `{entity_types}` was empirically harmful: it implicitly
licensed Mistral 7B to extend the closed set with new type names —
`phone_number`, `software`, `library`, `degree`, `acronym`, etc. — 23
invented types in one run on a CV transcript. The same prompt with no
taxonomy slot and only few-shot examples produced **0 invented types**.
Mistral treats explicit lists as "you can add to this"; examples anchor a
closed set without needing a rule.

**Declarative text stays minimal and concise — few-shot examples do the
hard work.** A prompt is a short headline (one sentence — what the model
is doing), a brief imperative core (the load-bearing structural rules —
schema fields, output shape), then the body: POSITIVE examples for the
right shape, NEGATIVE examples (`WRONG: ... → RIGHT: ...`) for the failure
modes you actually observe. Long declarative prose ("INTENT MATTERS:",
"PLAUSIBILITY:", "USE THE ASSISTANT'S RESPONSE", taxonomy bullets)
competes with examples for the model's attention budget; on Mistral 7B,
removing 50+ lines of such prose and keeping ~30 lines of examples
flipped contact-attribute capture (`email` / `phone` / `linkedin`) on the
speaker entity from absent → reliable. The principle generalizes:

- **Multi-task prompts split into labelled sections** (`## KEEP` /
  `## DROP`, `## Part 1 — RELATIONS` / `## Part 2 — SAME_AS`) with each
  section's POSITIVE + NEGATIVE block co-located. Labels prime
  attention; the imperatives stay one sentence; the examples teach. On
  `cloud_plausibility.txt`, splitting eliminated chunk-1 over-generation
  (1 input fact → 51 invented facts in the unified version → 0 in the
  split version).

- **Load-bearing structural contracts go at the top, not buried lower down.** When
  the downstream pipeline depends on a schema field, a brace-binding
  requirement, or a token like `[ESCALATE]` that the router parses, put
  it under the headline with its own POSITIVE + NEGATIVE pair. On
  `cloud_enrichment.txt`, hoisting the brace-binding contract for
  newly-minted entities to the top doubled the binding emission
  rate (6 → 16 per session) and recovered 41 personal facts per CV chunk
  that had been silently dropped at the deanon residual sweep.

- **NEGATIVE examples teach harder edges than POSITIVE alone.** Add them
  for the failure modes you observe, not hypothetical ones. On
  `cloud_graph_enrichment.txt`, a single `WRONG: (Alice, ..., "12 months")
  — literal value, not a graph node` NEGATIVE eliminated phantom-node
  introduction (2 violations → 0) without changing anything else.

**Closed-set vs. open-set fields behave differently in examples.** The
model treats fields differently based on whether the prompt examples
enumerate alternatives:

- `entity_type` — examples show only `person`, `organization`, `place`,
  `concept` and `preference`. Mistral stays inside this set on novel
  inputs (closed-set behaviour).
- `predicate` and attribute keys — examples show a handful of verbs
  (`works_at`, `lives_in`, `owns`, `sister_lives_in`) and attribute keys
  (`last_name`, `email`, `phone`, `linkedin`). On novel content Mistral
  coins reasonable new ones (`worked_for`, `led`, `delivered`,
  `country_of_residence`) — open-set behaviour.

This split is the contract: closed-set fields are constrained by example
exhaustiveness; open-set fields are filled by demonstration of shape. The
attribute-projection step (`paramem/graph/relation_prep.py`) auto-prefixes
attribute keys with `has_` so the prompt should emit bare keys (`email`, not
`has_email`).

### Calibration loop

Don't iterate prompts blindly. The calibration tool
(`scripts/dev/calibrate_prompts.py`) probes each pipeline phase live
against the running server with operator-supplied variants, captures the
per-phase trace (`paramem/graph/phase_trace.py`), and renders a
baseline-vs-candidate diff per phase. Workflow:

1. Drop a `calib_<original>.txt` variant into the calibration prompt
   directory, `paths.calibration/prompts/` (default
   `data/ha/calibration/prompts/`). Variants live there, never beside the
   production prompts under `configs/prompts/` — a run names the variant
   and the server resolves it strictly from that directory, so a typo is
   refused rather than silently falling back to the shipped prompt.
2. Run `python scripts/dev/calibrate_prompts.py --input <fixture>
   --baseline auto --stop-phase <phase>` (use `--stop-phase` to skip
   downstream phases when iterating on early stages — saves compute at
   ~50–70 s per skipped phase).
3. Read the per-phase diff in stdout (raw output deltas, parsed-summary
   changes per phase). Every run's full result — the response, plus anything
   the production code emitted while the run executed — is written by the
   server under `paths.calibration/artifacts/<run>/`, and each response
   names its own `artifact_dir`. The client keeps no copy: it writes a
   `runs.json` index of which run covered which chunk, plus the
   baseline/variance comparisons it computes across runs. `--seed-from`
   reads a recorded run back through that index, so a run's output can be
   posted back as the next run's seed.
4. Promote the variant to production only when the per-phase diff confirms
   the targeted improvement without regressions on other phases.

The calibration endpoint is gated by
`consolidation.calibrate_endpoint_enabled` in `configs/server.yaml`
(default OFF — it loads the live model and would race against scheduled
consolidation in production).

`transcript` sent to any `/calibrate/*` endpoint MUST be the turn-marked
production surface (`[user] <text>` / `[assistant] <text>`, rendered by
`SessionBuffer._format_turns`) — every prompt's few-shots are calibrated
on it, and a bare, unmarked transcript puts the model off-distribution
(the endpoints reject unmarked input with HTTP 400). `calibrate_prompts.py`
renders this automatically; a manual `curl` call must supply it explicitly.

### Editing checklist

Before editing any file under `configs/prompts/`:

1. **Pick a measurable target.** What signal in the per-phase trace are you
   trying to move? "Alex = `person` at `local_extract`", "contact attrs
   on the speaker entity at `local_extract`", "no invented entity_types at
   `local_extract`" — single-sentence targets that succeed.
2. **Write the variant as a `calib_<original>.txt` file.** Don't edit the
   production prompt directly until calibration confirms the change.
3. **Run the calibration probe.** Read the per-phase diff. If the targeted
   phase moved correctly and downstream phases didn't regress, promote.
4. **Don't add a verbatim taxonomy slot or a long prose rule** unless a
   per-phase calibration measurement justifies it. The empirical record is
   that they make Mistral 7B worse, not better.

The phase-trace and calibration-loop machinery is documented inline in
`paramem/graph/phase_trace.py` and `paramem/server/calibrate.py`.
