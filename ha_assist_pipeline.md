# Plan: F5.1 Home Assistant Assist Pipeline Integration

## Goal

Build a custom Home Assistant conversation agent that uses ParaMem as its memory
backend. The agent processes conversations through the indexed key pipeline,
consolidates knowledge during idle time, and answers from parametric memory when
relevant. This is the first real-world validation of ParaMem outside synthetic
benchmarks.

## Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│  Home Assistant                                            │
│                                                            │
│   Voice Satellite ──→ STT ──→ Conversation Agent ──→ TTS   │
│                                      │                     │
│                            ┌─────────┴───────────┐         │
│                            │ ParaMem HA Plugin   │         │
│                            │  (custom_component) │         │
│                            └─────────┬───────────┘         │
└──────────────────────────────────────┼─────────────────────┘
                                       │ REST/gRPC
                              ┌────────┴─────────┐
                              │  ParaMem Server  │
                              │  (standalone)    │
                              │                  │
                              │  ┌─ Model ─────┐ │
                              │  │ Base + LoRA │ │
                              │  └─────────────┘ │
                              │  ┌─ Registry ──┐ │
                              │  │ Keys + Meta │ │
                              │  └─────────────┘ │
                              │  ┌─ Graph ─────┐ │
                              │  │ NetworkX    │ │
                              │  └─────────────┘ │
                              └──────────────────┘
```

## Architecture Decision: External Service (DECIDED)

ParaMem runs as a standalone long-running daemon on a NAS or laptop, separate
from HA. The HA custom component is a thin REST client (~50 lines).

**Deployment model:**
- ParaMem server runs persistently, keeps model loaded in VRAM, serves inference
  over the network. Target hardware: NAS with GPU or laptop.
- Consolidation (training) runs once per day outside active hours — no
  inference/training conflict to manage.
- HA custom component is a lightweight HTTP client. No GPU deps in HA's env.

**Why external service?**
- Clean separation: HA handles voice/UI, ParaMem handles memory/inference.
- Independent lifecycle: restart ParaMem without restarting HA and vice versa.
- Can run on a different machine from HA (GPU box on the network).
- Dependency isolation (ParaMem conda env stays untouched).
- Crash isolation: CUDA/OOM errors don't take down HA.
- Once-per-day training eliminates the need for GPU resource locking.

---

## Component 1: ParaMem Server

A lightweight HTTP/gRPC server wrapping the existing ParaMem pipeline. Runs in
the `paramem` conda env alongside the existing codebase.

### API Endpoints

```
POST /chat          — Handle a conversation turn
POST /consolidate   — Trigger idle-time consolidation
GET  /status        — Server health, model loaded, keys count, last consolidation
GET  /memory/facts  — List all stored facts (for debugging/admin)
POST /memory/forget — Remove specific facts (privacy)
```

### /chat Request/Response

```json
// Request
{
  "text": "What's my favorite restaurant?",
  "conversation_id": "abc123",
  "history": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}

// Response
{
  "text": "Your favorite restaurant is Osteria da Mario.",
  "latency_ms": 1200
}
```

### Internal Flow (per /chat request)

```
1. Receive user text + conversation history
2. Check for temporal reference ("yesterday", "last week", etc.)
   ├─ Temporal query:
   │   a. Parse time reference → absolute date range
   │   b. Query registry for keys with last_seen_at in range
   │   c. Probe only those specific keys
   │   d. Format prompt with recalled facts as context
   │   e. Generate answer
   │   └─ Return response
   └─ Standard query:
       a. Format as chat prompt (with conversation history)
       b. Generate answer with adapter active
       c. Facts are in the weights — model answers directly
       └─ Return response
3. Append turn to session transcript buffer
```

The adapter is always active. Personal facts are in the weights — no key
enumeration or fact reconstruction needed for standard queries. The model just
answers.

The recall+reason path (probe specific keys → feed as context) is only needed
for temporal queries where the registry serves as an index: "What did we discuss
yesterday?" → parse date → filter registry by `last_seen_at` → probe matching
keys → answer from those facts.

### DISCUSSION POINT 2: Temporal query detection (SIMPLIFIED)

Temporal queries need to be detected to trigger registry-based key lookup.
Simple keyword/pattern matching for time references ("yesterday", "last week",
"on Monday", "this morning") should suffice — these are syntactically distinct.
HA's Assist pipeline already handles date parsing for device commands, which we
can reuse.

### DISCUSSION POINT 3: Latency budget for voice UX

What's the acceptable response latency? For voice assistants, >3s feels slow.

| Query type | Inference path | Est. latency |
|-----------|---------------|-------------|
| Standard (facts in weights) | Direct generation | ~1-2s |
| Temporal (registry lookup) | Parse date + probe N keys + generation | ~2-4s |

The question is whether 1-2s generation time on Mistral 7B (NF4) is fast enough
for voice UX, or whether we need a smaller/faster model for the inference path.

---

## Component 2: Idle-Time Consolidation

After each conversation ends, the consolidation loop runs in the background.
This is the "sleep consolidation" metaphor — the agent processes the day's
conversations and strengthens its memories.

### Trigger Conditions (v1: once-per-day)

- **Scheduled:** Daily consolidation at a configured hour (default: 2am).
  Processes all session transcripts accumulated since last consolidation.
- **Manual trigger:** `POST /consolidate` for testing/admin.
- **Future:** Idle-time trigger (no message for N min) as a refinement.

### Consolidation Pipeline

```
1. Extract transcript from session buffer
2. extract_graph(model, tokenizer, transcript, session_id)
3. Merge with cumulative knowledge graph (GraphMerger)
4. generate_qa_from_relations(relations, model, tokenizer)
5. assign_keys(qa_pairs, start_index=next_available_key)
6. build_enriched_registry(keyed_pairs, session_id, existing_registry)
7. Handle contradictions (graph predicate normalization)
8. format_indexed_training(all_active_keyed_pairs, tokenizer)
9. Reinitialize adapter: model.delete_adapter("episodic") + create_adapter()
10. train_adapter(model, tokenizer, dataset, "episodic", ...)
11. save_adapter + save_registry + save_graph
12. Rebuild fact cache (probe_all_keys on new adapter)
```

### Training/inference conflict (RESOLVED)

Consolidation runs once per day outside active hours (default: 2am). No
inference/training conflict in v1. The server returns a status flag via
`GET /status` so the HA plugin knows if consolidation is in progress — edge
case handling only (e.g., user talks to the assistant at 2am).

### DISCUSSION POINT 5: Incremental vs. full retrain per consolidation

Two approaches for consolidation:

**(a) Full retrain:** Delete adapter, create fresh, train on ALL active keyed
pairs. Clean slate every time. Validated in Tests 1-2. At 100 keys: ~50-70 min
(Gemma), ~30-50 min (Mistral).

**(b) Incremental retrain:** Keep existing adapter, add only new keys, retrain
on new + existing. Faster per session but accumulates adapter drift. Test 2b
showed this works for contradictions at 16 keys. Unvalidated at 100+.

The spec doesn't mandate one approach. Full retrain is simpler and proven.
Incremental is faster but riskier. Which do you prefer for v1?

---

## Component 3: HA Custom Component (thin client)

A minimal HA integration that forwards conversation turns to the ParaMem server.

### File Structure

```
custom_components/paramem/
├── __init__.py          # Setup, config entry
├── manifest.json        # HA integration metadata
├── conversation.py      # ConversationEntity implementation
├── config_flow.py       # UI configuration (server URL, etc.)
├── const.py             # Constants
└── strings.json         # UI strings
```

### ConversationEntity Implementation

```python
class ParaMemConversationEntity(ConversationEntity):

    async def _async_prepare(self, user_input):
        """Pre-check: is the ParaMem server available?"""
        # Ping /status, warn if consolidating

    async def _async_handle_message(self, user_input, chat_log):
        """Forward to ParaMem server, return response."""
        history = self._extract_history(chat_log)
        response = await self.hass.async_add_executor_job(
            self._call_paramem_server,
            user_input.text,
            user_input.conversation_id,
            history,
        )
        return ConversationResult(response=response["text"])

    def _call_paramem_server(self, text, conversation_id, history):
        """Sync HTTP call to ParaMem server."""
        return requests.post(
            f"{self.server_url}/chat",
            json={"text": text, "conversation_id": conversation_id, "history": history},
            timeout=30,
        ).json()
```

### Configuration

Configurable via HA UI (config flow):
- `server_url`: ParaMem server address (default: `http://localhost:8420`)
- `timeout`: Max response wait time (default: 30s)
- `voice_mode`: Enable concise voice-optimized responses (default: true)

---

## Component 4: Voice-First Response Shaping

### System Prompt (voice mode)

```
You are a personal memory assistant. Answer concisely in 1-2 spoken sentences.
Do not use markdown, lists, or structured formatting.
Do not say "based on my records" or "according to my memory."
Speak naturally as if you simply remember.
```

### Agent Role: Privacy-Gated Hybrid (DECIDED)

The local model is the privacy gatekeeper. It sees everything; the cloud sees
nothing personal.

**Flow:**
```
User query
  → Local model (adapter active, personal facts in weights)
  → Can I answer this well?
    ├─ YES → respond directly (nothing leaves device)
    └─ NO → rewrite query, resolving personal references:
            "Plan dinner at my favorite restaurant for my wife's birthday"
            → "Plan a birthday dinner at Osteria da Mario on March 15"
            → Forward rewritten query to cloud SOTA model
            → Forward cloud response VERBATIM to user (no postprocessing)
```

**Local-first with cloud escalation (MoE-style):**

```
User query → Local model (always, adapter active)
  ├─ Knows the answer → respond directly, done
  └─ Doesn't know → emit [ESCALATE] tag
       → Server forwards query to SOTA API
       → SOTA response returned verbatim to user
```

- **Local model** is always the first and only responder for personal facts.
- **Cloud model** is a fallback for general knowledge, called only when the
  local model signals it cannot answer.
- **No routing logic, no session switching, no conversation state management
  across models.** One decision point: can I answer this or not?

**Implementation:**
- System prompt instructs the local model: "If you cannot answer a question
  from your knowledge, respond ONLY with `[ESCALATE]` followed by the
  question to forward."
- The ParaMem server detects the `[ESCALATE]` tag in the model output,
  strips it, calls the cloud API with the forwarded query, and returns
  the cloud response directly to the user.
- If no cloud endpoint is configured, the local model's response is used
  as-is (fully offline mode).

**Critical rules:**
- Cloud responses are returned verbatim — no local postprocessing.
- The forwarded query should not contain personal information. The local
  model is instructed to only escalate questions it can't answer — by
  definition these are non-personal queries.
- If the local model is uncertain whether it knows the answer, it should
  attempt to answer rather than escalate. Err on the side of local.

**Privacy guarantees:**
- Local model always sees the query first
- Cloud only receives queries the local model could not answer (general
  knowledge by definition — personal facts are in the adapter weights)
- Full memory store never leaves the device
- Fully functional without cloud (graceful degradation)

**Configuration:**
- `cloud_endpoint`: URL of SOTA model API (optional — fully local if unset)
- `cloud_model`: Model identifier (e.g., "gpt-4o", "claude-sonnet")
- `cloud_api_key`: API key for cloud service
- `escalation_prompt`: System prompt that instructs the model on self-assessment
  and query rewriting

---

## Component 5: State Persistence

### Persistent Artifacts (survive restarts)

| Artifact | Location | Purpose |
|----------|----------|---------|
| Base model | HuggingFace cache | Unchanged, cached |
| LoRA adapter | `data/ha/adapters/episodic/` | Trained weights |
| SimHash registry | `data/ha/registry.json` | Key verification + temporal metadata |
| Knowledge graph | `data/ha/graph.json` | Cumulative entity/relation store |
| Fact cache | In-memory (rebuilt on start) | Cached reconstructed facts |
| Session buffer | `data/ha/sessions/` | Pending transcripts awaiting consolidation |
| Server config | `data/ha/config.yaml` | Model choice, training params, etc. |

### Startup Sequence

```
1. Load base model + quantize (30-60s)
2. Load LoRA adapter from disk
3. Load registry from disk
4. Reconstruct all facts (probe_all_keys) → populate fact cache
5. Start HTTP server, accept requests
```

### DISCUSSION POINT 7: Startup time

Loading Gemma 2 9B with NF4 quantization + CPU offload takes 30-60s. Probing
100 keys adds another 30-60s. Total cold start: ~1-2 min.

This means after a reboot or ParaMem restart, there's a ~1-2 min window where
the agent is unavailable. Is this acceptable? Options:
- Accept it (reboot is rare)
- Serve from fact cache file (persisted from last probe) during startup
- Use a lighter model (Mistral 7B loads in ~20s, or a 3B model in ~10s)

---

## Model Selection (DECIDED: configurable, default Mistral 7B)

The server model is a config option. The architecture is already model-agnostic
(AD-1): `load_base_model` takes a `ModelConfig`, adapters are created on top.
Switching models means changing one config value — the pipeline, adapter
management, and server code are identical.

**Default:** Mistral 7B Instruct v0.3 (fast, clean QA generation, no CPU offload)
**Tested:** Qwen 2.5 3B (lightweight, good for NAS), Gemma 2 9B (highest recall)

```yaml
# server config
model: mistral  # or: qwen3b, gemma
```

**Important:** Adapters are model-specific. Switching models requires retraining
(or migration via F5.3). The server validates that the loaded adapter matches
the configured model.

Model choice will evolve with next-gen releases. The modular architecture
ensures we can swap in future models without pipeline changes.

---

## Implementation Phases

### Phase 5.1a: ParaMem Server (MVP) — IMPLEMENTED (2026-03-17)
1. ✅ HTTP server with `/chat`, `/consolidate`, `/status` endpoints (`paramem/server/app.py`)
2. ✅ Model loading on startup with adapter auto-load (`paramem/server/app.py` lifespan)
3. ✅ Chat inference with adapter active — two paths: standard (facts in weights) and temporal (registry-based key lookup) (`paramem/server/inference.py`)
4. ✅ `[ESCALATE]` detection → cloud forwarding, query only, no history (`paramem/server/escalation.py`)
5. ✅ Session transcript buffering to JSONL on disk (`paramem/server/session_buffer.py`)
6. ✅ Server config with model selection, cloud, voice prompt (`paramem/server/config.py`, `configs/server.yaml`)
7. ✅ 19 unit tests passing
8. ⬜ GPU smoke test (start server with real model, POST /chat, verify response)

### Phase 5.1b: Consolidation Loop — IMPLEMENTED (code), NOT TESTED
1. ✅ Full extract → QA → train pipeline wrapping existing modules (`paramem/server/consolidation.py`)
2. ✅ Manual trigger via `POST /consolidate`
3. ✅ Daily scheduled consolidation (configurable hour, default 2am)
4. ✅ Registry + graph + adapter persistence
5. ⬜ GPU integration test (buffer a session, trigger consolidation, verify adapter retrained)

### Phase 5.1c: HA Custom Component — IMPLEMENTED (2026-03-18)

Files: `custom_components/paramem/`

| File | Purpose |
|------|---------|
| `__init__.py` | Entry setup, platform forwarding |
| `conversation.py` | `ParaMemConversationEntity` — async HTTP client to `/chat` |
| `config_flow.py` | `ParaMemConfigFlow` — server URL + timeout, connectivity test via `/status` |
| `manifest.json` | Integration metadata, depends on `conversation` |
| `strings.json` | UI labels and error messages |
| `const.py` | `DOMAIN`, defaults |

1. ✅ Thin REST client forwarding to ParaMem server (`conversation.py`)
2. ✅ Config flow with server URL and timeout, validates connectivity (`config_flow.py`)
3. ✅ Voice-mode system prompt (inherited from server config — no duplication)
4. ✅ History extraction from HA `ChatLog` → server `/chat` history format
5. ✅ Graceful error handling (server unreachable, timeout)
6. ✅ 12 unit tests passing (`tests/test_ha_component.py`)
7. ⬜ Integration test with real HA instance

### Phase 5.1d: Temporal Queries — IMPLEMENTED (2026-03-18)

Two-path inference in `paramem/server/inference.py`:
- **Standard path:** adapter active, facts in weights, model answers directly.
- **Temporal path:** detect time reference → resolve to date range → filter
  registry by `last_seen_at` → probe matching keys → feed recalled facts as
  context → generate answer.

| Module | Purpose |
|--------|---------|
| `paramem/server/temporal.py` | `detect_temporal_query()` — keyword pattern matching for 20+ temporal expressions (yesterday, last week, N days ago, day names, etc.) → `(start_date, end_date)` |
| `paramem/server/temporal.py` | `filter_registry_by_date()` — scans enriched registry for active keys with `last_seen_at` in range |
| `paramem/server/inference.py` | `_handle_temporal_query()` — probes matching keys, builds augmented prompt with recalled facts, generates answer |

1. ✅ Keyword pattern matching for 20+ time references → absolute date ranges
2. ✅ Registry date filtering by `last_seen_at` (active keys only)
3. ✅ Inference path: detect → filter → probe → context-augmented generation
4. ✅ Falls back to standard path if no keys match the date range
5. ✅ 24 unit tests passing (`tests/test_temporal.py`)
6. ⬜ GPU integration test (end-to-end temporal query with trained adapter)

### Phase 5.1e: Polish — PENDING
1. ⬜ Error handling, reconnection, graceful degradation
2. ⬜ Conversation end detection tuning
3. ⬜ Latency optimization (model warmup, response streaming)
4. ⬜ Admin UI for memory inspection and deletion

### Test Coverage Summary (2026-03-18)

| Test file | Tests | Scope |
|-----------|-------|-------|
| `tests/test_server.py` | 19 | Server config, escalation, session buffer |
| `tests/test_ha_component.py` | 12 | Manifest, config flow, history extraction |
| `tests/test_temporal.py` | 24 | Temporal detection (all patterns), registry filtering |
| Existing test suite | 243 | Core pipeline, training, evaluation |
| **Total** | **298** | **All passing** |

---

## Discussion Points

### Resolved

| # | Topic | Decision | Rationale |
|---|-------|----------|-----------|
| 1 | Architecture | External service | GPU workload too heavy for HA process; runs on NAS/laptop |
| 3 | Latency budget | No concern | Grok STT is fast; ParaMem inference ~1-2s; within voice UX norms |
| 6 | Agent role | Local-first + cloud escalation | MoE-style: local answers personal, escalates to SOTA when it can't answer; cloud response verbatim |
| 8 | Model choice | Configurable, default Mistral 7B | Model-agnostic architecture; also test Qwen 3B; adapters are model-specific |
| 4 | Training blocks inference | Once-per-day consolidation | Runs outside active hours; no conflict to manage |
| 5 | Incremental vs. full retrain | Full retrain for v1 | Once-per-day removes speed pressure; proven at 100 keys |
| 7 | Startup time | Accept (~1-2 min) | Server runs persistently; cold starts only after reboot |

### Deferred (implement and tune during development)

| # | Topic | Approach |
|---|-------|----------|
| 2 | Temporal query detection | Keyword pattern matching for time references; reuse HA date parsing |

---

## Dependencies

- **New:** FastAPI or Flask (HTTP server), possibly gRPC
- **Existing:** All ParaMem pipeline code (paramem package)
- **HA side:** `homeassistant.components.conversation` (built-in)
- **No new ML dependencies** — uses existing PyTorch/PEFT/bitsandbytes stack
