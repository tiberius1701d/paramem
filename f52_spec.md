# F5.2 Spec: Cloud Escalation and Tool Use

> **Status (2026-04):** Sections FR-3 through FR-6 describe the original
> ParaMem-side tool executor and `tools.yaml` manual config. That path was
> removed in phase 5.2e — HA's conversation agent is now the single tool
> execution channel. The implemented routing is the dual-graph tri-path router
> in `paramem/server/router.py` (see `ha_assist_pipeline.md` for the live
> design). Config schema was renamed `agents.general` → `agents.sota`
> (`CloudAgentConfig`). The sections below are retained for historical context.

## Problem Statement

The ParaMem server currently runs Mistral 7B locally for all queries. This means:
- No tool use (device control, web search, weather, music)
- Limited reasoning capability compared to SOTA cloud models
- Users who had Groq/OpenAI-based HA conversation agents lost functionality when switching to ParaMem

F5.2 adds a configurable cloud agent alongside the local memory agent, restoring
tool use and general-purpose capabilities while preserving the core privacy guarantee:
personal knowledge never leaves the device.

## User Stories

1. **As a user**, I ask "turn on the living room lights" and the system controls
   the device — even though Mistral doesn't support tool calling well.

2. **As a user**, I ask "what's the weather?" and get a real-time answer from
   my HA weather sensor or a web search — not a hallucinated guess.

3. **As a user**, I ask "where do I live?" and get an answer from parametric
   memory — without that fact ever touching a cloud API.

4. **As a user**, I ask "play Queen on the office speaker" and Music Assistant
   plays the right artist on the right device.

5. **As a user**, I can choose my cloud provider (OpenAI, Anthropic, Groq,
   Google, or others) without changing my setup.

6. **As a user**, my system works fully offline when the cloud is unreachable —
   personal queries still work, tool queries gracefully degrade.

7. **As a community contributor**, I can add a new cloud provider by
   implementing a single adapter class.

## Functional Requirements

### FR-1: Two-Agent Routing

The server maintains two agent roles:

- **Memory agent** (local): current Mistral pipeline with adapter. Handles
  personal knowledge queries. Always available.
- **General agent** (cloud): configurable provider. Handles tool use, general
  knowledge, and complex reasoning.

Routing logic (server-side, not model-side):

```
Query arrives
  → Graph entity check (in-RAM, milliseconds)
  → Entity match found?
    ├─ YES → memory agent (local Mistral, adapter active)
    └─ NO  → HA conversation agent (WebSocket, primary)
              ├─ HA available → response via conversation.process
              └─ HA unavailable → SOTA cloud fallback (Anthropic/OpenAI/Google)
```

When no entity match is found and a cloud agent is configured, the query goes
**directly to cloud** — no local inference. This avoids double inference
(Mistral + cloud) for every non-personal query.

The `[ESCALATE]` tag is a secondary mechanism only: if the graph falsely
routes a query to memory (entity match but the model cannot answer), Mistral
emits `[ESCALATE]` and the server forwards to cloud. This is the only case
where both models run for a single query.

### FR-2: Provider Abstraction

A `CloudAgent` base class with provider-specific adapters (all implemented):

- **OpenAI-compatible** (OpenAI, Groq, Mistral API, local ollama) — `openai_compat.py`
- **Anthropic** (Claude) — `anthropic.py` (optional dependency: `pip install paramem[anthropic]`)
- **Google** (Gemini) — `google.py` (optional dependency: `pip install paramem[google]`)

Each adapter handles:
- Authentication
- Message formatting
- Tool definition translation (standard → provider format)
- Tool call extraction from responses
- Response parsing

All providers use JSON Schema for tool parameter definitions. The differences
are structural (field naming, nesting) not semantic.

### FR-3: Tool Use

Tools are split into two categories by configuration:

**HA-proxied tools** — executed via HA REST API:
- Device control (`light.turn_on`, `media_player.*`, etc.)
- HA scripts with return values (`return_response: true`)
- State queries (`GET /api/states/<entity_id>`)
- Template rendering (`POST /api/template`)

**Cloud-native tools** — executed directly by the cloud model:
- Web search (Tavily, SerpAPI, etc. via cloud provider's tool calling)
- Any tool the cloud provider supports natively

Configuration determines which tools route where. A tool defined in both
HA and cloud config uses the HA path by default (closer to the data source).

### FR-4: HA Tool Execution

ParaMem server calls HA REST API to execute tools:

```
POST /api/services/<domain>/<service>
Authorization: Bearer <LONG_LIVED_TOKEN>
{"entity_id": "...", ...}
```

For scripts that return data:

```
POST /api/services/script/<name>
{"variables": {"query": "..."}, "return_response": true}
```

Authentication via HA long-lived access token. **Must be stored as env var
only** (e.g., `${HA_TOKEN}`), never inline in YAML config files.

### FR-5: Agentic Loop

When the cloud model requests a tool call:

1. Parse tool call from provider response
2. Classify: HA-proxied or cloud-native?
3. If HA-proxied: execute via HA REST API, collect result
4. Send result back to cloud model
5. Repeat until cloud model returns final text (max 5 rounds)

The loop is synchronous from the user's perspective — one query in, one
response out.

**Latency budget:** Total agentic loop must complete within 8 seconds
(voice UX threshold). Per-tool timeout: 3 seconds. If exceeded, the tool
returns an error to the cloud model rather than blocking indefinitely.
Connection pooling (`httpx.AsyncClient` with keep-alive) for HA calls
to avoid per-request TCP overhead.

### FR-6: Tool Discovery

Two sources for tool definitions, merged at startup:

**Auto-discovery from HA:**
- Query HA REST API for available services and scripts
- Filter to a configurable allowlist (**default-deny**: if no allowlist is
  configured, auto-discovery produces an empty tool set)
- Generate tool definitions from HA service schemas
- Sensitive domains (`alarm_control_panel`, `lock`, `person`, `device_tracker`)
  are excluded even if listed in the allowlist — require explicit override

**Superseded:** The earlier `tools.yaml` manual configuration path has been
removed. HA's conversation agent is now the single tool-execution path, so tool
definitions and entity resolution live entirely on the HA side. ParaMem no
longer maintains a tool registry or manual override file.

### FR-7: Configuration

```yaml
# Current schema (see configs/server.yaml for the canonical example).
agents:
  sota:
    enabled: true
    provider: anthropic         # openai | anthropic | google
    model: claude-sonnet-4-6
    api_key: ${ANTHROPIC_API_KEY}
    endpoint: ""                # optional custom endpoint
  sota_providers:               # registry of alternates for direct routing
    anthropic: {...}
    openai: {...}
    google: {...}
  ha_agent_id: conversation.groq  # HA conversation agent for tool use

tools:
  ha:
    url: ${HA_URL}
    token: ${HA_TOKEN}          # env var only — never inline
    auto_discover: true
    supported_languages: [en, de, fr, es]
    allowlist:
      - light.*
      - switch.*
      - script.*
      - climate.*
      - media_player.*
  tool_timeout_seconds: 10      # HA call timeout (VPN latency budget)
```

**Note:** The earlier `tools.yaml` / `tool_registry` / `tool_executor` /
`cloud_native` scheme has been removed. Tool execution is HA-only — the HA
conversation agent owns all tool definitions and entity resolution. ParaMem
does not maintain a separate tool registry.

**Config loader requirement:** `load_server_config` must implement env var
interpolation (`${VAR_NAME}` → `os.environ["VAR_NAME"]`). The existing
loader does plain `yaml.safe_load` — this needs extending.

**Migration:** The `CloudConfig` dataclass has been replaced by `CloudAgentConfig`
in the `agents.sota` structure. The `cloud:` key in existing configs is
accepted as a deprecated alias during the transition.

### FR-8: Graceful Degradation

| Condition | Behavior |
|-----------|----------|
| Cloud unreachable | Fall back to local memory agent |
| HA unreachable | Tool calls fail, cloud model informed, text response only |
| No cloud configured | Fully local operation (current behavior) |
| No tools configured | Cloud handles text queries only, no tool calling |

### FR-9: Progressive Capability

The server adapts to what is configured:

- **Nothing configured** → local-only, current behavior
- **Cloud agent only** → routing + cloud escalation, no tools
- **Cloud + HA tools** → full tool use via HA
- **Cloud + HA tools + cloud-native tools** → hybrid tool execution
- **Local agent for both roles** → both memory and general on same local model

## Out of Scope

- **STT/TTS** — handled by HA voice pipeline
- **Conversation history to cloud** — cloud agent is stateless in 5.2a;
  local model handles follow-up reformulation
- **Cloud model training or fine-tuning** — cloud models are used as-is
- **HA custom component changes** — the HA component stays a thin REST client
- **Streaming responses** — buffered responses only
- **Model-side routing** — the server decides routing, not the model. Primary escalation is via HA WebSocket (`conversation.process`), with SOTA cloud providers (Anthropic, OpenAI, Google) as fallback

## Open Questions

### OQ-1: Conversation History to Cloud — DECIDED

**Decision: No history to cloud in 5.2a. Cloud agent is stateless.**

The cloud model receives only the current query — no conversation history.
This prevents personal context from earlier turns leaking to the cloud.

For follow-up questions ("what about tomorrow?" after a weather query), the
local model handles continuity — it has full conversation history and can
reformulate a self-contained query for the cloud if needed.

Richer history handling (sanitized history, selective forwarding) is deferred
to 5.2c alongside PII sanitization.

### OQ-2: PII Sanitization (Security) — PARTIALLY ADDRESSED

Pattern-based sanitizer implemented with three modes (off/warn/block).
Default: "block" — queries with personal patterns fall back to local model.

Remaining gaps (documented, accepted for now):
- Implicit personal references resolved by local model ("my favorite
  restaurant" → "Trattoria Luna") still contain personal data
- Numeric quantifiers beyond "two/three" not caught
- "my mother-in-law" and hyphenated relations not matched
- Cloud model [ESCALATE] exfiltration mitigated by block mode but not
  structurally prevented

### OQ-3: Tool Response Size — RESOLVED

Tool responses truncated at 2000 characters before passing to cloud model.
Uses `json.dumps` for structured serialization. Sufficient for current
tool set; may need per-tool limits for verbose APIs.

### OQ-4: HA Auto-Discovery Granularity — RESOLVED

Default-deny: no allowlist = no tools exposed. Sensitive domains
(`alarm_control_panel`, `lock`, `person`, `device_tracker`) blocked
even in allowlist. Enforced in both auto-discovery and tools.yaml paths.

## Implementation Phases

### Phase 5.2a: Provider Abstraction — COMPLETE

- ✅ `CloudAgent` base class + OpenAI-compatible adapter (Groq, ollama)
- ✅ Config schema: `agents.sota` (`CloudAgentConfig`) with deprecated `cloud:` alias
- ✅ Env var interpolation (`${VAR}`) in config loader
- ✅ Provider dispatch via `get_cloud_agent()` registry
- ✅ Direct cloud routing for no-entity-match (no double inference)
- ✅ Graceful fallback when cloud unreachable
- ✅ Privacy integration test: entity-matched queries never reach cloud
- ✅ 22 unit tests (config, adapter, registry, privacy routing)

### Phase 5.2b: Tool Use — COMPLETE

- ✅ HA REST API client with eager connection pooling (`httpx.Client`)
- ✅ HA health check at startup (`GET /api/`)
- ✅ Tool registry: auto-discovery (default-deny) + `tools.yaml`
- ✅ Sensitive domain blocking on both discovery and YAML paths
- ✅ Agentic loop: 8s total timeout, 3s per-tool, max 5 rounds
- ✅ Tool response truncation (2000 chars, `json.dumps`)
- ✅ `dict()` copy before mutation in `call_service`
- ✅ 16 unit tests (registry, HA client, executor, config)
- ✅ Anthropic adapter (`paramem/server/cloud/anthropic.py`)
- ✅ Google adapter (`paramem/server/cloud/google.py`)
- ✅ OpenAI adapter (`paramem/server/cloud/openai_compat.py`) — covers OpenAI, Groq, ollama
- ⬜ HA version check (deferred)

### Phase 5.2c: PII Sanitization — COMPLETE

- ✅ Pattern-based sanitizer (possessive, self-referential, personal claims)
- ✅ Three modes: off/warn/block (default: block)
- ✅ Mode validation in config (`__post_init__`)
- ✅ Unknown mode fails closed (treated as block)
- ✅ Wired into ALL cloud paths (direct, [ESCALATE], fallback, legacy)
- ✅ 21 unit tests (patterns, modes, edge cases)
- ⬜ Conversation history sanitization (deferred — stateless in 5.2a)

### Phase 5.2d: Polish — PARTIALLY COMPLETE

- ⬜ Documentation (setup guide, provider configs, tool authoring)
- ✅ HA auto-discovery testing with real HA instance (43 services, 238 entities)
- ✅ Error recovery and timeout tuning
- ✅ Performance profiling (0.6s weather, 1.2s script tools, 2.2s search)

### Phase 5.2e: HA Conversation Agent Rearchitecture — COMPLETE

The tool execution architecture was rearchitected after live testing revealed
that HA's internal automation engine handles tool execution, prompt rendering,
and entity resolution better than replicating it externally.

**Before:** ParaMem → Groq API → tool call → ParaMem → HA REST/WebSocket →
back to ParaMem → Groq API again. Two Groq round-trips minimum.

**After:** ParaMem → HA `conversation.process` via WebSocket → done. HA runs
the full Groq pipeline internally (J.A.R.V.I.S. prompt, tools, entity
resolution, room awareness). One hop.

- ✅ `HAClient.conversation_process()` — WebSocket call to HA conversation agent
- ✅ Primary escalation via `_escalate_to_ha_agent()` with direct cloud fallback
- ✅ Cloud-only mode routes all queries through HA conversation agent
- ✅ GPU startup guard: auto cloud-only when GPU is occupied
- ✅ `--cloud-only` CLI flag for explicit GPU-free startup
- ✅ Latency: 0.6–2.2s end-to-end (vs 2+ round-trips before)
- ✅ Server deployed via systemd, auto-reclaim timer for GPU

**Retained for future direct cloud path:**
- Tool registry with Extended OpenAI Conversation format parser
- `execute_script_ws()` for raw HA sequence execution
- `render_template()` for Jinja rendering via HA API
- Agentic loop executor (5 rounds, 8s timeout)
- `configs/tools.yaml` with 7 tool definitions

**Design principle:** ParaMem owns memory. HA owns everything else (models,
prompts, tools, entity resolution, room context). No duplication.

### Code Review Fixes Applied

Four parallel reviews (3 code + 1 security) identified 14 issues. All fixed:
- Legacy [ESCALATE] path sanitized
- `_probe_and_reason` fallback sanitized + passes tools
- Sanitizer default "block" (was "warn")
- Tool result message ordering fixed
- `tools.yaml` sensitive domain check
- HA client dict mutation, thread-safe init
- `json.dumps` for tool results
- Empty choices guard, error body redaction
- Mode validation, unknown mode fails closed
- `is_available()` supports keyless providers
- Removed `render_template`/`get_state` (SSTI risk)
- "where was I born" pattern added
