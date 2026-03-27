# F5.2 Architecture: Cloud Escalation and Tool Use

## Chosen Tech Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Cloud adapters** | httpx (async) | Already in use by HF; async matches FastAPI server |
| **HA communication** | HA REST API + `return_response: true` | Simpler than WebSocket; script return values supported since HA 2024.1 |
| **HA auth** | Long-lived access token | Standard HA mechanism; no expiry; env var only (never inline config) |
| **Tool definitions** | JSON Schema (internal) | All providers use JSON Schema for parameters; adapters translate structure only |
| **Config format** | YAML (server.yaml extension) | Consistent with existing config; env var interpolation for secrets |
| **Tool config** | tools.yaml + HA auto-discovery | Auto-discovery for HA services; manual YAML for cloud-native and overrides |

## Alternatives Considered

### HA Communication

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| REST API with `return_response` | Simple HTTP, synchronous, script return values | Requires HA 2024.1+ | **Chosen** — simplest, sufficient |
| WebSocket API | Most capable, streaming | Connection management overhead, complex | Skip — overkill for occasional tool calls |
| Webhooks | No auth needed, event-driven | No return values, fire-and-forget only | Skip — cannot return tool results |
| Conversation API (`/api/conversation/process`) | Built-in NLU | Redundant — we already have a model reasoning | Skip — wrong abstraction level |

### Provider Abstraction

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Adapter pattern (base class + per-provider) | Clean separation, extensible | More files | **Chosen** — each provider needs different request/response handling |
| OpenAI-compatible only | Groq, Mistral, ollama all compatible | Anthropic and Google are not | Skip — locks out major providers |
| LiteLLM / unified proxy | One library handles all providers | Heavy dependency, version coupling | Skip — we need ~100 lines per adapter, not a framework |

### Routing

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| Graph entity check (server-side) | Milliseconds, no inference, deterministic | Misses rephrased references | **Chosen** — fast, private, good enough |
| Model-side `[ESCALATE]` tag only | Model decides autonomy | Unreliable — Mistral bad at self-assessment | Kept as fallback only |
| Lightweight classifier | Better accuracy | Extra inference step, latency | Skip — graph check is sufficient |
| Cloud model routes via `recall_memory` tool | Cloud decides when to use memory | Leaks query to cloud, privacy violation | **Rejected** — non-negotiable |

### Tool Execution Split

| Option | Pros | Cons | Decision |
|--------|------|------|----------|
| All tools via HA | Single execution path, HA controls everything | Latency for cloud-native tools; unnecessary proxy | Skip |
| All tools via cloud model | Simple — cloud handles everything | HA device control needs HA access | Skip |
| Configurable split (HA + cloud-native) | Best of both: HA for devices, cloud for search | Two execution paths | **Chosen** — matches reality |

## Key Architectural Decisions

### AD-F52-1: Server Routes, Not Model

The ParaMem server decides whether a query goes to the memory agent or the
general agent. The model does not make this decision.

Routing is a graph entity lookup (milliseconds, in-RAM). If any entity in
the query matches the knowledge graph, the query goes to the local memory
agent. Otherwise, it goes to the cloud agent (if configured) or falls back
to local.

When the graph check finds no entity match and a cloud agent is configured,
the query goes **directly to cloud** — no local inference. This avoids
double inference (Mistral + cloud) for every non-personal query.

The `[ESCALATE]` tag is a secondary mechanism only: if the graph falsely
routes to memory (entity match but the model cannot answer), Mistral emits
`[ESCALATE]` and the server forwards to cloud. This is the only case where
both models run for a single query.

Rationale: Local models are unreliable at self-assessment. Server-side routing
is deterministic, fast, and doesn't depend on model capability. Direct cloud
routing eliminates unnecessary Mistral inference for the majority of
non-personal queries.

### AD-F52-2: Privacy Model

Privacy protection operates at two levels:

**Structurally enforced (5.2a):**
- Queries matching known graph entities are routed to the local model only
- The cloud agent receives no conversation history (stateless)
- Adapter-stored knowledge (the 27MB LoRA weights) never leaves the device
- The cloud agent's `call()` method is never invoked for entity-matched queries

**Not yet enforced (5.2c, open):**
- Implicit personal context in queries without named entities ("what should
  I get my wife for her birthday?") bypasses entity matching and routes to
  cloud. The query itself leaks family context.
- Mixed queries where the local model resolves a personal reference and the
  resolved value goes to cloud ("book at Trattoria Luna" from "my favorite
  restaurant").

The structural protections are verified by an integration test. The implicit
leakage vectors are tracked as security concerns and addressed in Phase 5.2c
with an explicit PII sanitization module.

This is honest about what is and isn't protected. Named entities in the
knowledge graph are structurally protected. Everything else requires the
sanitization layer.

### AD-F52-3: HA REST API for Tool Execution

Tool calls that need HA (device control, HA scripts) use the REST API:

```
POST http://<HA_HOST>:8123/api/services/<domain>/<service>
Authorization: Bearer <TOKEN>
Content-Type: application/json

{"entity_id": "light.living_room", "brightness": 200}
```

For scripts that return data:

```
POST http://<HA_HOST>:8123/api/services/script/perform_tavily_search
{"variables": {"query": "weather tomorrow"}, "return_response": true}
```

Requires HA 2024.1+ for `return_response`. Long-lived access token for auth.

Rationale: Simplest approach that supports return values. WebSocket is
more capable but adds connection management complexity for a synchronous
use case. Webhooks cannot return results.

### AD-F52-4: Configurable Tool Split

Tools are categorized by where they execute:

- **HA-proxied**: device control, HA scripts, state queries, templates.
  Cloud model requests the tool call, ParaMem server executes via HA REST
  API, result flows back to cloud model.
- **Cloud-native**: web search, general knowledge tools. The cloud model
  executes these directly through its own tool-calling capability.

Configuration determines routing. A tool can be moved between categories
by changing config — no code changes needed.

Rationale: Device control must go through HA (it owns the devices). Web
search can go directly through the cloud provider's tool calling (faster,
no HA round-trip needed). Making this configurable lets users optimize
for their setup.

### AD-F52-5: Provider Adapter Pattern

```
paramem/server/
├── cloud/
│   ├── base.py              # CloudAgent ABC
│   ├── openai_compat.py     # OpenAI, Groq, Mistral API, ollama
│   ├── anthropic.py         # Claude
│   └── google.py            # Gemini
├── tools/
│   ├── ha_client.py         # HA REST API client
│   ├── registry.py          # Tool definition loading + merging
│   └── executor.py          # Agentic loop: tool call → execute → continue
```

Each adapter implements:
- `format_tools(tools) → provider_format`
- `call(messages, tools) → response`
- `extract_tool_calls(response) → list[ToolCall]`
- `format_tool_result(tool_call_id, result) → message`

Standard tool definitions (JSON Schema) are the internal format. Adapters
translate at call time.

### AD-F52-6: Tool Discovery + Manual Config

Two sources merged at startup:

1. **HA auto-discovery**: query HA for available services/scripts, generate
   tool definitions from HA's service schemas. **Default-deny:** if no
   allowlist is configured, auto-discovery produces an empty tool set.
   Sensitive domains (`alarm_control_panel`, `lock`, `person`,
   `device_tracker`) are excluded even if listed — require explicit override.

2. **Manual `tools.yaml`**: cloud-native tool definitions, overrides for
   auto-discovered tools, custom tools. Same format as the internal
   tool schema.

Auto-discovered tools can be disabled or overridden in `tools.yaml`. This
gives control without requiring manual definition of every HA service.

### AD-F52-7: Graceful Degradation

The system degrades progressively:

| Loss | Impact | Recovery |
|------|--------|----------|
| Cloud provider | No general agent, tool use, or complex reasoning | Local model handles all queries |
| HA connection | No device control or HA-proxied tools | Cloud model informed of tool failure; text response only |
| Both | Fully local operation (current F5.1 behavior) | Personal memory queries still work |

No hard dependencies. Every component is optional. The minimum viable
configuration is the local model alone (no config changes from F5.1).

### AD-F52-8: Stateless Cloud Agent

The cloud agent receives only the current query — no conversation history.
This prevents personal context from earlier turns leaking to the cloud.

For follow-up questions ("what about tomorrow?" after a weather query), the
local model handles continuity. It has full conversation history and can
reformulate a self-contained query before routing to cloud.

Rationale: Every prior conversation turn potentially contains personal data.
Sending history to cloud undermines the privacy model. The local model is
the only component that sees the full conversation.

### AD-F52-9: HA Client Connection Pooling

The HA REST API client uses `httpx.AsyncClient` with keep-alive connection
pooling. Each tool call in the agentic loop reuses the same TCP connection
to HA, eliminating per-request connection overhead (estimated 50-100ms
savings per round-trip).

The client is initialized at server startup with a health check against
`GET /api/`. HA version is checked to verify `return_response` support
(HA 2024.1+). Warnings are logged if HA is unreachable or below the
minimum version.

### AD-F52-10: Config Migration

The existing `CloudConfig` dataclass is replaced by the new `agents.general`
structure. For backward compatibility, the config loader accepts the
deprecated `cloud:` key and maps it to `agents.general`.

Env var interpolation (`${VAR_NAME}`) is added to the config loader.
Secrets (API keys, HA tokens) must use env vars — the loader warns if
a literal key-like string is detected in a secret field.

## Integration Points

| System | Protocol | Auth | Purpose |
|--------|----------|------|---------|
| HA conversation agent | WebSocket (`conversation.process`) | Long-lived access token | Primary escalation — full Groq pipeline with tools, prompts, room context |
| Cloud providers (Groq, etc.) | HTTPS REST | API key (per-provider) | Fallback escalation — direct cloud call when HA unavailable |
| Home Assistant (services) | HTTP REST + WebSocket | Long-lived access token | Tool auto-discovery, entity resolution, state queries, script execution |
| ParaMem local model | In-process | None | Memory agent inference (existing) |
| HA custom component | HTTP REST (existing `/chat`) | None (local network) | User queries in, responses out (unchanged) |

### Architecture Note (2026-03-27)

The primary escalation path was rearchitected from direct Groq API calls with
ParaMem-side tool execution to a single WebSocket call to HA's conversation
agent. This eliminates tool duplication, prompt duplication, and entity
resolution on the ParaMem side. HA is the control entity for models, prompts,
tools, and room context. ParaMem owns only memory.

The direct cloud API path is retained as fallback (with full tool
infrastructure) and reserved for future SOTA model routing where Groq+Llama
may not provide sufficient reasoning quality.

## Known Constraints and Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| PII leakage in escalated queries | Implicit personal context sent to cloud | Phase 5.2c: explicit sanitization module. Named entities protected structurally; implicit context is not. |
| HA REST API latency | Tool round-trips add 100-500ms each | Mitigated: primary path uses HA conversation.process (single hop, 0.6–2.2s). Direct cloud fallback retains 8s budget. |
| Provider API format drift | Adapter breaks on API update | Pin SDK versions; adapter tests with mock responses |
| Cloud rate limits | Degraded service during spikes | Fallback to local; exponential backoff |
| HA `return_response` version requirement | Won't work on old HA installs | Document HA 2024.1+ requirement; detect and warn at startup |
| Large tool responses | Cloud model context window overflow | Truncate/summarize before feeding back to cloud model |
| Auto-discovery exposes sensitive HA services | Cloud model told about alarm codes, locks, etc. | Default-deny: no allowlist = no tools. Sensitive domains blocked even if listed. |
| Config migration | Breaking change for existing `cloud:` config key | Deprecated alias maps `cloud:` → `agents.general` during transition |
| Double inference | Mistral + cloud both run for non-personal queries | Mitigated: cloud-only mode routes directly to HA agent. Local mode uses direct routing (no entity match → HA agent, no local inference). |
| Env var interpolation missing | API keys stored as literal `${VAR}` strings | Implement interpolation in config loader before 5.2a ships |
