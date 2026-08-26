# Security

> **Work in progress — not a finished security story.** The encryption-at-rest machinery described in this document is partial by design. It raises the bar for specific *separation* scenarios (see §2) and does not attempt to defend against an attacker who gains operator-level access to the host.
>
> **Two operator-level paths to plaintext, both trivial:**
>
> 1. **The Python codebase is editable.** `paramem/` on disk is plain source. An attacker with write access to the installed package path — which the operator user has by definition — can neutralize encryption with a small source edit and restart the server. There is no code-signing, no bytecode integrity check, no TPM-backed attestation.
> 2. **Config-level data exfiltration via `debug: true`.** The operator can flip one line in `configs/server.yaml` and restart. From that point on, `debug: true` forces retention of every consolidated session's transcript regardless of `consolidation.retain_sessions`, and additionally writes per-cycle debug artifacts under `data/ha/debug/episodic/.../cycle_<N>/` (snapshot JSON files) — the per-session JSONLs under `data/ha/sessions/` are already plaintext on disk while pending, independent of this flag; see §4 carve-outs for the exact retention geometry. No code edit, no crypto break — just the legitimate debug path used against the intent of a Security-ON deployment. This is intentional behaviour for debugging (see §4 carve-outs); it is named here because an attacker with config-write access can use it as a data-extraction primitive. (The simulate-mode graph store — each tier's own written graph payload — is encrypted by default and respects `require_encryption`; it is NOT a debug carve-out.)
>
> Closing either gap is outside what this project can do alone; both require host-level integrity tooling (IMA/EVM on Linux, equivalent on Windows) plus operator discipline on config-write permissions.
>
> Statements in this document describe the *current* implementation, not a finished target.

ParaMem is a personal memory service that stores conversational and personal facts as weight deltas in a local LoRA adapter, plus a small set of on-disk metadata files (registry, knowledge graph, session queue, voice profiles). It runs on a single host under a single admin and is designed for home / edge deployment — not multi-tenant or server-farm use.

**Scope — what ParaMem provides vs. what the operator owns.** ParaMem ships the cryptographic primitives (age envelopes, two-identity daily + recovery key model, passphrase-wrapped on-disk key file) and the key-lifecycle tooling (`generate-key`, `rotate-daily`, `rotate-recovery`, `change-passphrase`, `restore`). Everything above that boundary — host integrity, runtime attestation, hardware-backed key storage, physical isolation of keys from data at rest, who owns which filesystem path, backup-storage separation, network scoping — is a **deployment-shape decision that belongs to the operator**. The operator chooses their own threat model: bare metal with a YubiKey recipient, containerised under a read-only rootfs, keys on removable media unmounted at rest, data on a NAS while keys stay local, single host with defaults accepted, or any other shape. ParaMem does not prescribe one; it provides the foundation that makes multiple shapes workable. This document names what the primitives protect against and what they don't, so operators can make informed trust decisions for their own deployment.

This document describes what ParaMem defends today, what it does not, the trust boundaries in the design, and the operator contract for running it as securely as the current implementation allows. It is a living document; the security posture will tighten as work packages from the hardening plan land.

## 1. Data handled

| Artifact | Content | Format |
|---|---|---|
| Adapter weight tensors | Personal facts, preferences, episodic memories | `.safetensors` — opaque numerical tensors |
| Indexed key registry | Key identifiers (active + withheld), SimHash fingerprints | JSON |
| Per-key bookkeeping | Speaker binding, relation type, reinforcement count, first-seen/last-seen timestamps | JSON |
| Cumulative knowledge graph | Entities, predicates, relations | JSON (NetworkX) |
| Session queue | Transcript + speaker binding awaiting consolidation | JSON (atomic temp-file + rename) |
| Session snapshot | RAM state at graceful shutdown | age-encrypted when a key is configured |
| Speaker profiles | Voice embeddings + disclosed names | JSON (biometric data — see §8) |
| Background trainer resume state | Epoch counter + checkpoint references | JSON |
| Adapter manifest sidecars | Payload content digest, key count, plus base-model SHA, tokenizer fingerprint, and LoRA shape for a trained tier only — a graph tier's manifest carries no weight fingerprints, since it has no weights to fingerprint | JSON |

Adapter weights are the dominant artifact by volume and sensitivity. They are numerical — not directly readable for facts — but also not probe-resistant. Under Security ON they are encrypted at rest as age envelopes alongside the JSON metadata. The indexed-recall path still requires the encrypted registry for key enumeration; weight encryption adds defense-in-depth against blob-copy attackers. See §4 and §8 for the full picture.

## 2. Threat model

**Trust assumption.** The admin / operator of the host is a trusted authority. ParaMem does not attempt to protect data from an attacker who has the operator's OS credentials, process-memory access, or write access to the installed Python package. The operator holds the daily passphrase and the daily-key file; these travel in the same trust domain as the data they protect. A hostile process running as the operator can read the decrypted store from RAM, modify `paramem/` source (e.g. replace the encryption routine with a pass-through), or read the wrapped daily key plus passphrase and decrypt at rest. None of these are defended against.

**What Security ON actually buys — the narrow, honest claim.** When the data directory is separated from the key material (decoupled from the running server), the data directory alone is not decryptable. Concretely, encryption at rest narrows the blast radius in exactly these separation scenarios:

- **Accidental cloud-sync of `data/ha/` alone** (OneDrive, iCloud, rsync to NAS) — data appears at the sync destination but is unreadable without the key material kept at `~/.config/paramem/` and `PARAMEM_DAILY_PASSPHRASE`.
- **Backup exfiltration** (a backup copy of the data directory without the config dir) — same story.
- **Filesystem read by a different OS user on the same host** (mode `0600` on `~/.config/paramem/daily_key.age` and `.env` enforced at startup).
- **Theft of a powered-off host IF the passphrase is not co-located** (depends on operator discipline — typically a weak defense because `.env` lives on the same disk).

**In scope beyond data-at-rest:**
- LAN-adjacent attackers sending unauthenticated requests — mitigated by the bearer-token auth layer (see §5).
- Prompt-injection attempts via voice input.
- Careless maintainers, accidental commits, screenshots of on-disk state.

**Out of scope — explicitly:**
- Any attacker with operator-user OS credentials, root, or process memory access.
- Anyone with write access to the installed Python package (can neutralize encryption in three lines of source).
- Nation-state adversaries.
- Supply-chain compromise of the Python runtime or pinned dependencies beyond version pinning.
- Multi-user isolation on the same host (ParaMem is a single-admin service).
- Side-channel attacks on the CPU or GPU during inference.

## 3. Trust boundaries

- **User voice → STT.** Raw audio arrives on a Wyoming protocol port. Transcript and speaker embedding cross into the server process.
- **Home Assistant ↔ ParaMem.** A thin HA custom component POSTs to the `/chat` endpoint over HTTP on the LAN, carrying a bearer token minted into the token store (typically an unattributed `chat`-scope token). Bearer-token authentication is a two-posture model (OFF / ON) governed entirely by whether a per-user token store is wired — see §5 for the full model. When auth is OFF the server accepts any LAN request, announced at startup as an explicit open posture, not a silent one.
- **ParaMem → Home Assistant.** `agents.ha_agent_id` **must name a LOCAL HA conversation agent** — the built-in `conversation.home_assistant`, or a self-hosted LLM agent on the operator's own hardware. This hop is scrubbed under `sanitization.scrub` like the cloud hop — on every turn, regardless of the personal verdict — and the reply is restored the same way; entity and area names Home Assistant itself registered stay readable in the text sent to it, so device control keeps working for person-named devices. It is the leg that stays reachable when the cloud leg is closed by `cloud.enabled: false` or by `cloud.allow_degraded_serving: false` during an outage. The local-agent requirement stands for a different reason than the payload's own scrubbing: configuring a cloud-backed HA agent (`conversation.groq`, `conversation.openai`, …) forwards the household's turns to a third party outside every ParaMem switch, no matter how well the payload itself is scrubbed. (Unrelated and unaffected: the HA custom component's own fallback `agent_id`, which exists to prevent recursive routing when ParaMem is HA's default conversation agent.)
- **ParaMem → cloud.** Sanitized queries may be sent to a configured cloud agent for escalation or cloud enrichment. This path is opt-in via config; nothing is sent without an active cloud configuration. Whether a query is personal — and therefore kept on-device — is decided by the intent classifier, supplemented by an encoder-based "is this about the speaker?" check with multilingual exemplars under `configs/personal_referent/` (English token-set fallback when the encoder isn't loaded). A query that does egress under an anonymizing `cloud_mode` has its in-scope values marked by a small local detection model and replaced with placeholders before it leaves the device; no model rewrites the outbound text. No speaker name or id is ever sent to the cloud system prompt (see Routing-time intent classifier below). Coverage scales with the exemplar files and the classifier; see §7 and §8 for the operator's responsibility and the residual risk.
- **Routing-time intent classifier (privacy property).** Deciding which route a query takes runs before any recall and never receives the speaker's identity: under `intent.mode: llm` (default) the local model classifies on the query's content alone, with nothing naming the speaker in its prompt; under `intent.mode: embeddings` the classifier match never receives speaker identity either. The only point where speaker identity reaches a model-facing prompt at all is the response-time local reasoning leg — and even there the speaker is referenced by an internal anonymous handle, never a display name, resolved to a name only when the reply is assembled. A remembered fact whose object is a personal name is ordinary memory content on the local leg. The cloud leg carries no identity line and no recalled facts at all.
- **Unidentified caller → relay.** A request reaches the relay path only when every resolution step — bound token, voice-embedding match, session history, anonymous-speaker promotion — has failed to yield a speaker; it is then served on a relay path (HA / cloud / local base model only) with no knowledge-store access, no history egress, and no consolidation while it stays unattributed (a later voice or name-disclosure claim can still attribute and consolidate it normally). A personal interrogative on this path is caught by a dedicated short-circuit and gets the canned no-identity response instead of cloud.
- **Adapter files at rest.** The on-disk artifacts listed in §1 live under the configured data directory. At-rest encryption is governed by the binary switch in §4.
- **Backup at rest.** Session snapshots and every other piece of infrastructure metadata follow the Security-ON/OFF contract in §4 — encrypted as age envelopes when the daily identity is loaded, and plaintext only when no key is configured.

## 4. Encryption at rest

ParaMem operates in one of two modes, governed by the loaded key material. There are no partial states.

### Security ON
`PARAMEM_DAILY_PASSPHRASE` is set AND `~/.config/paramem/daily_key.age` exists. When `~/.config/paramem/recovery.pub` is also present, every new write is multi-recipient (daily + recovery).

All infrastructure metadata — registry, graph, queue, snapshots, speaker profiles, backup artifacts — is age-encrypted on disk and decrypted only into process RAM on load, with the exception of the per-slot manifest sidecar (`meta.json`) — see the plaintext-by-design carve-outs below. The read path recognizes an encrypted file automatically and decrypts it in place; a plaintext file is read as-is. On startup the server logs one of:
```
SECURITY: ON (age daily identity loaded, recovery recipient available)
SECURITY: ON (age daily identity loaded, recovery recipient missing — run `paramem generate-key` to re-enable multi-recipient writes)
```

### Security OFF
No key material is loaded. All infrastructure metadata is plaintext on disk. This is a **documented operator opt-out**, not a gap. On startup the server logs:
```
SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)
```
and surfaces `encryption: off` on the `/status` endpoint. The server does not silently degrade between modes: if the daily identity is loaded but on-disk files are plaintext (or vice versa), startup refuses with an actionable message. This extends past startup: a key that is present but unusable (wrong passphrase, corrupt or tampered daily-key file) fails loud at the point of use — every write that would otherwise silently degrade to plaintext raises instead, rather than degrading only on the next restart.

### Fail-loud opt-in: `security.require_encryption`

The Security-OFF opt-out is the operator's choice. Deployments that want a misconfiguration to fail loud rather than silently land plaintext on disk can set `security.require_encryption: true` in `configs/server.yaml`. When set, the server refuses to start unless the daily identity is loadable AND actually unwraps — the startup gate performs the real unlock, not just a file-presence + env-var check, so a wrong passphrase or a corrupt daily-key file is caught at boot rather than at the first write. This is a uniform startup gate covering every feature that writes to disk (snapshots, checkpoint shards, backups, infrastructure metadata). Default is `false` (the AUTO-everywhere posture described above).

### Refusal cases

- age files on disk without the daily identity loaded → startup refused with a clear message pointing at `PARAMEM_DAILY_PASSPHRASE` + the daily-key file path.
- Plaintext files alongside age envelopes → startup refused; reconcile the store before restart.
- Plaintext files while the daily identity is loaded → startup refused; migrate the store or unset the passphrase.

For the migration and reset runbooks when a refusal occurs, see [DEPLOYMENT.md — Encryption & recovery operations](DEPLOYMENT.md#encryption--recovery-operations).

### Plaintext-by-design carve-outs

Some on-disk artifacts are intentionally kept plaintext in both modes:

- `data/ha/state/trial.json` — migration-trial marker (paths, hashes, timestamps). Encrypting would brick recovery on key loss.
- `data/ha/state/backup.json` — scheduled-backup runner status. Same reasoning.
- `data/ha/backups/<kind>/<ts>/*.meta.json` — backup artifact sidecars (timestamp, ciphertext SHA-256, tier, label). Encrypting would turn a wrong-key restore into a silent "backup not found" instead of a clear decrypt error. The paired `*.bin.enc` payload remains encrypted.
- Every adapter slot's own manifest sidecar (`meta.json`) — trained-at timestamp, key count, and the payload's plaintext content digest, plus (for a trained tier only) the base-model, tokenizer, and LoRA-shape compatibility fingerprints. Encrypting it would turn a wrong-key restore, or an operator inspecting a slot after a crash, into a silent "nothing here" instead of a readable record: the digest is a verifier, not a secret, and `registry_sha256` has carried the same plaintext-by-design property since it existed. The payload itself — the adapter weights, or the graph under the `simulate` venue — remains encrypted; only the sidecar that describes it is plaintext.
- `data/ha/sessions/<session_id>.jsonl` — raw per-session transcripts, plaintext. A single long conversation can span several of these files — a session rotates on an idle gap or once it grows past what one consolidation pass can take in (see [DEPLOYMENT.md](DEPLOYMENT.md) for the rotation triggers). Written unconditionally the moment a turn is appended, independent of `debug` — there is no RAM-only mode for a pending session — and kept until the session consolidates. A session is retired only by a successful consolidation; any other outcome (a recall-gate rejection, an abort, a full interim ring) leaves it pending for the next cycle's attempt, with nothing to configure for that case. At consolidation, `consolidation.retain_sessions` decides the retired JSONL's fate: retained transcripts (including document-ingest chunks and their original bytes) move under the debug tree, `data/ha/debug/…`; otherwise the JSONL is deleted — this is the operator's choice, not a data-loss bug. `debug: true` forces retention regardless of `retain_sessions`. Knob defaults and the exact archive location are documented in [DEPLOYMENT.md](DEPLOYMENT.md). Plaintext throughout by design — the point is `tail`/`cat`/`grep` inspection, not a `debug`-only opt-in.
- Per-cycle debug artifacts under the debug tree, `data/ha/debug/…` (`episodic_rels_snapshot.json`, `procedural_rels_snapshot.json`, `graph_merged_snapshot.json`, `graph_enriched_snapshot.json`) — written only when `debug: true`. Always plaintext, inspection-first, regardless of Security posture. The simulate-mode tier's own stored graph (its written payload, plus the tier's `indexed_key_registry.json` — SimHash fingerprints live inside this file, not a separate one) is a SEPARATE, encrypted store and does NOT use this carve-out.
- Per-session extraction snapshots under the debug tree (`graph_snapshot.json`, `procedural_graph_snapshot.json`) — written by the consolidation loop when `debug: true` and `save_cycle_snapshots` is enabled. Same plaintext-inspection rationale as the per-cycle aggregates above.
- `response.json` under the calibration artifact root, `data/ha/calibration/artifacts/calibrate/<stage>/<stamp>/` — the full result of an operator-invoked `/calibrate/*` call (parsed graph or reply including diagnostics, phase records, raw model output). Written by `on_calibration_result`, unconditionally on every calibration run, regardless of `debug` — calibration artifacts are a separate scope from the debug tree above (both can be open at once, in which case the same artifact lands in both roots). `POST /calibrate/extract_pending` writes per-session graph snapshots of **real pending conversations** here — the same content a live fold would extract, not a supplied fixture — so this location holds plaintext personal data whenever that route has been used, independent of `debug`.
None of the first four carry user facts. The session-transcript bullet also carries user facts, but — unlike the three debug-artifact bullets that follow it, which are produced only at operator request via the `debug` flag — it is written unconditionally; only its post-consolidation fate (retain vs. delete) is flag-governed, by `consolidation.retain_sessions` (`debug: true` forcing retention regardless). Adapter weight blobs carry user facts as numerical patterns; see §8 for the probe-resistance limit that encryption does not fully close.

## 5. Authentication & authorization

The auth layer is independent of the encryption mode — it governs which REST requests are accepted, not how data is written to disk. All credentials live in one place — the token store, persisted at `user_tokens.json` — populated exclusively via `mint-user-token`; there is no separate shared-secret validation path. The startup log always emits exactly one `AUTH:` line naming the active posture:

| Posture | Condition | Effect |
|---------|-----------|--------|
| **OFF** | No per-user token store wired | The server is usable without credentials — conversational endpoints (`/chat`, `/voice`, `/push/*`, `/status`) accept any request. Fail-closed admin: the auth middleware stamps the non-admin **chat** scope on every pass-through request, so admin endpoints (`/gpu/*`, `/consolidate`, `/backup/*`, etc.) 403 via `require_admin` until a store is configured (i.e. until the first `mint-user-token`). Startup emits a loud `AUTH: OFF` warning. Default for a fresh install with `mobile_pwa.enabled: false` and no prior mint. |
| **ON** | A per-user token store is wired — either `mobile_pwa.enabled: true`, or `user_tokens.json` already exists from a prior mint | All endpoints require a per-user opaque bearer token. **Fail-closed**: a wired store with zero active tokens still 401s every request rather than reverting to open access. Each token carries a **scope** — `chat` (the secure default, including pre-scope-field tokens) or `admin`. Admin scope is required for operational endpoints. The `chat` scope reaches `/chat`, `/voice`, `/push/*`, and `/status`. |

Every accepted token — attributed to a `speaker_id` or not — additionally carries a capability scope:

| Scope | Endpoints reached | How to mint |
|-------|------------------|-------------|
| `admin` | All endpoints (conversational + operational) | `mint-user-token <speaker> --scope admin`, or `--unattributed --scope admin --force-admin` |
| `chat` | `/chat`, `/voice`, `/push/*`, `/status` only | `mint-user-token <speaker> --scope chat` (the default), or `--unattributed --scope chat` |

Token minting, revocation, and the `mint-user-token` CLI syntax are documented in [DEPLOYMENT.md — Per-user token management](DEPLOYMENT.md#per-user-token-management).

**`PARAMEM_API_TOKEN` — carrier, not a credential.** The environment variable name survives as the well-known place infrastructure consumers (the systemd consolidation-tick timer, the HA custom component) read their own bearer-token value from. The server itself never reads or validates it as a credential; the value placed in it must be a token actually minted into the token store (typically an unattributed admin token, `mint-user-token --unattributed --scope admin --force-admin`, for infrastructure callers; an unattributed chat token for the HA component). Minting the infrastructure token is a deployment step the operator performs once.

**Upgrading from the shared-token model.** A deployment that predates per-user tokens set `PARAMEM_API_TOKEN` as a single shared credential, validated by the server directly. That validation path is retired — the variable is only the carrier described above. A deployment that set the env var, never enabled `mobile_pwa.enabled`, and never ran `mint-user-token` lands in the **OFF** posture on upgrade: every REST endpoint is open, and the old token value is silently ignored. The server detects exactly this case — env var set, no per-user store wired — and emits an additional loud warning alongside the standard `AUTH: OFF` line, naming `mint-user-token` as the fix. The migration is one command: mint an admin-scope token (`paramem mint-user-token --unattributed --scope admin --force-admin`) and update `PARAMEM_API_TOKEN` (and any systemd drop-in / HA component config reading it) to the minted value.

**Security properties of per-user tokens:**

- Tokens are opaque random secrets. The plaintext token is displayed once at mint time and never stored or logged. Only the `sha256(token)` hash is persisted on disk, in `user_tokens.json`. Scope is a capability boundary — it is derived server-side from the stored record, never from a claim in the request.
- `user_tokens.json` follows the deployment-wide encryption posture: plaintext under Security OFF, age-encrypted when the daily key is loaded. It is covered by the startup mode-consistency check — a plaintext credential file alongside a loaded key is refused at startup.
- **Fail-closed:** revoking the last active token in the store keeps the auth layer fail-closed rather than silently reverting to open access.
- **Token-never-logged:** the plaintext token is never written to any log file. `user_tokens.json` stores only `sha256(token)`.

**Live reload.** Revocation and scope changes (re-mint + revoke) take effect on the running server without a restart: the token store re-reads `user_tokens.json` on the next authenticated request when the file's mtime changes. Accepted cross-process revocation race window: the narrow in-flight window between a revoke write and the next request; not a meaningful attack surface for typical deployment cadences.

**Rotation.** Rotating any token — the infrastructure carrier included — is revoke-then-mint: `revoke-user-token` the suspected-compromised token, `mint-user-token` a fresh one, and update every consumer that reads the old plaintext value (`.env`/systemd drop-in for the infrastructure carrier, device Settings for a per-user token, the HA component config for its token). There is no separate shared-secret revocation path any more — every credential goes through the same store.

**Revoking unattributed tokens.** Revoking without naming a specific speaker is refused rather than silently matching every unattributed token — preventing accidental bulk-revocation. Use `revoke-user-token --label <label>` to revoke an unattributed token by its device label.

**Web Push infrastructure files (when `mobile_pwa.push_enabled: true`):**

- `vapid_keys.json` — EC P-256 VAPID private key (PEM). Auto-generated on first startup when push is enabled; auto-loaded on subsequent startups. Both files follow the same encryption posture as `user_tokens.json`: plaintext under Security OFF, age-encrypted under Security ON, covered by the startup mode-consistency scan.
- `push_subscriptions.json` — per-speaker Web Push endpoint registrations. Schema: `{"version":1, "subscriptions": {"<speaker_id>": [{endpoint, keys:{p256dh,auth}}...]}}`.
- **VAPID key stability:** rotating `vapid_keys.json` invalidates all existing browser push subscriptions (browsers will not receive notifications until they re-subscribe). Treat the keypair as effectively immutable once browsers have subscribed. Key rotation is intentionally out of scope.
- **Notification-only ping posture:** no personal content passes through the push relay. The push payload is intentionally empty (or carries only a generic title); real content is fetched by the client after the user taps the notification.
- **Revocation** is per-token or per-speaker and takes effect immediately on the next request.
- **Token carriers:** `Authorization: Bearer <token>` HTTP header — this is the carrier the PWA uses in practice. The middleware also accepts the configured cookie name if one is presented by the client, but the server does not issue a cookie; the PWA stores the token in `localStorage` and sends it exclusively via the `Authorization` header.

**Path exemptions.** The following paths are exempt from bearer-token checks so the browser can load the PWA shell and liveness checks can operate before a token is presented:

- `/` — redirects to `/app/`; exempt so the browser follows the redirect before a token is presented
- `/app` — bare mount redirect (307 → `/app/`); exempt so it reaches the `StaticFiles` handler
- `/health` — unauthenticated liveness endpoint for HA binary sensors and external pollers
- `/app/` prefix — the PWA shell, its static assets, and the service worker (`/app/sw.js`)

All other endpoints enforce the active posture. The Wyoming STT/TTS ports have no protocol-level auth; see §6.

## 6. Network exposure & transport

**HTTPS/TLS is required.** Three features hard-fail on plain HTTP:

- **PWA install / `getUserMedia`** — browsers block microphone access and PWA service-worker registration on non-HTTPS origins (except `localhost`).
- **Web Push** — the Web Push standard mandates HTTPS; browsers reject subscriptions over plain HTTP.
- **Bearer tokens** — tokens are only confidential over TLS. Plain HTTP exposes them to any LAN observer.

For HTTPS setup and the Tailscale configuration see [DEPLOYMENT.md](DEPLOYMENT.md).

**Trust-boundary assumption.** The threat model assumes a Tailscale VPN or a trusted private LAN as the transport layer. The server is **never** intended for direct internet exposure. Specifically:

- Wyoming STT (port 10300) and Wyoming TTS ports have no protocol-level authentication. They must not be reachable from the public internet — secure via firewall or Tailscale ACLs.
- `/gpu/*`, `/consolidate`, `/backup/*`, `/admin/*`, `/calibrate/*`, and `/debug/*` are admin-only endpoints. Exposing them to the internet is a security risk even with a strong admin-scope token.
- `/debug/*` is not uniformly read-only: `POST /debug/erase-keys` (admin scope, `config.debug=true`, and an explicit confirmation in the request body) stale-marks the named memory keys — unrecallable immediately, with their stored content replaced at that key's own tier's next consolidation.
- The HA custom component reaches the server over HTTP on the LAN; place it behind a Tailscale exit node or restrict it to a dedicated VLAN.

An admin-scope token is the sole authentication barrier for the admin surface. Mint least-privilege `chat`-scope tokens for conversational endpoints (`mint-user-token <speaker> --scope chat`, or `--unattributed --scope chat` for a shared device) to narrow the blast radius if a token leaks; reserve `admin` scope for the operator and for the infrastructure carrier described in §5.

## 7. Recovery model

The security model follows BitLocker semantics: the key material is the only path to the data. Losing it is equivalent to losing the data; gaining it is equivalent to gaining the data (see §2 on the admin/operator trust model). There is no backdoor, no author escrow, no cloud recovery service.

The deployment uses two keys:

1. **Daily access key.** A per-host daily identity (age X25519) stored on disk as a passphrase-wrapped envelope at `~/.config/paramem/daily_key.age` (mode `0600`, parent directory `0700`). The passphrase is provided via the `PARAMEM_DAILY_PASSPHRASE` environment variable — loaded from the operator's environment or a systemd drop-in. Hardware-backed unlock (TPM2, Windows DPAPI, libsecret) is a future upgrade path behind the same loader interface and does not change the operator-facing contract. Rotatable without operator intervention.
2. **Recovery key.** A *separate* age X25519 identity (bech32 `AGE-SECRET-KEY-1…`), minted alongside the daily identity by `paramem generate-key`. The public recipient is persisted at `~/.config/paramem/recovery.pub` (mode `0644`) so every new envelope lists it alongside the daily recipient. The secret is printed *once* to stderr at generation time with a BitLocker-style warning — operators must confirm they have saved it before the key files are written — and is never persisted on this device. Store it offline: printed paper, metal seed plate, password-manager secure note, or a safe. Used only when the daily access path fails (passphrase loss, disk loss, hardware replacement). Survives hardware replacement; restoring decrypts the store and enrolls a fresh daily identity on the new host.

Both keys decrypt the same data. Loss of the daily key is routine (rotate it). Loss of the recovery key — with the daily path also unavailable — is unrecoverable.

**Rotation.** `paramem rotate-daily` mints a fresh daily identity, re-encrypts every age infrastructure file to `[daily_new, recovery]` — including every `adapter_model.safetensors` blob — and atomically swaps the new daily key file into place. The recovery recipient is preserved. `paramem rotate-recovery` mints a fresh recovery identity, prints the new bech32 secret once with the same refuse-without-confirm UX as `generate-key`, and re-encrypts every file to `[daily, recovery_new]`. Both commands are crash-safe: per-file atomic rename plus a rotation manifest at `~/.config/paramem/rotation.manifest.json` that records pending vs done files, so a crash resumes from where it left off (`rotate-recovery` excepted — the print-once secret cannot be resumed and must be restarted cleanly).

**Hardware replacement.** `paramem restore --recovery-key-file <path>` is the entry point after losing the original device. Given the recovery bech32 from paper, it sanity-checks against an on-disk age envelope, mints a fresh daily identity (new operator-supplied passphrase), writes `daily_key.age` + `recovery.pub` to the new machine, and re-encrypts every age file to `[daily_new, recovery]`. The recovery identity is reused on the envelopes — it is the thing that authorised the restore, and the operator's paper copy remains valid. Crash-safe via the same rotation-manifest mechanism; a typo in the bech32 aborts before any on-disk mutation. Distinct from `paramem backup-restore`, which restores a backup archive over REST.

**Backup restore across key rotation.** Age-encrypted backups do not carry a key fingerprint in the sidecar — the fingerprint concept does not map onto X25519 recipient lists. A stale daily identity surfaces as a decrypt error on restore (HTTP 500 `decrypt_invalid_token`), which is equally actionable: the operator either re-keys the backup via `rotate-daily` / `rotate-recovery` or restores from the recovery bech32. Backups written while Security was OFF are plaintext and always restore.

**Discarding the interim ring.** `POST /interim/discard` removes interim-tier facts from the live store and from disk, but — the same property that governs `/speaker/forget`, whose keys stop being served the moment the call returns while their stored content is replaced only at that tier's next consolidation — does not reach into backup bundles taken beforehand: a bundle captured before the discard still contains those slots, and restoring it brings them back.

**Erasing named keys.** `POST /debug/erase-keys` stale-marks an explicit, operator-supplied list of memory keys — an admin-scope, `config.debug=true`, explicitly confirmed operation. Each named key stops being served the moment the call returns; its stored content is replaced at that key's own tier's next consolidation. A tier that no consolidation rebuilds keeps the withheld record in place. Like the interim-ring discard, it does not reach into backup bundles taken beforehand.

**Full-snapshot restore (migration revert).** Beyond per-artifact config restores, `POST /backup/restore` with `restore_config: true` restores a complete `snapshot_bundle` — every tier's written payload (adapter weights, or a graph for a tier running under `consolidation.mode: simulate`), donor checkpoints, registries, `key_metadata.json`, speaker profiles, and `server.yaml` — verifying every file hash and decrypt-probing the daily identity *before* any mutation, and safety-snapshotting the current state first so the revert is itself reversible. This is the revert path for a migration that has already been accepted (its trial marker cleared): the pre-migration bundle is the rollback, restored over REST followed by a restart. It is refused during an active `TRIAL`/`STAGING` migration or while consolidation/training is running. Base-swap snapshot bundles (`pre_base_swap` tier) additionally retain a non-restored `server.yaml.candidate` sidecar — the candidate config that was staged for the swap — so the operator can extract it and retry after a rollback; these bundles are retention-immune for 30 days (same class as pre-migration snapshots), surviving pruning even after the trial marker is cleared.

**Infrastructure integrity check.** `paramem integrity` (and `GET /integrity`) verifies on-disk registries, simhashes, and manifests for validity and cross-tier consistency, the same way in both the `train` and `simulate` venues. It runs at startup, as a migration pre-flight gate, and on demand — surfacing a corrupt or half-written store (including a backup that no longer decrypts under the current daily identity) before it propagates. It also verifies that each tier's stored keys still correspond to a written payload whose content digest still matches its manifest — trained weights or a graph, whichever the tier holds — so a tier where they do not is a visible failure.

Biometric unlocks (Windows Hello, fingerprint, FIDO2) are supported as *access conveniences* for the daily path only. They are not a recovery mechanism: biometrics unlock a sealed key on specific hardware; they do not regenerate the key on a new device. Any sensible deployment pairs biometric-unlocked daily access with a printed recovery artifact.

For the encryption-lifecycle command reference and startup-gate reset runbook, see [DEPLOYMENT.md — Encryption & recovery operations](DEPLOYMENT.md#encryption--recovery-operations).

## 8. Operator responsibilities

ParaMem is a single-admin service. The operator — the person running the server — is responsible for:

- Generating and storing key material. Run `paramem generate-key` to mint the daily identity (stored passphrase-wrapped on this host) and the recovery identity (printed once — save it offline). Do not rely on a single storage location for the only copy of the recovery bech32.
- Scoping LAN exposure. Set `PARAMEM_LISTEN_IP` to the specific host interface that should accept incoming requests, and `PARAMEM_NAS_IP` to scope the Windows Firewall rule to the Home Assistant source host. Unset values default to an open posture with a loud startup warning.
- Choosing the appropriate auth posture (§5) for the deployment. Enable `mobile_pwa.enabled: true` (or run one `mint-user-token`, which wires the store the same way — after a restart; the store handle is assigned once at server startup, so a mint against a from-scratch, auth-OFF deployment does not flip the already-running server ON) for per-user tokens that carry speaker identity; mint a `--scope chat` token for the HA component and an `--unattributed --scope admin --force-admin` token for the systemd infrastructure carrier (`PARAMEM_API_TOKEN` in `.env`). Until the first mint (and the restart that picks it up) the server is usable by any reachable peer for conversational endpoints, but administrative endpoints (`/gpu/*`, `/consolidate`, `/backup/*`, etc.) 403 until a store is configured — a loud startup warning is emitted regardless.
- **Rotating a compromised token.** `revoke-user-token` the compromised token, `mint-user-token` a fresh one, and update every consumer that reads the old plaintext value: `.env`/systemd drop-in for the infrastructure carrier, device Settings for a per-user token, the HA component config for its token.
- Managing `.env` and per-secret files under `~/.config/paramem/secrets/` with file mode `0600` and directory mode `0700`. The server refuses to start if permissions are looser.
- Scoping the Home Assistant long-lived access token to a dedicated, minimal-privilege HA user — not to a full admin.
- **Keeping `agents.ha_agent_id` pointed at a LOCAL HA conversation agent.** See §3. The HA hop is scrubbed under `sanitization.scrub` like the cloud hop and its reply restored through the same exit gate, but a cloud-backed HA agent still forwards the household's turns to a third party outside every ParaMem switch, no matter how well the payload itself is scrubbed.
- **Deciding `cloud.allow_degraded_serving`.** When the local model becomes unavailable for a reason the operator did not choose (GPU held by another process, insufficient VRAM, a failed adapter reload, a persistent CUDA fault), the ship default `false` closes the cloud leg: HA still answers, and anything HA cannot serve returns a canned limited-mode reply. Setting it `true` accepts that the household's questions route to a third party during an outage. The egress policy still applies in that state: a personal query is still refused under `block`/`both`, and outbound text is still scrubbed under `anonymize`/`both` — if scrubbing cannot run, the cloud leg is refused rather than sent unscrubbed. Folding an attested self-introduction onto the speaker's identity does not happen in this state, since it needs the local model; recognizing the speaker's own enrolled name is unaffected, since that needs no model call. The personal-query check may run with reduced language coverage in this state if the GPU has been fully released. Only the current query's text leaves — the memory store is unreachable, so no stored facts egress, and conversation history is always assembled server-side from the stored transcript rather than sent by a client, so nothing a client fabricates can enter the model's context or a cloud payload.
- Handling backups. A backup that captures the data directory but not the master-key source defeats the encryption.
- Reviewing the cloud-egress classifier exemplar files for the languages the deployment serves. The sanitizer's first-person check is encoder-based with multilingual exemplars under `configs/personal_referent/<class>.<lang>.txt`; coverage on a language without dedicated exemplars relies on cross-lingual transfer in the multilingual encoder and may miss idioms or low-resource phrasings. For deployments serving non-English speakers, add a file pair (`about_speaker.<lang>.txt` + `not_about_speaker.<lang>.txt`) and verify with a probe set before going live. The same applies to `configs/sentence_types/` for the abstention gate.

## 9. Known limitations

The security properties are honest, not aspirational. The following are the limitations an operator should understand before deploying.

- **Adapter probe resistance is limited.** An attacker with (a) the adapter weight file, (b) the base model, and (c) knowledge of relevant entity names can extract a meaningful fraction of stored facts through systematic probing. The adapter is opaque to grep but not opaque to a model that asks the right questions. This is inherent to any LoRA-based parametric memory — the knowledge must be accessible to be useful.
- **Weight encryption narrows blob-copy risk but does not close probe surface.** ParaMem encrypts the key registry — per adapter tier, its `key_metadata.json` (bookkeeping rows) and `indexed_key_registry.json` (SimHash fingerprints live inside this file, not a separate registry) — and — under Security ON — the LoRA weight tensors (`adapter_model.safetensors`) as age envelopes. The registry encryption blocks the **systematic** extraction path: the indexed-recall template requires knowing the key string (`graph17`, `proc4`, …), and without the registry an attacker cannot enumerate keys. Encrypting the weight tensors adds defense-in-depth against an attacker who copies only the `adapters/` subtree without the key material. What remains: an attacker with (a) the decrypted weights, (b) the base model, and (c) knowledge of entity names can still extract facts through targeted natural-language probing ("what did Alex say about X?"), membership inference, and continued fine-tuning — these require running inference on the weights and are not closed by encryption alone.
- **Runtime exposure is identical to RAG.** While the server is reasoning over a recalled fact, that fact lives as plaintext in GPU / CPU RAM inside the server process. Any system reasoning over private data has this property; we isolate it to one process behind a local API rather than streaming recalled context to external tools.
- **Extraction-stage cloud enrichment narrows but does not eliminate PII egress, and completeness cannot be independently verified.** When `consolidation.extraction_enrichment_provider` is set to a cloud provider (default `""` = disabled), the pipeline sends an anonymized transcript and fact array to the cloud for coreference resolution, compound splitting, and dedup. Before sending, a small local detection model — the sole classifier, with no second, code-side detector — marks values belonging to the operator's `sanitization.scrub` vocabulary (default: name, phone number, postal address, online-identity hints); the pipeline itself, never a model, then substitutes placeholders into both the facts and the transcript, using an exact, case-**sensitive** match. Case sensitivity is load-bearing: it is the only signal separating a person named `Bill` from the common noun `bill` (an invoice), or `Will`/`will`, `Mark`/`mark`, `Rose`/`rose` — a case-insensitive match would fire on the common noun and effectively disable enrichment for that user. Given a correctly-marked value, substitution into both the facts and the transcript is exact and complete; there is nothing left for a post-hoc check to verify on that side.

  **With scrubbing active, the speaker's own name is folded onto their identity before anything crosses the cloud boundary, so it is scrubbed the same as any other name — including in the chat system prompt.** The speaker's enrolled display name is recognized on name equality alone, with no model call needed. When the speaker introduces themselves by name in conversation ("I'm Priya", "call me Priya"), the model that ran extraction is asked, once, which of the already-tagged names in that transcript belongs to the speaker; a match is accepted only when it is consistent with the enrolled name (the enrolled name itself, or a short form of it) — an unrelated namesake is refused and keeps an ordinary placeholder. No speaker name or id is ever placed in a cloud-facing system prompt regardless (see §3). This fold does not apply when person-name scrubbing is itself off (an empty or narrowed `scrub`), in which case names egress unscrubbed like any other content excluded from `scrub`.

  The genuine residual risk is, stated plainly: **(i) detection miss** — the detection model fails to mark an in-scope value as sensitive in the first place, so it is never scrubbed and is not code-recoverable once it has egressed. Mitigated by the shipped label vocabulary and, before ship, by a recall check over labeled PII — never by a runtime re-check. The detector is tuned so that over-marking, not under-marking, is the accepted failure direction, since under-scrubbing sends real PII to the cloud and can never be undone. It is still a genuine omission risk in a different way: nothing checks a given pass for completeness against the actual content — a pass that catches only some of the in-scope values in a transcript is indistinguishable downstream from one that correctly found all of them, and the unmarked values go out verbatim. **(ii) Unmarked-occurrence substitution miss** — substitution only touches occurrences the detector actually marked; a differently-inflected form of an already-marked value (observed for German surnames) can reach the cloud unscrubbed even when the base form was caught, since mechanical substitution cannot generalize across word forms the way a model rewriting prose in principle could. A marked variant differing only in case or diacritics is not affected by this gap — every distinct marked surface is substituted. **(iii) Single-model classification with no independent second opinion.** This is deliberate: no code-side rule can distinguish a name from a same-spelled common word by context (only a model reading the sentence can), and the `scrub` vocabulary itself can't be expressed as a code-side type rule (`postal address`, scrubbed, and `city`, deliberately left verbatim so the cloud can still reason about places, are the same underlying entity type). Building a rule-based backstop would mean re-authoring `scrub` as a second, closed vocabulary — the exact thing this design avoids. The detection model is a pinned release loaded as weights only, running no code of its own.

  Free-form secrets (API keys, passwords, tokens) are not a `scrub` category and are not scrubbed by this pass; for documents that may contain machine credentials, keep `extraction_enrichment_provider=""` or scrub credentials before ingest. Place names and organization names are deliberately left verbatim under the default `scrub` so the cloud can still reason about them (e.g. "What's a good restaurant in Berlin?"). `scrub` is configurable via `sanitization.scrub`; narrowing or broadening it is a privacy-vs-utility tradeoff the operator makes consciously, per deployment — a configuration narrower than the shipped default can shift how a borderline value is classified, and the guarantees above are measured against the shipped default category set. The one hard guarantee, independent of all of the above, is the master switch: `cloud.enabled: false` stops all cloud egress, including the graph-tier pass below.

  **The graph-tier enrichment pass** runs the same anonymize-then-cloud-then-restore round trip as extraction-stage enrichment, but over the accumulated cross-session knowledge graph rather than a single session's transcript, and carries the same single-classifier limitation described above. It fails closed at two levels: if the detection model is unavailable, the whole pass is skipped for that fold (content already enriched in earlier folds is unaffected); within a fold, content that cannot be reliably matched back to the graph's own entities after detection is held back from the cloud call rather than sent unmasked, while the rest of that fold's content still goes out — this guard only ever removes content from what egresses, never adds anything. An empty `sanitization.scrub` sends this tier's content to the cloud unscrubbed, the same opt-out contract as every other cloud-egress path in this document. On the way back, any cloud response naming a fact through an unresolved placeholder is dropped rather than merged into the graph, checked per individual fact. Because this pass has no access to a transcript, it cannot recognize a speaker introducing themselves by name the way extraction-stage enrichment can; under the default `scrub`, the accepted consequence is that the cloud cannot recognize two different name-forms as the same person once both are opaque placeholders — coreference for organizations and places is unaffected, since those stay verbatim under the default `scrub`. This pass runs only at a full consolidation fold, never during an interim cycle, so it reaches strictly less content than a pass that ran on every cycle would.
- **Escalation to either external leg can leak.** Both the HA leg and the cloud leg apply the same scrub — the HA leg unconditionally, the cloud leg under `sanitization.cloud_mode` — and the residual below is shared by both, not just the cloud leg. The sanitizer applied before escalation has two arms: a known-entity scrub (substitution against the speaker's graph entities) and a self-reference gate (encoder-based "is this about the speaker?" classifier with multilingual exemplars under `configs/personal_referent/`, falling back to an English token-set when the encoder isn't loaded). The self-reference gate classifies purely from text content — it does not require a resolved `speaker_id` to fire. On the relay path (no speaker resolved at all), history is always empty, so this is scoped to the current turn only: a personal interrogative is caught by a dedicated no-identity short-circuit before either leg is even tried, and a personal declarative is governed by the same `sanitization.cloud_mode` policy every other leg applies (with a live local model to anonymize it, when one is loaded) on the cloud leg, and by the HA leg's own unconditional scrub. Residual risk: the encoder operates on lexical/semantic shape; the local model can still rewrite a query in a form that embeds a personal fact while passing the gate. Cross-lingual transfer in the multilingual encoder lifts coverage past the languages with explicit exemplars (en/de today) but is not guaranteed for every locale or idiom — adding `<class>.<lang>.txt` exemplar files for production languages tightens the bound.
- **LAN authentication is operator-provisioned.** When no per-user token store is wired (no `mobile_pwa.enabled: true` and no prior mint), conversational REST endpoints are accessible to any LAN peer (Security OFF posture); administrative endpoints remain 403 fail-closed until a store is configured. Wyoming STT / TTS ports do not support protocol-level auth at all and rely on network-layer scoping (firewall rule) for access control.
- **No auth rate-limiting.** The bearer-token layer does not implement brute-force throttling. The design relies on high-entropy opaque tokens (infeasible to guess) and network-layer scoping (Tailscale / LAN) rather than rate limiting. If a token leaks, revoke it immediately.
- **Key loss is total.** No backdoor, no recovery service, no escrow. The recovery key *is* the backdoor; losing it is losing the data.
- **Biometrics are convenience, not security.** Biometric unlock binds to specific hardware and specific OS sessions. A new device or a TPM clear invalidates the daily path. Biometrics cannot be rotated if compromised and are not cryptographic secrets.
- **Supply chain pinning is not auditing.** Dependency versions are pinned in `pyproject.toml`, including the CUDA-specific `bitsandbytes` development wheel required for RTX 50-series hardware. Pinning prevents silent updates but does not constitute a reviewed supply chain.
- **Voice embeddings are biometric data.** Under GDPR Article 9 (EU) voice embeddings are special-category personal data. They are encrypted at rest under Security-ON; losing the recovery key is privacy-protective for this data, but *sharing* the key exports biometrics.
- **`/health` is unauthenticated.** The `/health` endpoint is exempt from token checks by design (HA binary sensors and pollers need it without credentials). It returns only liveness state, not personal data.

> The biggest limit — that the Python package is not attested and can be trivially tampered with by an operator-level attacker — is named in the top-of-document disclaimer, not repeated here.

## 10. Vulnerability reporting

Please do not open a public GitHub issue for suspected security vulnerabilities.

Contact: **Tobias Preusser — `tobias.preusser75@gmail.com`**.

When reporting, include:
- Affected version / commit
- Deployment configuration (Security ON / OFF, cloud enabled / disabled, HA connected)
- A clear reproduction or the minimum data needed to reason about the issue

ParaMem is research software maintained by a single author. There is no formal SLA for response times. Responsible disclosure is appreciated; public coordination will be on a best-effort basis.

## 11. References

- `README.md` — project overview, configuration, setup
- `DEPLOYMENT.md` — installation, configuration, encryption lifecycle, token management, backup & migration
- `paramem/server/auth.py`, `paramem/server/user_tokens.py`, `paramem/server/secret_store.py` — runtime entry points for the boundaries described above
- The internal hardening plan and empirical probe results live outside the public repository; enquiries should be routed through the disclosure channel in §10.
