# Security

> **Work in progress — not a finished security story.** The encryption-at-rest machinery described in this document is partial by design. It raises the bar for specific *separation* scenarios (see §2) and does not attempt to defend against an attacker who gains operator-level access to the host.
>
> **Two operator-level paths to plaintext, both trivial:**
>
> 1. **The Python codebase is editable.** `paramem/` on disk is plain source. An attacker with write access to the installed package path — which the operator user has by definition — can neutralize encryption with a three-line edit to `paramem/backup/encryption.py::envelope_encrypt_bytes` and restart the server. There is no code-signing, no bytecode integrity check, no TPM-backed attestation.
> 2. **Config-level data exfiltration via `debug: true`.** The operator can flip one line in `configs/server.yaml` and restart. From that point on, every consolidation cycle writes plaintext copies of user facts into `data/ha/sessions/*.jsonl` and per-cycle debug artifacts under `data/ha/debug/episodic/.../cycle_<N>/` (snapshot JSON files). No code edit, no crypto break — just the legitimate debug path used against the intent of a Security-ON deployment. This is intentional behaviour for debugging (see §4 carve-outs); it is named here because an attacker with config-write access can use it as a data-extraction primitive. (The simulate-mode graph store — `graph.json` under `<adapter_dir>/<tier>/` — is encrypted by default and respects `require_encryption`; it is NOT a debug carve-out.)
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
| Indexed key registry | Key identifiers, SimHash fingerprints, timestamps | JSON |
| Cumulative knowledge graph | Entities, predicates, relations | JSON (NetworkX) |
| Session queue | Transcript + speaker binding awaiting consolidation | JSON (atomic temp-file + rename) |
| Session snapshot | RAM state at graceful shutdown | age-encrypted when a key is configured |
| Speaker profiles | Voice embeddings + disclosed names | JSON (biometric data — see §8) |
| Background trainer resume state | Epoch counter + checkpoint references | JSON |
| Adapter manifest sidecars | Base-model SHA, tokenizer fingerprint, LoRA shape | JSON |

Adapter weights are the dominant artifact by volume and sensitivity. They are numerical — not directly readable for facts — but also not probe-resistant. Under Security ON they are encrypted at rest as age envelopes alongside the JSON metadata. The indexed-recall path still requires the encrypted registry for key enumeration; weight encryption adds defense-in-depth against blob-copy attackers. See §4 and §8 for the full picture.

## 2. Threat model

**Trust assumption.** The admin / operator of the host is a trusted authority. ParaMem does not attempt to protect data from an attacker who has the operator's OS credentials, process-memory access, or write access to the installed Python package. The operator holds the daily passphrase and the daily-key file; these travel in the same trust domain as the data they protect. A hostile process running as the operator can read the decrypted store from RAM, modify `paramem/` source (e.g. replace `envelope_encrypt_bytes` with a pass-through), or read the wrapped daily key plus passphrase and decrypt at rest. None of these are defended against.

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

- **User voice → STT.** Raw audio arrives on a Wyoming protocol port. Transcript + speaker embedding cross into the FastAPI process on the shared asyncio event loop.
- **Home Assistant ↔ ParaMem.** A thin HA custom component POSTs to the `/chat` endpoint over HTTP on the LAN. Bearer-token authentication has four postures governed by `PARAMEM_API_TOKEN` and `mobile_pwa.enabled` — see §5 for the full model. When auth is OFF the server accepts any LAN request, announced at startup as an explicit open posture, not a silent one.
- **ParaMem → Home Assistant.** `agents.ha_agent_id` **must name a LOCAL HA conversation agent** — the built-in `conversation.home_assistant`, or a self-hosted LLM agent on the operator's own hardware. ParaMem sends this hop **cleartext**, gated by intent classification only, and treats it as on-premise: it is the leg that stays reachable when the cloud leg is closed by `cloud.enabled: false` or by `cloud.allow_degraded_serving: false` during an outage. Configuring a cloud-backed HA agent (`conversation.groq`, `conversation.openai`, …) silently re-opens cloud egress one hop away, outside every ParaMem switch, with no sanitization applied. (Unrelated and unaffected: the HA custom component's own fallback `agent_id`, which exists to prevent recursive routing when ParaMem is HA's default conversation agent.)
- **ParaMem → cloud.** Sanitized queries (and speaker name, as persona anchor) may be sent to a configured cloud agent for escalation or cloud enrichment. This path is opt-in via config; nothing is sent without an active cloud configuration. Whether a query is personal — and therefore kept on-device — is decided by the intent classifier, supplemented by an encoder-based "is this about the speaker?" check with multilingual exemplars under `configs/personal_referent/` (English token-set fallback when the encoder isn't loaded). A query that does egress under an anonymizing `cloud_mode` is rewritten to placeholders by the local model first. Coverage scales with the exemplar files and the classifier; see §7 and §8 for the operator's responsibility and the residual risk.
- **Routing-time intent classifier (privacy property).** The intent classifier in `paramem/server/intent.py` runs *before* any retrieval and *outside* the PA reasoning path. Under `intent.mode: llm` (default) the local Mistral 7B is invoked with the focused classifier section of `configs/prompts/pa_voice.txt` only — `_personalize_prompt` is **not** applied, so the speaker name is not injected into the classifier system message and the query is classified on content alone. Under `intent.mode: embeddings` the sentence-encoder cosine match never receives speaker identity. Either path keeps routing-time classification orthogonal to personal-context exposure: the speaker-name leak surface is the *response-time* PA path (which intentionally uses `_personalize_prompt`), gated by the cloud-egress sanitizer above.
- **Adapter files at rest.** The on-disk artifacts listed in §1 live under the configured data directory. At-rest encryption is governed by the binary switch in §4.
- **Backup at rest.** Session snapshots and every other piece of infrastructure metadata follow the Security-ON/OFF contract in §4 — encrypted as age envelopes when the daily identity is loaded, and plaintext only when no key is configured.

## 4. Encryption at rest

ParaMem operates in one of two modes, governed by the loaded key material. There are no partial states.

### Security ON
`PARAMEM_DAILY_PASSPHRASE` is set AND `~/.config/paramem/daily_key.age` exists. When `~/.config/paramem/recovery.pub` is also present, every new write is multi-recipient (daily + recovery).

All infrastructure metadata — registry, graph, queue, snapshots, speaker profiles, manifest sidecars, backup artifacts — is age-encrypted on disk and decrypted only into process RAM on load. The universal read path sniffs the envelope magic at the start of each file (the literal bytes `age-encryption.org/v1` followed by a newline) and routes to the decryptor; plaintext is passed through verbatim. On startup the server logs one of:
```
SECURITY: ON (age daily identity loaded, recovery recipient available)
SECURITY: ON (age daily identity loaded, recovery recipient missing — run `paramem generate-key` to re-enable multi-recipient writes)
```

### Security OFF
No key material is loaded. All infrastructure metadata is plaintext on disk. This is a **documented operator opt-out**, not a gap. On startup the server logs:
```
SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)
```
and surfaces `encryption: off` on the `/status` endpoint. The server does not silently degrade between modes: if the daily identity is loaded but on-disk files are plaintext (or vice versa), startup refuses with an actionable message.

### Fail-loud opt-in: `security.require_encryption`

The Security-OFF opt-out is the operator's choice. Deployments that want a misconfiguration to fail loud rather than silently land plaintext on disk can set `security.require_encryption: true` in `configs/server.yaml`. When set, the server refuses to start unless the daily identity is loadable — a uniform startup gate covering every feature that writes to disk (snapshots, checkpoint shards, backups, infrastructure metadata). Default is `false` (the AUTO-everywhere posture described above).

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
- `data/ha/sessions/*.jsonl` and `data/ha/sessions/archive/*.jsonl` — raw conversation transcripts, written only when `debug: true`. Explicit operator opt-in to plaintext persistence for inspection; the whole point of debug mode is to see the transcripts with `tail`/`cat`/`grep`.
- Per-cycle debug artifacts under `data/ha/debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/` (`episodic_rels_snapshot.json`, `procedural_rels_snapshot.json`, `graph_merged_snapshot.json`, `graph_enriched_snapshot.json`) — written only when `debug: true`. Always plaintext, inspection-first, regardless of Security posture. The simulate-mode output under `<adapter_dir>/<tier>/` (`graph.json`, `simhash_registry.json`, `indexed_key_registry.json`) is a SEPARATE, encrypted store and does NOT use this carve-out.
- Per-session extraction snapshots under `data/ha/debug/episodic/.../cycle_<N>/run_<run_id>/sessions/<session_id>/` (`graph_snapshot.json`, `procedural_graph_snapshot.json`) — written by the consolidation loop when `debug: true` and `save_cycle_snapshots` is enabled. Same plaintext-inspection rationale as the per-cycle aggregates above.
- `data/ha/debug/episodic/.../cycle_<N>/run_<run_id>/calibrate_extract_<session_id>_<ts>.json` — the full result of an operator-invoked `/calibrate/extract` call (parsed graph including diagnostics, phase records, raw model output). Written only when `debug: true`; the write self-gates off under `debug: false`. Same plaintext-inspection rationale as the other per-cycle debug artifacts.
None of the first three carry user facts. The next four *do* carry user facts but are produced only at operator request via the `debug` flag. Adapter weight blobs carry user facts as numerical patterns; see §8 for the probe-resistance limit that encryption does not fully close.

## 5. Authentication & authorization

The auth layer is independent of the encryption mode — it governs which REST requests are accepted, not how data is written to disk. Two knobs interact: `PARAMEM_API_TOKEN` (environment variable; shared bearer token) and `mobile_pwa.enabled` (config; per-user bearer tokens). The startup log always emits exactly one `AUTH:` line naming the active posture:

| Posture | Condition | Effect |
|---------|-----------|--------|
| **OFF** | Neither configured | The server is usable without credentials — conversational endpoints (`/chat`, `/voice`, `/push/*`, `/status`) accept any request. Fail-closed admin: the auth middleware stamps the non-admin **chat** scope on every pass-through request, so admin endpoints (`/gpu/*`, `/consolidate`, `/backup/*`, etc.) 403 via `require_admin` until a shared token or per-user store is configured. Startup emits a loud `AUTH: OFF` warning. Default for new installs. |
| **ON-shared** | `PARAMEM_API_TOKEN` set | All endpoints require the single shared bearer token. The shared token always carries **admin** scope (full reach). Requests are **unattributed** — no `speaker_id` is attached. |
| **ON-per-user** | `mobile_pwa.enabled: true` | All endpoints require a per-user opaque bearer token. **Fail-closed.** Each token carries a **scope** — `chat` (the secure default, including pre-scope-field tokens) or `admin`. Admin scope is required for operational endpoints. The `chat` scope reaches `/chat`, `/voice`, `/push/*`, and `/status`. |
| **ON-both** | Both configured | Shared token (admin scope) checked first; per-user store is the fallback. |

**Token scope dimension.** Within ON-shared / ON-per-user / ON-both, every accepted token additionally carries a capability scope:

| Scope | Endpoints reached | How to mint |
|-------|------------------|-------------|
| `admin` | All endpoints (conversational + operational) | Shared `PARAMEM_API_TOKEN`, or `mint-user-token <speaker> --scope admin`, or `--unattributed --scope admin --force-admin` |
| `chat` | `/chat`, `/voice`, `/push/*`, `/status` only | `mint-user-token <speaker> --scope chat` (the default), or `--unattributed --scope chat` |

Token minting, revocation, and the `mint-user-token` CLI syntax are documented in [DEPLOYMENT.md — Per-user token management](DEPLOYMENT.md#per-user-token-management).

**Security properties of per-user tokens:**

- Tokens are opaque random secrets. The plaintext token is displayed once at mint time and never stored or logged. Only the `sha256(token)` hash is persisted on disk, in `user_tokens.json`. Scope is a capability boundary — it is derived server-side from the stored record, never from a claim in the request.
- `user_tokens.json` follows the deployment-wide encryption posture: plaintext under Security OFF, age-encrypted when the daily key is loaded. It is covered by the startup mode-consistency check — a plaintext credential file alongside a loaded key is refused at startup.
- **Fail-closed:** revoking the last active token in the store keeps the auth layer fail-closed rather than silently reverting to open access.
- **Token-never-logged:** the plaintext token is never written to any log file. `user_tokens.json` stores only `sha256(token)`.

**Live reload.** Revocation and scope changes (re-mint + revoke) take effect on the running server without a restart: `UserTokenStore` re-reads `user_tokens.json` on the next authenticated request when the file's mtime changes. Accepted cross-process revocation race window: the narrow in-flight window between a revoke write and the next request; not a meaningful attack surface for typical deployment cadences.

**Revoking a compromised shared token.** If `PARAMEM_API_TOKEN` is suspected compromised: update the value in `.env` (or the systemd drop-in), restart the server, and re-provision all devices and the HA component with the new token. The shared token is stateless — there is no revocation record to update; the old value simply stops being accepted after restart.

**Revoking unattributed tokens.** `revoke_speaker()` skips entries whose `speaker_id` is `None` and raises `ValueError` if called with `None` — preventing accidental bulk-revocation. Use `revoke-user-token --label <label>` to revoke an unattributed token by its device label.

**Web Push infrastructure files (when `mobile_pwa.push_enabled: true`):**

- `vapid_keys.json` — EC P-256 VAPID private key (PEM). Auto-generated on first startup when push is enabled; auto-loaded on subsequent startups. Both files follow the same encryption posture as `user_tokens.json`: plaintext under Security OFF, age-encrypted under Security ON, covered by the startup mode-consistency scan via `infra_paths()`.
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
- `/gpu/*`, `/consolidate`, `/backup/*`, `/admin/*`, `/calibrate/*`, and `/debug/*` are admin-only endpoints. Exposing them to the internet is a security risk even with a strong `PARAMEM_API_TOKEN`.
- The HA custom component reaches the server over HTTP on the LAN; place it behind a Tailscale exit node or restrict it to a dedicated VLAN.

When `PARAMEM_API_TOKEN` is set, the shared token is the sole authentication barrier for the admin surface. Use per-user tokens (`mobile_pwa.enabled: true`) with `--scope chat` for conversational endpoints to narrow the blast radius if a token leaks.

## 7. Recovery model

The security model follows BitLocker semantics: the key material is the only path to the data. Losing it is equivalent to losing the data; gaining it is equivalent to gaining the data (see §2 on the admin/operator trust model). There is no backdoor, no author escrow, no cloud recovery service.

The deployment uses two keys:

1. **Daily access key.** A per-host daily identity (age X25519) stored on disk as a passphrase-wrapped envelope at `~/.config/paramem/daily_key.age` (mode `0600`, parent directory `0700`). The passphrase is provided via the `PARAMEM_DAILY_PASSPHRASE` environment variable — loaded from the operator's environment or a systemd drop-in. Hardware-backed unlock (TPM2, Windows DPAPI, libsecret) is a future upgrade path behind the same loader interface and does not change the operator-facing contract. Rotatable without operator intervention.
2. **Recovery key.** A *separate* age X25519 identity (bech32 `AGE-SECRET-KEY-1…`), minted alongside the daily identity by `paramem generate-key`. The public recipient is persisted at `~/.config/paramem/recovery.pub` (mode `0644`) so every new envelope lists it alongside the daily recipient. The secret is printed *once* to stderr at generation time with a BitLocker-style warning — operators must confirm they have saved it before the key files are written — and is never persisted on this device. Store it offline: printed paper, metal seed plate, password-manager secure note, or a safe. Used only when the daily access path fails (passphrase loss, disk loss, hardware replacement). Survives hardware replacement; restoring decrypts the store and enrolls a fresh daily identity on the new host.

Both keys decrypt the same data. Loss of the daily key is routine (rotate it). Loss of the recovery key — with the daily path also unavailable — is unrecoverable.

**Rotation.** `paramem rotate-daily` mints a fresh daily identity, re-encrypts every age infrastructure file to `[daily_new, recovery]` — including all `adapter_model.safetensors` blobs enumerated via `infra_paths` — and atomically swaps the new daily key file into place. The recovery recipient is preserved. `paramem rotate-recovery` mints a fresh recovery identity, prints the new bech32 secret once with the same refuse-without-confirm UX as `generate-key`, and re-encrypts every file to `[daily, recovery_new]`. Both commands are crash-safe: per-file atomic rename plus a rotation manifest at `~/.config/paramem/rotation.manifest.json` that records pending vs done files, so a crash resumes from where it left off (`rotate-recovery` excepted — the print-once secret cannot be resumed and must be restarted cleanly).

**Hardware replacement.** `paramem restore --recovery-key-file <path>` is the entry point after losing the original device. Given the recovery bech32 from paper, it sanity-checks against an on-disk age envelope, mints a fresh daily identity (new operator-supplied passphrase), writes `daily_key.age` + `recovery.pub` to the new machine, and re-encrypts every age file to `[daily_new, recovery]`. The recovery identity is reused on the envelopes — it is the thing that authorised the restore, and the operator's paper copy remains valid. Crash-safe via the same rotation-manifest mechanism; a typo in the bech32 aborts before any on-disk mutation. Distinct from `paramem backup-restore`, which restores a backup archive over REST.

**Backup restore across key rotation.** Age-encrypted backups do not carry a key fingerprint in the sidecar — the fingerprint concept does not map onto X25519 recipient lists. A stale daily identity surfaces as a decrypt error on restore (HTTP 500 `decrypt_invalid_token`), which is equally actionable: the operator either re-keys the backup via `rotate-daily` / `rotate-recovery` or restores from the recovery bech32. Backups written while Security was OFF are plaintext and always restore.

**Full-snapshot restore (migration revert).** Beyond per-artifact config restores, `POST /backup/restore` with `restore_config: true` restores a complete `snapshot_bundle` — every tier's adapter weights, registries, `key_metadata.json`, speaker profiles, and `server.yaml` — verifying every file hash and decrypt-probing the daily identity *before* any mutation, and safety-snapshotting the current state first so the revert is itself reversible. This is the revert path for a migration that has already been accepted (its trial marker cleared): the pre-migration bundle is the rollback, restored over REST followed by a restart. It is refused during an active `TRIAL`/`STAGING` migration or while consolidation/training is running. Base-swap snapshot bundles (`pre_base_swap` tier) additionally retain a non-restored `server.yaml.candidate` sidecar — the candidate config that was staged for the swap — so the operator can extract it and retry after a rollback; these bundles are retention-immune for 30 days (same class as pre-migration snapshots), surviving pruning even after the trial marker is cleared.

**Infrastructure integrity check.** `paramem integrity` (and `GET /integrity`) verifies on-disk registries, simhashes, manifests, and per-tier graphs for validity and cross-tier consistency. It runs at startup, as a migration pre-flight gate, and on demand — surfacing a corrupt or half-written store (including a backup that no longer decrypts under the current daily identity) before it propagates.

Biometric unlocks (Windows Hello, fingerprint, FIDO2) are supported as *access conveniences* for the daily path only. They are not a recovery mechanism: biometrics unlock a sealed key on specific hardware; they do not regenerate the key on a new device. Any sensible deployment pairs biometric-unlocked daily access with a printed recovery artifact.

For the encryption-lifecycle command reference and startup-gate reset runbook, see [DEPLOYMENT.md — Encryption & recovery operations](DEPLOYMENT.md#encryption--recovery-operations).

## 8. Operator responsibilities

ParaMem is a single-admin service. The operator — the person running the server — is responsible for:

- Generating and storing key material. Run `paramem generate-key` to mint the daily identity (stored passphrase-wrapped on this host) and the recovery identity (printed once — save it offline). Do not rely on a single storage location for the only copy of the recovery bech32.
- Scoping LAN exposure. Set `PARAMEM_LISTEN_IP` to the specific host interface that should accept incoming requests, and `PARAMEM_NAS_IP` to scope the Windows Firewall rule to the Home Assistant source host. Unset values default to an open posture with a loud startup warning.
- Choosing the appropriate auth posture (§5) for the deployment. Set `PARAMEM_API_TOKEN` to protect the server with a single shared token; enable `mobile_pwa.enabled: true` for per-user tokens that carry speaker identity. When neither is configured the server is usable by any reachable peer for conversational endpoints, but administrative endpoints (`/gpu/*`, `/consolidate`, `/backup/*`, etc.) 403 until a token or per-user store is configured — a loud startup warning is emitted regardless.
- **Rotating a compromised shared token.** If `PARAMEM_API_TOKEN` is suspected compromised: update the value in `.env` (or the systemd drop-in), restart the server, and re-provision all devices and the HA component with the new token.
- Managing `.env` and per-secret files under `~/.config/paramem/secrets/` with file mode `0600` and directory mode `0700`. The server refuses to start if permissions are looser.
- Scoping the Home Assistant long-lived access token to a dedicated, minimal-privilege HA user — not to a full admin.
- **Keeping `agents.ha_agent_id` pointed at a LOCAL HA conversation agent.** See §3. A cloud-backed HA agent turns the one hop ParaMem treats as on-premise into unsanitized cloud egress that none of ParaMem's switches can close.
- **Deciding `cloud.allow_degraded_serving`.** When the local model becomes unavailable for a reason the operator did not choose (GPU held by another process, insufficient VRAM, a failed adapter reload/apply, a persistent CUDA fault), the ship default `false` closes the cloud leg: HA still answers, and anything HA cannot serve returns a canned limited-mode reply. Setting it `true` accepts that the household's questions route to a third party during an outage. In that state there is no local model — so no personal verdict and no anonymization are computed, and the current turn is sent to the cloud verbatim. Only the query's own text leaves; the memory store is unreachable, so no stored facts egress. History turns are still drop-gated.
- Handling backups. A backup that captures the data directory but not the master-key source defeats the encryption.
- Reviewing the cloud-egress classifier exemplar files for the languages the deployment serves. The sanitizer's first-person check is encoder-based with multilingual exemplars under `configs/personal_referent/<class>.<lang>.txt`; coverage on a language without dedicated exemplars relies on cross-lingual transfer in the multilingual encoder and may miss idioms or low-resource phrasings. For deployments serving non-English speakers, add a file pair (`about_speaker.<lang>.txt` + `not_about_speaker.<lang>.txt`) and verify with a probe set before going live. The same applies to `configs/sentence_types/` for the abstention gate.

## 9. Known limitations

The security properties are honest, not aspirational. The following are the limitations an operator should understand before deploying.

- **Adapter probe resistance is limited.** An attacker with (a) the adapter weight file, (b) the base model, and (c) knowledge of relevant entity names can extract a meaningful fraction of stored facts through systematic probing. The adapter is opaque to grep but not opaque to a model that asks the right questions. This is inherent to any LoRA-based parametric memory — the knowledge must be accessible to be useful.
- **Weight encryption narrows blob-copy risk but does not close probe surface.** ParaMem encrypts the key registry (`registry/key_metadata.json`) and the per-tier SimHash registries (`<store>/<tier>/simhash_registry.json`) and — under Security ON — the LoRA weight tensors (`adapter_model.safetensors`) as age envelopes. The registry encryption blocks the **systematic** extraction path: the indexed-recall template requires knowing the key string (`graph17`, `proc4`, …), and without the registry an attacker cannot enumerate keys. Encrypting the weight tensors adds defense-in-depth against an attacker who copies only the `adapters/` subtree without the key material. What remains: an attacker with (a) the decrypted weights, (b) the base model, and (c) knowledge of entity names can still extract facts through targeted natural-language probing ("what did Alex say about X?"), membership inference, and continued fine-tuning — these require running inference on the weights and are not closed by encryption alone.
- **Runtime exposure is identical to RAG.** While the server is reasoning over a recalled fact, that fact lives as plaintext in GPU / CPU RAM inside the server process. Any system reasoning over private data has this property; we isolate it to one process behind a local API rather than streaming recalled context to external tools.
- **Extraction-stage cloud enrichment narrows but does not eliminate PII egress, and completeness cannot be independently verified.** When `consolidation.extraction_enrichment_provider` is set to a cloud provider (default `""` = disabled), the extraction pipeline runs a local-anonymization pass — the SAME local model that ran extraction, on a second prompt — before sending an anonymized transcript and fact array to the cloud for coreference resolution, compound splitting, and dedup. The local model is the **sole classifier**: given the operator's `sanitization.scrub` vocabulary (default: person name, email address, phone number, physical address, social/professional profile URL), it decides per value whether it is in scope, mints a type-tagged placeholder for it (e.g. `Person_1`), and returns that `{real → placeholder}` mapping. There is no code-side entity-type gate downstream of that decision, and no second, independent classifier — this is a deliberate design choice, argued below, not an oversight.

  The two payload surfaces this pass builds have different guarantees. **Facts** are anonymized mechanically and deterministically: the script substitutes the model's mapping into `graph.relations`' subject/object fields via an exact, edge-aware, case-**sensitive** substitution primitive — the predicate is never a substitution target. Given a correct mapping, fact substitution is exact and complete; there is nothing left for a post-hoc scan to verify on this surface. Case sensitivity is load-bearing here: it is the only signal separating a person named `Bill` from the common noun `bill` (an invoice), or `Will`/`will`, `Mark`/`mark`, `Rose`/`rose`. A case-insensitive match would fire on the common noun for any entity-derived name and silently disable cloud enrichment for that user on every session, so the pipeline never performs one. **The transcript**, by contrast, is authored by the model itself rather than mechanically rebuilt from the mapping — the model rewrites the prose in-context, replacing only the values it placeholdered and leaving everything else unchanged. Prose classification is context-dependent in a way the case-sensitive fact fields are not — "Will" the name and "Will I?" the verb are identical bytes — so only the model, holding sentence context, can make that call; there is no mechanical exactness guarantee on this surface, and no runtime check re-verifies it.

  The genuine residual risk is, stated plainly: **(i) classification miss** — the model fails to label an in-scope value as sensitive. Not code-recoverable once it has egressed. Mitigated by prompt design (a canonical vocabulary and few-shot boundary examples) and, before ship, by the `/calibrate/extract` recall gate over labeled PII (names and attribute-borne PII) — never by a runtime re-check. For instance-level doubt (the category IS in scope; is this particular value really an instance of it), the prompt's posture is the opposite of leaving it verbatim: it instructs placeholdering the value, since under-scrubbing sends real PII to the cloud and can never be undone — over-scrubbing, not under-scrubbing, is the accepted failure direction there. It is still a genuine omission risk in a different way: egress fails closed only when the anonymizer itself fails to produce a usable result (a parse failure, or a missing/empty rewritten transcript); it proceeds whenever the anonymizer runs successfully, including with a mapping that is empty (a legitimate "nothing in scope" verdict) or one that names only some of the in-scope content — there is no guard that checks a returned mapping's completeness against the actual content. A mapping that names 2 of 10 in-scope persons and silently drops the other 8 still egresses; the remaining 8 names go out verbatim. **(ii) Transcript-authoring fidelity** — because the model rewrites the prose rather than the script substituting it, a fidelity failure (content dropped or reworded beyond the intended substitutions) is possible in principle; this is verified offline at the same calibration gate (transcript fidelity, not just recall), not at runtime. **(iii) Single-model classification with no independent second opinion.** This is deliberate: a second, code-side detector cannot resolve the homograph problem above — only the model holds the sentence context needed to tell "Will" the name from "Will I?" the verb. And no code-side rule can express the `scrub` vocabulary in the first place: `physical address` (scrubbed) and `city` (deliberately left verbatim, so the cloud can still reason about places) are the same graph entity type, so a rule keyed on entity type would either over-scrub places or under-scrub addresses — there is no type-level line to draw. Building a rule-based backstop would mean re-authoring `scrub` as a second, closed code-side vocabulary, which is the exact artifact this design avoids everywhere.

  Free-form secrets (API keys, passwords, tokens) are not a `scrub` category and are not scrubbed by this pass. Place names and organization names are deliberately left verbatim under the default `scrub` so the cloud can still reason about them (e.g. "What's a good restaurant in Berlin?"). Document ingestion goes through the same pipeline; for documents that may contain machine credentials, keep `extraction_enrichment_provider=""` or scrub credentials before ingest. `scrub` is configurable via `sanitization.scrub`; narrowing or broadening it is a privacy-vs-utility tradeoff the operator makes consciously, per deployment. That residual is why the ship default is `cloud.enabled: false` — the one master switch for all cloud egress (`refinement_enrichment: "off"` for the graph-tier pass below): the operator's only hard guarantee is the off switch.

  **The graph-tier enrichment pass** (`paramem.training.graph_enrich.enrich_graph`, dispatched via `GraphTierRefiner.run_enrichment`, the post-merge cross-session second-order-relation pass) runs the SAME anonymize → cloud → de-anonymize chain, through `paramem.cloud` (`anonymize.py` / `deanonymize.py`) — the one round-trip contract every cloud-egress path shares — over the cumulative cross-session graph — it is not a separate, unprotected cloud call, and it carries the identical single-model-classification limitation described above. The cumulative fold graph carries no reliable entity types of its own (registry-derived relations have none), and this pass does not attempt to derive one: before each chunk's cloud call it runs `anonymize` (the SAME chain session-tier extraction uses), scoped against the same `scrub`. Because the local model's mapping keys are surface strings it produced independently of the fold graph's own canonical node-key text, each key is first reconciled to the graph's node identities via `canonical()` internally — a re-cased, separator-varied, or diacritic-varied key still resolves against the node it names, with the model's own placeholder preserved verbatim; an entry that matches no node in its chunk (or an ambiguous multiple) is dropped rather than guessed. This tier fails closed when the local anonymizer itself fails to parse (mirroring the extraction stage's guard); a local mapping that comes back empty is a legitimate "nothing in scope" verdict and proceeds, exactly like the extraction stage. The one additional guard here is narrower: when the local mapping names something but reconciliation onto this chunk's actual node keys drops all of it for a chunk with real (non-speaker) content, that residual is a classification/identity failure rather than evidence of out-of-scope content, and the chunk's cloud call is skipped (fail-closed) rather than sent unmasked, counted in `privacy_skipped_chunks`. **Operator opt-out is an explicit privacy-posture change at this tier**: an empty `sanitization.scrub` short-circuits before any local-anonymizer call and the chunk's triples egress to the cloud VERBATIM — this tier previously always ran the local anonymizer regardless of `scrub`, so an operator who had set `scrub: []` expecting the prior (accidentally-still-scrubbing) behavior now gets true unmasked egress at this tier; the fix aligns it with the session tier's and chat-egress's existing opt-out contract, but it is a real behavior change operators relying on the old default should be aware of. On the RESPONSE side, no unresolvable/orphan placeholder token ever reaches the graph — that fail-closed property is unchanged — but the unit it is enforced against is the individual item, not the whole chunk delta: every relation this tier's cloud response returns is treated as new (there is no local baseline here to fall back to), so `deanonymize_facts`'s residual sweep drops only the relation(s) actually carrying an unresolvable token, counted in `dropped_relations`; a `same_as` pair with either member unresolved is dropped the same way, per pair, via `deanonymize_text`. This is distinct from `privacy_skipped_chunks`, which fires on the OUTBOUND side before any cloud call is made — `dropped_relations` is the fail-closed backstop on what comes back. Neither outbound guard is a totality check on the OUTBOUND content: a chunk in which the model names only some of the real content, or whose mapping only partially survives reconciliation, still passes, and the unnamed/dropped names egress verbatim. The anonymous `speaker{N}` handle is never tokenised at this tier either (the local anonymizer prompt forbids mapping it, matching the session tier) — it reaches the payload bare by design, since it carries no identifying information. Accepted consequence: person-level `same_as` coreference (recognizing two surface forms of the same person, e.g. a nickname or honorific variant) can no longer be detected by the cloud judge once both surfaces are opaque tokens — the name-surface signal that coreference depends on is gone for people under the default `scrub`. Coreference for organizations, places, and other non-person entities is unaffected (those surfaces stay verbatim under the default `scrub`).
- **Cloud escalation can leak.** The sanitizer applied before escalation has two arms: a known-entity scrub (substitution against the speaker's graph entities) and a self-reference gate (encoder-based "is this about the speaker?" classifier with multilingual exemplars under `configs/personal_referent/`, falling back to an English token-set when the encoder isn't loaded). The self-reference gate fires only when an identified `speaker_id` is present — voice-resolved or post-greeting. Residual risk: the encoder operates on lexical/semantic shape; the local model can still rewrite a query in a form that embeds a personal fact while passing the gate. Cross-lingual transfer in the multilingual encoder lifts coverage past the languages with explicit exemplars (en/de today) but is not guaranteed for every locale or idiom — adding `<class>.<lang>.txt` exemplar files for production languages tightens the bound. Speaker name is sent structurally to the cloud persona and is not scrubbed by default.
- **LAN authentication is operator-provisioned.** When neither `PARAMEM_API_TOKEN` nor `mobile_pwa.enabled` is configured, conversational REST endpoints are accessible to any LAN peer (Security OFF posture); administrative endpoints remain 403 fail-closed until a credential is configured. Wyoming STT / TTS ports do not support protocol-level auth at all and rely on network-layer scoping (firewall rule) for access control.
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
