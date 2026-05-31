# Security

> ⚠️ **Work in progress — not a finished security story.** The encryption-at-rest machinery described in this document is partial by design. It raises the bar for specific *separation* scenarios (see §2) and does not attempt to defend against an attacker who gains operator-level access to the host.
>
> **Two operator-level paths to plaintext, both trivial:**
>
> 1. **The Python codebase is editable.** `paramem/` on disk is plain source. An attacker with write access to the installed package path — which the operator user has by definition — can neutralize encryption with a three-line edit to `paramem/backup/encryption.py::envelope_encrypt_bytes` and restart the server. There is no code-signing, no bytecode integrity check, no TPM-backed attestation.
> 2. **Config-level data exfiltration via `debug: true`.** The operator can flip one line in `configs/server.yaml` and restart. From that point on, every consolidation cycle writes plaintext copies of user facts into `data/ha/sessions/*.jsonl` and `data/ha/debug/cycle_*/{graph_snapshot.json,episodic_qa_snapshot.json,procedural_rels_snapshot.json}`. No code edit, no crypto break — just the legitimate debug path used against the intent of a Security-ON deployment. This is intentional behaviour for debugging (see §4 carve-outs); it is named here because an attacker with config-write access can use it as a data-extraction primitive. (The simulate-mode graph store — `graph.json` under `<adapter_dir>/<tier>/` — is encrypted by default and respects `require_encryption`; it is NOT a debug carve-out.)
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
| Speaker profiles | Voice embeddings + disclosed names | JSON (biometric data — see §7) |
| Background trainer resume state | Epoch counter + checkpoint references | JSON |
| Adapter manifest sidecars | Base-model SHA, tokenizer fingerprint, LoRA shape | JSON |

Adapter weights are the dominant artifact by volume and sensitivity. They are numerical — not directly readable for facts — but also not probe-resistant. Under Security ON they are encrypted at rest as age envelopes alongside the JSON metadata. The indexed-recall path still requires the encrypted registry for key enumeration; weight encryption adds defense-in-depth against blob-copy attackers. See §4 and §7 for the full picture.

## 1.5. Quick start: enable encryption

A fresh install runs **Security-OFF** by default — infrastructure metadata is plaintext on disk and the server emits a loud startup warning (`SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)`). To enable:

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

After this the startup log reads `SECURITY: ON (age daily identity loaded, recovery recipient available)` and `/status` reports `encryption: on`. If anything in the chain fails, the server refuses to start with an actionable message rather than silently degrade — see §4 for the mode-consistency rules.

New writes land as age envelopes from the first consolidation onward. A pre-existing plaintext data directory must be reconciled manually before startup; the server refuses to mix plaintext with age envelopes on disk.

Day-to-day key operations after that point:

| Command | When to use |
|---|---|
| `paramem change-passphrase` | Change the passphrase that wraps `daily_key.age`. Identity itself is unchanged — no re-encrypt of the data store. |
| `paramem rotate-daily` | Periodic hygiene, or suspected compromise of the daily identity (not just the passphrase). Mints a fresh X25519 identity and re-encrypts every envelope. Recovery recipient is preserved. |
| `paramem rotate-recovery` | Rotate the printed paper (new bech32, old one invalidated). Daily identity is unchanged. |
| `paramem restore --recovery-key-file PATH` | Hardware replacement — given the recovery bech32 from paper, mints a fresh daily identity on the new host and re-keys every envelope. |
| `paramem dump PATH` | Decrypt a single envelope for debugging. |

All lifecycle commands are per-file atomic + idempotent + resumable via a crash-safe rotation manifest (`~/.config/paramem/rotation.manifest.json`).

## 2. Threat model

**Trust assumption.** The admin / operator of the host is a trusted authority. ParaMem does not attempt to protect data from an attacker who has the operator's OS credentials, process-memory access, or write access to the installed Python package. The operator holds the daily passphrase and the daily-key file; these travel in the same trust domain as the data they protect. A hostile process running as the operator can read the decrypted store from RAM, modify `paramem/` source (e.g. replace `envelope_encrypt_bytes` with a pass-through), or read the wrapped daily key plus passphrase and decrypt at rest. None of these are defended against.

**What Security ON actually buys — the narrow, honest claim.** When the data directory is separated from the key material (decoupled from the running server), the data directory alone is not decryptable. Concretely, encryption at rest narrows the blast radius in exactly these separation scenarios:

- **Accidental cloud-sync of `data/ha/` alone** (OneDrive, iCloud, rsync to NAS) — data appears at the sync destination but is unreadable without the key material kept at `~/.config/paramem/` and `PARAMEM_DAILY_PASSPHRASE`.
- **Backup exfiltration** (a backup copy of the data directory without the config dir) — same story.
- **Filesystem read by a different OS user on the same host** (mode `0600` on `~/.config/paramem/daily_key.age` and `.env` enforced at startup).
- **Theft of a powered-off host IF the passphrase is not co-located** (depends on operator discipline — typically a weak defense because `.env` lives on the same disk).

**In scope beyond data-at-rest:**
- LAN-adjacent attackers sending unauthenticated requests — mitigated by the bearer-token auth layer (see §4.1).
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
- **Home Assistant ↔ ParaMem.** A thin HA custom component POSTs to the `/chat` endpoint over HTTP on the LAN. Bearer-token authentication has four postures governed by `PARAMEM_API_TOKEN` and `mobile_pwa.enabled` — see §4.1 for the full model. When auth is OFF the server accepts any LAN request, announced at startup as an explicit open posture, not a silent one.
- **ParaMem → cloud.** Sanitized queries (and speaker name, as persona anchor) may be sent to a configured cloud agent for escalation or SOTA enrichment. This path is opt-in via config; nothing is sent without an active cloud configuration. The cloud-egress sanitizer has two arms: a known-entity scrub against the speaker's graph entities (language-agnostic) and an encoder-based "is this about the speaker?" classifier with multilingual exemplars under `configs/personal_referent/`, falling back to an English token-set when the encoder isn't loaded. Coverage scales with the exemplar files; see §6 and §7 for the operator's responsibility and the residual risk.
- **Routing-time intent classifier (privacy property).** The intent classifier in `paramem/server/intent.py` runs *before* any retrieval and *outside* the PA reasoning path. Under `intent.mode: llm` (default) the local Mistral 7B is invoked with the focused classifier section of `configs/prompts/pa_voice.txt` only — `_personalize_prompt` is **not** applied, so the speaker name is not injected into the classifier system message and the query is classified on content alone. Under `intent.mode: embeddings` the sentence-encoder cosine match never receives speaker identity. Either path keeps routing-time classification orthogonal to personal-context exposure: the speaker-name leak surface is the *response-time* PA path (which intentionally uses `_personalize_prompt`), gated by the cloud-egress sanitizer above.
- **Adapter files at rest.** The on-disk artifacts listed in §1 live under the configured data directory. At-rest encryption is governed by the binary switch in §4.
- **Backup at rest.** Session snapshots and every other piece of infrastructure metadata follow the Security-ON/OFF contract in §4 — encrypted as age envelopes when the daily identity is loaded, and plaintext only when no key is configured.

## 4. Security modes

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

### 4.1 Authentication postures

The auth layer is independent of the encryption mode — it governs which REST requests are accepted, not how data is written to disk.  Two knobs interact: `PARAMEM_API_TOKEN` (environment variable; shared bearer token) and `mobile_pwa.enabled` (config; per-user bearer tokens).  The startup log always emits exactly one `AUTH:` line naming the active posture:

| Posture | Condition | Effect |
|---------|-----------|--------|
| **OFF** | Neither configured | All REST endpoints — conversational and admin (`/gpu/*`, `/consolidate`, `/backup/*`, etc.) — accept any request without credentials. The server is open by design: the auth middleware stamps **admin** scope on every pass-through request, so `require_admin` allows. Startup emits a loud `AUTH: OFF` warning. Default for new installs. |
| **ON-shared** | `PARAMEM_API_TOKEN` set | All endpoints require the single shared bearer token. The shared token always carries **admin** scope (full reach). Requests are **unattributed** — no `speaker_id` is attached. |
| **ON-per-user** | `mobile_pwa.enabled: true` | All endpoints require a per-user opaque bearer token. **Fail-closed.** Each token carries a **scope** — `chat` (the secure default, including pre-scope-field tokens) or `admin`. Admin scope is required for operational endpoints. The `chat` scope reaches `/chat`, `/voice`, `/push/*`, and `/status`. |
| **ON-both** | Both configured | Shared token (admin scope) checked first; per-user store is the fallback. |

**Token scope dimension.** Within ON-shared / ON-per-user / ON-both, every accepted token additionally carries a capability scope:

| Scope | Endpoints reached | How to mint |
|-------|------------------|-------------|
| `admin` | All endpoints (conversational + operational) | Shared `PARAMEM_API_TOKEN`, or `mint-user-token <speaker> --scope admin`, or `--unattributed --scope admin --force-admin` |
| `chat` | `/chat`, `/voice`, `/push/*`, `/status` only | `mint-user-token <speaker> --scope chat` (the default), or `--unattributed --scope chat` |

Secure defaults: missing `scope` field in a pre-existing token record reads as `chat` at runtime. In an ON posture a `chat`-scope (or otherwise non-admin) request is denied by `require_admin`. Auth-OFF mode is the deliberate open posture — the middleware stamps `admin` scope on pass-through requests so all endpoints stay reachable without credentials.

**Live reload.** Revocation and scope changes (re-mint + revoke) take effect on the running server without a restart: `UserTokenStore` re-reads `user_tokens.json` on the next authenticated request when the file's mtime changes. Encryption mode (Security ON/OFF) is honoured on reload. Accepted cross-process revocation race window: the narrow in-flight window between a revoke write and the next request; not a meaningful attack surface for typical deployment cadences.

**Revoking unattributed tokens.** `revoke_speaker()` skips entries whose `speaker_id` is `None` and raises `ValueError` if called with `None` — preventing accidental bulk-revocation. Use `revoke-user-token --label <label>` to revoke an unattributed token by its device label.

The shared token covers legacy deployments (HA custom component with a fixed key in `.env`). The per-user path covers the mobile PWA, where each device is issued its own token that resolves to a speaker identity on every request.

**Per-user token properties:**

- Tokens are opaque random secrets. The plaintext token is displayed once at mint time (via `paramem mint-user-token`; see `README.md`) and never stored or logged. Only the `sha256(token)` hash is persisted on disk, in `user_tokens.json`.
- `user_tokens.json` follows the deployment-wide encryption posture: plaintext under Security OFF, age-encrypted when the daily key is loaded. It is covered by the startup mode-consistency check — a plaintext credential file alongside a loaded key is refused at startup.

**Web Push infrastructure files (when `mobile_pwa.push_enabled: true`):**

- `vapid_keys.json` — EC P-256 VAPID private key (PEM). Auto-generated on first startup when push is enabled; auto-loaded on subsequent startups. Both files follow the same encryption posture as `user_tokens.json`: plaintext under Security OFF, age-encrypted under Security ON, covered by the startup mode-consistency scan via `infra_paths()`.
- `push_subscriptions.json` — per-speaker Web Push endpoint registrations. Schema: `{"version":1, "subscriptions": {"<speaker_id>": [{endpoint, keys:{p256dh,auth}}...]}}`.
- **VAPID key stability:** rotating `vapid_keys.json` invalidates all existing browser push subscriptions (browsers will not receive notifications until they re-subscribe). Treat the keypair as effectively immutable once browsers have subscribed. Key rotation is intentionally out of scope.
- **Notification-only ping posture:** no personal content passes through the push relay. The push payload is intentionally empty (or carries only a generic title); real content is fetched by the client after the user taps the notification.
- **Revocation** is per-token or per-speaker and takes effect immediately on the next request. Revoking the last active token in the store keeps the auth layer fail-closed rather than silently reverting to open access.
- **Token carriers:** `Authorization: Bearer <token>` HTTP header, or an `httpOnly; Secure; SameSite=Strict` cookie set by the server during the PWA onboarding flow (same-origin — the PWA is served from the ParaMem server itself). Header takes precedence over cookie.

**Path exemptions:** the PWA shell (`/` and `/app`) and the web app manifest (`/manifest.json`) are exempt from token checks so the login page loads before a token exists. All other endpoints enforce the active posture.

### Fail-loud opt-in: `security.require_encryption`

The Security-OFF opt-out is the operator's choice. Deployments that want a misconfiguration to fail loud rather than silently land plaintext on disk can set `security.require_encryption: true` in `configs/server.yaml`. When set, the server refuses to start unless the daily identity is loadable — a uniform startup gate covering every feature that writes to disk (snapshots, checkpoint shards, backups, infrastructure metadata). Default is `false` (the AUTO-everywhere posture described above).

### Refusal cases

- age files on disk without the daily identity loaded → startup refused with a clear message pointing at `PARAMEM_DAILY_PASSPHRASE` + the daily-key file path.
- Plaintext files alongside age envelopes → startup refused; reconcile the store before restart.
- Plaintext files while the daily identity is loaded → startup refused; migrate the store or unset the passphrase.

### Recovering from a startup-gate refusal

The error message from each refusal points at one of two paths.

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

### Plaintext-by-design carve-outs

Some on-disk artifacts are intentionally kept plaintext in both modes:

- `data/ha/state/trial.json` — migration-trial marker (paths, hashes, timestamps). Encrypting would brick recovery on key loss.
- `data/ha/state/backup.json` — scheduled-backup runner status. Same reasoning.
- `data/ha/backups/<kind>/<ts>/*.meta.json` — backup artifact sidecars (timestamp, ciphertext SHA-256, tier, label). Encrypting would turn a wrong-key restore into a silent "backup not found" instead of a clear decrypt error. The paired `*.bin.enc` payload remains encrypted.
- `data/ha/sessions/*.jsonl` and `data/ha/sessions/archive/*.jsonl` — raw conversation transcripts, written only when `debug: true`. Explicit operator opt-in to plaintext persistence for inspection; the whole point of debug mode is to see the transcripts with `tail`/`cat`/`grep`.
- `data/ha/debug/cycle_<N>/graph_snapshot.json`, `data/ha/debug/cycle_<N>/episodic_qa_snapshot.json`, `data/ha/debug/cycle_<N>/procedural_rels_snapshot.json` — per-cycle debug artifacts, written only when `debug: true`. Always plaintext, inspection-first, regardless of Security posture. The simulate-mode output under `<adapter_dir>/<tier>/` (`graph.json`, `simhash_registry.json`, `indexed_key_registry.json`) is a SEPARATE, encrypted store and does NOT use this carve-out.
- `data/ha/debug/cycle_<N>/sessions/<session_id>/graph_snapshot.json` and `data/ha/debug/cycle_<N>/sessions/<session_id>/procedural_graph_snapshot.json` — per-session extraction snapshots written by the consolidation loop when `debug: true` and `save_cycle_snapshots` is enabled. Same plaintext-inspection rationale as the per-cycle aggregates above.
None of the first three carry user facts. The next two *do* carry user facts but are produced only at operator request via the `debug` flag. Adapter weight blobs carry user facts as numerical patterns; see §7 for the probe-resistance limit that encryption does not fully close.

## 5. Recovery model

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

## 6. Operator responsibilities

ParaMem is a single-admin service. The operator — the person running the server — is responsible for:

- Generating and storing key material. Run `paramem generate-key` to mint the daily identity (stored passphrase-wrapped on this host) and the recovery identity (printed once — save it offline). Do not rely on a single storage location for the only copy of the recovery bech32.
- Scoping LAN exposure. Set `PARAMEM_LISTEN_IP` to the specific host interface that should accept incoming requests, and `PARAMEM_NAS_IP` to scope the Windows Firewall rule to the Home Assistant source host. Unset values default to an open posture with a loud startup warning.
- Choosing the appropriate auth posture (§4.1) for the deployment. Set `PARAMEM_API_TOKEN` to protect the server with a single shared token; enable `mobile_pwa.enabled: true` for per-user tokens that carry speaker identity. When neither is configured the server accepts any request from a reachable peer, with a loud startup warning.
- Managing `.env` and per-secret files under `~/.config/paramem/secrets/` with file mode `0600` and directory mode `0700`. The server refuses to start if permissions are looser.
- Scoping the Home Assistant long-lived access token to a dedicated, minimal-privilege HA user — not to a full admin.
- Handling backups. A backup that captures the data directory but not the master-key source defeats the encryption.
- Reviewing the cloud-egress classifier exemplar files for the languages the deployment serves. The sanitizer's first-person check is encoder-based with multilingual exemplars under `configs/personal_referent/<class>.<lang>.txt`; coverage on a language without dedicated exemplars relies on cross-lingual transfer in the multilingual encoder and may miss idioms or low-resource phrasings. For deployments serving non-English speakers, add a file pair (`about_speaker.<lang>.txt` + `not_about_speaker.<lang>.txt`) and verify with a probe set before going live. The same applies to `configs/sentence_types/` for the abstention gate.

## 7. Known limitations

The security properties are honest, not aspirational. The following are the limitations an operator should understand before deploying.

- **Adapter probe resistance is limited.** An attacker with (a) the adapter weight file, (b) the base model, and (c) knowledge of relevant entity names can extract a meaningful fraction of stored facts through systematic probing. The adapter is opaque to grep but not opaque to a model that asks the right questions. This is inherent to any LoRA-based parametric memory — the knowledge must be accessible to be useful.
- **Weight encryption narrows blob-copy risk but does not close probe surface.** ParaMem encrypts the key registry (`registry/key_metadata.json`) and the per-tier SimHash registries (`<store>/<tier>/simhash_registry.json`) and — under Security ON — the LoRA weight tensors (`adapter_model.safetensors`) as age envelopes. The registry encryption blocks the **systematic** extraction path: the indexed-recall template requires knowing the key string (`graph17`, `proc4`, …), and without the registry an attacker cannot enumerate keys. Encrypting the weight tensors adds defense-in-depth against an attacker who copies only the `adapters/` subtree without the key material. What remains: an attacker with (a) the decrypted weights, (b) the base model, and (c) knowledge of entity names can still extract facts through targeted natural-language probing ("what did Alex say about X?"), membership inference, and continued fine-tuning — these require running inference on the weights and are not closed by encryption alone.
- **Runtime exposure is identical to RAG.** While the server is reasoning over a recalled fact, that fact lives as plaintext in GPU / CPU RAM inside the server process. Any system reasoning over private data has this property; we isolate it to one process behind a local API rather than streaming recalled context to external tools.
- **Extraction-stage SOTA enrichment narrows but does not eliminate PII egress.** When `consolidation.extraction_noise_filter` is set to a SOTA provider (default `""` = disabled), the extraction pipeline runs a local-anonymization pass on person names — replacing them with placeholders (`{Person_N}`) — before sending the anonymized transcript and facts to the cloud for coreference resolution, compound splitting, and dedup. A forward-path verification step (`verify_anonymization`, default on) blocks the cloud call when a real name leaked past the local anonymizer. The default cloud-egress scope is `{"person"}`; place names, organization names, phone numbers, email addresses, and free-form secrets (API keys, passwords, tokens) flow to the SOTA provider inside the anonymized payload. The scope is configurable via `consolidation.extraction_pii_scope`, but broadening it strips the structured context a personal assistant relies on to be useful — privacy-vs-utility is the operator's call, made consciously and per deployment. Document ingestion goes through the same pipeline; for documents that may contain machine credentials, keep `extraction_noise_filter=""` or scrub credentials before ingest.
- **Cloud escalation can leak.** The sanitizer applied before escalation has two arms: a known-entity scrub (substitution against the speaker's graph entities) and a self-reference gate (encoder-based "is this about the speaker?" classifier with multilingual exemplars under `configs/personal_referent/`, falling back to an English token-set when the encoder isn't loaded). The self-reference gate fires only when an identified `speaker_id` is present — voice-resolved or post-greeting. Residual risk: the encoder operates on lexical/semantic shape; the local model can still rewrite a query in a form that embeds a personal fact while passing the gate. Cross-lingual transfer in the multilingual encoder lifts coverage past the languages with explicit exemplars (en/de today) but is not guaranteed for every locale or idiom — adding `<class>.<lang>.txt` exemplar files for production languages tightens the bound. Speaker name is sent structurally to SOTA persona and is not scrubbed by default.
- **LAN authentication is operator-provisioned.** When neither `PARAMEM_API_TOKEN` nor `mobile_pwa.enabled` is configured, the REST endpoints are accessible to any LAN peer (Security OFF posture). Wyoming STT / TTS ports do not support protocol-level auth at all and rely on network-layer scoping (firewall rule) for access control.
- **Key loss is total.** No backdoor, no recovery service, no escrow. The recovery key *is* the backdoor; losing it is losing the data.
- **Biometrics are convenience, not security.** Biometric unlock binds to specific hardware and specific OS sessions. A new device or a TPM clear invalidates the daily path. Biometrics cannot be rotated if compromised and are not cryptographic secrets.
- **Supply chain pinning is not auditing.** Dependency versions are pinned in `pyproject.toml`, including the CUDA-specific `bitsandbytes` development wheel required for RTX 50-series hardware. Pinning prevents silent updates but does not constitute a reviewed supply chain.
- **Voice embeddings are biometric data.** Under GDPR Article 9 (EU) voice embeddings are special-category personal data. They are encrypted at rest under Security-ON; losing the recovery key is privacy-protective for this data, but *sharing* the key exports biometrics.

> The biggest limit — that the Python package is not attested and can be trivially tampered with by an operator-level attacker — is named in the top-of-document disclaimer, not repeated here.

## 8. Vulnerability reporting

Please do not open a public GitHub issue for suspected security vulnerabilities.

Contact: **Tobias Preusser — `tobias.preusser75@gmail.com`**.

When reporting, include:
- Affected version / commit
- Deployment configuration (Security ON / OFF, cloud enabled / disabled, HA connected)
- A clear reproduction or the minimum data needed to reason about the issue

ParaMem is research software maintained by a single author. There is no formal SLA for response times. Responsible disclosure is appreciated; public coordination will be on a best-effort basis.

## 9. References

- `README.md` — project overview, configuration, setup
- `spec.md` — architecture + design decisions, including the F5.4 privacy/security feature track
- `paramem/server/auth.py`, `paramem/server/user_tokens.py`, `paramem/server/secret_store.py` — runtime entry points for the boundaries described above
- The internal hardening plan and empirical probe results live outside the public repository; enquiries should be routed through the disclosure channel in §8.
