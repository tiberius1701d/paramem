# Security

> ⚠️ **Work in progress — not a finished security story.** The encryption-at-rest machinery described in this document is partial by design. It raises the bar for specific *separation* scenarios (see §2) and does not attempt to defend against an attacker who gains operator-level access to the host.
>
> **Two operator-level paths to plaintext, both trivial:**
>
> 1. **The Python codebase is editable.** `paramem/` on disk is plain source. An attacker with write access to the installed package path — which the operator user has by definition — can neutralize encryption with a three-line edit to `paramem/backup/encryption.py::envelope_encrypt_bytes` and restart the server. There is no code-signing, no bytecode integrity check, no TPM-backed attestation.
> 2. **Config-level data exfiltration via `debug: true`.** The operator can flip one line in `configs/server.yaml` and restart. From that point on, every consolidation cycle writes plaintext copies of user facts into `data/ha/sessions/*.jsonl` and `data/ha/debug/cycle_*/{graph_snapshot.json,episodic_qa_snapshot.json,procedural_rels_snapshot.json}`. No code edit, no crypto break — just the legitimate debug path used against the intent of a Security-ON deployment. This is intentional behaviour for debugging (see §4 carve-outs); it is named here because an attacker with config-write access can use it as a data-extraction primitive. (The simulate-mode JSON store under `paths.simulate` is encrypted by default and respects `require_encryption`; it is NOT a debug carve-out.)
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
- LAN-adjacent attackers sending unauthenticated requests — mitigated by `PARAMEM_API_TOKEN`.
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
- **Home Assistant ↔ ParaMem.** A thin HA custom component POSTs to the `/chat` endpoint over HTTP on the LAN. Bearer-token authentication is opt-in via the `PARAMEM_API_TOKEN` environment variable. When unset, the server accepts any LAN request — this is announced at startup as an explicit open posture, not a silent one.
- **ParaMem → cloud.** Sanitized queries (and speaker name, as persona anchor) may be sent to a configured cloud agent for escalation or SOTA enrichment. This path is opt-in via config; nothing is sent without an active cloud configuration. Sanitization is regex-based and is documented as incomplete — see §7.
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
- `data/ha/debug/cycle_<N>/graph_snapshot.json`, `data/ha/debug/cycle_<N>/episodic_qa_snapshot.json`, `data/ha/debug/cycle_<N>/procedural_rels_snapshot.json` — per-cycle debug artifacts, written only when `debug: true`. Always plaintext, inspection-first, regardless of Security posture. The simulate peer-storage output under `paths.simulate/<tier>/keyed_pairs.json` is a SEPARATE, encrypted store and does NOT use this carve-out.
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

Biometric unlocks (Windows Hello, fingerprint, FIDO2) are supported as *access conveniences* for the daily path only. They are not a recovery mechanism: biometrics unlock a sealed key on specific hardware; they do not regenerate the key on a new device. Any sensible deployment pairs biometric-unlocked daily access with a printed recovery artifact.

## 6. Operator responsibilities

ParaMem is a single-admin service. The operator — the person running the server — is responsible for:

- Generating and storing key material. Run `paramem generate-key` to mint the daily identity (stored passphrase-wrapped on this host) and the recovery identity (printed once — save it offline). Do not rely on a single storage location for the only copy of the recovery bech32.
- Scoping LAN exposure. Set `PARAMEM_LISTEN_IP` to the specific host interface that should accept incoming requests, and `PARAMEM_NAS_IP` to scope the Windows Firewall rule to the Home Assistant source host. Unset values default to an open posture with a loud startup warning.
- Setting `PARAMEM_API_TOKEN` to require bearer-token authentication on all REST endpoints. When unset, the server accepts any request from a reachable peer.
- Managing `.env` and per-secret files under `~/.config/paramem/secrets/` with file mode `0600` and directory mode `0700`. The server refuses to start if permissions are looser.
- Scoping the Home Assistant long-lived access token to a dedicated, minimal-privilege HA user — not to a full admin.
- Handling backups. A backup that captures the data directory but not the master-key source defeats the encryption.

## 7. Known limitations

The security properties are honest, not aspirational. The following are the limitations an operator should understand before deploying.

- **Adapter probe resistance is limited.** An attacker with (a) the adapter weight file, (b) the base model, and (c) knowledge of relevant entity names can extract a meaningful fraction of stored facts through systematic probing. The adapter is opaque to grep but not opaque to a model that asks the right questions. This is inherent to any LoRA-based parametric memory — the knowledge must be accessible to be useful.
- **Weight encryption narrows blob-copy risk but does not close probe surface.** ParaMem encrypts the SimHash registry (`<store>/<tier>/keyed_pairs.json`) and — under Security ON — the LoRA weight tensors (`adapter_model.safetensors`) as age envelopes. The registry encryption blocks the **systematic** extraction path: the indexed-recall template requires knowing the key string (`graph17`, `proc4`, …), and without the registry an attacker cannot enumerate keys. Encrypting the weight tensors adds defense-in-depth against an attacker who copies only the `adapters/` subtree without the key material. What remains: an attacker with (a) the decrypted weights, (b) the base model, and (c) knowledge of entity names can still extract facts through targeted natural-language probing ("what did Alex say about X?"), membership inference, and continued fine-tuning — these require running inference on the weights and are not closed by encryption alone.
- **Runtime exposure is identical to RAG.** While the server is reasoning over a recalled fact, that fact lives as plaintext in GPU / CPU RAM inside the server process. Any system reasoning over private data has this property; we isolate it to one process behind a local API rather than streaming recalled context to external tools.
- **Extraction-stage SOTA enrichment narrows but does not eliminate PII egress.** When `consolidation.extraction_noise_filter` is set to a SOTA provider (default `""` = disabled), the extraction pipeline runs a local-anonymization pass on person names — replacing them with placeholders (`{Person_N}`) — before sending the anonymized transcript and facts to the cloud for coreference resolution, compound splitting, and dedup. A forward-path verification step (`verify_anonymization`, default on) blocks the cloud call when a real name leaked past the local anonymizer. The default cloud-egress scope is `{"person"}`; place names, organization names, phone numbers, email addresses, and free-form secrets (API keys, passwords, tokens) flow to the SOTA provider inside the anonymized payload. The scope is configurable via `consolidation.extraction_pii_scope`, but broadening it strips the structured context a personal assistant relies on to be useful — privacy-vs-utility is the operator's call, made consciously and per deployment. Document ingestion goes through the same pipeline; for documents that may contain machine credentials, keep `extraction_noise_filter=""` or scrub credentials before ingest.
- **Cloud escalation can leak.** The sanitizer applied before escalation is regex-based and does not inspect every variant of personal reference. Deploying with cloud escalation enabled means accepting a residual risk that the local model rewrites a query in a form that embeds a personal fact. Speaker name is sent structurally to SOTA persona and is not scrubbed by default.
- **LAN authentication is operator-provisioned.** When `PARAMEM_API_TOKEN` is unset, the REST endpoints are accessible to any LAN peer. Wyoming STT / TTS ports do not support protocol-level auth at all and rely on network-layer scoping (firewall rule) for access control.
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
- `paramem/server/auth.py`, `paramem/server/secret_store.py` — runtime entry points for the boundaries described above
- The internal hardening plan and empirical probe results live outside the public repository; enquiries should be routed through the disclosure channel in §8.
