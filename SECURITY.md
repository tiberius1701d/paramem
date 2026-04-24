# Security

ParaMem is a personal memory service that stores conversational and personal facts as weight deltas in a local LoRA adapter, plus a small set of on-disk metadata files (registry, knowledge graph, session queue, voice profiles). It runs on a single host under a single admin and is designed for home / edge deployment — not multi-tenant or server-farm use.

This document describes what ParaMem defends, what it does not, the trust boundaries in the design, and the operator contract for running it securely. It is a living document; the security posture will tighten as work packages from the hardening plan land.

## 1. Data handled

| Artifact | Content | Format |
|---|---|---|
| Adapter weight tensors | Personal facts, preferences, episodic memories | `.safetensors` — opaque numerical tensors |
| Indexed key registry | Key identifiers, SimHash fingerprints, timestamps | JSON |
| Cumulative knowledge graph | Entities, predicates, relations | JSON (NetworkX) |
| Session queue | Transcript + speaker binding awaiting consolidation | JSON (atomic temp-file + rename) |
| Session snapshot | RAM state at graceful shutdown | Fernet-encrypted when a key is configured |
| Speaker profiles | Voice embeddings + disclosed names | JSON (biometric data — see §7) |
| Background trainer resume state | Epoch counter + checkpoint references | JSON |
| Adapter manifest sidecars | Base-model SHA, tokenizer fingerprint, LoRA shape | JSON |

Adapter weights are the dominant artifact by volume and sensitivity. They are numerical — not directly readable for facts — but also not probe-resistant (see §7).

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

paramem encrypt-infra                # wrap any existing plaintext infra files (PMEM1 envelope)
paramem migrate-to-age               # flip PMEM1 → age multi-recipient [daily, recovery]
systemctl --user restart paramem-server
```

After this the startup log reads `SECURITY: ON (age daily identity loaded, recovery recipient available)` and `/status` reports `encryption: on`. If anything in the chain fails, the server refuses to start with an actionable message rather than silently degrade — see §4 for the mode-consistency rules.

The two migration commands (`encrypt-infra` then `migrate-to-age`) are only needed on an existing deployment with plaintext or legacy-Fernet data on disk. A completely fresh install can skip both: just run `generate-key`, set the env var, and start the server — new writes land as age from the first consolidation onward.

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

**In scope** — we attempt to defend against:
- Filesystem read access by a non-root process on the same host
- Accidental backup or cloud-sync exposure of the data directory
- Physical theft of the host while powered off
- Careless maintainers, accidental commits, screenshots of on-disk state
- LAN-adjacent attackers sending unauthenticated requests to the server
- Prompt-injection attempts via voice input

**Out of scope** — ParaMem does not defend against:
- Attackers with live root or RAM access on the host
- Nation-state adversaries
- Supply-chain compromise of the Python runtime or of major pinned dependencies beyond version pinning
- Multi-user isolation on the same host (ParaMem is a single-admin service)
- Side-channel attacks on the CPU or GPU during inference

## 3. Trust boundaries

- **User voice → STT.** Raw audio arrives on a Wyoming protocol port. Transcript + speaker embedding cross into the FastAPI process on the shared asyncio event loop.
- **Home Assistant ↔ ParaMem.** A thin HA custom component POSTs to the `/chat` endpoint over HTTP on the LAN. Bearer-token authentication is opt-in via the `PARAMEM_API_TOKEN` environment variable. When unset, the server accepts any LAN request — this is announced at startup as an explicit open posture, not a silent one.
- **ParaMem → cloud.** Sanitized queries (and speaker name, as persona anchor) may be sent to a configured cloud agent for escalation or SOTA enrichment. This path is opt-in via config; nothing is sent without an active cloud configuration. Sanitization is regex-based and is documented as incomplete — see §7.
- **Adapter files at rest.** The on-disk artifacts listed in §1 live under the configured data directory. At-rest encryption is governed by the binary switch in §4.
- **Backup at rest.** Session snapshots are Fernet-encrypted when a master key is present. Other infrastructure metadata follows the Security-ON/OFF contract in §4.

## 4. Security modes

ParaMem operates in one of two modes, governed by the loaded key material. There are no partial states.

### Security ON
Any of the following combinations counts as ON:
- `PARAMEM_MASTER_KEY` is set (Fernet base64 key, 32 bytes of entropy) — legacy PMEM1 envelope path.
- `PARAMEM_DAILY_PASSPHRASE` is set AND `~/.config/paramem/daily_key.age` exists — age two-identity path. When `~/.config/paramem/recovery.pub` is also present, every new write is multi-recipient (daily + recovery).

All infrastructure metadata — registry, graph, queue, snapshots, speaker profiles, manifest sidecars, backup artifacts — is encrypted on disk and decrypted only into process RAM on load. The universal read path sniffs the envelope magic (`age-encryption.org/v1\n` or `PMEM1\n`) and routes to the appropriate decryptor, so a mixed on-disk state during an age migration is transparent to callers. On startup the server logs one of:
```
SECURITY: ON (age daily identity loaded, recovery recipient available)
SECURITY: ON (age daily identity loaded, recovery recipient missing — run `paramem generate-key` to re-enable multi-recipient writes)
SECURITY: ON (PARAMEM_MASTER_KEY set)
```
The age posture takes precedence when both Fernet and age identities are loaded (the transitional state during a migration).

### Security OFF
No key material is loaded. All infrastructure metadata is plaintext on disk. This is a **documented operator opt-out**, not a gap. On startup the server logs:
```
SECURITY: OFF (no key — all infrastructure metadata is plaintext on disk)
```
and surfaces `encryption: off` on the `/status` endpoint. The server does not silently degrade between modes: if a key is loaded but on-disk files are plaintext (or vice versa), startup refuses until an explicit `paramem encrypt-infra` / `paramem migrate-to-age` / `paramem decrypt-infra --i-accept-plaintext` migration is performed.

### Transitional state (age migration in progress)

When both `PARAMEM_MASTER_KEY` and the daily age identity are loaded and the on-disk store contains a mix of PMEM1 and age envelopes, the server accepts the state and logs a WARN naming the number of PMEM1 files still pending. New writes route through age as soon as the daily identity is loaded; PMEM1 files on disk are read via the Fernet path until the operator runs `paramem migrate-to-age` to rewrite them as age multi-recipient envelopes. The command is per-file atomic (`<path>.tmp` → fsync → rename → fsync parent) and idempotent — already-age files are skipped and re-runs are safe after a crash.

After a successful `migrate-to-age` the operator can remove `PARAMEM_MASTER_KEY` from the environment one release later. Fernet support is retained until then as the rollback safety net for backups taken under the old envelope format.

Refusal cases:
- age files on disk without the daily identity loaded → unreadable.
- PMEM1 files on disk without the Fernet master key → unreadable.
- Plaintext files alongside any encryption magic → startup refused regardless of which keys are loaded.
- `migrate-to-age` with `recovery.pub` missing → refused (would strip the recovery safety net). The `--allow-daily-only` flag opts out of this refusal but is strongly discouraged: losing the daily passphrase after such a run makes the data unrecoverable.

### Plaintext-by-design carve-outs

Three control-plane metadata files are kept plaintext in both modes:
- `data/ha/state/trial.json` — migration-trial marker (paths, hashes, timestamps).
- `data/ha/state/backup.json` — scheduled-backup runner status (paths, timestamps, counts).
- `data/ha/backups/<kind>/<ts>/*.meta.json` — backup artifact sidecars (timestamp, ciphertext SHA-256, key fingerprint, tier, label). The paired `*.bin.enc` payload remains encrypted.

None of these files contain user facts. Encrypting them would either brick recovery on key loss (`trial.json`, `backup.json`) or turn a wrong-key restore into a silent "backup not found" instead of a clear decrypt-error (backup sidecars). These carve-outs are the **only** exceptions to the binary-mode contract.

## 5. Recovery model

The security model follows BitLocker semantics: the master key is the only path to the data. Losing it is equivalent to losing the data. There is no backdoor, no author escrow, no cloud recovery service.

The deployment uses two keys:

1. **Daily access key.** A per-host daily identity (age X25519) stored on disk as a passphrase-wrapped envelope at `~/.config/paramem/daily_key.age` (mode `0600`, parent directory `0700`). The passphrase is provided via the `PARAMEM_DAILY_PASSPHRASE` environment variable — loaded from the operator's environment or a systemd drop-in. Hardware-backed unlock (TPM2, Windows DPAPI, libsecret) is a future upgrade path behind the same loader interface and does not change the operator-facing contract. Rotatable without operator intervention.
2. **Recovery key.** A *separate* age X25519 identity (bech32 `AGE-SECRET-KEY-1…`), minted alongside the daily identity by `paramem generate-key`. The public recipient is persisted at `~/.config/paramem/recovery.pub` (mode `0644`) so every new envelope lists it alongside the daily recipient. The secret is printed *once* to stderr at generation time with a BitLocker-style warning — operators must confirm they have saved it before the key files are written — and is never persisted on this device. Store it offline: printed paper, metal seed plate, password-manager secure note, or a safe. Used only when the daily access path fails (passphrase loss, disk loss, hardware replacement). Survives hardware replacement; restoring decrypts the store and enrolls a fresh daily identity on the new host.

Both keys decrypt the same data. Loss of the daily key is routine (rotate it). Loss of the recovery key — with the daily path also unavailable — is unrecoverable.

**Rotation.** `paramem rotate-daily` mints a fresh daily identity, re-encrypts every age infrastructure file to `[daily_new, recovery]`, and atomically swaps the new daily key file into place. The recovery recipient is preserved. `paramem rotate-recovery` mints a fresh recovery identity, prints the new bech32 secret once with the same refuse-without-confirm UX as `generate-key`, and re-encrypts every file to `[daily, recovery_new]`. Both commands are crash-safe: per-file atomic rename plus a rotation manifest at `~/.config/paramem/rotation.manifest.json` that records pending vs done files, so a crash resumes from where it left off (`rotate-recovery` excepted — the print-once secret cannot be resumed and must be restarted cleanly).

**Hardware replacement.** `paramem restore --recovery-key-file <path>` is the entry point after losing the original device. Given the recovery bech32 from paper, it sanity-checks against an on-disk age envelope, mints a fresh daily identity (new operator-supplied passphrase), writes `daily_key.age` + `recovery.pub` to the new machine, and re-encrypts every age file to `[daily_new, recovery]`. The recovery identity is reused on the envelopes — it is the thing that authorised the restore, and the operator's paper copy remains valid. Crash-safe via the same rotation-manifest mechanism; a typo in the bech32 aborts before any on-disk mutation. Distinct from `paramem backup-restore`, which restores a backup archive over REST.

**Backup restore across key rotation.** Every encrypted backup sidecar carries the 16-hex-char fingerprint of the key it was written under. `POST /backup/restore` compares the sidecar's fingerprint against the current master key; a mismatch is refused with HTTP 400 `fingerprint_mismatch`. To restore a backup taken under a prior key, keep that key set in `PARAMEM_MASTER_KEY` and pass `force_rotate_key=true` (REST body) or `--force-rotate-key` (CLI) — the server logs a WARN and proceeds. Backups written while Security was OFF have no stored fingerprint and always restore.

Biometric unlocks (Windows Hello, fingerprint, FIDO2) are supported as *access conveniences* for the daily path only. They are not a recovery mechanism: biometrics unlock a sealed key on specific hardware; they do not regenerate the key on a new device. Any sensible deployment pairs biometric-unlocked daily access with a printed recovery artifact.

## 6. Operator responsibilities

ParaMem is a single-admin service. The operator — the person running the server — is responsible for:

- Generating and storing the master key. If you set `PARAMEM_MASTER_KEY`, also save a recovery copy offline. Do not rely on a single storage location for the only copy.
- Scoping LAN exposure. Set `PARAMEM_LISTEN_IP` to the specific host interface that should accept incoming requests, and `PARAMEM_NAS_IP` to scope the Windows Firewall rule to the Home Assistant source host. Unset values default to an open posture with a loud startup warning.
- Setting `PARAMEM_API_TOKEN` to require bearer-token authentication on all REST endpoints. When unset, the server accepts any request from a reachable peer.
- Managing `.env` and per-secret files under `~/.config/paramem/secrets/` with file mode `0600` and directory mode `0700`. The server refuses to start if permissions are looser.
- Scoping the Home Assistant long-lived access token to a dedicated, minimal-privilege HA user — not to a full admin.
- Handling backups. A backup that captures the data directory but not the master-key source defeats the encryption.

## 7. Known limitations

The security properties are honest, not aspirational. The following are the limitations an operator should understand before deploying.

- **Adapter probe resistance is limited.** An attacker with (a) the adapter weight file, (b) the base model, and (c) knowledge of relevant entity names can extract a meaningful fraction of stored facts through systematic probing. The adapter is opaque to grep but not opaque to a model that asks the right questions. This is inherent to any LoRA-based parametric memory — the knowledge must be accessible to be useful.
- **Runtime exposure is identical to RAG.** While the server is reasoning over a recalled fact, that fact lives as plaintext in GPU / CPU RAM inside the server process. Any system reasoning over private data has this property; we isolate it to one process behind a local API rather than streaming recalled context to external tools.
- **Cloud escalation can leak.** The sanitizer applied before escalation is regex-based and does not inspect every variant of personal reference. Deploying with cloud escalation enabled means accepting a residual risk that the local model rewrites a query in a form that embeds a personal fact. Speaker name is sent structurally to SOTA persona and is not scrubbed by default.
- **LAN authentication is operator-provisioned.** When `PARAMEM_API_TOKEN` is unset, the REST endpoints are accessible to any LAN peer. Wyoming STT / TTS ports do not support protocol-level auth at all and rely on network-layer scoping (firewall rule) for access control.
- **Key loss is total.** No backdoor, no recovery service, no escrow. The recovery key *is* the backdoor; losing it is losing the data.
- **Biometrics are convenience, not security.** Biometric unlock binds to specific hardware and specific OS sessions. A new device or a TPM clear invalidates the daily path. Biometrics cannot be rotated if compromised and are not cryptographic secrets.
- **Supply chain pinning is not auditing.** Dependency versions are pinned in `pyproject.toml`, including the CUDA-specific `bitsandbytes` development wheel required for RTX 50-series hardware. Pinning prevents silent updates but does not constitute a reviewed supply chain.
- **Voice embeddings are biometric data.** Under GDPR Article 9 (EU) voice embeddings are special-category personal data. They are encrypted at rest under Security-ON; losing the recovery key is privacy-protective for this data, but *sharing* the key exports biometrics.

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
