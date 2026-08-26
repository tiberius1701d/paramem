"""Model-agnostic loader with QLoRA and multi-adapter support.

This module isolates all model-specific logic behind a clean interface.
Swapping the base model requires only changing the model_id in config.
"""

import contextlib
import logging
import os
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from paramem.utils.config import AdapterConfig, ModelConfig
from paramem.utils.tiers import MAIN_TIERS
from paramem.utils.tokens import RenderedPrompt, encode_rendered
from paramem.utils.vram_guard import safe_empty_cache, vram_measure

logger = logging.getLogger(__name__)


@contextmanager
def grad_checkpointing_disabled(model):
    """Disable gradient checkpointing for the scope and restore it to its
    ENTRY state on exit. HF silently disables the KV cache while checkpointing
    is active, so any generate() during a training-configured session must
    toggle it off; a model that entered with checkpointing OFF exits OFF."""
    was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    if was_checkpointing:
        model.gradient_checkpointing_disable()
    try:
        yield
    finally:
        if was_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )


@contextmanager
def base_model_inference(model: PeftModel):
    """Run generation on the base weights in a clean inference state.

    Yields with the active LoRA adapter disabled so the base model drives
    output, and with gradient checkpointing turned off for the duration of the
    scope.  HF silently disables the KV cache whenever gradient checkpointing is
    active, which makes ``model.generate()`` produce garbage; disabling it here
    keeps the cache live.  The entry state of both is restored on exit —
    checkpointing is re-enabled only when it was already on at scope entry (a
    model that enters with checkpointing OFF exits OFF).  Without this,
    structured-generation calls (QA distillation, extraction, dedup) drift
    across cycles: the train path leaves checkpointing enabled after episodic
    training, so the next ``generate`` runs without the KV cache and falls
    through to a degraded fallback path.

    The base model's object identity is fixed at load time
    (:func:`load_base_model` / :func:`ensure_resident_tiers`) and it is
    always wrapped with at least one tier resident, so *model* is always a
    ``PeftModel`` here — never a raw base model to fall through to.

    Args:
        model: The live ``PeftModel`` to run inference on.

    Raises:
        TypeError: *model* is not a ``PeftModel``.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(f"base_model_inference requires a PeftModel, got {type(model).__name__}")

    with grad_checkpointing_disabled(model):
        with model.disable_adapter():
            yield


def generate_adapter_off(
    model,
    tokenizer,
    messages: list[dict],
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
) -> str:
    """Run one deterministic, adapter-off generate and return the decoded text.

    THE one adapter-off raw-generate implementation shared by every local
    structured-output call that is not a full conversational reply
    (:func:`paramem.server.intent._classify_via_llm` and
    :func:`paramem.server.temporal_selection.select_date_groups`) — it
    lives here, alongside :func:`base_model_inference` (the primitive it
    runs inside), rather than mirrored per caller.
    :func:`paramem.server.inference._generate_local_reply` is a separate,
    larger reply-shaping pipeline (system prompt assembly, cap-hit
    detection) and is not folded in.

    Two-step tokenization (chat template → string → tensors) mirrors the
    production inference path: feeding ``apply_chat_template``'s tensor
    output straight into ``generate()`` crashes on transformers >= 5
    because that call returns a ``BatchEncoding`` (no ``.shape``
    attribute), so the template is rendered to a string first and
    tokenized as its own step.

    Runs inside :func:`base_model_inference`, which disables gradient
    checkpointing (so the KV cache stays live, since HF silently disables
    it while checkpointing is active) and, on a ``PeftModel``, disables
    the active adapter for the duration — every caller of this function
    runs its judgment on the base weights, never biased by an adapter
    trained for a different objective (e.g. the PA adapter's personal-key
    recall bias).

    Args:
        model: The (optionally PEFT-wrapped) model to generate from.
        tokenizer: The model's tokenizer.
        messages: Chat-template messages (``[{"role", "content"}, ...]``).
        max_new_tokens: Generation cap.
        temperature: Sampling temperature; ``0.0`` selects greedy decode
            (``do_sample=False`` always — every caller of this function
            wants deterministic structured output).

    Returns:
        The decoded generated tail (prompt tokens excluded), with special
        tokens stripped.

    Raises:
        Exception: Any tokenization, generation, or decode failure
            propagates — this function has no fail-open contract of its
            own; each caller owns its own try/except and its own
            fail-shape (``Intent.UNKNOWN``, ``DateSelection(all=True, ...)``).
    """
    prompt = render_chat_prompt(messages, tokenizer, add_generation_prompt=True)
    inputs = encode_rendered(tokenizer, prompt, return_tensors="pt").to(model.device)

    with base_model_inference(model):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=temperature,
                pad_token_id=getattr(tokenizer, "pad_token_id", None)
                or getattr(tokenizer, "eos_token_id", None),
            )

    generated = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


@dataclass
class _BackupScope:
    """Handle yielded by :func:`tier_backup_scope`.

    The base model's object identity is fixed at load time — ``create_adapter``
    always mutates in place and never returns a new object — so this scope
    does not track the model reference; the caller's own ``model`` stays
    valid for the whole scope without resyncing.

    ``vram`` carries the ``free_before``/``free_after``/``delta``/``total``
    mapping :func:`~paramem.utils.vram_guard.vram_measure` captured around
    this scope's own snapshot (the backup adapter's ``create_adapter`` +
    ``copy_adapter_weights`` pair) — populated once, at scope entry, and
    left untouched afterwards. This scope records nothing itself; the
    telemetry write belongs to the caller, which owns the fold's telemetry
    directory and cycle stamp.
    """

    vram: "Mapping[str, int]" = field(default_factory=dict)


def _switch_off(model: PeftModel, adapter_name: str, tiers: tuple[str, ...]) -> None:
    """Move the active adapter off ``adapter_name`` onto a resident main tier.

    No-op when ``adapter_name`` is not currently active — callers must never
    delete an adapter without first confirming it is not the active one.
    """
    if active_adapter_name(model) != adapter_name:
        return
    for tier in tiers:
        if tier in model.peft_config:
            model.set_adapter(tier)
            return


@contextmanager
def tier_backup_scope(model: PeftModel, config: AdapterConfig, tier: str) -> Iterator[_BackupScope]:
    """Snapshot *tier*'s resident adapter into a transient ``<tier>_backup``
    adapter, restored on any exception.

    Manages the BACKUP adapter ONLY — never interim adapters, never the
    on-disk scratch a training event leaves behind.  On entry, if *tier* is
    resident in ``model.peft_config`` it is copied into a ``<tier>_backup``
    adapter (any pre-existing, leaked backup is discarded first so a stale
    snapshot can never be restored over good weights).  On ANY exception
    raised inside the ``with`` body, the tier is restored from its backup
    before the ORIGINAL exception propagates unchanged. *tier* itself is
    never deleted or recreated inside this scope — a config mismatch is
    caught by ``ensure_adapter_matching`` BEFORE this scope is entered, and
    every tier's transient staging slot warm-starts uniformly regardless of
    door (an ordinary full event and the RECONCILE door alike, decided
    inside :func:`~paramem.training.trainer.train_adapter`; its own
    cold-start arm fires only on a LoRA shape mismatch or first boot) — so
    the only thing this scope's restore ever unwinds is a tier trained warm
    in place (the funnel's staging-copy-and-promote); a failure can never
    leave production holding a zero-init adapter.  On every exit (success
    or exception), the backup adapter is freed — it is VRAM-only and never
    read by anything outside this scope.

    No tier's weights are activated live until the whole bundle has written
    (the tandem go-live design), so this scope's only job is the unwind of
    a tier that never went live: it covers exactly one tier's training,
    never a whole event's worth of tiers.

    Args:
        model: The live ``PeftModel`` carrying *tier*, if resident.  Must
            already be a ``PeftModel`` — this CM does not perform the
            initial ``get_peft_model`` wrap.
        config: *tier*'s ``AdapterConfig`` — sizes the backup identically.
        tier: The single tier name to snapshot (the driver's tier loop
            variable).

    Yields:
        _BackupScope: carries ``.vram`` — the
            ``free_before``/``free_after``/``delta``/``total`` mapping
            :func:`~paramem.utils.vram_guard.vram_measure` captured around
            the snapshot itself; an empty ``{}`` when *tier* was not
            resident (no snapshot taken).  This scope records nothing —
            the caller writes it into the fold telemetry ring.  *model*
            itself is never reassigned — its object identity is fixed at
            load time, so the caller's own reference stays valid throughout.

    Raises:
        RuntimeError: if ``model`` is not a ``PeftModel`` (runtime contract
            check — NOT an ``assert``, which is stripped under ``-O``).
    """
    if not isinstance(model, PeftModel):
        raise RuntimeError("tier_backup_scope requires a PeftModel with main tiers resident")

    scope = _BackupScope()
    backup = f"{tier}_backup"
    snapshotted = False
    try:
        if tier in model.peft_config:
            if backup in model.peft_config:
                # Leaked from a prior aborted event — discard before
                # re-snapshotting so the stale backup can never clobber good
                # weights on a later restore.
                _switch_off(model, backup, MAIN_TIERS)
                model.delete_adapter(backup)
            with vram_measure("backup_creation") as _vram:
                create_adapter(model, config, backup)
                copy_adapter_weights(model, src=tier, dst=backup)
            scope.vram = dict(_vram)
            snapshotted = True
        yield scope
    except BaseException:
        # Restore only when this scope actually snapshotted the tier — a
        # tier whose backup was never populated (partial-enter double-fault)
        # must not be touched.
        if snapshotted and backup in model.peft_config and tier in model.peft_config:
            try:
                copy_adapter_weights(model, src=backup, dst=tier)
            except Exception:  # noqa: BLE001  # boundary: best-effort restore on
                # an exception path — the original exception below must
                # always propagate unchanged, so a restore failure here is
                # logged and swallowed, not raised.
                logger.warning("tier_backup_scope: restore failed for tier %s", tier, exc_info=True)
        raise  # re-raise the ORIGINAL exception, untransformed
    finally:
        # Every exit (success or exception) frees the backup — it is
        # VRAM-only and outlives nothing outside this scope.  This block
        # runs on the exception path too (Python runs `finally` after
        # `except` re-raises), so a teardown failure here must NEVER replace
        # the in-flight exception.  Every step is therefore best-effort: log
        # and continue, never raise — a leaked backup adapter is vastly
        # preferable to a misrouted exception.
        try:
            _switch_off(model, backup, MAIN_TIERS)
        except Exception:  # noqa: BLE001  # boundary: best-effort teardown; must
            # never replace an in-flight exception — see the block comment above.
            logger.warning("tier_backup_scope: switch-off-backup failed", exc_info=True)
        if backup in model.peft_config:
            try:
                model.delete_adapter(backup)
            except Exception:  # noqa: BLE001  # boundary: best-effort teardown; must
                # never replace an in-flight exception — see the block comment above.
                logger.warning(
                    "tier_backup_scope: could not delete backup %s", backup, exc_info=True
                )


# Cache for system role support per tokenizer class to avoid repeated try/except
_system_role_cache: dict[str, bool] = {}


def supports_system_role(tokenizer: PreTrainedTokenizer) -> bool:
    """Check if a tokenizer's chat template actually renders system content.

    Some templates (Gemma 2) reject system messages with an error.
    Others (Mistral v0.3) silently drop them. Both cases need folding.
    We verify by checking that a marker string appears in the rendered output.
    """
    key = getattr(tokenizer, "name_or_path", id(tokenizer))
    if key not in _system_role_cache:
        marker = "SYSROLE_CHECK_MARKER"
        try:
            rendered = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": marker},
                    {"role": "user", "content": "t"},
                    {"role": "assistant", "content": "t"},
                ],
                tokenize=False,
            )
            _system_role_cache[key] = marker in rendered
        except Exception:
            _system_role_cache[key] = False
    return _system_role_cache[key]


def adapt_messages(messages: list[dict], tokenizer: PreTrainedTokenizer) -> list[dict]:
    """Adapt chat messages for the model's template.

    If the model doesn't support system roles (e.g. Gemma 2), folds
    system content into the first user message. Otherwise returns
    messages unchanged.
    """
    if supports_system_role(tokenizer):
        return messages

    system_parts = []
    other_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            other_messages.append(msg)

    if not system_parts:
        return other_messages

    # Prepend system content to first user message
    adapted = []
    system_prefix = "\n\n".join(system_parts)
    prepended = False
    for msg in other_messages:
        if msg["role"] == "user" and not prepended:
            adapted.append({"role": "user", "content": f"{system_prefix}\n\n{msg['content']}"})
            prepended = True
        else:
            adapted.append(msg)

    if not prepended:
        # No user message to prepend to — add as first user message
        adapted.insert(0, {"role": "user", "content": system_prefix})

    return adapted


def render_chat_prompt(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    *,
    add_generation_prompt: bool = True,
) -> RenderedPrompt:
    """THE one production chat-template renderer.

    Applies :func:`adapt_messages` internally (folding system content for a
    tokenizer whose template doesn't render a system role, e.g. Mistral
    v0.3), then ``tokenizer.apply_chat_template(adapted, tokenize=False,
    add_generation_prompt=...)``, and wraps the result in
    :class:`~paramem.utils.tokens.RenderedPrompt` — the marker type
    :func:`~paramem.utils.tokens.encode_rendered` requires before it will
    tensorize the text.

    This is the ONLY production call site of ``apply_chat_template`` for
    text that will later be generated on. The one deliberate exception is
    :func:`supports_system_role`, which renders a probe string that is
    never encoded — it stays a direct ``apply_chat_template`` call.

    Args:
        messages: Chat message dicts (``[{"role", "content"}, ...]``),
            UN-adapted — this function applies :func:`adapt_messages` so
            callers must not adapt a second time.
        tokenizer: The tokenizer whose chat template renders the text.
        add_generation_prompt: Forwarded to ``apply_chat_template``. ``True``
            (default) appends the assistant-turn generation prefix; callers
            rendering a full (already-completed) conversation for training
            pass ``False``.

    Returns:
        The rendered prompt string as a :class:`RenderedPrompt`.
    """
    adapted = adapt_messages(messages, tokenizer)
    rendered = tokenizer.apply_chat_template(
        adapted, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return RenderedPrompt(rendered)


def _get_quantization_config(model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
    """Build quantization config from model settings."""
    if model_config.quantization == "none":
        return None

    compute_dtype = getattr(torch, model_config.compute_dtype)

    extra_kwargs = {}
    if model_config.cpu_offload:
        extra_kwargs["llm_int8_enable_fp32_cpu_offload"] = True

    if model_config.quantization == "int8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            **extra_kwargs,
        )

    if model_config.quantization == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            **extra_kwargs,
        )

    if model_config.quantization == "fp4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=compute_dtype,
            **extra_kwargs,
        )

    raise ValueError(f"Unknown quantization type: {model_config.quantization}")


def _verify_device_placement(model: PreTrainedModel, model_config: ModelConfig) -> None:
    """Verify model is on the expected device after loading.

    When cpu_offload=False, ALL parameters must be on GPU.
    When cpu_offload=True, a mix of GPU and CPU is expected.
    Raises RuntimeError if placement violates the configuration.
    """
    devices = {str(p.device) for p in model.parameters()}
    param_count = sum(p.numel() for p in model.parameters())

    if model_config.cpu_offload:
        logger.info(
            "Model loaded: %s (%.1fM params, quantization=%s, devices=%s, cpu_offload=True)",
            model_config.model_id,
            param_count / 1e6,
            model_config.quantization,
            devices,
        )
    else:
        cpu_params = sum(p.numel() for p in model.parameters() if "cpu" in str(p.device))
        if cpu_params > 0:
            raise RuntimeError(
                f"Model {model_config.model_id} has {cpu_params / 1e6:.1f}M params on CPU "
                f"but cpu_offload=False. This means the model does not fit in GPU VRAM. "
                f"Either free GPU memory, enable cpu_offload, or use a smaller model."
            )
        logger.info(
            "Model loaded: %s (%.1fM params, quantization=%s, device=cuda, cpu_offload=False)",
            model_config.model_id,
            param_count / 1e6,
            model_config.quantization,
        )


def _apply_wsl2_async_load_workaround() -> None:
    """Disable Transformers' threaded weight loading if requested via env var.

    WSL2's dxg paravirt layer can fail with ENOMEM (dxgkio_make_resident)
    when multiple threads call tensor.to('cuda') concurrently during model
    loading. Setting HF_DEACTIVATE_ASYNC_LOAD=1 forces sequential loading,
    which eliminates the race. This is a no-op if the env var is not set.
    """
    if os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1":
        logger.debug("HF_DEACTIVATE_ASYNC_LOAD=1: threaded weight loading disabled")


def load_base_model(
    model_config: ModelConfig,
    adapters: Mapping[str, AdapterConfig],
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a quantized base model, wrap it with every tier in *adapters*, and
    return the tokenizer.

    Supports CPU offload for models that don't fit entirely in GPU VRAM
    (e.g. Gemma 2 9B on 8GB).

    On WSL2 with RTX 50-series GPUs, Transformers' threaded weight loading
    can race the dxg memory mapper. Set HF_DEACTIVATE_ASYNC_LOAD=1 in .env
    to force sequential loading if you hit "CUDA driver error: device not ready".

    The base model's object identity is fixed here: the returned model is
    always a ``PeftModel`` carrying every tier in *adapters*, and every
    adapter operation afterwards mutates that one object in place — nothing
    downstream ever returns, rebinds, or unwraps it again.

    Args:
        model_config: Base-model load settings (quantization, device map,
            offload).
        adapters: The tiers that exist, in ``MAIN_TIERS`` order.  Non-empty
            — :func:`ensure_resident_tiers` raises ``ValueError`` on an
            empty map, since an adapter-less ``PeftModel`` cannot run
            ``forward``/``generate``/``disable_adapter``.  The active
            adapter on return is the FIRST key.
            PRODUCTION SOURCE: ``config.tier_config_map()`` —
            ``app._load_model_into_state`` passes it. Experiments/scripts
            pass their own map for the tier(s) they will mount or train.

    Returns:
        The wrapped ``PeftModel`` and its tokenizer.
    """
    logger.info("Loading base model: %s", model_config.model_id)

    _apply_wsl2_async_load_workaround()

    quantization_config = _get_quantization_config(model_config)

    if model_config.cpu_offload:
        # Intentional partial offload (e.g. Gemma 2 9B on 8GB GPU)
        load_kwargs = {
            "device_map": "auto",
            "max_memory": {
                0: model_config.max_memory_gpu,
                "cpu": model_config.max_memory_cpu,
            },
            "trust_remote_code": model_config.trust_remote_code,
        }
    else:
        # Force all layers to GPU — fail loudly if it doesn't fit
        load_kwargs = {
            "device_map": {"": 0},
            "trust_remote_code": model_config.trust_remote_code,
        }

    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["torch_dtype"] = getattr(torch, model_config.compute_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_id,
        **load_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_id,
        trust_remote_code=model_config.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Llama 3.1 declares multiple EOS tokens (turn-end markers); take the
        # first since pad_token_id must be a single int.
        eos = model.config.eos_token_id
        model.config.pad_token_id = eos[0] if isinstance(eos, list) else eos

    _verify_device_placement(model, model_config)

    model = ensure_resident_tiers(model, adapters)

    return model, tokenizer


def unload_model(model, tokenizer=None) -> None:
    """Free a model from GPU/CPU memory.

    Routes through ``safe_empty_cache`` (:mod:`paramem.utils.vram_guard`, a
    leaf module with no ``paramem`` dependency of its own) so cuBLAS
    workspaces and allocator-pool slack are released too — otherwise
    process teardown leaves a ~280 MiB ghost CUDA context that
    ``_gpu_has_compute_processes`` sees on the next boot and forces
    gpu_conflict mode.
    """
    del model
    if tokenizer is not None:
        del tokenizer
    safe_empty_cache()
    logger.info("Model unloaded, CUDA cache cleared")


def lora_shape_fields(adapter_config: AdapterConfig) -> dict:
    """Return the ``AdapterConfig`` fields that determine a LoRA adapter's
    tensor topology: ``r`` (rank — determines tensor shape directly),
    ``lora_alpha`` (the scaling factor PEFT applies at forward time), and
    ``target_modules`` (which layers are adapted at all).

    The single source both :func:`create_adapter` (which builds a real
    ``peft.LoraConfig`` from these) and :func:`ensure_adapter_matching`
    (which compares a resident adapter's ``LoraConfig`` against these) read
    from, so a future field added here to make it shape-relevant is picked
    up by the mismatch guard automatically rather than silently escaping it.

    Deliberately excludes ``dropout``: it changes training regularization,
    not tensor shape or topology, so it is not compared by
    :func:`ensure_adapter_matching` — a warm-kept resident adapter's
    ``lora_dropout`` stays whatever it was created with. An operator dropout
    edit in config therefore takes effect only the next time the adapter is
    actually (re)created (first boot, or a shape-relevant mismatch), never
    on a routine warm-kept fold — the reconcile door (``/reconsolidate``)
    warm-starts like every other fold and applies no dropout edit by
    itself — see ``configs/server.yaml.example``'s adapters section.
    """
    return {
        "r": adapter_config.rank,
        "lora_alpha": adapter_config.alpha,
        "target_modules": list(adapter_config.target_modules)
        if adapter_config.target_modules
        else [],
    }


def create_adapter(
    model: PeftModel,
    adapter_config: AdapterConfig,
    adapter_name: str = "default",
) -> None:
    """Create a new LoRA adapter on *model*, in place, and activate it.

    Adds the adapter via ``add_adapter`` — never re-wraps: every tier is
    created through :func:`ensure_resident_tiers` at load time, so this
    function only ever adds a NEW adapter onto an already-wrapped model,
    never the first (wrap) adapter. Re-wrapping an already-wrapped model
    does not reset it — it aliases the same ``peft_config`` dict and
    accumulates adapters (PEFT ``tuners_utils.py:281-293``) — which is why
    this function has a ``PeftModel`` precondition rather than a
    raw-model fallback.

    Args:
        model: The live ``PeftModel``. Mutated in place; nothing is
            returned.
        adapter_config: LoRA shape (rank, alpha, target_modules, dropout).
        adapter_name: Name to register the new adapter under.

    Raises:
        TypeError: *model* is not a ``PeftModel``.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(f"create_adapter requires a PeftModel, got {type(model).__name__}")

    lora_config = LoraConfig(
        **lora_shape_fields(adapter_config),
        lora_dropout=adapter_config.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Add adapter to existing PeftModel — no re-wrapping, which would
    # cause nested tensor names on save (breaking reload).
    model.add_adapter(adapter_name, lora_config)
    model.set_adapter(adapter_name)

    # Ensure base_model_name_or_path is set for save/reload
    if model.peft_config[adapter_name].base_model_name_or_path is None:
        base_name = getattr(model.get_base_model().config, "_name_or_path", None)
        if base_name:
            model.peft_config[adapter_name].base_model_name_or_path = base_name

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Adapter '%s' created: rank=%d, trainable=%.2fM / %.1fM total (%.2f%%)",
        adapter_name,
        adapter_config.rank,
        trainable_params / 1e6,
        total_params / 1e6,
        100 * trainable_params / total_params,
    )


def ensure_resident_tiers(
    model: "PreTrainedModel | PeftModel",
    adapters: Mapping[str, AdapterConfig],
) -> PeftModel:
    """Wrap or extend *model* so every tier in *adapters* is resident.

    THE sole :func:`peft.get_peft_model` call site in ``paramem``, and the
    one place a raw base model becomes a ``PeftModel``. Called from exactly
    two places: :func:`load_base_model` (first wrap, at boot) and
    :func:`~paramem.server.app._remount_adapters_from_disk` (a live
    ``PeftModel`` that :func:`detach_adapters` may just have emptied — a
    detach immediately followed by the create that repairs it here, with
    nothing using the model in between).

    A raw ``PreTrainedModel`` is wrapped with the FIRST entry of *adapters*
    via ``get_peft_model`` and every remaining entry is added via
    :func:`create_adapter`. An already-wrapped ``PeftModel`` has any entry
    not already resident in ``peft_config`` added the same way — entries
    already resident are left untouched (their trained weights, if any, are
    preserved). Either way the active adapter on return is the FIRST key of
    *adapters*, fixing ``peft_config`` order and the active adapter together
    immediately after a wrap.

    Args:
        model: Raw base model or an already-wrapped ``PeftModel``.
        adapters: ``{tier_name: AdapterConfig}``, in the order tiers should
            be created — ``MAIN_TIERS`` order in production
            (``ServerConfig.tier_config_map()``).

    Returns:
        The wrapped ``PeftModel`` carrying every tier in *adapters* (plus
        any tier already resident on an already-wrapped *model*).

    Raises:
        ValueError: *adapters* is empty — an adapter-less ``PeftModel``
            cannot run ``forward``, ``generate`` or ``disable_adapter``.
    """
    if not adapters:
        raise ValueError("ensure_resident_tiers requires at least one tier")

    names = list(adapters)

    if isinstance(model, PeftModel):
        for name in names:
            if name not in model.peft_config:
                create_adapter(model, adapters[name], name)
        model.set_adapter(names[0])
        return model

    # The ONE raw base model -> PeftModel transition. Every subsequent
    # adapter, on this model or any other already-wrapped one, goes through
    # create_adapter's add_adapter branch instead.
    first_name = names[0]
    first_config = adapters[first_name]
    lora_config = LoraConfig(
        **lora_shape_fields(first_config),
        lora_dropout=first_config.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config, adapter_name=first_name)

    if peft_model.peft_config[first_name].base_model_name_or_path is None:
        base_name = getattr(peft_model.get_base_model().config, "_name_or_path", None)
        if base_name:
            peft_model.peft_config[first_name].base_model_name_or_path = base_name

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "Adapter '%s' created: rank=%d, trainable=%.2fM / %.1fM total (%.2f%%)",
        first_name,
        first_config.rank,
        trainable_params / 1e6,
        total_params / 1e6,
        100 * trainable_params / total_params,
    )

    for name in names[1:]:
        create_adapter(peft_model, adapters[name], name)
    peft_model.set_adapter(first_name)
    return peft_model


def mount_adapter(model: PeftModel, slot: str | Path, adapter_name: str) -> None:
    """Mount an on-disk adapter slot onto *model* as *adapter_name*, in place.

    The one "mount a saved slot" primitive: resolves *slot* through
    :func:`_adapter_slot_for_load` (transparently decrypting an age-encrypted
    ``adapter_model.safetensors`` into memfd-backed plaintext; a no-op for a
    plaintext slot), calls ``model.load_adapter``, and patches
    ``peft_config[adapter_name].base_model_name_or_path`` from the base
    model's own config when PEFT left it ``None`` — its behaviour for
    second-and-later adapters (:func:`create_adapter` carries the same
    patch for adapters created rather than mounted).

    Args:
        model: The live ``PeftModel``. *slot* is mounted onto it in place;
            nothing is returned.
        slot: The directory that DIRECTLY contains ``adapter_model.safetensors``
            (not a parent holding kind/timestamp subdirectories).
        adapter_name: Name to register the mounted adapter under.

    Raises:
        TypeError: *model* is not a ``PeftModel``.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(f"mount_adapter requires a PeftModel, got {type(model).__name__}")

    with _adapter_slot_for_load(Path(slot)) as load_path:
        model.load_adapter(str(load_path), adapter_name=adapter_name)

    if model.peft_config[adapter_name].base_model_name_or_path is None:
        base_name = getattr(model.get_base_model().config, "_name_or_path", None)
        if base_name:
            model.peft_config[adapter_name].base_model_name_or_path = base_name


def switch_adapter(model: PeftModel, adapter_name: str) -> None:
    """Switch the active adapter on a multi-adapter model."""
    model.set_adapter(adapter_name)
    logger.debug("Switched to adapter: %s", adapter_name)


def active_adapter_name(model: PeftModel) -> Optional[str]:
    """Return the single active adapter name, normalizing PEFT's str/list return.

    PEFT 0.18+ normally exposes ``model.active_adapter`` as a string, but some
    layouts return a list.
    """
    raw = model.active_adapter
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


def drop_adapter_slot(model: PeftModel, name: str, *, fallback_adapter: str) -> None:
    """Delete transient PEFT slot *name* from *model* if resident.

    Switch-off-before-delete: PEFT refuses to leave the model with no active
    adapter, so deleting *name* while it is the ACTIVE adapter is only safe
    once *fallback_adapter* (a production tier guaranteed resident) is
    confirmed switched onto. No-op when *name* is already absent from
    ``model.peft_config``.

    When *name* is currently active, the delete proceeds ONLY when
    *fallback_adapter* is resident AND the switch to it actually lands
    (re-checked via :func:`active_adapter_name` after the switch attempt — a
    raised switch is treated identically to an absent fallback). Otherwise
    the delete is SKIPPED entirely and the leaked slot is logged: deleting
    the model's own active adapter with no confirmed successor would leave
    PeftModel with a stale/absent active config — the exact state the
    project's PEFT rule forbids (never delete the last/active adapter
    without a switch already landed; see ``paramem/server/gates.py``'s sole-
    adapter guard and ``paramem/memory/interim_adapter.py``'s SOLE-ADAPTER
    TRAP NOTE). The trade: a caller that mints a transient slot can be left
    holding a leaked one when its own fallback is broken, but the lifecycle
    backstop (``paramem.training.trainer.assert_staging_absent`` /
    ``_ensure_staging_slot``) surfaces that leak loudly at the next training
    event rather than this primitive silently breaking the live model.

    When *name* is NOT the active adapter, no switch is needed and the
    delete always proceeds.

    The one implementation of the "delete a transient slot" primitive —
    shared by the staging lifecycle (``paramem.training.trainer``), the donor
    build's transient slot, and any other caller that mints a PEFT adapter
    for the duration of one operation and must tear it down afterward.
    """
    if name not in model.peft_config:
        return
    if active_adapter_name(model) == name:
        if fallback_adapter not in model.peft_config:
            logger.warning(
                "drop_adapter_slot: skipping delete of active adapter %r — fallback "
                "%r is not resident; leaving %r in place to avoid an active-adapter-"
                "less PeftModel (the lifecycle backstop will catch the leaked slot "
                "at the next training event)",
                name,
                fallback_adapter,
                name,
            )
            return
        try:
            switch_adapter(model, fallback_adapter)
        except Exception:  # noqa: BLE001  # boundary: PEFT's own switch call is an
            # external API that can genuinely fail on a sufficiently broken model
            # state; a failed switch must be treated identically to an absent
            # fallback (skip the delete), never propagate out of this primitive.
            logger.warning(
                "drop_adapter_slot: switch to fallback %r failed — skipping delete "
                "of active adapter %r (the lifecycle backstop will catch the leaked "
                "slot at the next training event)",
                fallback_adapter,
                name,
                exc_info=True,
            )
            return
        if active_adapter_name(model) != fallback_adapter:
            logger.warning(
                "drop_adapter_slot: switch to fallback %r did not land — skipping "
                "delete of active adapter %r",
                fallback_adapter,
                name,
            )
            return
    model.delete_adapter(name)


def detach_adapters(model: PeftModel, names: Iterable[str]) -> list[str]:
    """Delete every adapter in *names* from a live PeftModel, deterministically.

    Before the first delete, the active adapter is moved onto a survivor —
    the first of :data:`~paramem.utils.tiers.MAIN_TIERS` order that is
    resident and NOT in *names*, else any resident adapter not in *names*,
    else no switch (the caller is emptying peft_config deliberately and
    owns the restore).
    PEFT's delete_adapter silently reassigns the active adapter when the
    deleted one was active, and leaves it STALE when nothing survives, so the
    switch-before-delete lives here rather than at each call site.

    The switch runs unconditionally whenever a survivor exists — not only
    when the current active adapter happens to be one of *names* — so the
    post-reap active adapter is deterministic for every caller, not just the
    ones that happen to already be sitting on the survivor.

    Returns the sorted names actually deleted; ``[]`` when none were
    resident. Never raises on an absent name. A caller holding a model that
    may not be a ``PeftModel`` (e.g. the disk venue's bare graph store) must
    check that itself before calling — see
    :func:`~paramem.memory.interim_adapter.unload_interim_adapters`'s
    explicit ``model is None`` branch.

    Args:
        model: The live ``PeftModel``.
        names: Adapter names to delete.  Names absent from
            ``model.peft_config`` are silently skipped — never raises.

    Returns:
        Sorted list of adapter names actually deleted from
        ``model.peft_config``.

    Raises:
        TypeError: *model* is not a ``PeftModel``.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(f"detach_adapters requires a PeftModel, got {type(model).__name__}")

    names_set = set(names)
    resident = sorted(n for n in names_set if n in model.peft_config)
    if not resident:
        return []

    survivor: Optional[str] = None
    for tier in MAIN_TIERS:
        if tier in model.peft_config and tier not in names_set:
            survivor = tier
            break
    if survivor is None:
        for candidate in model.peft_config:
            if candidate not in names_set:
                survivor = candidate
                break
    if survivor is not None:
        model.set_adapter(survivor)
        logger.debug("detach_adapters: switched active adapter to survivor '%s'", survivor)

    for name in resident:
        model.delete_adapter(name)
        logger.info("Deleted adapter from PEFT: %s", name)

    return resident


def save_adapter(
    model: PeftModel,
    path: str | Path,
    adapter_name: str,
    *,
    manifest=None,
) -> Path:
    """Save a specific adapter to disk via the atomic slot-dir path.

    Thin forwarder to :func:`atomic_save_adapter`.  All callers receive
    slot-dir layout (``path/<ts>/adapter_*.*``) automatically.

    Args:
        model: PeftModel whose adapter weights are saved.
        path: Parent directory that will hold the timestamped slot.
        adapter_name: Name of the adapter to save.
        manifest: Optional :class:`paramem.adapters.manifest.AdapterManifest`
            to write alongside the adapter files.  When ``None``, no
            ``meta.json`` is written.

    Returns:
        Path to the final (promoted) slot directory, as returned by
        :func:`atomic_save_adapter`.
    """
    return atomic_save_adapter(model, path, adapter_name, manifest=manifest)


def atomic_save_adapter(
    model: PeftModel,
    target_dir: str | Path,
    adapter_name: str,
    *,
    manifest=None,
) -> Path:
    """Save adapter atomically into a timestamped slot directory.

    The PEFT-specific payload writer handed to
    :func:`paramem.adapters.slot.write_slot`, which owns the promotion
    sequence (pending-dir creation, collision-safe timestamp, manifest
    digest stamping, fsync, atomic rename) shared by every payload kind.
    This function's own closure performs only the weight-specific write:

    1. ``model.save_pretrained(pending_slot, selected_adapters=[adapter_name])``.
       PEFT may write ``pending_slot/<adapter_name>/`` (nested) or directly
       into ``pending_slot/`` (flat — some PEFT versions).
    2. **Flatten inside ``pending_slot``**: if ``pending_slot/<adapter_name>/``
       exists, iterate its children and rename each up one level into
       ``pending_slot/``, then rmdir the now-empty nested directory.  If the
       nested directory is absent, this step is a no-op.
    3. **Encrypt ``adapter_model.safetensors`` in-place** via
       :func:`_encrypt_adapter_safetensors`.  When the daily age identity is
       loaded, the plaintext tensor bytes are replaced by an age envelope.
       When no key is configured the file is rewritten unchanged (plaintext
       pass-through, zero overhead).

    ``write_slot`` then computes the plaintext SHA-256 of the resulting
    ``adapter_model.safetensors``, stamps it into *manifest*'s
    ``payload.sha256``, writes ``meta.json`` (when *manifest* is not
    ``None``), and promotes the slot — every caller of this function
    therefore inherits the payload digest for free.

    All staging happens inside ``.pending/`` which
    :func:`paramem.backup.backup.sweep_orphan_pending` cleans on startup.

    Args:
        model: PeftModel whose adapter weights are saved.
        target_dir: Adapter-kind directory.  Slots are created as
            ``target_dir/<ts>/``.
        adapter_name: Name of the adapter (must be in
            ``model.peft_config``).
        manifest: Optional :class:`paramem.adapters.manifest.AdapterManifest`
            to write into the pending slot as ``meta.json`` before
            promotion.  When ``None``, no ``meta.json`` is written.

    Returns:
        Path to the final (promoted) slot directory.
    """
    from paramem.adapters.slot import write_slot

    def _write_payload(pending_slot: Path) -> None:
        # PEFT save into pending slot
        model.save_pretrained(str(pending_slot), selected_adapters=[adapter_name])

        # Flatten inside pending_slot — PEFT may write
        # <pending_slot>/<adapter_name>/adapter_*.*
        nested = pending_slot / adapter_name
        if nested.exists() and nested.is_dir():
            for child in list(nested.iterdir()):
                child.rename(pending_slot / child.name)
            nested.rmdir()

        # Encrypt adapter_model.safetensors in-place (age when key loaded,
        # plaintext pass-through when no key is configured).
        _encrypt_adapter_safetensors(pending_slot)

    final_slot = write_slot(Path(target_dir), manifest=manifest, write_payload=_write_payload)
    logger.info("Adapter '%s' saved to slot %s", adapter_name, final_slot)
    return final_slot


# ---------------------------------------------------------------------------
# Adapter safetensors encryption helpers
# ---------------------------------------------------------------------------

_SAFETENSORS_FILENAME = "adapter_model.safetensors"


def _encrypt_adapter_safetensors(slot: Path) -> None:
    """Encrypt ``adapter_model.safetensors`` inside *slot* in-place.

    Reads the plaintext tensor file written by PEFT's ``save_pretrained``,
    passes the bytes through :func:`paramem.backup.encryption.envelope_encrypt_bytes`
    (age multi-recipient when a daily identity is configured, plaintext
    pass-through when none is), then atomically replaces the file via
    :func:`paramem.backup.encryption._atomic_write_bytes`.

    When no daily identity is configured this is a no-op: ``envelope_encrypt_bytes``
    returns the bytes unchanged and ``_atomic_write_bytes`` overwrites with the
    same plaintext — which is logically identical to the pre-encrypt state and
    safe to perform.  When a daily identity IS configured but cannot be
    unwrapped, ``envelope_encrypt_bytes`` raises instead of falling back to
    plaintext, so this call propagates that failure rather than leaving a
    plaintext tensor file behind.

    Called from inside :func:`atomic_save_adapter`'s ``_write_payload``
    closure, after the PEFT save-and-flatten steps and before that closure
    returns control to :func:`~paramem.adapters.slot.write_slot` — the
    envelope that owns the promotion sequence shared by both venues.
    ``write_slot`` computes the plaintext payload digest and writes
    ``meta.json`` only after ``_write_payload`` (this encryption step
    included) has finished, so the manifest is always stamped from the
    bytes actually on disk, encrypted or not.  The pending slot is not yet
    promoted so a failure here aborts the entire save without leaving an
    inconsistent slot in the live tree.

    Args:
        slot: Pending slot directory.  Must contain ``adapter_model.safetensors``
            at the top level (PEFT flatten step already done).

    Raises:
        OSError: On any filesystem error during read or atomic write.
        RuntimeError: A daily identity is configured but could not be
            unwrapped (see :func:`paramem.backup.encryption.envelope_encrypt_bytes`).
    """
    from paramem.backup.encryption import _atomic_write_bytes, envelope_encrypt_bytes

    safetensors_path = slot / _SAFETENSORS_FILENAME
    if not safetensors_path.exists():
        # PEFT wrote no safetensors (e.g. adapter_model.bin layout or unit-test
        # mock that omits the file) — skip silently.
        logger.debug(
            "_encrypt_adapter_safetensors: %s absent in %s — skipping",
            _SAFETENSORS_FILENAME,
            slot,
        )
        return
    plaintext = safetensors_path.read_bytes()
    encrypted = envelope_encrypt_bytes(plaintext)
    _atomic_write_bytes(safetensors_path, encrypted)


@contextlib.contextmanager
def _adapter_slot_for_load(slot: Path):
    """Context manager that yields a readable slot directory for PEFT loading.

    When ``adapter_model.safetensors`` inside *slot* is an age envelope,
    decrypts it into an anonymous in-memory file (``os.memfd_create`` on
    Linux; ``/dev/shm/paramem-<random>`` with mode ``0700`` as a fallback)
    and yields a *temporary* copy of the slot directory where the safetensors
    file is replaced by the decrypted bytes.  The caller passes the yielded
    path to ``model.load_adapter`` or ``PeftModel.from_pretrained``.  All
    temporary files are removed on context exit regardless of exceptions.

    When ``adapter_model.safetensors`` is plaintext (or absent) the original
    *slot* is yielded unchanged — zero overhead.

    The memfd / shm file is opened with ``O_RDWR`` + ``F_SEAL_WRITE`` (Linux
    memfd sealing, best-effort) so the kernel never needs to page the tensor
    to a swap file.  The fd is closed on exit; no ``/proc/self/fd/<fd>``
    symlink persists after the context.

    Usage::

        with _adapter_slot_for_load(slot) as load_path:
            model.load_adapter(str(load_path), adapter_name=name)

    Args:
        slot: Path to the slot directory written by :func:`atomic_save_adapter`.

    Yields:
        Path — either the original *slot* (plaintext case) or a temporary
        directory with the decrypted safetensors at the same relative path.
    """
    import shutil
    import tempfile

    from paramem.backup.age_envelope import AGE_MAGIC

    safetensors_path = slot / _SAFETENSORS_FILENAME

    # Fast path — plaintext or file absent: yield original slot unchanged.
    if not safetensors_path.exists() or not safetensors_path.read_bytes()[:22].startswith(
        AGE_MAGIC[:22]
    ):
        # Yield as a plain generator (no cleanup needed).
        yield slot
        return

    # Encrypted path — need to decrypt into a temporary location.
    from paramem.backup.encryption import read_maybe_encrypted

    plaintext = read_maybe_encrypted(safetensors_path)

    # Try memfd_create (Linux anonymous in-memory file, no on-disk residue).
    # Fall back to /dev/shm with mode 0700 when memfd is unavailable.
    tmp_dir: Path | None = None
    memfd: int | None = None
    shm_path: Path | None = None

    try:
        try:
            memfd = os.memfd_create("paramem_adapter", flags=0)  # type: ignore[attr-defined]
            os.write(memfd, plaintext)
            os.lseek(memfd, 0, os.SEEK_SET)
            # Expose via /proc/self/fd/<fd> so PEFT can open it as a regular path.
            memfd_path = Path(f"/proc/self/fd/{memfd}")
            # Build a temporary directory that mirrors the slot layout but
            # replaces adapter_model.safetensors with a symlink to the memfd.
            tmp_dir = Path(tempfile.mkdtemp(prefix="paramem_slot_"))
            tmp_dir.chmod(0o700)
            # Hardlink every non-safetensors file (adapter_config.json, meta.json).
            for child in slot.iterdir():
                if child.name != _SAFETENSORS_FILENAME:
                    shutil.copy2(child, tmp_dir / child.name)
            # Symlink safetensors → /proc/self/fd/<fd>.
            (tmp_dir / _SAFETENSORS_FILENAME).symlink_to(memfd_path)
            yield tmp_dir

        except (AttributeError, OSError):
            # memfd_create not available (non-Linux) or /proc not mounted.
            # Fall back to /dev/shm with a random name and mode 0700.
            import secrets

            shm_dir = Path("/dev/shm")
            if not shm_dir.exists():
                shm_dir = Path(tempfile.gettempdir())
            shm_name = f"paramem-{secrets.token_hex(8)}"
            tmp_dir = shm_dir / shm_name
            tmp_dir.mkdir(mode=0o700, exist_ok=False)
            for child in slot.iterdir():
                if child.name != _SAFETENSORS_FILENAME:
                    shutil.copy2(child, tmp_dir / child.name)
            shm_path = tmp_dir / _SAFETENSORS_FILENAME
            shm_path.write_bytes(plaintext)
            shm_path.chmod(0o600)
            yield tmp_dir

    finally:
        if memfd is not None:
            with contextlib.suppress(OSError):
                os.close(memfd)
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def copy_adapter_weights(model: PeftModel, src: str, dst: str) -> None:
    """Copy LoRA adapter weights (weight + bias) from src to dst in-memory.

    Uses named_parameters suffix matching to find adapter-keyed tensors.
    Preserves device placement — each parameter is copied in-place on its
    existing device. No disk I/O.

    Handles both `.{name}.weight` and `.{name}.bias` suffixes so configs
    with `bias="lora_only"` or `bias="all"` are covered. Asserts that the
    set of source and destination parameter keys match exactly — a mismatch
    means src and dst adapters have different target_modules or configs
    and the copy would be silently incomplete.
    """
    if src not in model.peft_config:
        raise ValueError(f"Source adapter '{src}' not found")
    if dst not in model.peft_config:
        raise ValueError(f"Destination adapter '{dst}' not found")

    def _index(adapter: str) -> dict:
        out = {}
        for name, p in model.named_parameters():
            for suffix in (f".{adapter}.weight", f".{adapter}.bias"):
                if name.endswith(suffix):
                    # key: (base path without adapter name, suffix type)
                    base = name[: -len(suffix)]
                    out[(base, suffix[-len(".weight") :] if "weight" in suffix else ".bias")] = p
                    break
        return out

    src_index = _index(src)
    dst_index = _index(dst)

    if not src_index or not dst_index:
        raise RuntimeError(
            f"No adapter-keyed parameters found for src='{src}' (count={len(src_index)}) "
            f"or dst='{dst}' (count={len(dst_index)})"
        )

    if set(src_index.keys()) != set(dst_index.keys()):
        missing_in_dst = set(src_index.keys()) - set(dst_index.keys())
        missing_in_src = set(dst_index.keys()) - set(src_index.keys())
        raise RuntimeError(
            f"Adapter parameter sets differ between '{src}' and '{dst}' — cannot copy. "
            f"Missing in dst: {len(missing_in_dst)}. Missing in src: {len(missing_in_src)}. "
            f"Adapters likely have different target_modules or bias configs."
        )

    with torch.no_grad():
        for key, src_p in src_index.items():
            dst_index[key].data.copy_(src_p.data)

    logger.debug("Copied %d tensors from adapter '%s' to '%s'", len(src_index), src, dst)


def get_adapter_info(model: PeftModel) -> dict:
    """Return summary info about all loaded adapters."""
    info = {}
    for name in model.peft_config:
        config = model.peft_config[name]
        info[name] = {
            "rank": config.r,
            "alpha": config.lora_alpha,
            "target_modules": list(config.target_modules),
            "dropout": config.lora_dropout,
        }
    return info


# Below this norm, an adapter's LoRA-B weights are considered a fresh
# (cold, LoRA-zero) init rather than a trained-and-warm one. PEFT's
# create_adapter leaves LoRA-B at EXACTLY zero (identity residual), so a
# bit-exact-zero check would work for a freshly-created adapter; the
# headroom is for bf16 denormals and for a warm adapter whose weights have
# barely moved off zero (e.g. one optimizer step before an abort) --
# neither is "cold" and the threshold must not misclassify them.
LORA_B_COLD_NORM_THRESHOLD = 1e-6


class LoraTensorsNotFound(RuntimeError):
    """No ``lora_B`` tensor was found for the requested adapter name.

    Distinct from a bare ``RuntimeError`` so callers can catch this specific
    condition (wrong/missing adapter name) without also swallowing unrelated
    ``RuntimeError`` subclasses such as a CUDA OOM or "device lost" error
    raised from inside the same ``named_parameters()`` iteration.
    """


def lora_b_frobenius_norm(model: PeftModel, adapter_name: str) -> float:
    """Return the total Frobenius norm of all LoRA-B tensors for *adapter_name*.

    LoRA-B is zero-initialised by PEFT at adapter creation (identity
    residual), so a zero norm immediately after ``create_adapter`` and a
    non-zero norm after training together prove cold init actually ran end
    to end. Shared by the consolidation fold-telemetry measurement
    (:func:`measured_adapter_init_state`) and
    ``experiments/test20_smallN_cold_gate.py`` (its Hard Assertion #3) —
    one implementation, not a copy in each caller.

    Args:
        model: PeftModel carrying *adapter_name*.
        adapter_name: Adapter whose LoRA-B norm is computed.

    Returns:
        Total Frobenius norm of all LoRA-B tensors for the adapter (float).

    Raises:
        LoraTensorsNotFound: When no ``lora_B`` tensors are found for the
            adapter (wrong adapter name / not yet created).
    """
    total_norm = 0.0
    count = 0
    for name, param in model.named_parameters():
        if f"lora_B.{adapter_name}.weight" in name:
            total_norm += param.data.norm().item()
            count += 1
    if count == 0:
        raise LoraTensorsNotFound(
            f"No lora_B tensors found for adapter '{adapter_name}' — check adapter name"
        )
    return total_norm


def measured_adapter_init_state(model: PeftModel, adapter_name: str) -> "str | None":
    """Return ``"cold"``/``"warm"`` from the measured LoRA-B Frobenius norm.

    Wraps :func:`lora_b_frobenius_norm` for the fold-telemetry ring
    (``paramem.server.fold_telemetry``): a diagnostics measurement must
    never block or crash a production fold on a condition specific to the
    measurement itself, so a :class:`LoraTensorsNotFound` failure (no
    ``lora_B`` tensors indexed under *adapter_name* — e.g. a test double
    standing in for the model) degrades to ``None`` (the caller omits the
    ``init`` field) rather than propagating. This is the ONLY degradation:
    an ``AttributeError`` (e.g. *model* has no ``named_parameters``, or is on
    the meta device) or any other exception type still propagates
    unchanged — those indicate a genuinely broken caller state, not an
    unmeasurable-but-otherwise-healthy adapter. In production the model is
    always a real ``PeftModel`` with *adapter_name* already created by
    ``create_adapter`` (a full fold's main tiers) or ``create_interim_adapter``
    (an interim event's own slot) before this is called, so this path is
    expected to always measure successfully; the ``None`` branch exists for
    the introspection boundary, not as a normal outcome.

    Args:
        model: PeftModel carrying *adapter_name*.
        adapter_name: Adapter whose measured init state is classified.

    Returns:
        ``"cold"`` when the norm is below :data:`LORA_B_COLD_NORM_THRESHOLD`,
        ``"warm"`` otherwise, or ``None`` when the norm could not be measured.
    """
    try:
        norm = lora_b_frobenius_norm(model, adapter_name)
    except LoraTensorsNotFound:
        return None
    return "cold" if norm < LORA_B_COLD_NORM_THRESHOLD else "warm"


def has_prior_trained_weights(model: PeftModel, adapter_name: str) -> bool:
    """True when *adapter_name* carries weights worth starting a staging slot from.

    The one predicate for "this tier has prior trained weights" — resident
    in ``model.peft_config`` AND measuring ``"warm"`` via
    :func:`measured_adapter_init_state`. Absence and a resident-but-cold
    adapter (freshly created, never trained) both read ``False``: neither
    has anything a staging slot could usefully warm-start from.

    Named once here and read by both the training funnel's donor
    resolution (:meth:`~paramem.training.consolidation.ConsolidationLoop.
    _resolve_donor_checkpoint` — a donor applies only where there are no
    prior trained weights) and :func:`~paramem.training.trainer.
    train_adapter`'s own staging-init branch, so the LoRA-B threshold rule
    is stated exactly once rather than re-derived at each call site.

    Args:
        model: The live model, or ``None`` (callers such as ``/status``
            check ``model is not None`` first). Not required to be a
            ``PeftModel`` instance otherwise — presence in ``peft_config``
            is the discriminator.
        adapter_name: Adapter/tier name to classify.

    Returns:
        ``True`` only when *adapter_name* is resident and measures
        ``"warm"``; ``False`` on absence, a measured ``"cold"`` adapter, or
        an unmeasurable adapter (:data:`LoraTensorsNotFound` degrades to
        ``None`` inside :func:`measured_adapter_init_state`, which reads as
        ``False`` here).
    """
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None or adapter_name not in peft_config:
        return False
    return measured_adapter_init_state(model, adapter_name) == "warm"


def ensure_adapter_matching(
    model: PeftModel,
    adapter_config: AdapterConfig,
    adapter_name: str,
) -> None:
    """Ensure *adapter_name* exists and matches *adapter_config*'s LoRA topology, in place.

    The single config-mismatch guard for the warm-init default: warm init
    keeps a resident adapter's trained weights across events, so every
    warm-init entrance (the per-tier build/write driver, called for each
    tier before ``tier_backup_scope`` is entered — and interim-slot mint)
    must call this instead of unconditionally deleting and recreating.
    Three outcomes:

    - Absent: cold birth via :func:`create_adapter` — there are no weights
      to preserve, so there is nothing to compare or keep warm.
    - Present, config matches (:func:`lora_shape_fields`: ``r``,
      ``lora_alpha``, ``target_modules``): no-op. This is the warm path —
      the caller's staging copy (``train_adapter``'s production→staging
      ``copy_adapter_weights`` call, right after ``_ensure_staging_slot``)
      then warm-starts from these weights.
    - Present, config differs: deleted and recreated cold, with a warning
      naming the mismatched field(s). The comparison is on the PEFT
      ``LoraConfig`` fields, deliberately never on parameter key sets — a
      rank change keeps the same key names (same ``target_modules``) but
      different tensor shapes, so a parameter-key-set comparison would
      pass and the mismatch would only surface later as a tensor-shape
      ``RuntimeError`` inside :func:`copy_adapter_weights` /
      ``model.generate()``. Calling this ahead of any weight-touching
      operation on the tier (in particular ahead of
      :func:`tier_backup_scope`, which snapshots the resident tier via
      :func:`copy_adapter_weights`) is what makes config comparison
      sufficient to catch it first.

    Args:
        model: The live ``PeftModel``. Mutated in place; nothing is
            returned.
        adapter_config: The tier's target ``AdapterConfig`` (rank, alpha,
            target_modules, dropout) to create or validate against.
        adapter_name: Adapter/tier name to validate or create.

    Raises:
        TypeError: *model* is not a ``PeftModel``.
    """
    if not isinstance(model, PeftModel):
        raise TypeError(f"ensure_adapter_matching requires a PeftModel, got {type(model).__name__}")

    if adapter_name not in model.peft_config:
        create_adapter(model, adapter_config, adapter_name)
        return

    resident = model.peft_config[adapter_name]
    target_fields = lora_shape_fields(adapter_config)
    mismatches: list[str] = []
    for field_name, target_value in target_fields.items():
        resident_value = getattr(resident, field_name, None)
        if field_name == "target_modules":
            resident_value = set(resident_value) if resident_value else set()
            target_value = set(target_value)
        if resident_value != target_value:
            mismatches.append(f"{field_name}: resident={resident_value} target={target_value}")

    if not mismatches:
        return

    logger.warning(
        "ensure_adapter_matching: adapter '%s' config mismatch (%s) — recreating cold",
        adapter_name,
        "; ".join(mismatches),
    )
    model.delete_adapter(adapter_name)
    create_adapter(model, adapter_config, adapter_name)
