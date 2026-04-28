"""Tuning tool: regenerate intent-classifier exemplars from upstream community data.

Static fetch.  Output goes to ``configs/intents/upstream/<class>.<lang>.txt``
so the canonical hand-curated ``configs/intents/<class>.<lang>.txt`` files
are never overwritten -- operator diffs the upstream draft against the
canonical set and manually merges the lines they want.

This is a dev / tuning tool, not a runtime concern.  The shipped pipeline
loads ``configs/intents/<class>.<lang>.txt`` directly; this script just
seeds the manual-curation step when refreshing the exemplar pool against
new community data.

Sources:
* ``home-assistant/intents`` (GitHub) -- HA-specific imperatives across
  many locales.  Best fit: COMMAND class.
* ``AmazonScience/MASSIVE`` (S3 tarball) -- ~1M crowd-sourced
  utterances across 51 locales with intent labels.  Best fit:
  GENERAL class (factoid / Q&A patterns) and supplementary COMMAND.

Both sources fetched with stdlib only -- no new project dependencies.
HA via ``git clone`` (subprocess), MASSIVE via ``urllib`` + ``tarfile``.

Cache layout (default ``$HOME/.cache/paramem/intent_exemplars/``):
  intents-repo/         home-assistant/intents working copy (kept fresh)
  massive-1.0.tar.gz    MASSIVE archive (cached, ~80 MB)
  massive/              extracted MASSIVE JSONL files

Usage::

    python scripts/dev/build_intent_exemplars.py
    diff -u configs/intents/command.en.txt configs/intents/upstream/command.en.txt
    # ... manually merge desired lines ...

CLI flags:
  --ha / --no-ha           include / skip the HA-intents source (default include)
  --massive / --no-massive include / skip the MASSIVE source (default include)
  --langs en,de,...        locales to emit (default: en,de -- match the
                           canonical shipped set)
  --output-dir PATH        where to write upstream drafts
                           (default: configs/intents/upstream)
  --cache-dir PATH         where to cache downloaded sources
  --max-per-class N        cap output per (class, lang) at N lines
                           (default 100; set to 0 to disable)
  --skip-fetch             use cached sources only, don't fetch fresh
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


_HA_INTENTS_REPO = "https://github.com/home-assistant/intents.git"
_MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
)

_HA_COMMAND_SUBSTRINGS = (
    "Turn",
    "Set",
    "Play",
    "Pause",
    "Stop",
    "Resume",
    "Open",
    "Close",
    "Lock",
    "Unlock",
    "Increase",
    "Decrease",
    "Mute",
    "Unmute",
    "Cancel",
    "Start",
    "Add",
    "Remove",
    "Climate",
    "Light",
    "Cover",
    "Vacuum",
    "Media",
)

# MASSIVE intent → our intent class.  PERSONAL is left empty: MASSIVE doesn't
# carry personal-context queries (it's crowd-sourced generic intents).
# Hand-curated PERSONAL stays the source of truth for that class.
_MASSIVE_INTENT_TO_CLASS: dict[str, str] = {
    # Device control / smart-home actions -> COMMAND
    "iot_hue_lightoff": "command",
    "iot_hue_lightup": "command",
    "iot_hue_lightdim": "command",
    "iot_hue_lighton": "command",
    "iot_hue_lightchange": "command",
    "iot_cleaning": "command",
    "iot_coffee": "command",
    "iot_wemo_on": "command",
    "iot_wemo_off": "command",
    "alarm_set": "command",
    "alarm_remove": "command",
    "play_music": "command",
    "play_radio": "command",
    "play_podcasts": "command",
    "play_audiobook": "command",
    "play_game": "command",
    "music_likeness": "command",
    "audio_volume_up": "command",
    "audio_volume_down": "command",
    "audio_volume_mute": "command",
    # General knowledge / world Q&A -> GENERAL
    "general_quirky": "general",
    "qa_definition": "general",
    "qa_factoid": "general",
    "qa_currency": "general",
    "qa_maths": "general",
    "qa_stock": "general",
    "weather_query": "general",
    "news_query": "general",
    "general_joke": "general",
    "datetime_query": "general",
    "datetime_convert": "general",
    "transport_query": "general",
}

# MASSIVE locale codes use BCP-47 (e.g. "en-US", "de-DE").  Map our 2-letter
# language codes to the most common locale code.
_LANG_TO_MASSIVE_LOCALE: dict[str, str] = {
    "en": "en-US",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "it": "it-IT",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "pt": "pt-PT",
}


def _run_git(*args: str, cwd: Path | None = None) -> str:
    """Subprocess wrapper for git -- raises on non-zero exit."""
    logger.info("git %s (cwd=%s)", " ".join(args), cwd or "")
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _fetch_ha_intents(cache_dir: Path, *, skip_fetch: bool = False) -> Path:
    """Clone or pull home-assistant/intents.  Returns the working-copy path."""
    repo_dir = cache_dir / "intents-repo"
    if repo_dir.exists():
        if skip_fetch:
            logger.info("HA intents: using cached %s (--skip-fetch)", repo_dir)
            return repo_dir
        try:
            _run_git("pull", "--ff-only", cwd=repo_dir)
        except subprocess.CalledProcessError as e:
            logger.warning("HA intents: git pull failed (%s); using existing checkout", e)
        return repo_dir
    if skip_fetch:
        raise FileNotFoundError(f"HA intents cache not found at {repo_dir} and --skip-fetch is set")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _run_git("clone", "--depth", "1", _HA_INTENTS_REPO, str(repo_dir))
    return repo_dir


def _fetch_massive(cache_dir: Path, *, skip_fetch: bool = False) -> Path:
    """Download + extract the MASSIVE archive.  Returns the JSONL data dir."""
    archive = cache_dir / "massive-1.0.tar.gz"
    extracted = cache_dir / "massive"
    if extracted.exists() and any(extracted.iterdir()):
        logger.info("MASSIVE: using cached %s", extracted)
        return extracted
    if skip_fetch:
        raise FileNotFoundError(f"MASSIVE cache not found at {extracted} and --skip-fetch is set")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        logger.info("MASSIVE: downloading %s -> %s (~80 MB)", _MASSIVE_URL, archive)
        with urllib.request.urlopen(_MASSIVE_URL) as resp, archive.open("wb") as f:
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    logger.info("MASSIVE: extracting to %s", extracted)
    extracted.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extracted)
    return extracted


_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|\([^()]+\)|\[[^\[\]]+\]")


def _normalise_template(template: str) -> str | None:
    """Resolve a Home Assistant intent template to a plain sentence.

    HA templates use ``{slot}`` for variables and ``( ... )`` /
    ``[ ... ]`` for optional groups.  We strip the markup but keep the
    surrounding text -- precision over creativity.  Returns ``None`` for
    templates we can't safely resolve (deeply nested, empty after strip).
    """
    if not template or template.startswith(("$", "#")):
        return None
    out = template
    # Iteratively collapse nested groups.  Five passes is plenty for HA.
    for _ in range(5):
        new = _TEMPLATE_PLACEHOLDER_RE.sub(
            lambda m: " " if m.group(0).startswith("{") else m.group(0)[1:-1],
            out,
        )
        if new == out:
            break
        out = new
    out = re.sub(r"\s+", " ", out).strip().strip(".,;:!?")
    if not out or len(out.split()) < 2:
        return None
    return out


def _ha_extract(repo_dir: Path, lang: str) -> list[str]:
    """Walk sentences/<lang>/*.yaml for HA-action sentence templates.

    Returns deduplicated, normalised sentences whose source intent matches
    one of ``_HA_COMMAND_SUBSTRINGS``.  No YAML library required -- we
    line-scan for ``- "..."`` or ``- '...'`` entries inside the
    sentences blocks.  Imperfect but robust enough for a tuning tool.
    """
    sentences_dir = repo_dir / "sentences" / lang
    if not sentences_dir.exists():
        logger.warning("HA intents: no sentences dir for lang %r at %s", lang, sentences_dir)
        return []

    out: set[str] = set()
    for yaml_path in sorted(sentences_dir.glob("*.yaml")):
        # Filename shape: <Action>_<Domain>.yaml (e.g. HassTurnOn_light.yaml)
        if not any(sub in yaml_path.name for sub in _HA_COMMAND_SUBSTRINGS):
            continue
        for line in yaml_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            # Match `- "..."` and `- '...'` (yaml list items wrapping strings).
            m = re.match(r'^- ["\'](.+)["\']\s*$', stripped)
            if not m:
                continue
            template = m.group(1)
            normalised = _normalise_template(template)
            if normalised:
                out.add(normalised)
    return sorted(out)


def _massive_extract(massive_dir: Path, lang: str) -> dict[str, list[str]]:
    """Read MASSIVE JSONL for the given language and bucket utts by class.

    Returns ``{class_name: [utterance, ...]}`` for our PERSONAL/COMMAND/GENERAL
    classes.  PERSONAL bucket stays empty (MASSIVE doesn't cover it).
    """
    locale = _LANG_TO_MASSIVE_LOCALE.get(lang)
    if locale is None:
        logger.warning("MASSIVE: no locale mapping for lang %r", lang)
        return {}
    # MASSIVE 1.0 layout: massive/1.0/data/<locale>.jsonl
    jsonl = massive_dir / "1.0" / "data" / f"{locale}.jsonl"
    if not jsonl.exists():
        # Some archives place the data at the root.  Try fallback.
        candidates = list(massive_dir.rglob(f"{locale}.jsonl"))
        if not candidates:
            logger.warning("MASSIVE: no JSONL found for locale %r", locale)
            return {}
        jsonl = candidates[0]

    bucket: dict[str, list[str]] = defaultdict(list)
    with jsonl.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            intent = rec.get("intent", "")
            klass = _MASSIVE_INTENT_TO_CLASS.get(intent)
            if klass is None:
                continue
            utt = (rec.get("utt") or "").strip()
            if utt and len(utt.split()) >= 2:
                bucket[klass].append(utt)
    return bucket


def _write_output(
    output_dir: Path,
    klass: str,
    lang: str,
    sentences: list[str],
    *,
    source_label: str,
    max_per_class: int,
) -> None:
    """Write deduplicated sentences to ``<output_dir>/<class>.<lang>.txt``.

    Existing file (from a previous run / source) is appended to without
    duplication so multiple sources can contribute to the same class.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{klass}.{lang}.txt"
    existing: list[str] = []
    if path.exists():
        existing = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    seen = set(existing)
    new = [s for s in sentences if s not in seen]
    if max_per_class > 0:
        # Cap considers existing + new combined.
        budget = max(0, max_per_class - len(existing))
        new = new[:budget]
    combined = existing + new

    with path.open("w", encoding="utf-8") as f:
        f.write(f"# Intent class: {klass.upper()} -- upstream draft (auto-generated).\n")
        f.write("# DO NOT load this file directly into the runtime.  Diff against\n")
        f.write(f"# configs/intents/{klass}.{lang}.txt and merge desired lines manually.\n")
        f.write(f"# Last contributor: {source_label}\n")
        f.write(f"# Total lines: {len(combined)} (added {len(new)} this pass).\n")
        f.write("#\n")
        for s in combined:
            f.write(s + "\n")
    logger.info("wrote %s: +%d (total %d, source=%s)", path, len(new), len(combined), source_label)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ha", dest="ha", action="store_true", default=True)
    parser.add_argument("--no-ha", dest="ha", action="store_false")
    parser.add_argument("--massive", dest="massive", action="store_true", default=True)
    parser.add_argument("--no-massive", dest="massive", action="store_false")
    parser.add_argument(
        "--langs",
        type=str,
        default="en,de",
        help="Comma-separated language codes to emit (default: en,de)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/intents/upstream"),
        help="Output directory for upstream drafts",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.path.expanduser("~/.cache/paramem/intent_exemplars")),
        help="Cache dir for downloaded sources",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=100,
        help="Cap per (class, lang) output (default 100; 0 disables)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Use cached sources only; do not fetch fresh data",
    )
    args = parser.parse_args(argv)

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    if not langs:
        print("ERROR: --langs must list at least one language code", file=sys.stderr)
        return 2

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.ha:
        try:
            repo_dir = _fetch_ha_intents(args.cache_dir, skip_fetch=args.skip_fetch)
            for lang in langs:
                sentences = _ha_extract(repo_dir, lang)
                logger.info("HA: %s -> %d sentences", lang, len(sentences))
                _write_output(
                    args.output_dir,
                    "command",
                    lang,
                    sentences,
                    source_label="home-assistant/intents",
                    max_per_class=args.max_per_class,
                )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("HA intents source failed: %s", e)

    if args.massive:
        try:
            massive_dir = _fetch_massive(args.cache_dir, skip_fetch=args.skip_fetch)
            for lang in langs:
                buckets = _massive_extract(massive_dir, lang)
                for klass, utts in buckets.items():
                    logger.info("MASSIVE: %s/%s -> %d utts", klass, lang, len(utts))
                    _write_output(
                        args.output_dir,
                        klass,
                        lang,
                        utts,
                        source_label="AmazonScience/MASSIVE",
                        max_per_class=args.max_per_class,
                    )
        except (urllib.error.URLError, FileNotFoundError, tarfile.TarError) as e:
            logger.error("MASSIVE source failed: %s", e)

    print()
    print("=" * 72)
    print(f"Upstream drafts in: {args.output_dir}")
    print("=" * 72)
    for path in sorted(args.output_dir.glob("*.txt")):
        with path.open() as f:
            count = sum(1 for line in f if line.strip() and not line.strip().startswith("#"))
        print(f"  {path}  ({count} lines)")
    print()
    print("Next: diff each upstream/<class>.<lang>.txt against the canonical")
    print("configs/intents/<class>.<lang>.txt and merge desired lines by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
