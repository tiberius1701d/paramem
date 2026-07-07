# Tooling

## Pinned versions

- **StrictDoc**: `0.26.0` exactly (`requirements.in`). Verified empirically
  against this exact version — see the deviations noted below for the two
  places where behavior differed from initial assumptions.
- **Python**: StrictDoc 0.26.0 requires Python >= 3.10. CI
  (`.github/workflows/strictdoc.yml`) pins **3.11** via
  `actions/setup-python`. Locally, `make venv` uses whatever `python3`
  resolves to on `PATH` (see `Makefile` header comment) — pass
  `PYTHON=python3.11` explicitly if you have that interpreter installed
  and want to match CI exactly.

## Lock file: hash-locked, generated under the target Python (3.11)

`requirements.txt` is a fully hash-locked, transitively-pinned lock
generated from `requirements.in` with `pip-tools`. For a tool-qualification
deliverable the lock must be resolved under the same Python version CI
installs against (3.11), not whatever interpreter happens to be on the
authoring host — a resolver running under a newer Python can silently pick
newer transitive versions that don't support 3.11 (this was caught in
practice: see below). The lock was generated in a **throwaway conda env**
(never the `paramem` env), created solely for this purpose and removed
immediately after:

```bash
conda create -y -n baseline-lock python=3.11
conda run -n baseline-lock pip install pip-tools   # pip-tools==7.5.3
cd baseline
conda run -n baseline-lock pip-compile --generate-hashes requirements.in -o requirements.txt
conda env remove -y -n baseline-lock
```

The resulting `requirements.txt` pins 84 packages (`strictdoc` + its full
transitive closure) with sha256 hashes for every published artifact of
each pinned version (i.e. the hash list is not filtered to the resolving
interpreter's platform tag — e.g. `numpy==2.4.6` carries 72 hashes covering
its full wheel matrix). This is pip-compile's default `--generate-hashes`
behavior.

**This lock was first generated under Python 3.13** (no `python3.11` was
available on the authoring host at that point), then **regenerated under
Python 3.11** once a throwaway conda env made that interpreter available.
The two resolutions differed in one real, load-bearing way: `numpy`
resolved to `2.5.1` under 3.13 but `2.4.6` under 3.11, because numpy 2.5.x
dropped Python 3.11 support (confirmed via
`pip index versions numpy` under the 3.11 env, whose highest listed version
is `2.4.6`). Everything else — package set (84 pinned packages both times)
and all other versions — was identical; the only other diff was two
`# via` comment lines (`anyio`, `starlette` additionally listed as
consumers of `typing-extensions` under 3.11, since that dependency is
declared with a `python_version < "3.12"` marker upstream — cosmetic, not
a version change). This confirms the 3.13-resolved lock would have
installed a numpy release incompatible with CI's Python 3.11; the
regeneration under 3.11 is the correct, committed lock.

To regenerate the lock (e.g. after bumping `strictdoc` in
`requirements.in`), repeat the throwaway-conda-env recipe above — always
against Python 3.11 to match CI.

## Reproducing the toolchain from scratch

```bash
make venv install
.venv/bin/strictdoc version   # should print 0.26.0
```

## Build / export commands

```bash
make check   # validation-only re-export (sdoc format), cheapest CI-equivalent check
make html    # HTML documentation
make reqif   # ReqIF export (_build/reqif/output.reqif)
make all     # html + reqif-sdoc + sdoc, identical to CI
make clean   # remove _build/
```

Under the hood these all call `strictdoc export . --formats=... --output-dir _build`
run from the venv (`.venv/bin/strictdoc`) **against the `baseline/` folder**,
not against `srb.sdoc` directly — StrictDoc needs the containing folder to
resolve `[DOCUMENT_FROM_FILE]` includes and the `@shared_grammar` alias.

## Empirically confirmed facts (this install, StrictDoc 0.26.0)

- `ProjectConfig.__init__` accepts `project_title`, `project_features`,
  `grammars`, and `exclude_doc_paths` as documented in `strictdoc_config.py`
  — confirmed via `inspect.signature(ProjectConfig.__init__)` against the
  installed 0.26.0 package.
- `exclude_doc_paths` in `strictdoc_config.py` (`.venv/`, `_build/`,
  `README.md`, `TOOLING.md`, `CONVENTIONS.md`) is required when running
  `strictdoc export .` from inside `baseline/` itself (as opposed to a
  parent directory): without excluding `.venv/`/`_build/`, StrictDoc's
  file scanner walks into them and tries to parse third-party
  `.md`-shaped files it finds there, which fails export with unrelated
  parse errors; without excluding the three top-level `.md` docs, they
  get auto-discovered and rendered as extra standalone HTML documents
  (with their own TABLE/TRACE views) alongside the actual requirements
  tree.
- `OPTIONS: AUTO_LEVELS: On` (not `Off`) is used in every `.sdoc` file.
  `AUTO_LEVELS: Off` requires every non-composite node (including `[TEXT]`)
  to carry an explicit `LEVEL:` field, or export fails with
  `[TEXT].LEVEL field is not provided`. Since this skeleton has no
  hand-maintained numbering scheme, `On` (StrictDoc's default automatic
  section/requirement numbering) is used instead; it does not affect
  `TITLE` fields, which are always explicit regardless of this setting.
- Composition via `[DOCUMENT_FROM_FILE]` does not double-render chapters
  as separate top-level HTML documents — exporting the folder produces a
  single `srb.html` (root) with `srb-TABLE.html` / `srb-TRACE.html`
  companion views; chapters do not get their own `<chapter>.html`.
- ReqIF export format token is `reqif-sdoc`, not `reqif`, and requires
  `"REQIF"` in `project_features`.
