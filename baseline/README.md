# ParaMem System Requirements Baseline

This is the requirements backbone for ParaMem, built on
[StrictDoc](https://strictdoc.readthedocs.io/) 0.26.0. It is a self-contained
tooling skeleton: no requirement content yet, just the document structure,
grammar, and build tooling needed to author and export requirements.

It currently lives as a gitignored subdirectory of the main `paramem`
repository (see the root `.gitignore`) and is intended to become its own
standalone repository.

## Document structure

- `srb.sdoc` — root document (`UID: SRB`). Includes every chapter via
  `[DOCUMENT_FROM_FILE]` so the whole tree exports and renders as one
  document.
- `grammars/srb.sgra` — the shared custom grammar (`@shared_grammar`),
  imported by every `.sdoc` file. Defines the `SECTION` and `REQUIREMENT`
  element types and their fields.
- `chapters/` — the ten chapters/annexes, each a standalone `.sdoc`
  document with its own `[[SECTION]]` and a one-line scope `[TEXT]` note.
  Currently empty placeholders — no `[REQUIREMENT]` nodes yet.
- `strictdoc_config.py` — StrictDoc project configuration: enabled
  features, the `@shared_grammar` alias, and doc-path excludes for
  `.venv/` and `_build/`.

See `CONVENTIONS.md` for the UID scheme and custom requirement fields, and
`TOOLING.md` for exactly how the toolchain is pinned and reproduced.

## Building locally

```bash
make venv install all
```

This creates `.venv/`, installs the hash-locked `requirements.txt`, and
exports HTML, ReqIF, and SDoc formats to `_build/`. Individual targets:

- `make html` — HTML documentation only.
- `make reqif` — ReqIF export only (`_build/reqif/output.reqif`).
- `make check` (alias `make build`) — validation-only SDoc re-export; the
  cheapest way to confirm the tree parses and the grammar is satisfied.
- `make clean` — remove `_build/`.

`make all` runs the exact same `strictdoc export` invocation as CI
(`.github/workflows/strictdoc.yml`); see that file's header comment for why
the workflow is dormant until `baseline/` splits into its own repository.

## Status

Greenfield scaffolding. No requirements have been written yet — chapters
are empty placeholders with scope notes only.
