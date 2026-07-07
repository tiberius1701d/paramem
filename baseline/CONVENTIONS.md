# Conventions

## UID scheme

| Prefix | Meaning | Chapter | Example |
|---|---|---|---|
| `OBJ-###` | Objective | `03_objectives.sdoc` | `OBJ-001` |
| `FR-<CAPABILITY>-###` | Functional requirement | `05_functional.sdoc` | `FR-RECALL-001` |
| `QR-<AREA>-###` | Quality requirement | `06_quality.sdoc` | `QR-LATENCY-001` |
| `CON-###` | Constraint | `07_constraints.sdoc` | `CON-001` |
| `ADR-###` | Architecture/requirement decision record | `annex_a_decisions.sdoc` | `ADR-001` |
| `OQ-###` | Open question | `annex_c_open_questions.sdoc` | `OQ-001` |

`<CAPABILITY>` / `<AREA>` are short uppercase tokens naming the capability
or quality attribute the requirement belongs to (e.g. `RECALL`, `LATENCY`).
Sections within `05_functional.sdoc` / `06_quality.sdoc` are organized by
capability/area, one section per token, to be defined as requirements are
added.

## Custom requirement fields

Every `[REQUIREMENT]` node carries these fields, in this order (defined in
`grammars/srb.sgra`):

| Field | Required | Purpose |
|---|---|---|
| `UID` | Yes | The identifier, following the scheme above. |
| `STATUS` | Yes | Enforced enum: `draft`, `reviewed`, or `frozen`. |
| `PROVENANCE` | No | Evidence pointer — where the requirement came from (design doc, decision, upstream requirement). |
| `VERIFICATION` | No | How the requirement is verified: a test reference, a measurement method, or `TBD`. |
| `STATEMENT` | Yes | The requirement text itself. |

`STATUS` is a `SingleChoice` field — StrictDoc rejects any value outside
`draft` / `reviewed` / `frozen` with a semantic error at export time
(`Requirement field has an invalid SingleChoice value`), which is the CI
failure mechanism for a malformed status.

## Grammar enforcement is per-file

These fields are enforced by the **shared grammar**
(`grammars/srb.sgra`), not by StrictDoc's built-in default grammar. Every
`.sdoc` file — the root `srb.sdoc` and every chapter under `chapters/` —
must carry its own

```
[GRAMMAR]
IMPORT_FROM_FILE: @shared_grammar
```

block. If a file omits this, StrictDoc silently falls back to its default
built-in grammar for that document, and the custom fields (`PROVENANCE`,
`VERIFICATION`, the `STATUS` enum) will not be recognized or enforced in
that file.
