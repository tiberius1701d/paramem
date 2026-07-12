"""Structural guard: only the restore handlers may rename onto the live server.yaml.

``paramem.server.migration.promote_config`` is the only function allowed to promote
an operator-supplied *candidate* over the live ``configs/server.yaml``. It reads the
candidate from disk, re-checks its hash against the staged one, and **constructs it
as if it already sat at the live path** (``build_server_config``) before the rename —
so a candidate that cannot boot never becomes the live config.

A hand-rolled ``os.rename(x, live_config_path)`` in a request handler bypasses all of
that: it renames first and discovers the config is unbootable at the next
``load_server_config`` — by which point the server is already dead on the next boot.
That was the defect this guard exists to prevent recurring.

Design: default-deny, not default-allow
----------------------------------------
An earlier version of this guard scanned a hardcoded list of three "promoting
functions" for ``os.rename`` calls. Mutation-testing it (adding a fourth handler
with a raw ``os.rename(candidate, live_config_path)``) showed the guard passed
regardless — a new bypass anywhere else in the module was invisible to it.

This version walks the **whole module** for anything rename-shaped
(``os.rename``, a ``from os import rename`` alias, ``shutil.move``, or a
``Path.replace(dest)``-shaped call — one positional argument, no keywords, which
distinguishes it from ``dataclasses.replace(obj, **changes)``) whose destination
expression mentions ``live_config_path`` or ``DEFAULT_SERVER_CONFIG_PATH`` — the two
names this codebase consistently binds the live config path to. Every hit must be
inside an allowlisted function; anything else fails by default.

``migration_rollback`` and ``backup_restore`` are the only allowlisted renamers.
They restore *previously-live, already-validated* bytes from a backup slot — a
different operation from promoting an operator's candidate, so they must NOT go
through ``promote_config`` (there is no candidate file at a path and no staged hash
to check). Instead they call ``migration.validate_candidate`` on the decrypted
backup bytes before their own rename, so the "unbootable config goes live" property
still holds for them without reusing the candidate-promotion primitive.
"""

from __future__ import annotations

import ast
from pathlib import Path

APP_PY = Path(__file__).resolve().parent.parent / "paramem" / "server" / "app.py"

# The only functions permitted to rename bytes onto the live config path. Both
# validate (via migration.validate_candidate) before they do it — see module
# docstring.
_ALLOWED_LIVE_CONFIG_RENAMERS = frozenset({"migration_rollback", "backup_restore"})

# The two names this codebase consistently binds the live server.yaml path to.
# See e.g. paramem/server/app.py's ~15 `live_config_path = ...` assignments and
# `DEFAULT_SERVER_CONFIG_PATH` in paramem/server/config.py.
_LIVE_CONFIG_NAMES = frozenset({"live_config_path", "DEFAULT_SERVER_CONFIG_PATH"})


def _build_parent_map(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Return a ``child -> parent`` map so a node's enclosing function can be found."""
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _enclosing_function_name(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    """Walk *node*'s ancestors to the nearest enclosing (possibly nested) function."""
    cur = node
    while cur in parents:
        cur = parents[cur]
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
    return "<module level>"


def _mentions_live_config(expr: ast.expr) -> bool:
    """Return True when *expr* (e.g. wrapped in ``str(...)``) references a live-config name."""
    for node in ast.walk(expr):
        if isinstance(node, ast.Name) and node.id in _LIVE_CONFIG_NAMES:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _LIVE_CONFIG_NAMES:
            return True
    return False


def _rename_like_calls(tree: ast.Module) -> list[tuple[ast.Call, ast.expr]]:
    """Return ``(call_node, destination_expr)`` for every rename/move/replace-shaped call.

    Covers ``os.rename(src, dst)``, a bare call via ``from os import rename [as x]``,
    ``shutil.move(src, dst)``, and ``<obj>.replace(dst)`` — the last one restricted to
    exactly one positional argument and no keywords, which is ``Path.replace``'s
    signature and not ``dataclasses.replace(obj, **changes)``'s (always at least one
    keyword change in this codebase's usage).
    """
    rename_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            for alias in node.names:
                if alias.name == "rename":
                    rename_aliases.add(alias.asname or alias.name)

    hits: list[tuple[ast.Call, ast.expr]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and (
                (func.value.id == "os" and func.attr == "rename")
                or (func.value.id == "shutil" and func.attr == "move")
            )
            and node.args
        ):
            hits.append((node, node.args[-1]))
            continue

        if isinstance(func, ast.Name) and func.id in rename_aliases and node.args:
            hits.append((node, node.args[-1]))
            continue

        if (
            isinstance(func, ast.Attribute)
            and func.attr == "replace"
            and len(node.args) == 1
            and not node.keywords
        ):
            hits.append((node, node.args[0]))

    return hits


def test_only_the_restore_handlers_rename_onto_the_live_config_path():
    """Default-deny: any rename-shaped call targeting the live config must be allowlisted."""
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    parents = _build_parent_map(tree)

    offenders: list[str] = []
    for call, dest in _rename_like_calls(tree):
        if not _mentions_live_config(dest):
            continue
        fn_name = _enclosing_function_name(call, parents)
        if fn_name not in _ALLOWED_LIVE_CONFIG_RENAMERS:
            offenders.append(f"  {fn_name} — app.py:{call.lineno}")

    assert not offenders, (
        "A rename-shaped call targets the live config path from outside the "
        "allowlisted restore handlers. The live config is promoted only via "
        "migration.promote_config (candidates) or a validate-then-rename inside "
        "migration_rollback / backup_restore (backups) — a new bypass here "
        "reintroduces the rename-before-validate defect:\n" + "\n".join(offenders)
    )


def test_app_has_no_rename_config_symbol():
    """The retired ``_rename_config`` helper is gone — promotion has one route."""
    source = APP_PY.read_text(encoding="utf-8")
    assert "_rename_config" not in source, (
        "paramem/server/app.py still references _rename_config. The atomic config "
        "swap lives in paramem.server.migration.promote_config, which validates the "
        "candidate before renaming; a second rename helper reintroduces the "
        "rename-before-validate defect."
    )


def test_confirm_and_base_swap_promote_through_promote_config():
    """Both mutating candidate-promotion handlers call ``promote_config``.

    Complements the default-deny scan above: this pins that the two candidate
    paths reach the filesystem through the validated primitive, not merely that
    no *other* function does.
    """
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    parents = _build_parent_map(tree)
    calls_promote_config = {
        _enclosing_function_name(node, parents)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "promote_config"
    }
    for fn_name in ("migration_confirm", "_run_base_swap_orchestration"):
        assert fn_name in calls_promote_config, (
            f"{fn_name} does not call promote_config — the live config swap must go "
            "through the single validated-promotion route."
        )


def test_restore_handlers_validate_before_their_rename():
    """``migration_rollback`` / ``backup_restore`` call ``validate_candidate`` before renaming.

    They are allowlisted renamers precisely because they validate first — this pins
    that the call is actually present, not merely that no OTHER function renames.
    """
    tree = ast.parse(APP_PY.read_text(encoding="utf-8"))
    parents = _build_parent_map(tree)
    calls_validate_candidate = {
        _enclosing_function_name(node, parents)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_candidate"
    }
    for fn_name in _ALLOWED_LIVE_CONFIG_RENAMERS:
        assert fn_name in calls_validate_candidate, (
            f"{fn_name} renames onto the live config path but does not call "
            "validate_candidate first — a backup written under an older schema "
            "could go live unbootable."
        )
