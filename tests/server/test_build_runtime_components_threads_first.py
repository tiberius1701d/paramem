"""``paramem.server.app._build_runtime_components`` — structural check that
``torch.set_num_threads`` is the first statement.
"""

from __future__ import annotations

import ast
import inspect

import paramem.server.app as app_module


class TestBuildRuntimeComponentsSetsThreadsFirst:
    """``torch.set_num_threads(config.cpu.threads)`` is the FIRST statement
    of ``_build_runtime_components`` — a structural check on the source,
    since exercising the routine behaviourally would load every runtime
    component (speaker embedding, TTS/STT, cloud agent, HA client)."""

    def test_torch_set_num_threads_is_the_first_statement_in_the_source(self):
        source = inspect.getsource(app_module._build_runtime_components)
        tree = ast.parse(source)
        (func_def,) = tree.body
        assert isinstance(func_def, ast.FunctionDef)

        body = func_def.body
        # body[0] is the docstring (an Expr wrapping a str Constant) when
        # present; the first EXECUTABLE statement is the one right after it.
        first_statement = body[1] if _is_docstring(body[0]) else body[0]

        assert isinstance(first_statement, ast.Expr)
        call = first_statement.value
        assert isinstance(call, ast.Call)
        assert ast.unparse(call.func) == "torch.set_num_threads"
        assert ast.unparse(call.args[0]) == "config.cpu.threads"


def _is_docstring(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )
