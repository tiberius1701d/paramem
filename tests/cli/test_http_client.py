"""Tests for ``paramem.cli.http_client.ServerHTTPError``.

Covers ``__str__``'s body-inclusion behaviour: the response body is folded
into the exception's string rendering (when non-empty after stripping) so a
handler that renders the exception directly — rather than reading
``.body``/``parse_error_detail`` itself — still surfaces the server's actual
message.
"""

from __future__ import annotations

from paramem.cli.http_client import ServerHTTPError


def test_str_includes_non_empty_body():
    """A non-empty body is appended after the status line."""
    exc = ServerHTTPError(409, "http://x/migration/confirm", "candidate_invalid_config: boom")
    assert str(exc) == ("HTTP 409 from http://x/migration/confirm: candidate_invalid_config: boom")


def test_str_omits_empty_body():
    """An empty body falls back to the bare status line — no trailing colon."""
    exc = ServerHTTPError(500, "http://x/status", "")
    assert str(exc) == "HTTP 500 from http://x/status"


def test_str_omits_whitespace_only_body():
    """A whitespace-only body is treated as empty — no trailing colon,
    no literal whitespace leaking into the rendered message."""
    exc = ServerHTTPError(500, "http://x/status", "   \n\t  ")
    assert str(exc) == "HTTP 500 from http://x/status"
