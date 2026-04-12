"""Unit tests for HA client (no real API calls)."""

from unittest.mock import MagicMock, patch

from paramem.server.tools.ha_client import HAClient


class TestHAClient:
    @patch("paramem.server.tools.ha_client.httpx.Client")
    def test_health_check_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": "API running."}
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.is_closed = False
        mock_client_cls.return_value = mock_client

        client = HAClient("http://localhost:8123", "token")
        client._client = mock_client
        result = client.health_check()
        assert result == {"message": "API running."}

    @patch("paramem.server.tools.ha_client.httpx.Client")
    def test_call_service_with_return_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "search results"}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        mock_client_cls.return_value = mock_client

        client = HAClient("http://localhost:8123", "token")
        client._client = mock_client
        result = client.call_service(
            "script",
            "tavily_search",
            data={"variables": {"query": "test"}},
            return_response=True,
        )
        assert result == {"content": "search results"}
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["return_response"] is True
