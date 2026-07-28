from __future__ import annotations

from pathlib import Path

import httpx
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent

from asr_mcp.client import AsrHttpClient, BridgeSettings
from asr_mcp.server import create_mcp_server


async def test_mcp_server_exposes_asr_tools_and_calls_capability_discovery(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"models": [{"id": "qwen3-asr-1.7b", "status": "unloaded"}]},
        )

    client = AsrHttpClient(
        BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
        transport=httpx.MockTransport(handler),
    )
    server = create_mcp_server(client)

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        tools = await session.list_tools()
        result = await session.call_tool("asr_list_models", {})

    assert {tool.name for tool in tools.tools} == {
        "asr_list_models",
        "asr_get_model_status",
        "asr_transcribe_file",
        "asr_create_job",
        "asr_get_job",
        "asr_cancel_job",
        "asr_load_model",
        "asr_unload_model",
    }
    tools_by_name = {tool.name: tool for tool in tools.tools}
    assert tools_by_name["asr_list_models"].annotations is not None
    assert tools_by_name["asr_list_models"].annotations.readOnlyHint is True
    assert tools_by_name["asr_unload_model"].annotations is not None
    assert tools_by_name["asr_unload_model"].annotations.destructiveHint is True
    assert result.isError is False
    assert result.structuredContent == {
        "models": [{"id": "qwen3-asr-1.7b", "status": "unloaded"}]
    }


async def test_mcp_transcribe_tool_uploads_allowed_local_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"
    audio_path.write_bytes(b"mcp-audio")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        assert b"mcp-audio" in request.content
        assert b"product-name" in request.content
        return httpx.Response(
            200,
            json={"id": "tr_mcp", "model": "qwen3-asr-1.7b", "text": "MCP 转录成功"},
        )

    server = create_mcp_server(
        AsrHttpClient(
            BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(tmp_path,)),
            transport=httpx.MockTransport(handler),
        )
    )

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        result = await session.call_tool(
            "asr_transcribe_file",
            {
                "path": str(audio_path),
                "model": "qwen3-asr-1.7b",
                "language": "zh",
                "context": "product-name",
            },
        )

    assert result.isError is False
    assert result.structuredContent == {
        "id": "tr_mcp",
        "model": "qwen3-asr-1.7b",
        "text": "MCP 转录成功",
    }


async def test_mcp_tool_reports_disallowed_local_path_as_tool_error(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"not-allowed")
    server = create_mcp_server(
        AsrHttpClient(
            BridgeSettings(base_url="http://asr.test:18080", allowed_roots=(allowed_root,)),
        )
    )

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("asr_transcribe_file", {"path": str(outside_audio)})

    assert result.isError is True
    content = result.content[0]
    assert isinstance(content, TextContent)
    assert "ASR_MCP_ALLOWED_ROOTS" in content.text
