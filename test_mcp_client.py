"""
MCP Client Test Harness
=======================
Tests mcp_server.py end-to-end over stdio JSON-RPC transport using the official MCP Client SDK.
Verifies tool discovery and executes query_vector_db.
"""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_test():
    workspace_dir = Path(__file__).resolve().parent
    server_script = workspace_dir / "mcp_server.py"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(workspace_dir),
    )

    print(f"Connecting to MCP server via stdio: {sys.executable} {server_script}...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Session initialized successfully.")

            # 1. Discover registered tools
            tools_response = await session.list_tools()
            tool_names = [t.name for t in tools_response.tools]
            print(f"Discovered {len(tool_names)} tool(s) in registry: {tool_names}")

            for t in tools_response.tools:
                print(f"\nTool: {t.name}")
                print(f"Description: {t.description}")
                print(f"Input Schema: {json.dumps(t.input_schema, indent=2)}")

            # 2. Execute query_vector_db tool
            query_args = {
                "query": "What are the operating hours for Apna Indian Kitchen?",
                "task_type_filter": "Data2txt"
            }
            print(f"\nCalling tool 'query_vector_db' with arguments: {query_args}...")

            call_result = await session.call_tool(
                "query_vector_db",
                arguments=query_args
            )

            print("\n=== RAW TOOL OUTPUT ===")
            for content in call_result.content:
                if hasattr(content, "text"):
                    print(content.text)
                else:
                    print(content)
            print("========================")


if __name__ == "__main__":
    asyncio.run(run_test())
