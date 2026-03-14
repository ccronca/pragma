#!/usr/bin/env python3
"""
Pragma MCP Server

Exposes Pragma's historical MR search as MCP tools that AI assistants can call
to get institutional knowledge from past code changes and review discussions.
"""

import asyncio
import json
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


PRAGMA_API_URL = os.getenv("PRAGMA_API_URL", "http://localhost:8000")

app = Server("pragma-mcp")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """Expose Pragma search functions as callable MCP tools."""
    return [
        Tool(
            name="search",
            description=(
                "Search for similar merge requests using semantic search. "
                "Use 'query' for natural language (discussions) or 'code_diff' for code patterns (diffs)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query (best for finding discussions and past decisions)",
                    },
                    "code_diff": {
                        "type": "string",
                        "description": "Code diff to search for (best for finding similar code patterns)",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (1-20)",
                        "default": 5,
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum similarity score threshold (0-1)",
                        "default": 0.0,
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Filter by content type: 'diff' or 'discussion'",
                        "enum": ["diff", "discussion"],
                    },
                    "repository": {
                        "type": "string",
                        "description": (
                            "Filter results to a specific repository "
                            "(format: 'owner/name', e.g. 'product-security/pdm')"
                        ),
                    },
                },
            },
        ),
        Tool(
            name="get_mr",
            description="Get full details of a specific merge request by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "mr_id": {
                        "type": "integer",
                        "description": "The merge request IID",
                    }
                },
                "required": ["mr_id"],
            },
        ),
        Tool(
            name="list_mrs",
            description="List all indexed merge requests",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of MRs to return (1-200)",
                        "default": 50,
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Pagination offset",
                        "default": 0,
                    },
                    "repository": {
                        "type": "string",
                        "description": (
                            "Filter by repository "
                            "(format: 'owner/name', e.g. 'product-security/pdm')"
                        ),
                    },
                },
            },
        ),
        Tool(
            name="list_repositories",
            description="List all indexed repositories with MR counts",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls by routing to the Pragma REST API."""
    async with httpx.AsyncClient() as client:
        if name == "search":
            payload = {
                "query": arguments.get("query"),
                "code_diff": arguments.get("code_diff"),
                "top_k": arguments.get("top_k", 5),
                "min_score": arguments.get("min_score", 0.0),
                "content_type": arguments.get("content_type"),
            }
            if arguments.get("repository"):
                payload["repository"] = arguments["repository"]
            response = await client.post(f"{PRAGMA_API_URL}/search", json=payload)
        elif name == "get_mr":
            response = await client.get(f"{PRAGMA_API_URL}/mrs/{arguments['mr_id']}")
        elif name == "list_mrs":
            params = {
                "limit": arguments.get("limit", 50),
                "offset": arguments.get("offset", 0),
            }
            if arguments.get("repository"):
                params["repository"] = arguments["repository"]
            response = await client.get(f"{PRAGMA_API_URL}/mrs", params=params)
        elif name == "list_repositories":
            response = await client.get(f"{PRAGMA_API_URL}/repositories")
        else:
            raise ValueError(f"Unknown tool: {name}")

        response.raise_for_status()
        return [TextContent(type="text", text=json.dumps(response.json(), indent=2))]


async def main():
    """Run the MCP server on stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
