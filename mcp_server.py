"""
MCP Server Entry Point

Starts the A-MEM MCP Server.
"""

import asyncio
from src.a_mem.main import main

if __name__ == "__main__":
    asyncio.run(main())



