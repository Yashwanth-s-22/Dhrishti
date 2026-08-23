"""
Drishti Unified MCP Server
==========================
Unified Model Context Protocol (MCP) server exposing Drishti's tools:
1. `search_gdelt_news`: Real-time news search via GDELT DOC 2.0 API.
2. `run_drishti_ml_cascade`: Quantitative Multi-Stage ML Cascade execution (Models A -> B -> C -> D).
3. `get_historical_event`: Historical event knowledge from event_store.json.
4. `get_stakeholder_analysis`: Deterministic rules-based stakeholder disaggregation.

Can be run as a standalone stdio/SSE server or imported directly by agents.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is on sys.path when running script directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.mcpserver import MCPServer
from config.settings import MCP_SERVER_NAME, MCP_SERVER_VERSION
from drishti_mcp.tools.gdelt_tool import search_gdelt_news
from drishti_mcp.tools.ml_cascade_tool import run_drishti_ml_cascade
from drishti_mcp.tools.event_store_tool import get_historical_event
from drishti_mcp.tools.stakeholder_tool import get_stakeholder_analysis

# Initialize official MCP server
mcp_server = MCPServer(
    name=MCP_SERVER_NAME,
    instructions="Drishti Macro-Agricultural Resilience Decision Intelligence Server.",
    version=MCP_SERVER_VERSION,
)


@mcp_server.tool(
    name="search_gdelt_news",
    description="Search GDELT DOC 2.0 API for recent news articles on agricultural trade, policy, and geopolitical shocks.",
)
def tool_search_gdelt_news(
    query: str,
    timespan: str = "7d",
    max_records: int = 5,
) -> Dict[str, Any]:
    """Search GDELT DOC 2.0 API for news articles."""
    return search_gdelt_news(query=query, timespan=timespan, max_records=max_records)


@mcp_server.tool(
    name="run_drishti_ml_cascade",
    description="Execute the validated multi-stage econometric ML cascade (Models A -> B -> C -> D) for quantitative predictions.",
)
def tool_run_drishti_ml_cascade(
    country: str,
    trade_type: str,
    hs4: int,
    year: int = 2024,
    month: int = 1,
    shock_intensity: float = 1.0,
    trade_share: float = 5.0,
) -> Dict[str, Any]:
    """Execute Drishti ML cascade across Models A, B, C, and D."""
    return run_drishti_ml_cascade(
        country=country,
        trade_type=trade_type,
        hs4=hs4,
        year=year,
        month=month,
        shock_intensity=shock_intensity,
        trade_share=trade_share,
    )


@mcp_server.tool(
    name="get_historical_event",
    description="Query the curated historical event store for past geopolitical shocks, observed trade/price impacts, and model validations.",
)
def tool_get_historical_event(
    event_id: str = "",
    query: str = "",
    commodity: str = "",
    country: str = "",
) -> Dict[str, Any]:
    """Retrieve historical event analysis from Drishti event store."""
    return get_historical_event(
        event_id=event_id,
        query=query,
        commodity=commodity,
        country=country,
    )


@mcp_server.tool(
    name="get_stakeholder_analysis",
    description="Compute structured stakeholder impacts (farmers, consumers, exporters, importers, regional, government) using the deterministic rule engine.",
)
def tool_get_stakeholder_analysis(
    country: str,
    trade_type: str,
    hs4: int,
    commodity_name: str = "",
    trade_pred: Optional[float] = None,
    prod_growth: Optional[float] = None,
    prod_risk: str = "Medium",
    price_pred: Optional[float] = None,
    gva_pred: Optional[float] = None,
    infl_pred: Optional[float] = None,
    trade_share: float = 5.0,
    effective_shock: float = 0.0,
    canonical_shock_direction: str = "supply_contraction",
    month: int = 1,
    year: int = 2024,
) -> Dict[str, Any]:
    """Evaluate stakeholder disaggregation rules."""
    return get_stakeholder_analysis(
        country=country,
        trade_type=trade_type,
        hs4=hs4,
        commodity_name=commodity_name,
        trade_pred=trade_pred,
        prod_growth=prod_growth,
        prod_risk=prod_risk,
        price_pred=price_pred,
        gva_pred=gva_pred,
        infl_pred=infl_pred,
        trade_share=trade_share,
        effective_shock=effective_shock,
        canonical_shock_direction=canonical_shock_direction,
        month=month,
        year=year,
    )


def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Direct synchronous helper to execute MCP tools from agents with unified exception handling.
    """
    tools_map = {
        "search_gdelt_news": tool_search_gdelt_news,
        "run_drishti_ml_cascade": tool_run_drishti_ml_cascade,
        "get_historical_event": tool_get_historical_event,
        "get_stakeholder_analysis": tool_get_stakeholder_analysis,
    }

    if tool_name not in tools_map:
        return {
            "status": "error",
            "message": f"Tool '{tool_name}' not found on Drishti MCP server.",
            "available_tools": list(tools_map.keys()),
        }

    try:
        return tools_map[tool_name](**arguments)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error executing MCP tool '{tool_name}': {e}",
            "arguments": arguments,
        }


async def main():
    """Run MCP server in stdio mode."""
    print(f"Starting {MCP_SERVER_NAME} v{MCP_SERVER_VERSION} (stdio mode)...")
    await mcp_server.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
