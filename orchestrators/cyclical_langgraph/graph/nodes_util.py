from typing import Dict, Any
from .state import AgentState
from mcp.stub_mcp_client_util import call_tool
from fault_engine_util import FaultEngine

def classifier_node(state: AgentState) -> Dict[str, Any]:
    cycle_count = state.get("cycle_count", 0)
    
    if cycle_count < 2:
        routing_flag = "tool"
    else:
        routing_flag = "analyze"
        
    return {
        "messages": ["Classifier determined next step: " + routing_flag],
        "active_context": {"last_action": "classify"},
        "routing_flag": routing_flag
    }

def mcp_tool_node(state: AgentState, fault_engine: FaultEngine) -> Dict[str, Any]:
    try:
        cycle_count = state.get("cycle_count", 0)
        tool_result = call_tool("invoice_db_lookup", {"query": "test"}, fault_engine, cycle_count)
        result_msg = f"Tool result: {tool_result}"
    except Exception as e:
        raise e 
        
    return {
        "messages": [result_msg],
        "active_context": {"last_tool": "invoice_db_lookup", "tool_data": tool_result},
        "routing_flag": "analyze"
    }

def analyst_node(state: AgentState) -> Dict[str, Any]:
    cycle_count = state.get("cycle_count", 0)
    max_cycles = state.get("max_cycles", 5)

    if cycle_count >= max_cycles:
        routing_flag = "end"
    else:
        routing_flag = "classify"

    return {
        "messages": ["Analyst reviewed data. Next step: " + routing_flag],
        "active_context": {"analysis_complete": routing_flag == "end"},
        "routing_flag": routing_flag,
        "cycle_count": cycle_count + 1
    }
