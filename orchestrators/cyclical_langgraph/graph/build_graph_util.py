from langgraph.graph import StateGraph, START, END
from .state import AgentState
from .nodes_util import classifier_node, mcp_tool_node, analyst_node
from fault_engine_util import FaultEngine
import functools

def route_from_classifier(state: AgentState) -> str:
    flag = state.get("routing_flag", "")
    if flag == "tool":
        return "mcp_tool"
    elif flag == "analyze":
        return "analyst"
    return "analyst"

def route_from_analyst(state: AgentState) -> str:
    flag = state.get("routing_flag", "")
    if flag == "end":
        return END
    return "classifier"

def build_cyclical_graph(fault_engine: FaultEngine):
    workflow = StateGraph(AgentState)
    
    workflow.add_node("classifier", classifier_node)
    
    bound_tool_node = functools.partial(mcp_tool_node, fault_engine=fault_engine)
    workflow.add_node("mcp_tool", bound_tool_node)
    
    workflow.add_node("analyst", analyst_node)
    
    workflow.add_edge(START, "classifier")
    
    workflow.add_conditional_edges(
        "classifier",
        route_from_classifier,
        {
            "mcp_tool": "mcp_tool",
            "analyst": "analyst"
        }
    )
    
    workflow.add_edge("mcp_tool", "analyst")
    
    workflow.add_conditional_edges(
        "analyst",
        route_from_analyst,
        {
            END: END,
            "classifier": "classifier"
        }
    )
    
    return workflow
