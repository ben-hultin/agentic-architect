from typing import TypedDict, Annotated, List, Dict, Any
import operator

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not a:
        return b
    if not b:
        return a
    merged = a.copy()
    for k, v in b.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    active_context: Annotated[Dict[str, Any], merge_dicts]
    routing_flag: str
    cycle_count: int
    max_cycles: int
    thread_id: str
