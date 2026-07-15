from typing import Dict, Any, List
from fault_engine_util import FaultEngine

def list_tools() -> List[str]:
    return ["invoice_db_lookup", "document_search"]

def call_tool(name: str, args: Dict[str, Any], fault_engine: FaultEngine, step_index: int) -> Dict[str, Any]:
    fault_engine.trigger_tool_fault(step_index, name)
    
    if name == "invoice_db_lookup":
        return {
            "invoice_id": "INV-12345",
            "amount": 1500.00,
            "status": "PAID",
            "vendor": "Acme Corp"
        }
    elif name == "document_search":
        return {
            "results": [
                {"doc_id": "DOC-99", "snippet": "Acme Corp payment terms are Net 30."}
            ]
        }
    else:
        return {"error": f"Tool '{name}' not found."}
