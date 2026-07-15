import random
from typing import Optional, Dict, Any

class ToolError(Exception):
    def __init__(self, message: str, code: int, response: Dict[str, Any]):
        super().__init__(message)
        self.message = message
        self.code = code
        self.response = response

class FaultEngine:
    def __init__(self, fault_profile: Optional[str] = None):
        self.fault_profile = fault_profile

    def trigger_tool_fault(self, step_index: int, tool_name: str):
        if not self.fault_profile:
            return

        if self.fault_profile == "Tool_Network_Drop_503":
            if random.random() < 0.2:
                raise ToolError(
                    message=f"Service Unavailable: Tool '{tool_name}' failed.",
                    code=503,
                    response={"status": "error", "type": "network_timeout", "details": "upstream_reset"}
                )
        
        if self.fault_profile == "Step_2_Failure" and step_index == 1:
            raise ToolError(
                message=f"Timeout: Step 2 tool '{tool_name}' exceeded 30s limit.",
                code=408,
                response={"status": "timeout", "limit": 30000}
            )

        if self.fault_profile == "Step_1_Failure" and step_index == 0:
            raise ToolError(
                message=f"Unauthorized: Authentication failed for tool '{tool_name}'.",
                code=401,
                response={"status": "unauthorized", "provider": "google_cloud", "reason": "expired_token"}
            )

    def inject_corruption(self, data: str) -> str:
        if self.fault_profile == "Data_Corruption":
            chars = list(data)
            for _ in range(len(chars) // 10):
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = chr(random.randint(33, 126))
            return "".join(chars)
        return data
