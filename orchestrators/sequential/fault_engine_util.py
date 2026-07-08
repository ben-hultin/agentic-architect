import random
from typing import Optional

class FaultEngine:
    def __init__(self, fault_profile: Optional[str] = None):
        self.fault_profile = fault_profile

    def should_fail_tool(self, step_index: int, _tool_name: str) -> bool:
        if not self.fault_profile:
            return False

        if self.fault_profile == "Tool_Network_Drop_503":
            # 20% chance of failure for any tool call
            return random.random() < 0.2
        
        if self.fault_profile == "Step_2_Failure" and step_index == 1:
            # Force failure on the second step (index 1)
            return True

        return False

    def inject_corruption(self, data: str) -> str:
        if self.fault_profile == "Data_Corruption":
            # Randomly flip some characters
            chars = list(data)
            for _ in range(len(chars) // 10):
                idx = random.randint(0, len(chars) - 1)
                chars[idx] = chr(random.randint(33, 126))
            return "".join(chars)
        return data
