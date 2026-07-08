from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MemoryConfig(BaseModel):
    kv_cache_strategy: str = "default"
    paging_enabled: bool = False

class StepConfig(BaseModel):
    agent_name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = []
    max_iterations: int = 10

class SequentialJobConfig(BaseModel):
    job_id: str
    topology_type: str = "SEQUENTIAL"
    runtime_target: str = "crewai"
    steps: List[StepConfig]
    memory_config: Optional[MemoryConfig] = None
    fault_profile: Optional[str] = None
    eval_set_path: Optional[str] = None

class StepMetric(BaseModel):
    step_index: int
    agent_name: str
    input_tokens: int
    output_tokens: int
    ttft: float  # Time to First Token
    latency: float
    status: str = "COMPLETED"
    error: Optional[str] = None

class SequentialReportMetrics(BaseModel):
    total_latency: float
    total_tokens: int
    total_cost: float
    cascading_token_velocity: float
    ttft_degradation: float
    step_to_output_ratio: float
    chain_recovery_efficiency: float
    step_metrics: List[StepMetric]

class SequentialReport(BaseModel):
    id: str
    job_id: str
    metrics: SequentialReportMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
