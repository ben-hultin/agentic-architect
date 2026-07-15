from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class MemoryConfig(BaseModel):
    kv_cache_strategy: str = "default"
    paging_enabled: bool = False

class BenchmarkConfig(BaseModel):
    framework: str
    suite: str
    user_task_ids: List[int] = []
    with_injections: bool = False
    attack_profile: Optional[str] = None

class NodeConfig(BaseModel):
    node_name: str
    role: str
    goal: str
    tools: List[str] = []

class CyclicalJobConfig(BaseModel):
    job_id: str
    topology_type: str = "STATEFUL"
    runtime_target: str = "langgraph"
    task: str
    nodes: List[NodeConfig]
    max_cycles: int = 5
    memory_config: Optional[MemoryConfig] = None
    fault_profile: Optional[str] = None
    checkpoint_backend: str = "memory"
    benchmark_config: Optional[BenchmarkConfig] = None

class NodeMetric(BaseModel):
    step_index: int
    node_name: str
    cycle_index: int
    input_tokens: int
    output_tokens: int
    ttft: float
    latency: float
    status: str = "COMPLETED"
    error: Optional[str] = None
    route_decision: Optional[str] = None

class CyclicalReportMetrics(BaseModel):
    total_latency: float
    total_tokens: int
    total_cost: float
    trajectory_drift: float
    cascading_token_consumption: float
    thread_recovery_latency: float
    cycles_executed: int
    node_metrics: List[NodeMetric]

class CyclicalReport(BaseModel):
    id: str
    job_id: str
    metrics: CyclicalReportMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)
