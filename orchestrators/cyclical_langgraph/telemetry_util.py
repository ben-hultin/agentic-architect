from typing import List
from models import NodeMetric, CyclicalReportMetrics

class TelemetryUtil:
    @staticmethod
    def calculate_metrics(node_metrics: List[NodeMetric], max_cycles: int, thread_recovery_latency: float) -> CyclicalReportMetrics:
        if not node_metrics:
            return CyclicalReportMetrics(
                total_latency=0,
                total_tokens=0,
                total_cost=0,
                trajectory_drift=0,
                cascading_token_consumption=0,
                thread_recovery_latency=0,
                cycles_executed=0,
                node_metrics=[]
            )

        total_latency = sum(m.latency for m in node_metrics)
        total_tokens = sum(m.input_tokens + m.output_tokens for m in node_metrics)
        total_cost = (total_tokens / 1000) * 0.01

        cycles_executed = max(m.cycle_index for m in node_metrics) if node_metrics else 0
        trajectory_drift = min(1.0, cycles_executed / max_cycles) if max_cycles > 0 else 0.0

        velocities = []
        for i in range(1, len(node_metrics)):
            if node_metrics[i-1].input_tokens > 0:
                velocities.append(node_metrics[i].input_tokens / node_metrics[i-1].input_tokens)
        cascading_token_consumption = sum(velocities) / len(velocities) if velocities else 1.0

        return CyclicalReportMetrics(
            total_latency=total_latency,
            total_tokens=total_tokens,
            total_cost=total_cost,
            trajectory_drift=trajectory_drift,
            cascading_token_consumption=cascading_token_consumption,
            thread_recovery_latency=thread_recovery_latency,
            cycles_executed=cycles_executed,
            node_metrics=node_metrics
        )
