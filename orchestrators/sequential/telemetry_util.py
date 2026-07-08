from typing import List
from .models import StepMetric, SequentialReportMetrics

class TelemetryUtil:
    @staticmethod
    def _calculate_cascading_token_velocity(step_metrics: List[StepMetric]) -> float:
        if len(step_metrics) <= 1:
            return 1.0
        velocities = []
        for i in range(1, len(step_metrics)):
            if step_metrics[i-1].input_tokens > 0:
                velocities.append(step_metrics[i].input_tokens / step_metrics[i-1].input_tokens)
        return sum(velocities) / len(velocities) if velocities else 1.0

    @staticmethod
    def _calculate_ttft_degradation(step_metrics: List[StepMetric]) -> float:
        if len(step_metrics) <= 1:
            return 0.0
        degradations = []
        for i in range(1, len(step_metrics)):
            degradations.append(step_metrics[i].ttft - step_metrics[i-1].ttft)
        return sum(degradations) / len(degradations) if degradations else 0.0

    @staticmethod
    def calculate_metrics(step_metrics: List[StepMetric]) -> SequentialReportMetrics:
        if not step_metrics:
            return SequentialReportMetrics(
                total_latency=0,
                total_tokens=0,
                total_cost=0,
                cascading_token_velocity=0,
                ttft_degradation=0,
                step_to_output_ratio=0,
                chain_recovery_efficiency=0,
                step_metrics=[]
            )

        total_latency = sum(m.latency for m in step_metrics)
        total_tokens = sum(m.input_tokens + m.output_tokens for m in step_metrics)
        total_cost = (total_tokens / 1000) * 0.01

        cascading_token_velocity = TelemetryUtil._calculate_cascading_token_velocity(step_metrics)
        ttft_degradation = TelemetryUtil._calculate_ttft_degradation(step_metrics)
        step_to_output_ratio = total_tokens / len(step_metrics)
        
        successful_steps = sum(1 for m in step_metrics if m.status == "COMPLETED")
        chain_recovery_efficiency = successful_steps / len(step_metrics)

        return SequentialReportMetrics(
            total_latency=total_latency,
            total_tokens=total_tokens,
            total_cost=total_cost,
            cascading_token_velocity=cascading_token_velocity,
            ttft_degradation=ttft_degradation,
            step_to_output_ratio=step_to_output_ratio,
            chain_recovery_efficiency=chain_recovery_efficiency,
            step_metrics=step_metrics
        )
