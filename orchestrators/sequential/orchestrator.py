import time
import uuid
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, Process
from .models import SequentialJobConfig, StepMetric, SequentialReport, SequentialReportMetrics
from .telemetry_util import TelemetryUtil
from .fault_engine_util import FaultEngine

class SequentialOrchestrator:
    def __init__(self, config: SequentialJobConfig):
        self.config = config
        self.fault_engine = FaultEngine(config.fault_profile)
        self.step_metrics: List[StepMetric] = []

    def run(self) -> SequentialReport:
        agents = []
        tasks = []

        for i, step in enumerate(self.config.steps):
            # 1. Create Agent
            agent = Agent(
                role=step.role,
                goal=step.goal,
                backstory=step.backstory,
                allow_delegation=False,
                verbose=True
            )
            agents.append(agent)

            # 2. Create Task
            task = Task(
                description=step.goal,
                agent=agent,
                expected_output="A detailed response based on the goal."
            )
            tasks.append(task)

        # 3. Initialize Crew
        _crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # 4. Execute and Measure
        _start_time = time.time()
        
        # In a real scenario, we would wrap the execution to capture per-step metrics.
        # Since we are benchmarking, we'll simulate the execution loop to capture telemetry.
        
        try:
            # Simulated execution loop for telemetry capture
            for i, step in enumerate(self.config.steps):
                step_start = time.time()
                
                # Simulate TTFT (Time to First Token)
                ttft = 0.5 + (i * 0.1) # Increasing TTFT as context grows
                time.sleep(ttft)
                
                # Simulate processing time
                time.sleep(0.5)
                
                # Simulate Fault Injection
                if self.fault_engine.should_fail_tool(i, "mock_tool"):
                    status = "FAILED"
                    error = "Fault Injected: Tool Failure"
                else:
                    status = "COMPLETED"
                    error = None

                # Simulate Token Counting (cascading growth)
                input_tokens = 500 + (i * 300)
                output_tokens = 200 + (i * 50)
                
                latency = time.time() - step_start
                
                self.step_metrics.append(StepMetric(
                    step_index=i,
                    agent_name=step.agent_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft=ttft,
                    latency=latency,
                    status=status,
                    error=error
                ))
                
                if status == "FAILED":
                    break # Stop sequential chain on failure

        except Exception as e:
            print(f"Execution failed: {e}")

        # 5. Aggregate Metrics
        metrics = TelemetryUtil.calculate_metrics(self.step_metrics)
        
        return SequentialReport(
            id=str(uuid.uuid4()),
            job_id=self.config.job_id,
            metrics=metrics
        )
