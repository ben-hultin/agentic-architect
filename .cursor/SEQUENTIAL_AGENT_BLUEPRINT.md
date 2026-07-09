To benchmark, execute, and monitor Sequential and Linear Control Topologies (e.g., CrewAI or structural chaining patterns) without falling into the traditional trapping of treating them as static black boxes, your platform requires a strict operational blueprint.
In sequential multi-agent or multi-step patterns, state behaves as an append-only rolling text window, where downstream tasks inherit a progressively growing context footprint. This linear structure introduces specific bottlenecks: compounding token degradation, high prefill latency overhead, and cascading error vulnerabilities.

Here is the decoupled architectural blueprint for orchestrating, executing, and benchmarking Sequential Agentic Topologies natively on GCP and Firebase.

1. Architectural Layout: Sequential AgOps Execution Pipeline
   To isolate variable network overhead and accurately compute token processing metrics, the execution path leverages decoupled Cloud Run targets driven by a single-concurrency Google Cloud Tasks worker.
   [ Next.js UI / Dashboard Control ]
   │
   ▼
   [ Firestore: Job Specification ]
   │
   ▼
   [ Google Cloud Tasks Queue ] ──► (maxConcurrentDispatches = 1 / Enforces Quota Isolation)
   │
   ▼
   [ Cloud Run: AgOps Load Thrower ]
   │
   (Injects Seed Payload & Tool Latency Faults)
   │
   ▼
   [ Cloud Run: Framework Runner Target ] ──► (Hosts CrewAI / Sequential Linear Graph)
   │
   ├──► Persists Task Artifacts via Firebase Data Connect
   └──► Pipes Operational Metrics to BigQuery
2. The 5-Layer Architectural Blueprint

Layer 1: Configuration & Manifest Layer (Next.js + Firestore)

- Dashboard Configuration: The user spins up a benchmark profile specifying a linear agent chain (e.g., Agent A: Triage $\rightarrow$ Agent B: Enrichment $\rightarrow$ Agent C: Output Generation).
- Job Registry: Firestore captures this schema within a jobs/{jobId} document under a SEQUENTIAL topology flag, mapping exact downstream dependencies, explicit tools allowed per step, and target iteration boundaries (max_iterations).

Layer 2: Serialized Ingestion Layer (Cloud Tasks)

- Provider Quota Protection: Multi-agent chains generate intense bursts of RPM (Requests Per Minute) and TPM (Tokens Per Minute) due to intermediate reasoning loops and validation steps. Cloud Tasks serializes runs (maxConcurrentDispatches = 1) to ensure execution profiles aren't corrupted by random provider rate-limiting exceptions or transient network degradation.

Layer 3: The Framework Runner Target (Cloud Run)

- Stateless Workspace Container: Cloud Run spins up the runtime target hosting your sequential code footprint (e.g., CrewAI or native pipeline chains).
- State Mechanics: In this linear topology, state maps as a rigid sequence of tasks. Each step is treated as a blocking operation. The output of Task[i] is automatically packed, formatted, and injected into the system prompt window of Task[i+1].
- Data Persistence: Long-term storage of final text outputs and structured metadata artifacts maps cleanly into Cloud Firestore, guaranteeing highly typed, queryable data formats.

Layer 4: Automated Testing Sandbox & Fault Injection (Isolated Compute)

- The Compute Separation Rule: The Load Thrower container runs completely isolated from the framework runner to prevent resource contention.
- The Linear Fault Engine: Unlike cyclical topologies where a tool can be retried indefinitely, a tool failure in a strict sequential loop often breaks the chain entirely. The Load Thrower tests this by programmatically corrupting downstream tool requests at specific step indices (e.g., forcing a mock database tool to throw an HTTP 500 during Task 2). This explicitly benchmarks whether the sequential framework can execute a graceful fallback or if it triggers a terminal failure.

Layer 5: Aggregation & Performance Layer (BigQuery)

- Telemetry Streaming: Mid-execution runtime parameters are continuously written to BigQuery. The dashboard monitors these metrics to track how the agent's context footprint expands step-by-step.

4. Implementation Reference (Python + CrewAI)

The sequential orchestrator is implemented in Python using the CrewAI framework, located in `orchestrators/sequential/`.

### Directory Structure
```text
orchestrators/sequential/
├── dao/
│   └── sequential_dao.py      # Firestore persistence for jobs and reports
├── services/
│   └── firestore_service.py   # Singleton Firestore client
├── Dockerfile                 # Container definition for Cloud Run
├── main.py                    # FastAPI entry point for Cloud Tasks
├── models.py                  # Pydantic models for config and metrics
├── orchestrator.py            # Core CrewAI execution logic
├── telemetry_util.py          # Metric calculation engine
├── fault_engine_util.py       # Programmatic fault injection
└── requirements.txt           # Python dependencies (crewai, fastapi, etc.)
```

### Core Components
- **Orchestrator**: Uses `crewai.Crew` with `Process.sequential` to execute a linear chain of agents. It wraps the execution to capture per-step telemetry.
- **Telemetry Engine**: Implements the benchmarking matrix, calculating cascading token velocity and TTFT degradation.
- **Fault Engine**: Injects simulated failures based on a `fault_profile` (e.g., `Tool_Network_Drop_503`) to measure recovery efficiency.
- **API Layer**: A FastAPI server that listens for `/run` POST requests from Cloud Tasks to trigger asynchronous benchmarking jobs.

### Deployment Workflow
1. **Build**: `gcloud builds submit --tag [IMAGE_URL] .`
2. **Deploy**: `gcloud run deploy [SERVICE_NAME] --image [IMAGE_URL] --region [REGION]`
3. **Trigger**: Cloud Tasks sends a signed OIDC request to the `/run` endpoint with a `jobId`.

5. Targeted Benchmarking Matrix for Sequential Topologies
   Because sequential workflows pass growing context footprints forward, the benchmarking layer must focus heavily on tracking cascading expenses and context-window degradation:
   ┌────────────────────────────────────────┐
   │ SEQUENTIAL EVALUATION MATRIX │
   └────────────────────────────────────────┘
   │
   ┌───────────────────┬────────┴───────────┬───────────────────┐
   ▼ ▼ ▼ ▼
   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
   │ CASCADING COST │ │ CONTEXT WINDOW │ │ TRAJECTORY CODE │ │ RESILIENCY CODE │
   └─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
   • Compound Token • Prefill Growth • Step-to-Output • Chain Recovery
   Velocity • Prefix Sharing Ratio • Silent Failure
   • Input/Output • TTFT Degradation • Task Delegation Tracking
   Asymmetry Ratios • Fallback Accuracy

- Compound Token Velocity: Tracks the exact token inflation curve across the linear chain. Because the output of each task expands the prompt context of the next, input costs scale quadratically rather than linearly. This metric isolates the precise step index where cost-efficiency breaks down.

- Prefill & TTFT Degradation: Measures Time to First Token (TTFT) at each distinct link in the chain. As the conversation and context history scale forward, the engine tracks how severely model latency degrades during the prefill parsing phase.

- Step-to-Output Ratio: Measures the structural efficiency of the sequence. It calculates how many intermediate reasoning tokens and sub-steps were consumed to generate the final useful task output, exposing hidden execution inefficiencies in structural managers or orchestrator nodes.

- Chain Recovery Efficiency: Tracks behavior when an intermediate node receives an error. If Agent 2 of a 4-agent chain receives corrupted data, the engine measures if the framework can handle the exception locally and pass a safe state down to Agent 3, or if the system drops out completely.
