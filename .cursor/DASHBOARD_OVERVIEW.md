# Agent Ops Dashboard Overview

The Agent Ops Dashboard is a Next.js-based benchmarking engine designed to monitor, run, and test various agentic topologies, ranging from standard application-layer frameworks (like CrewAI, LangGraph, and Google ADK) to experimental low-level operating system patterns (like the Agentic Microkernel).

## Tech Stack

- **Framework**: [Next.js 14](https://nextjs.org/) (App Router)
- **Language**: [TypeScript](https://www.typescriptlang.org/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/)
- **Icons**: [Lucide React](https://lucide.dev/)
- **Database/Backend**: [Firebase](https://firebase.google.com/) (Firestore)
- **Infrastructure**: Google Cloud Tasks (for job enqueuing)

## Directory Structure

```text
dashboard/
├── src/
│   ├── app/                # Next.js App Router pages and API routes
│   │   ├── api/            # Backend API endpoints (e.g., task enqueuing)
│   │   ├── builds/         # Build listing and detailed build views
│   │   ├── insights/       # Cross-topology comparison and recommendations
│   │   ├── layout.tsx      # Root layout with Sidebar
│   │   └── page.tsx        # Main Dashboard (KPI overview)
│   ├── components/         # Reusable UI components (KPI cards, charts, tables)
│   ├── dao/                # Data Access Objects (Firebase/Firestore hooks)
│   ├── services/           # External service integrations (Firebase, Cloud Tasks)
│   ├── types/              # TypeScript interfaces and type definitions
│   └── utils/              # Utility functions (currently empty)
├── public/                 # Static assets
└── tailwind.config.ts      # Tailwind CSS configuration
```

## Key Concepts

### 1. Control Topologies
The platform categorizes agents by their Control Topology & State Paradigms:
- **SEQUENTIAL**: Linear workflows (e.g., CrewAI).
- **STATEFUL**: Cyclical, node-based graphs (e.g., LangGraph).
- **ADAPTIVE**: Code-first dynamic routing graphs (e.g., Google ADK).
- **OS_KERNEL**: Low-level virtual machine layers using LLMs as CPUs (Agentic Microkernel).

### 2. Jobs
A "Job" represents a single benchmarking run. Jobs are triggered from the UI and enqueued via Google Cloud Tasks to be executed by a remote runner.
- **Status**: QUEUED, RUNNING, COMPLETED, FAILED.
- **Payload**: Includes `topology_type`, `runtime_target`, `memory_config`, and `fault_profile`.

### 3. Reports & Metrics
After a job completes, a "Report" is generated with detailed metrics. The dashboard displays different metrics based on the topology:
- **App-Layer Metrics**: Graph Trajectory Accuracy, Cascading Token Consumption, Thread Recovery Overhead.
- **Microkernel Metrics**: Context Compression Ratio, Scheduler Preemption Latency, Wasm Tool Initialisation Speed.

## Data Flow

1. **Trigger**: User clicks "Run Bench Test" in the Build Details page.
2. **Enqueue**: The frontend calls `enqueueBenchmarkingJob` (`services/cloudTasks.ts`), which hits the `/api/tasks/enqueue` endpoint.
3. **Execution**: The API route (ideally) enqueues a task in Google Cloud Tasks, which triggers the remote runner container.
4. **Monitoring**: The dashboard listens to Firestore changes via `useJobs` and `useReports` hooks (`dao/` directory) to provide live updates.

## UI Components

- **KPICard**: Displays high-level metrics (Success Rate, Error Rate, etc.) with trend indicators.
- **TrendChart**: Visualizes performance trends over time.
- **BuildsTable**: Lists all active and historical builds with their status and key metrics.
- **AttentionList**: Highlights builds that require immediate investigation (e.g., high error rates).
- **Sidebar**: Main navigation for the dashboard.
