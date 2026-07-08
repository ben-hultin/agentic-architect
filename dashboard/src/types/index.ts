export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";
export type TopologyType = "SEQUENTIAL" | "STATEFUL" | "ADAPTIVE" | "OS_KERNEL";

export interface MemoryConfig {
  kv_cache_strategy: string;
  paging_enabled: boolean;
}

export interface Job {
  id: string;
  topology_type: TopologyType;
  runtime_target: string;
  memory_config?: MemoryConfig;
  fault_profile?: string;
  status: JobStatus;
  createdAt: any;
  updatedAt: any;
  successRate?: number;
  errorRate?: number;
  cost?: number;
  useCase?: string;
  complexity?: string;
  lastRun?: string;
}

export interface Report {
  id: string;
  jobId: string;
  metrics: {
    performance: {
      latency: number;
      ttft: number;
      tokenCount: number;
      cost: number;
    };
    security: {
      violationRate: number;
    };
    accuracy: {
      relevance: number;
      factualCorrectness: number;
    };
    robustness: {
      trajectoryDeviation: number;
      stepSuccessRate: number;
    };
    microkernel?: {
      contextCompressionRatio: number;
      preemptionLatency: number;
      wasmInitSpeed: number;
    };
    appLayer?: {
      trajectoryDrift: number;
      cascadingTokenConsumption: number;
      threadRecovery: number;
    };
  };
  timestamp: any;
}
