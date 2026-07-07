export type JobStatus = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED";

export interface Job {
  id: string;
  framework: string;
  pattern: string;
  status: JobStatus;
  createdAt: any;
  updatedAt: any;
  successRate?: number;
  errorRate?: number;
  cost?: number;
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
  };
  timestamp: any;
}
