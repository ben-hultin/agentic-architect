import { TopologyType, MemoryConfig } from "@/types";

/**
 * Service for interacting with Google Cloud Tasks.
 * This will be used to enqueue benchmarking jobs.
 */

export interface TaskPayload {
  jobId: string;
  topology_type: TopologyType;
  runtime_target: string;
  memory_config?: MemoryConfig;
  fault_profile?: string;
  evalSetPath: string;
}

export const enqueueBenchmarkingJob = async (payload: TaskPayload) => {
  // In a production environment, this would call a Next.js API route 
  // which then uses the @google-cloud/tasks SDK to enqueue the task.
  // For now, we'll implement the client-side trigger.
  
  const response = await fetch("/api/tasks/enqueue", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error("Failed to enqueue benchmarking job");
  }

  return response.json();
};
