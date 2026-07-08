"use client";

import { useJobs } from "@/dao/jobs/useJobs";
import { Search, Plus, ChevronDown } from "lucide-react";
import { clsx } from "clsx";
import { Job } from "@/types";

export default function BuildsPage() {
  const { jobs } = useJobs();

  const mockJobs: Job[] = [
    { id: "Support-Triage-Orchestrator", useCase: "Customer support", topology_type: "SEQUENTIAL", runtime_target: "crewai", complexity: "High", successRate: 94, errorRate: 2.1, cost: 412, lastRun: "4m ago", status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Invoice-Extraction-ReAct", useCase: "Finance ops", topology_type: "STATEFUL", runtime_target: "langgraph", complexity: "Low", successRate: 88, errorRate: 5.4, cost: 189, lastRun: "11m ago", status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Sales-Research-MultiAgent", useCase: "Sales research", topology_type: "ADAPTIVE", runtime_target: "gemini-adk", complexity: "High", successRate: 76, errorRate: 11.2, cost: 902, lastRun: "2m ago", status: "FAILED" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Code-Review-Reflexion", useCase: "Engineering", topology_type: "STATEFUL", runtime_target: "langgraph", complexity: "Medium", successRate: 91, errorRate: 3, cost: 267, lastRun: "18m ago", status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Onboarding-Planner", useCase: "HR & onboarding", topology_type: "OS_KERNEL", runtime_target: "microkernel-v1", complexity: "Medium", successRate: 82, errorRate: 6.8, cost: 154, lastRun: "32m ago", status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Doc-QA-ToolChain", useCase: "Document QA", topology_type: "SEQUENTIAL", runtime_target: "crewai", complexity: "Medium", successRate: 69, errorRate: 14.5, cost: 338, lastRun: "6m ago", status: "FAILED" as const, createdAt: new Date(), updatedAt: new Date() },
  ];

  const displayJobs = jobs.length > 0 ? jobs : mockJobs;

  return (
    <div>
      <div className="flex justify-between items-end mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-[26px] font-semibold text-text-hi font-heading tracking-tight">Builds</h1>
          <div className="text-[13.5px] text-text-dim mt-1.5">24 builds across 4 control topologies</div>
        </div>
        <button className="flex items-center gap-2 bg-gradient-to-r from-cyan to-magenta text-[#08111A] text-[13.5px] font-semibold py-2.5 px-4.5 rounded-lg border-none">
          <Plus className="w-3.5 h-3.5 stroke-[2.2]" />
          New build
        </button>
      </div>

      <div className="flex items-center gap-3 mb-5 flex-wrap">
        <div className="flex-1 min-w-[220px] flex items-center gap-2.5 bg-surface border border-border rounded-lg p-2.5 px-3.5">
          <Search className="w-3.5 h-3.5 text-text-dim stroke-[1.8]" />
          <input 
            type="text" 
            placeholder="Search builds by name or use case" 
            className="bg-transparent border-none outline-none text-text-hi text-[13.5px] w-full placeholder:text-text-dim"
          />
        </div>
        <div className="flex items-center gap-2 bg-surface border border-border text-text-body text-[13px] font-medium p-2.5 px-3.5 rounded-lg cursor-pointer">
          Topology: All
          <ChevronDown className="w-3 h-3 text-text-dim stroke-[1.8]" />
        </div>
        <div className="flex items-center gap-2 bg-surface border border-border text-text-body text-[13px] font-medium p-2.5 px-3.5 rounded-lg cursor-pointer">
          Use case: All
          <ChevronDown className="w-3 h-3 text-text-dim stroke-[1.8]" />
        </div>
        <div className="flex bg-surface border border-border rounded-lg p-0.5 gap-0.5">
          <button className="bg-surface-2 text-text-hi text-[12.5px] font-medium py-1.5 px-3 rounded-md">All</button>
          <button className="text-text-dim text-[12.5px] font-medium py-1.5 px-3 rounded-md">Active</button>
          <button className="text-text-dim text-[12.5px] font-medium py-1.5 px-3 rounded-md">Attention</button>
        </div>
      </div>

      <section className="bg-surface border border-border rounded-xl overflow-hidden">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Build</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Topology</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Complexity</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Task success</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Error rate</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Spend, 7d</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Last run</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-3.5 px-5 border-b border-border whitespace-nowrap">Status</th>
            </tr>
          </thead>
          <tbody>
            {displayJobs.map((job: any) => (
              <tr key={job.id} className="hover:bg-white/[0.02] transition-colors cursor-pointer">
                <td className="p-4 px-5 border-b border-border last:border-0">
                  <div className="flex flex-col gap-1">
                    <span className="text-[13.5px] text-text-hi font-medium">{job.id}</span>
                    <span className="text-[11.5px] text-text-dim">{job.useCase || "N/A"}</span>
                  </div>
                </td>
                <td className="p-4 px-5 border-b border-border last:border-0">
                  <span className="inline-block text-[11px] text-text-body bg-surface-2 border border-border py-0.5 px-2 rounded-md whitespace-nowrap">{job.topology_type}</span>
                </td>
                <td className="p-4 px-5 border-b border-border last:border-0">
                  <div className="flex items-center gap-1.5 text-[11.5px] text-text-body">
                    <ComplexityBars complexity={job.complexity} />
                    {job.complexity || "Medium"}
                  </div>
                </td>
                <td className="p-4 px-5 border-b border-border last:border-0">
                  <div className="flex items-center gap-2 whitespace-nowrap">
                    <div className="w-14 h-1 rounded-[2px] bg-surface-2 overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-cyan to-magenta" style={{ width: `${job.successRate || 0}%` }} />
                    </div>
                    <span className="font-mono text-[12.5px] text-text-hi min-w-[34px]">{job.successRate || 0}%</span>
                  </div>
                </td>
                <td className="p-4 px-5 text-[13px] font-mono text-text-hi border-b border-border last:border-0">{job.errorRate || 0}%</td>
                <td className="p-4 px-5 text-[13px] font-mono text-text-hi border-b border-border last:border-0">${job.cost || 0}</td>
                <td className="p-4 px-5 text-[12px] font-mono text-text-dim border-b border-border last:border-0 whitespace-nowrap">{job.lastRun || "N/A"}</td>
                <td className="p-4 px-5 border-b border-border last:border-0">
                  <div className="flex items-center gap-2 whitespace-nowrap">
                    <StatusDot status={job.status} />
                    {job.status === "RUNNING" ? "Active" : job.status === "FAILED" ? "Attention" : "Paused"}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="flex justify-between items-center p-4 px-5 text-[12.5px] text-text-dim">
          <span>Showing {displayJobs.length} of 24 builds</span>
          <div className="flex gap-1.5">
            <button className="bg-surface-2 border border-border text-text-hi border-cyan text-[12px] py-1.5 px-2.5 rounded-md">1</button>
            <button className="bg-surface-2 border border-border text-text-body text-[12px] py-1.5 px-2.5 rounded-md">2</button>
            <button className="bg-surface-2 border border-border text-text-body text-[12px] py-1.5 px-2.5 rounded-md">3</button>
            <button className="bg-surface-2 border border-border text-text-body text-[12px] py-1.5 px-2.5 rounded-md">Next</button>
          </div>
        </div>
      </section>
    </div>
  );
}

const StatusDot = ({ status }: { status: string }) => {
  const statusClasses = {
    RUNNING: "bg-cyan shadow-[0_0_6px_rgba(0,240,255,0.7)] animate-pulse",
    FAILED: "bg-magenta shadow-[0_0_6px_rgba(189,0,255,0.7)]",
    PAUSED: "bg-text-dim",
    COMPLETED: "bg-text-dim",
  };

  return (
    <span className={clsx(
      "w-[7px] h-[7px] rounded-full",
      statusClasses[status as keyof typeof statusClasses] || statusClasses.COMPLETED
    )} />
  );
};

const ComplexityBars = ({ complexity }: { complexity: string }) => {
  const isLow = complexity === "Low" || complexity === "Medium" || complexity === "High";
  const isMedium = complexity === "Medium" || complexity === "High";
  const isHigh = complexity === "High";

  return (
    <div className="flex gap-0.5 items-end">
      <span className={clsx("w-[3px] rounded-[1px] h-[6px]", isLow ? "bg-text-body" : "bg-border")} />
      <span className={clsx("w-[3px] rounded-[1px] h-[9px]", isMedium ? "bg-text-body" : "bg-border")} />
      <span className={clsx("w-[3px] rounded-[1px] h-[12px]", isHigh ? "bg-text-body" : "bg-border")} />
    </div>
  );
};
