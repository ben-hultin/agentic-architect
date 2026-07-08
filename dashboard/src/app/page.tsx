"use client";

import { KPICard } from "@/components/KPICard";
import { TrendChart } from "@/components/TrendChart";
import { AttentionList } from "@/components/AttentionList";
import { BuildsTable } from "@/components/BuildsTable";
import { useJobs } from "@/dao/jobs/useJobs";
import { Job } from "@/types";

export default function Home() {
  const { jobs } = useJobs();

  // Mock data for initial display if loading or empty
  const mockJobs: Job[] = [
    { id: "Support-Triage-Orchestrator", topology_type: "SEQUENTIAL", runtime_target: "crewai", successRate: 94, errorRate: 2.1, cost: 412, status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Invoice-Extraction-ReAct", topology_type: "STATEFUL", runtime_target: "langgraph", successRate: 88, errorRate: 5.4, cost: 189, status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Sales-Research-MultiAgent", topology_type: "ADAPTIVE", runtime_target: "gemini-adk", successRate: 76, errorRate: 11.2, cost: 902, status: "FAILED" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Code-Review-Reflexion", topology_type: "STATEFUL", runtime_target: "langgraph", successRate: 91, errorRate: 3, cost: 267, status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Onboarding-Planner", topology_type: "OS_KERNEL", runtime_target: "microkernel-v1", successRate: 82, errorRate: 6.8, cost: 154, status: "RUNNING" as const, createdAt: new Date(), updatedAt: new Date() },
    { id: "Doc-QA-ToolChain", topology_type: "SEQUENTIAL", runtime_target: "crewai", successRate: 69, errorRate: 14.5, cost: 338, status: "FAILED" as const, createdAt: new Date(), updatedAt: new Date() },
  ];

  const displayJobs = jobs.length > 0 ? jobs : mockJobs;

  return (
    <div>
      <div className="flex justify-between items-end mb-8 flex-wrap gap-4">
        <div>
          <h1 className="text-[26px] font-semibold text-text-hi font-heading tracking-tight">Dashboard</h1>
          <div className="text-[13.5px] text-text-dim mt-1.5">Cross-build KPI overview · 24 builds monitored</div>
        </div>
        <div className="flex items-center gap-1.5 bg-surface border border-border text-text-body text-[13px] font-medium py-2 px-3.5 rounded-lg cursor-pointer">
          Last 7 days
          <svg className="w-[13px] h-[13px] stroke-text-dim fill-none stroke-[1.8]" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
        </div>
      </div>

      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KPICard label="Active builds" value="24" delta="3 this week" trend="up" />
        <KPICard label="Avg task success rate" value="84.6%" delta="2.1 pts" trend="up" />
        <KPICard label="Avg error rate" value="6.9%" delta="0.8 pts" trend="down" />
        <KPICard label="Spend, 7d" value="$6,284" delta="12% vs prior" trend="warn" />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
        <div className="lg:col-span-2">
          <TrendChart />
        </div>
        <div>
          <AttentionList />
        </div>
      </section>

      <BuildsTable jobs={displayJobs} />
    </div>
  );
}
