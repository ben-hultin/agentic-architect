"use client";

import Link from "next/link";
import { ArrowLeft, Play } from "lucide-react";
import { useState } from "react";
import { enqueueBenchmarkingJob } from "@/services/cloudTasks";

export default function BuildDetailsPage({ params }: Readonly<{ params: { id: string } }>) {
  const buildId = params.id;
  const [isRunning, setIsRunning] = useState(false);

  const handleRunTest = async () => {
    setIsRunning(true);
    try {
      await enqueueBenchmarkingJob({
        jobId: buildId,
        framework: "langgraph", // Example
        evalSetPath: "evalsets/default.json"
      });
      alert("Benchmarking job enqueued!");
    } catch (error) {
      console.error(error);
      alert("Failed to enqueue job.");
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div>
      <Link href="/builds" className="inline-flex items-center gap-1.5 text-[12.5px] text-text-dim mb-4 hover:text-text-body transition-colors">
        <ArrowLeft className="w-3.5 h-3.5 stroke-[2]" />
        Back to builds
      </Link>

      <div className="flex justify-between items-end mb-8 flex-wrap gap-4">
        <div>
          <h1 className="text-[26px] font-semibold text-text-hi font-heading tracking-tight">{buildId}</h1>
          <div className="flex items-center gap-2.5 mt-2 flex-wrap">
            <span className="inline-block text-[11.5px] font-medium text-text-hi bg-cyan/10 border border-cyan/30 py-1 px-2.5 rounded-md">
              Agentic Pattern: Orchestrator-worker
            </span>
            <div className="flex items-center gap-2 text-[13px] font-medium">
              <span className="w-[7px] h-[7px] rounded-full bg-cyan shadow-[0_0_6px_rgba(0,240,255,0.7)] animate-pulse" />
              Live Monitored
            </div>
          </div>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleRunTest}
            disabled={isRunning}
            className="flex items-center gap-2 bg-gradient-to-r from-cyan to-magenta text-[#08111A] text-[13.5px] font-semibold py-2.5 px-4.5 rounded-lg border-none disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {isRunning ? "Running..." : "Run Bench Test"}
          </button>
          <button className="bg-surface border border-border text-text-body text-[13.5px] font-medium py-2.5 px-4 rounded-lg hover:border-text-dim hover:text-text-hi transition-all">
            Configure Workflow
          </button>
        </div>
      </div>

      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        <div className="bg-surface border border-border rounded-xl p-[22px] px-6 relative before:content-[''] before:absolute before:top-0 before:left-0 before:right-0 before:h-[2px] before:bg-gradient-to-r before:from-cyan before:to-magenta before:opacity-70">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Task Success Rate</div>
            <div className="text-[12px] text-text-dim font-mono">Target &gt;90%</div>
          </div>
          <div className="font-mono text-[36px] font-semibold text-text-hi my-2 tracking-tighter">94%</div>
          <p className="text-[12.5px] text-text-body leading-relaxed">
            Measures if the agent successfully achieves its predefined workflow goals completely autonomous without requiring human fallback intervention.
          </p>
          <div className="mt-3.5">
            <div className="flex justify-between text-[12px] text-text-dim mb-1.5">
              <span>Optimal Threshold</span><span>94%</span>
            </div>
            <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden">
              <div className="h-full bg-gradient-to-r from-cyan to-magenta" style={{ width: "94%" }} />
            </div>
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Complexity of Build</div>
            <div className="text-[12px] text-text-dim font-mono">High Architectural Footprint</div>
          </div>
          <div className="font-mono text-[36px] font-semibold text-text-hi my-2 tracking-tighter">Tier 3</div>
          <p className="text-[12.5px] text-text-body leading-relaxed">
            Tracks the absolute structural payload of this system framework. Calculated directly via node dependencies, dynamic sub-agents, and routing depth.
          </p>
          <div className="flex flex-col gap-2.5 mt-3">
            <div className="flex justify-between items-center pb-2.5 border-b border-surface-2 text-[13px]">
              <span className="text-text-body">Sub-Agents Deployed</span>
              <span className="font-mono text-text-hi font-medium">5 Nodes</span>
            </div>
            <div className="flex justify-between items-center pb-2.5 border-b border-surface-2 text-[13px]">
              <span className="text-text-body">Tool Bindings</span>
              <span className="font-mono text-text-hi font-medium">12 Dependencies</span>
            </div>
          </div>
          <div className="flex gap-1 mt-3">
            <span className="flex-1 h-2 bg-gradient-to-r from-cyan to-magenta rounded-sm" />
            <span className="flex-1 h-2 bg-gradient-to-r from-cyan to-magenta rounded-sm" />
            <span className="flex-1 h-2 bg-gradient-to-r from-cyan to-magenta rounded-sm" />
            <span className="flex-1 h-2 bg-border rounded-sm" />
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Cost & Latency</div>
            <div className="text-[12px] text-text-dim font-mono">7d Aggregated Loop Spend</div>
          </div>
          <div className="font-mono text-[36px] font-semibold text-text-hi my-2 tracking-tighter">$412.35</div>
          <p className="text-[12.5px] text-text-body leading-relaxed">
            Because recursive agentic workflows utilize complex dynamic logic loops, token consumption averages 10x to 50x higher than basic static API prompts.
          </p>
          <div className="flex flex-col gap-2.5 mt-3">
            <div className="flex justify-between items-center pb-2.5 border-b border-surface-2 text-[13px]">
              <span className="text-text-body">Avg Exec Time</span>
              <span className="font-mono text-text-hi font-medium">4.2s / Run</span>
            </div>
            <div className="flex justify-between items-center pb-2.5 border-b border-surface-2 text-[13px]">
              <span className="text-text-body">Total LLM Calls</span>
              <span className="font-mono text-text-hi font-medium">14,240 calls</span>
            </div>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">
        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Quality & Reasoning Parameters</div>
            <div className="text-[12px] text-text-dim font-mono">LLM Context & Evaluators</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <MetricCard 
              title="Quality of Output" 
              body="Evaluates critical macro generation indicators across production cycles, validating granular data accuracy, strict schema compliance, contextual relevance, structural clarity, and rigid adherence to system guidelines."
              score="92 / 100"
              progress={92}
            />
            <MetricCard 
              title="Reasoning Coherence" 
              body="Audits internal system trajectory execution logic. Validates state graphs and path choices to prevent agents from executing counter-intuitive choices or fracturing into broken, unproductive thinking cycles."
              score="91 / 100"
              progress={91}
            />
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Size of Context Managed</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Flags systematic "context bloat." Tracks the trajectory data profile over multi-turn interactions, highlighting where historical responses recursively append to systemic payloads, blowing out costs.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Avg Working Window:</span>
                <span className="font-mono text-[14px] font-semibold text-magenta">18.4k tokens</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Resilience & Tool Infrastructure</div>
            <div className="text-[12px] text-text-dim font-mono">Runtime Diagnostics</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Error Rate & Resilience</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Determines infrastructure fault-tolerance and dynamic correction response when encountering unpredictable environment mutations, unexpected server downtime, rate limits, or corrupted logic payloads.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">System Resilience Rating:</span>
                <span className="font-mono text-[14px] font-semibold text-[#7de8a8]">High (2.1% Err)</span>
              </div>
            </div>
            <MetricCard 
              title="Trajectory Efficiency" 
              body="Analyzes step execution vectors. Tallies path optimization to map out how cleanly and linearly the engine evaluates tasks without looping on redundant tools or triggering circular operations."
              score="88% Eff"
              progress={88}
            />
            <MetricCard 
              title="Tool Call Accuracy" 
              body="Monitors the system API abstraction layer. Ensures strict schema verification and parameter typing, validating that generated commands strictly map to target environments without parameter hallucinations."
              score="96% Acc"
              progress={96}
            />
          </div>
        </div>
      </section>
    </div>
  );
}

const MetricCard = ({ title, body, score, progress }: { title: string; body: string; score: string; progress: number }) => (
  <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
    <div className="text-[13px] font-semibold text-text-hi">{title}</div>
    <p className="text-[12px] text-text-body leading-relaxed flex-grow">{body}</p>
    <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
      <div className="h-1.5 bg-border rounded-full overflow-hidden w-3/4">
        <div className="h-full bg-cyan" style={{ width: `${progress}%` }} />
      </div>
      <span className="font-mono text-[14px] font-semibold text-cyan">{score}</span>
    </div>
  </div>
);
