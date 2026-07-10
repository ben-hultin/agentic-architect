"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";

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

const TopologyMetrics = ({ topology_type }: { topology_type: string }) => {
  if (topology_type === "OS_KERNEL") {
    return (
      <>
        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Hardware & Memory Efficiency</div>
            <div className="text-[12px] text-text-dim font-mono">Microkernel Execution</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <MetricCard 
              title="Context Compression Ratio" 
              body="Measures memory reduction achieved when multiple agents share a base context window using PolyKV caching compared to raw memory expansion."
              score="3.2x savings"
              progress={82}
            />
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Wasm Tool Initialisation Speed</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Tracks cold start differences between running isolated tools in a WebAssembly sandbox vs. traditional Python subprocess executions.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Avg Init Latency:</span>
                <span className="font-mono text-[14px] font-semibold text-cyan">42ms</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Resource & Threat Isolation</div>
            <div className="text-[12px] text-text-dim font-mono">Syscall Layer</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Scheduler Preemption Latency</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Quantifies the overhead required to freeze a runaway thread, save its trajectory, and pass execution back to the scheduling queue.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Overhead Delay:</span>
                <span className="font-mono text-[14px] font-semibold text-[#7de8a8]">12.4ms</span>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  if (topology_type === "SEQUENTIAL") {
    return (
      <>
        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Linear Chain Performance</div>
            <div className="text-[12px] text-text-dim font-mono">Sequential Execution</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <MetricCard 
              title="Chain Recovery Efficiency" 
              body="Tracks behavior when an intermediate node receives an error. Measures if the framework can handle exceptions locally."
              score="94% Eff"
              progress={94}
            />
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">TTFT Degradation</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Measures how severely model latency degrades during the prefill parsing phase as context history scales forward.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Avg Increase:</span>
                <span className="font-mono text-[14px] font-semibold text-cyan">+120ms / step</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
          <div className="flex justify-between items-baseline mb-4">
            <div className="text-[14.5px] font-semibold text-text-hi font-heading">Context & Cost Scaling</div>
            <div className="text-[12px] text-text-dim font-mono">Telemetry Layer</div>
          </div>
          <div className="flex flex-col gap-5 mt-2.5">
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Cascading Token Velocity</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Tracks the exact token inflation curve across the linear chain as each task expands the prompt context of the next.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Inflation Rate:</span>
                <span className="font-mono text-[14px] font-semibold text-magenta">1.8x growth</span>
              </div>
            </div>
            <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
              <div className="text-[13px] font-semibold text-text-hi">Step-to-Output Ratio</div>
              <p className="text-[12px] text-text-body leading-relaxed flex-grow">
                Calculates how many intermediate reasoning tokens were consumed to generate the final useful task output.
              </p>
              <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
                <span className="text-[12px] text-text-dim">Efficiency:</span>
                <span className="font-mono text-[14px] font-semibold text-[#7de8a8]">0.82 ratio</span>
              </div>
            </div>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
        <div className="flex justify-between items-baseline mb-4">
          <div className="text-[14.5px] font-semibold text-text-hi font-heading">Framework Capabilities & Resiliency</div>
          <div className="text-[12px] text-text-dim font-mono">App-Layer Execution</div>
        </div>
        <div className="flex flex-col gap-5 mt-2.5">
          <MetricCard 
            title="Graph Trajectory Accuracy" 
            body="Measures how frequently the agent strays from defined state boundaries or gets stuck in cyclical tool-calling loops."
            score="92% Acc"
            progress={92}
          />
          <MetricCard 
            title="Thread Recovery Overhead" 
            body="Recovery latency required to log an error checkpoint to a database, instantiate a new turn, and resume execution after failure."
            score="88% Eff"
            progress={88}
          />
        </div>
      </div>

      <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
        <div className="flex justify-between items-baseline mb-4">
          <div className="text-[14.5px] font-semibold text-text-hi font-heading">Consumption & State Profile</div>
          <div className="text-[12px] text-text-dim font-mono">Graph Diagnostics</div>
        </div>
        <div className="flex flex-col gap-5 mt-2.5">
          <div className="bg-surface-2 border border-border rounded-lg p-4 flex flex-col gap-2">
            <div className="text-[13px] font-semibold text-text-hi">Cascading Token Consumption</div>
            <p className="text-[12px] text-text-body leading-relaxed flex-grow">
              Tracks how quickly costs compound when a single instruction triggers multi-tiered manager/worker delegation loops.
            </p>
            <div className="flex items-center justify-between mt-1 pt-2 border-t border-white/5">
              <span className="text-[12px] text-text-dim">Avg Multiplier:</span>
              <span className="font-mono text-[14px] font-semibold text-magenta">4.2x baseline</span>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default function BuildDetailsPage({ params }: Readonly<{ params: { id: string } }>) {
  const buildId = params.id;
  
  // Mock determination of topology_type based on buildId
  const isMicrokernel = buildId.includes("Planner");
  const topology_type = isMicrokernel ? "OS_KERNEL" : "SEQUENTIAL";
  const runtime_target = isMicrokernel ? "microkernel-v1-image" : "crewai";

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
              Topology: {topology_type}
            </span>
            <div className="flex items-center gap-2 text-[13px] font-medium">
              <span className="w-[7px] h-[7px] rounded-full bg-cyan shadow-[0_0_6px_rgba(0,240,255,0.7)] animate-pulse" />
              Live Monitored
            </div>
          </div>
        </div>
        <div className="flex gap-3">
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
        <TopologyMetrics topology_type={topology_type} />
      </section>
    </div>
  );
}
