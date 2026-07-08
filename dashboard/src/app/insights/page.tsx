"use client";

import { ArrowRight } from "lucide-react";
import { clsx } from "clsx";
import Link from "next/link";

export default function InsightsPage() {
  return (
    <div>
      <div className="mb-7">
        <h1 className="text-[26px] font-semibold text-text-hi font-heading tracking-tight">Insights</h1>
        <div className="text-[13.5px] text-text-dim mt-1.5">Compare builds and surface best practices across topologies and use cases</div>
      </div>

      {/* Build Comparison */}
      <section className="bg-surface border border-border rounded-xl p-6 px-[26px] mb-5">
        <div className="flex justify-between items-baseline mb-5 flex-wrap gap-2.5">
          <div className="text-[15px] font-semibold text-text-hi font-heading">Build comparison</div>
          <div className="text-[12px] text-text-dim">6 core metrics, side by side</div>
        </div>

        <div className="flex items-center gap-3.5 mb-6 flex-wrap">
          <div className="flex items-center gap-2.5 bg-surface-2 border border-border rounded-lg p-2 px-3.5 text-[13px]">
            <span className="w-2 h-2 rounded-[2px] bg-cyan flex-shrink-0" />
            <select className="bg-transparent border-none text-text-hi font-medium outline-none cursor-pointer">
              <option>Support-Triage-Orchestrator</option>
            </select>
          </div>
          <span className="text-[12px] text-text-dim font-mono">vs</span>
          <div className="flex items-center gap-2.5 bg-surface-2 border border-border rounded-lg p-2 px-3.5 text-[13px]">
            <span className="w-2 h-2 rounded-[2px] bg-magenta flex-shrink-0" />
            <select className="bg-transparent border-none text-text-hi font-medium outline-none cursor-pointer">
              <option>Doc-QA-ToolChain</option>
            </select>
          </div>
        </div>

        <div className="flex flex-col gap-[18px]">
          <CompareRow label="Task success rate" aValue="94%" aProgress={94} bValue="69%" bProgress={69} />
          <CompareRow label="Error rate" aValue="2.1%" aProgress={14} bValue="14.5%" bProgress={97} />
          <CompareRow label="Trajectory efficiency" aValue="88%" aProgress={88} bValue="61%" bProgress={61} />
          <CompareRow label="Tool call accuracy" aValue="96%" aProgress={96} bValue="74%" bProgress={74} />
          <CompareRow label="Reasoning coherence" aValue="91 / 100" aProgress={91} bValue="68 / 100" bProgress={68} />
          <CompareRow label="Cost per run" aValue="$0.42" aProgress={37} bValue="$1.15" bProgress={100} />
        </div>
      </section>

      {/* Topology Leaderboard */}
      <section className="bg-surface border border-border rounded-xl p-6 px-[26px] mb-5">
        <div className="flex justify-between items-baseline mb-5">
          <div className="text-[15px] font-semibold text-text-hi font-heading">Topology leaderboard</div>
          <div className="text-[12px] text-text-dim">Ranked by avg task success rate</div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">Topology</th>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">Builds</th>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">Avg success</th>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">Avg error</th>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">Avg cost / run</th>
                <th className="text-left text-[11px] font-medium text-text-dim uppercase tracking-[0.4px] pb-3 border-b border-border">7d trend</th>
              </tr>
            </thead>
            <tbody>
              <LeaderboardRow rank="01" name="STATEFUL" builds={8} success={91.2} error="3.4%" cost="$0.38" trend="▲ 1.8 pts" trendUp />
              <LeaderboardRow rank="02" name="SEQUENTIAL" builds={6} success={89.5} error="3.8%" cost="$0.31" trend="▲ 0.6 pts" trendUp />
              <LeaderboardRow rank="03" name="OS_KERNEL" builds={4} success={87} error="4.9%" cost="$0.19" trend="▲ 2.4 pts" trendUp />
              <LeaderboardRow rank="04" name="ADAPTIVE" builds={6} success={82.1} error="7.9%" cost="$0.27" trend="▼ 0.9 pts" />
            </tbody>
          </table>
        </div>
      </section>

      {/* Recommendations */}
      <section>
        <div className="flex justify-between items-baseline mb-4">
          <div className="text-[15px] font-semibold text-text-hi font-heading">Recommendations</div>
          <div className="text-[12px] text-text-dim">Generated from live build data</div>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <RecommendationCard 
            type="pos" 
            tag="Best practice" 
            title="OS_KERNEL context efficiency shines in high-turn chats"
            body="Across deep workflow builds, OS_KERNEL achieves a 3.2x context compression ratio versus ADAPTIVE. Default to it for heavy tool-usage scenarios with PolyKV."
          />
          <RecommendationCard 
            type="warn" 
            tag="Investigate" 
            title="SEQUENTIAL topologies show high cascading token consumption"
            body="Doc-QA-ToolChain runs an 14.5% error rate, triggering recursive delegation loops that blow out token budgets. Review explicit state boundaries and preemption."
          />
          <RecommendationCard 
            type="pos" 
            tag="Best practice" 
            title="STATEFUL routing drift is minimized in cyclical flows"
            body="Invoice-Extraction-ReAct reaches 96% success at $0.14 per run. Using Postgres checkpointers prevents trajectory drift during dynamic non-deterministic flows."
          />
        </div>
      </section>
    </div>
  );
}

const CompareRow = ({ label, aValue, aProgress, bValue, bProgress }: any) => (
  <div>
    <div className="flex justify-between text-[12.5px] text-text-body mb-2">
      <span className="text-text-hi font-medium">{label}</span>
    </div>
    <div className="flex flex-col gap-1.5">
      <div className="h-[7px] rounded-full bg-surface-2 overflow-hidden relative">
        <div className="h-full bg-cyan rounded-full" style={{ width: `${aProgress}%` }} />
      </div>
      <div className="font-mono text-[11.5px] text-text-dim flex justify-between">
        <span>Support-Triage-Orchestrator</span><span>{aValue}</span>
      </div>
      <div className="h-[7px] rounded-full bg-surface-2 overflow-hidden relative mt-1">
        <div className="h-full bg-magenta rounded-full" style={{ width: `${bProgress}%` }} />
      </div>
      <div className="font-mono text-[11.5px] text-text-dim flex justify-between">
        <span>Doc-QA-ToolChain</span><span>{bValue}</span>
      </div>
    </div>
  </div>
);

const LeaderboardRow = ({ rank, name, builds, success, error, cost, trend, trendUp }: any) => (
  <tr className="hover:bg-white/[0.02] transition-colors">
    <td className="py-3.5 border-b border-border">
      <div className="flex items-center gap-2.5">
        <span className="font-mono text-[12px] text-text-dim w-4">{rank}</span>
        <span className="text-text-hi font-medium">{name}</span>
      </div>
    </td>
    <td className="py-3.5 border-b border-border font-mono text-[12px] text-text-dim">{builds}</td>
    <td className="py-3.5 border-b border-border">
      <div className="flex items-center gap-2">
        <div className="w-[70px] h-[5px] rounded-full bg-surface-2 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-cyan to-magenta" style={{ width: `${success}%` }} />
        </div>
        <span className="font-mono text-[13px] text-text-hi">{success}%</span>
      </div>
    </td>
    <td className="py-3.5 border-b border-border font-mono text-[13px] text-text-hi">{error}</td>
    <td className="py-3.5 border-b border-border font-mono text-[13px] text-text-hi">{cost}</td>
    <td className={clsx("py-3.5 border-b border-border font-mono text-[12.5px]", trendUp ? "text-cyan" : "text-magenta")}>{trend}</td>
  </tr>
);

const RecommendationCard = ({ type, tag, title, body }: any) => (
  <div className={clsx(
    "bg-surface border border-border rounded-xl p-5 relative overflow-hidden before:content-[''] before:absolute before:top-0 before:left-0 before:bottom-0 before:w-[2px]",
    type === "pos" ? "before:bg-cyan" : "before:bg-magenta"
  )}>
    <span className={clsx(
      "inline-block text-[10.5px] font-semibold tracking-wider uppercase py-1 px-2 rounded-md mb-3",
      type === "pos" ? "text-cyan bg-cyan/10" : "text-magenta bg-magenta/10"
    )}>
      {tag}
    </span>
    <div className="text-[14.5px] font-semibold text-text-hi font-heading mb-2 leading-tight">{title}</div>
    <p className="text-[12.5px] text-text-body leading-relaxed mb-3.5">{body}</p>
    <Link href="/builds" className="text-[12px] text-text-hi font-medium inline-flex items-center gap-1.5 hover:underline">
      View builds <ArrowRight className="w-3 h-3 stroke-[2]" />
    </Link>
  </div>
);
