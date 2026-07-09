import { Job } from "@/types";

export interface BuildsTableProps {
  jobs: Job[];
}

export const BuildsTable = ({ jobs }: BuildsTableProps) => {
  return (
    <section className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="p-[22px] px-6 pb-0">
        <div className="text-[14.5px] font-semibold text-text-hi font-heading">Builds overview</div>
        <div className="text-[12px] text-text-dim mt-1">Showing {jobs.length} builds</div>
      </div>
      
      <div className="mt-3.5 overflow-x-auto">
        <table className="w-full border-collapse">
          <thead>
            <tr>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Build</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Topology</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Task success</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Error rate</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Spend, 7d</th>
              <th className="text-left text-[11.5px] font-medium text-text-dim uppercase tracking-[0.4px] p-2.5 px-6 border-b border-border">Status</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id} className="hover:bg-white/[0.02] transition-colors">
                <td className="p-3.5 px-6 text-[13.5px] text-text-hi font-medium border-b border-border last:border-0">{job.id}</td>
                <td className="p-3.5 px-6 text-[13.5px] border-b border-border last:border-0">
                  <span className="inline-block text-[11px] text-text-body bg-surface-2 border border-border py-0.5 px-2 rounded-md">{job.topology_type}</span>
                </td>
                <td className="p-3.5 px-6 text-[13.5px] border-b border-border last:border-0">
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-1 rounded-[2px] bg-surface-2 overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-cyan to-magenta" style={{ width: `${job.successRate || 0}%` }} />
                    </div>
                    <span className="font-mono text-[12.5px] text-text-hi min-w-[38px]">{job.successRate || 0}%</span>
                  </div>
                </td>
                <td className="p-3.5 px-6 text-[13.5px] font-mono text-text-hi border-b border-border last:border-0">{job.errorRate || 0}%</td>
                <td className="p-3.5 px-6 text-[13.5px] font-mono text-text-hi border-b border-border last:border-0">${job.cost || 0}</td>
                <td className="p-3.5 px-6 text-[13.5px] border-b border-border last:border-0">
                  <div className="flex items-center gap-2">
                    <StatusDot status={job.status} />
                    {job.status === "RUNNING" 
                      ? (job.currentStep !== undefined ? `Active (Step ${job.currentStep + 1})` : "Active") 
                      : job.status === "FAILED" ? "Attention" : "Completed"}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
};

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

function clsx(...classes: any[]) {
  return classes.filter(Boolean).join(' ');
}
