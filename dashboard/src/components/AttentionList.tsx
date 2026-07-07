export const AttentionList = () => {
  const items = [
    { name: "Doc-QA-ToolChain", pattern: "Tool-use chain", value: "14.5%" },
    { name: "Sales-Research-MultiAgent", pattern: "Multi-agent debate", value: "11.2%" },
    { name: "Refund-Approval-Planner", pattern: "Planner-executor", value: "10.4%" },
  ];

  return (
    <div className="bg-surface border border-border rounded-xl p-[22px] px-6">
      <div className="flex justify-between items-baseline mb-[18px]">
        <div className="text-[14.5px] font-semibold text-text-hi font-heading">Needs attention</div>
        <div className="text-[12px] text-text-dim">Error rate &gt; 10%</div>
      </div>

      <div className="flex flex-col">
        {items.map((item, idx) => (
          <div key={idx} className="flex items-center justify-between py-[11px] border-b border-border last:border-0">
            <div className="flex items-center gap-2.5">
              <span className="w-1.5 h-1.5 rounded-full bg-magenta shadow-[0_0_6px_rgba(189,0,255,0.6)] flex-shrink-0" />
              <div>
                <div className="text-[13px] text-text-hi font-medium">{item.name}</div>
                <div className="text-[11.5px] text-text-dim mt-[3px]">{item.pattern}</div>
              </div>
            </div>
            <div className="font-mono text-[13px] text-magenta font-semibold">{item.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};
