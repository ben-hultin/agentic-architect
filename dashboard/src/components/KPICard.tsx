import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export interface KPICardProps {
  label: string;
  value: string | number;
  delta: string;
  trend: "up" | "down" | "warn";
}

export const KPICard = ({ label, value, delta, trend }: KPICardProps) => {
  return (
    <div className="bg-surface border border-border rounded-xl p-5 pt-5 pb-4.5 relative overflow-hidden before:content-[''] before:absolute before:top-0 before:left-0 before:right-0 before:h-[2px] before:bg-gradient-to-r before:from-cyan before:to-magenta before:opacity-70">
      <div className="text-[12.5px] font-medium text-text-dim">{label}</div>
      <div className="font-mono text-[28px] font-semibold text-text-hi mt-2.5 tracking-tight">{value}</div>
      <div className={cn(
        "flex items-center gap-1 text-xs font-medium mt-2.5 font-mono",
        trend === "up" && "text-cyan",
        trend === "down" && "text-[#7DE8A8]",
        trend === "warn" && "text-magenta"
      )}>
        {trend === "up" ? "▲" : trend === "down" ? "▼" : "▲"} {delta}
      </div>
    </div>
  );
};
