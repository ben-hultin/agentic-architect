"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, Box, BarChart3, Settings } from "lucide-react";
import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const Sidebar = () => {
  const pathname = usePathname();

  const navItems = [
    { label: "Dashboard", href: "/", icon: LayoutDashboard },
    { label: "Builds", href: "/builds", icon: Box },
    { label: "Insights", href: "/insights", icon: BarChart3 },
    { label: "Settings", href: "/settings", icon: Settings },
  ];

  return (
    <aside className="w-[232px] flex-shrink-0 bg-surface border-right border-border flex flex-col p-7 px-5 sticky top-0 h-screen">
      <div className="flex items-center gap-2.5 font-heading font-semibold text-base text-text-hi mb-10 tracking-tight">
        <div className="w-[26px] h-[26px] rounded-[7px] bg-gradient-to-br from-cyan to-magenta flex-shrink-0" />
        Agent Ops
      </div>
      
      <nav className="flex flex-col gap-[2px]">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;
          
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 p-2.5 px-3 rounded-lg text-sm font-medium transition-colors border-l-2 border-transparent",
                isActive 
                  ? "text-text-hi bg-cyan/5 border-cyan" 
                  : "text-text-body hover:bg-white/5 hover:text-text-hi"
              )}
            >
              <Icon className="w-4 h-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto pt-4 border-t border-border text-[12px] color-text-dim flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-cyan shadow-[0_0_6px_var(--cyan)]" />
        gcp-agent-ops-prod
      </div>
    </aside>
  );
};
