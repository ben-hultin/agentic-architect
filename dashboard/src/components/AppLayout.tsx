"use client";

import { useAuth } from "@/dao/auth/useAuth";
import { Sidebar } from "@/components/Sidebar";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

export const AppLayout = ({ children }: { children: React.ReactNode }) => {
  const { user, loading } = useAuth();
  const pathname = usePathname();
  const router = useRouter();

  const isLoginPage = pathname === "/login";

  useEffect(() => {
    if (!loading && !user && !isLoginPage) {
      router.push("/login");
    }
  }, [user, loading, isLoginPage, router]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-screen bg-bg">
        <div className="w-8 h-8 rounded-full border-2 border-cyan border-t-transparent animate-spin" />
      </div>
    );
  }

  // Allow rendering the login page without the sidebar
  if (isLoginPage) {
    return <main className="flex-1 flex flex-col min-w-0 min-h-screen">{children}</main>;
  }

  // If not authenticated and not on login page, don't render anything while redirecting
  if (!user) {
    return null;
  }

  return (
    <>
      <Sidebar />
      <main className="flex-1 p-9 pb-15 min-w-0 overflow-y-auto min-h-screen">
        {children}
      </main>
    </>
  );
};
