import { LogPanel, SystemInfoBar } from "@/components/DevPanel";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { cn } from "@/lib/utils";
import { AlertCircle } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";

const navItems = [
  { to: "/", label: "Home", end: true },
  { to: "/setup", label: "Setup" },
  { to: "/build", label: "Build" },
  { to: "/tests", label: "Tests" },
  { to: "/docs", label: "Pipeline docs" },
] as const;

export function AppShell() {
  const {
    config,
    checks,
    logs,
    running,
    lastResult,
    error,
    logEndRef,
    refresh,
    clearLogs,
  } = useDevPanelContext();

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <header className="border-b border-border px-6 py-4">
        <div className="mb-3 flex items-baseline gap-3">
          <h1 className="text-lg font-semibold tracking-tight">bobr dev panel</h1>
          <span className="text-xs text-muted-foreground">
            buttons → ./bobr · agents use CLI directly
          </span>
        </div>
        <SystemInfoBar config={config} checks={checks} onRefresh={refresh} />
        {error ? (
          <div className="mt-3 flex items-center gap-2 text-xs text-destructive">
            <AlertCircle className="size-3.5" />
            {error}
          </div>
        ) : null}
      </header>

      <div className="flex flex-1">
        <nav className="w-44 shrink-0 border-r border-border bg-muted/30 px-3 py-4">
          <ul className="space-y-1">
            {navItems.map(({ to, label, ...rest }) => (
              <li key={to}>
                <NavLink
                  to={to}
                  end={"end" in rest ? rest.end : false}
                  className={({ isActive }) =>
                    cn(
                      "block rounded-md px-3 py-2 text-sm transition-colors",
                      isActive
                        ? "bg-primary text-primary-foreground font-medium"
                        : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
                    )
                  }
                >
                  {label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <main className="grid flex-1 gap-6 p-6 lg:grid-cols-[1fr_420px]">
          <section className="min-h-0">
            <Outlet />
          </section>

          <aside className="min-h-0">
            <LogPanel
              logs={logs}
              running={running}
              lastExit={lastResult?.exitCode}
              onClear={clearLogs}
              logEndRef={logEndRef}
            />
          </aside>
        </main>
      </div>
    </div>
  );
}
