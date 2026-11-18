import { PageHeader } from "@/components/ActionList";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { relativize } from "@/lib/format";
import { environment, fixFor, severity, sortChecks, summarize, type Severity } from "@/lib/health";
import { PAGES } from "@/lib/pages";
import type { DoctorCheck } from "@/types";
import { CircleAlert, CircleCheck, CircleX, Info } from "lucide-react";

export function OverviewPage() {
  const { checks, checksError } = useDevPanelContext();
  const page = PAGES[0];

  return (
    <div className="@container flex flex-col gap-6 px-8 py-7">
      <PageHeader title={page.label} description={page.description} />

      <div className="grid grid-cols-2 gap-3 @3xl:grid-cols-4">
        {checks
          ? environment(checks).map((tile) => (
              <div key={tile.label} className="flex flex-col gap-0.5 rounded-lg border px-4 py-3.5">
                <span className="text-xs text-muted-foreground">{tile.label}</span>
                <span className="truncate text-lg font-semibold tracking-tight">{tile.value}</span>
                <span className={tile.ok ? "truncate text-xs text-muted-foreground" : "truncate text-xs text-destructive"}>
                  {tile.sub}
                </span>
              </div>
            ))
          : Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[86px] rounded-lg" />)}
      </div>

      <section aria-labelledby="health-title" className="overflow-hidden rounded-lg border">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b px-4 py-3.5">
          <h2 id="health-title" className="text-[15px] font-semibold">
            Health
          </h2>
          {checks && !checksError ? <span className="text-sm text-muted-foreground">{summarize(checks).text}</span> : null}
          <span className="flex-1" />
          <code className="font-mono text-xs text-muted-foreground">./bobr doctor --json</code>
        </div>
        {checksError ? (
          <div role="alert" className="flex items-start gap-2 px-4 py-3 text-sm text-destructive">
            <CircleX className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <pre className="min-w-0 font-sans whitespace-pre-wrap">{checksError}</pre>
          </div>
        ) : checks ? (
          <ul className="divide-y">
            {sortChecks(checks).map((check) => (
              <HealthRow key={check.id} check={check} />
            ))}
          </ul>
        ) : (
          <div className="flex flex-col gap-3 px-4 py-3.5" aria-label="Running doctor">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} className="h-4" />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

const SEVERITY_ICON: Record<Severity, { icon: typeof CircleCheck; className: string; label: string }> = {
  fail: { icon: CircleX, className: "text-destructive", label: "failing" },
  warn: { icon: CircleAlert, className: "text-warning", label: "needs attention" },
  note: { icon: Info, className: "text-muted-foreground", label: "note" },
  ok: { icon: CircleCheck, className: "text-success", label: "passing" },
};

function HealthRow({ check }: { check: DoctorCheck }) {
  const { config, runningRun, requestRun } = useDevPanelContext();
  const { icon: Icon, className, label } = SEVERITY_ICON[severity(check)];
  const fix = fixFor(check);
  const message = relativize(check.message, config?.projectRoot);

  return (
    <li className="flex min-h-10 items-center gap-3 px-4 py-1.5">
      <Icon className={`size-4 shrink-0 ${className}`} role="img" aria-label={label} />
      <span className="w-28 shrink-0 font-mono text-xs text-foreground/80">{check.id}</span>
      <span className="min-w-0 flex-1 truncate text-[13px] text-muted-foreground" title={message}>
        {message}
      </span>
      {fix ? (
        <Button
          variant="outline"
          size="xs"
          disabled={runningRun !== null}
          onClick={() => requestRun(fix.actionId)}
        >
          {fix.label}
        </Button>
      ) : null}
    </li>
  );
}
