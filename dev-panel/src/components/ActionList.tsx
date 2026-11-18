import { RunStatusIcon, useNow } from "@/components/RunStatusIcon";
import { Button } from "@/components/ui/button";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { commandFor } from "@/hooks/useDevPanel";
import { formatClock, formatDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { Action, ActionGroup } from "@/types";
import { Play, Square } from "lucide-react";

export function PageHeader({ title, description }: { title: string; description: string }) {
  return (
    <div className="flex flex-col gap-1">
      <h1 className="text-[22px] font-semibold tracking-tight">{title}</h1>
      <p className="text-muted-foreground">{description}</p>
    </div>
  );
}

export function ActionList({ groups }: { groups: ActionGroup[] }) {
  // A page with one group already says what the group is in its own header
  const headings = groups.length > 1;
  return (
    <div className="flex flex-col gap-7">
      {groups.map((group) => (
        <section
          key={group.id}
          aria-labelledby={headings ? `group-${group.id}` : undefined}
          aria-label={headings ? undefined : group.label}
          className="flex flex-col gap-2.5"
        >
          {headings ? (
            <div className="flex flex-wrap items-baseline gap-x-2.5">
              <h2 id={`group-${group.id}`} className="text-[15px] font-semibold">
                {group.label}
              </h2>
              {group.description ? <p className="text-sm text-muted-foreground">{group.description}</p> : null}
            </div>
          ) : null}
          <div className="divide-y overflow-hidden rounded-lg border">
            {group.actions.map((action) => (
              <ActionRow key={action.id} action={action} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function ActionRow({ action }: { action: Action }) {
  const { runningRun, lastResults, requestRun, stopAction, selectRun } = useDevPanelContext();
  const isRunning = runningRun?.actionId === action.id;
  const busy = runningRun !== null;
  const last = lastResults[action.id];
  const now = useNow(isRunning);
  const destructive = action.variant === "destructive";

  return (
    <div
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_auto_auto] items-center gap-x-4 px-4 py-3",
        isRunning && "bg-muted/50",
      )}
    >
      <div className="flex min-w-0 flex-col items-start gap-1">
        <span className={cn("font-medium", destructive && "text-destructive")}>{action.label}</span>
        {action.description ? <span className="text-sm text-muted-foreground">{action.description}</span> : null}
        <code className="max-w-full truncate rounded-md bg-muted px-2 py-0.5 font-mono text-xs text-foreground/80">
          {commandFor(action)}
        </code>
      </div>
      <div className="flex min-w-16 items-center gap-1.5 text-xs whitespace-nowrap text-muted-foreground">
        {isRunning ? (
          <button
            type="button"
            className="flex items-center gap-1.5 text-foreground hover:underline"
            onClick={() => selectRun(runningRun.runId)}
            title="Show output"
          >
            <RunStatusIcon status="running" />
            {formatClock(now - runningRun.startedAt)}
          </button>
        ) : last ? (
          <>
            <RunStatusIcon status={last.status} />
            {last.status === "stopped" ? "Stopped" : formatDuration(last.durationMs)}
          </>
        ) : (
          "Never run"
        )}
      </div>
      {isRunning ? (
        <Button variant="destructive" size="sm" onClick={() => void stopAction()}>
          <Square className="fill-current" aria-hidden="true" />
          Stop
        </Button>
      ) : (
        <Button
          variant="outline"
          size="sm"
          disabled={busy}
          onClick={() => requestRun(action.id)}
          title={busy ? `Wait for ${runningRun?.label} to finish` : undefined}
          className={cn(destructive && "text-destructive hover:text-destructive")}
        >
          <Play className="fill-current" aria-hidden="true" />
          Run
        </Button>
      )}
    </div>
  );
}
