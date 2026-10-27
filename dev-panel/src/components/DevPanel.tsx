import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { Action, ActionGroup } from "@/types";
import { Loader2, Play, RefreshCw } from "lucide-react";

type Props = {
  groups: ActionGroup[];
  running: boolean;
  onRun: (actionId: string) => void;
};

function ActionButton({
  action,
  disabled,
  onRun,
}: {
  action: Action;
  disabled: boolean;
  onRun: (id: string) => void;
}) {
  const destructive = action.variant === "destructive";

  return (
    <Button
      variant={destructive ? "destructive" : "secondary"}
      size="sm"
      className="justify-start gap-2"
      disabled={disabled}
      onClick={() => onRun(action.id)}
      title={action.description || action.label}
    >
      {disabled ? (
        <Loader2 className="size-3.5 animate-spin" />
      ) : (
        <Play className="size-3.5 opacity-60" />
      )}
      <span className="truncate">{action.label}</span>
    </Button>
  );
}

export function ActionGrid({ groups, running, onRun }: Props) {
  return (
    <div className="space-y-4">
      {groups.map((group) => (
        <Card key={group.id} className="border-border bg-card">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <CardTitle className="text-sm font-medium">{group.label}</CardTitle>
              <Badge variant="outline" className="text-[10px]">
                {group.actions.length}
              </Badge>
            </div>
            {group.description ? (
              <CardDescription className="text-xs">{group.description}</CardDescription>
            ) : null}
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {group.actions.map((action) => (
                <ActionButton
                  key={action.id}
                  action={action}
                  disabled={running}
                  onRun={onRun}
                />
              ))}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export function LogPanel({
  logs,
  running,
  lastExit,
  onClear,
  logEndRef,
}: {
  logs: { stream: string; line: string }[];
  running: boolean;
  lastExit?: number;
  onClear: () => void;
  logEndRef: React.RefObject<HTMLDivElement | null>;
}) {
  return (
    <Card className="flex h-full min-h-0 flex-col border-border bg-card">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
        <div>
          <CardTitle className="text-sm font-medium">Output</CardTitle>
          <CardDescription className="text-xs">
            {running
              ? "Running…"
              : lastExit !== undefined
                ? `Last exit code: ${lastExit}`
                : "Ready"}
          </CardDescription>
        </div>
        <Button variant="ghost" size="sm" onClick={onClear} disabled={running}>
          Clear
        </Button>
      </CardHeader>
      <Separator />
      <CardContent className="min-h-0 flex-1 p-0">
        <ScrollArea className="h-[420px] w-full">
          <pre className="p-4 font-mono text-xs leading-relaxed">
            {logs.length === 0 ? (
              <span className="text-muted-foreground">
                Run an action to see output here.
              </span>
            ) : (
              logs.map((entry, i) => (
                <div
                  key={`${i}-${entry.line}`}
                  className={
                    entry.stream === "stderr"
                      ? "text-destructive"
                      : entry.stream === "meta"
                        ? "text-primary"
                        : "text-foreground/90"
                  }
                >
                  {entry.line}
                </div>
              ))
            )}
            <div ref={logEndRef} />
          </pre>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

export function SystemInfoBar({
  config,
  checks,
  onRefresh,
}: {
  config: { projectRoot: string; gpuArch: string } | null;
  checks: { id: string; level: string; message: string; ok: boolean }[];
  onRefresh: () => void;
}) {
  const failed = checks.filter((c) => !c.ok && c.level === "fail").length;
  const cuda = checks.find((c) => c.id === "nvcc");
  const gpu = checks.find((c) => c.id === "gpu");
  const bobr = checks.find((c) => c.id === "bobr_import");

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
      <Badge variant={failed === 0 ? "default" : "destructive"}>
        {failed === 0 ? "doctor OK" : `${failed} failed`}
      </Badge>
      {cuda ? (
        <Badge variant={cuda.ok ? "outline" : "destructive"} title={cuda.message}>
          CUDA
        </Badge>
      ) : null}
      {gpu?.message ? <Badge variant="outline">{gpu.message}</Badge> : null}
      {bobr ? (
        <Badge variant={bobr.ok ? "outline" : "destructive"}>
          {bobr.ok ? "import bobr" : "no bobr"}
        </Badge>
      ) : null}
      <span className="hidden max-w-md truncate md:inline" title={config?.projectRoot}>
        {config?.projectRoot}
      </span>
      <Button variant="ghost" size="icon" className="ml-auto size-7" onClick={onRefresh}>
        <RefreshCw className="size-3.5" />
      </Button>
    </div>
  );
}
