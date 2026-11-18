import type { RunStatus } from "@/hooks/useDevPanel";
import { cn } from "@/lib/utils";
import { CircleCheck, CircleSlash, CircleX, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

const LABELS: Record<RunStatus, string> = {
  running: "running",
  passed: "passed",
  failed: "failed",
  stopped: "stopped",
};

// tone="dark" for the terminal surface, where the light-theme status colors are too dim
export function RunStatusIcon({
  status,
  tone = "light",
  className,
}: {
  status: RunStatus;
  tone?: "light" | "dark";
  className?: string;
}) {
  const props = { role: "img", "aria-label": LABELS[status], className: cn("size-3.5 shrink-0", className) };
  const dark = tone === "dark";
  switch (status) {
    case "running":
      return <Loader2 {...props} className={cn(props.className, "animate-spin")} />;
    case "passed":
      return <CircleCheck {...props} className={cn(props.className, dark ? "text-green-400" : "text-success")} />;
    case "failed":
      return <CircleX {...props} className={cn(props.className, dark ? "text-red-400" : "text-destructive")} />;
    case "stopped":
      return <CircleSlash {...props} className={cn(props.className, dark ? "text-zinc-400" : "text-muted-foreground")} />;
  }
}

// Current time, re-read every second while `active`; drives elapsed-time displays
export function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}
