import { RunStatusIcon, useNow } from "@/components/RunStatusIcon";
import { Kbd } from "@/components/ui/kbd";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useDevPanelContext } from "@/context/DevPanelContext";
import type { Run } from "@/hooks/useDevPanel";
import { formatClock, formatDuration } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { LogLine } from "@/types";
import { ArrowDown, Check, Copy, Square, Trash2, WrapText } from "lucide-react";
import { useLayoutEffect, useRef, useState, type ComponentProps } from "react";

const WRAP_KEY = "bobr-panel:log-wrap";

function readWrapPref() {
  try {
    return localStorage.getItem(WRAP_KEY) !== "false";
  } catch {
    return true;
  }
}

// Small icon/text button for the dark terminal surface
function TermButton({ className, active, ...props }: ComponentProps<"button"> & { active?: boolean }) {
  return (
    <button
      type="button"
      className={cn(
        "inline-flex h-7 shrink-0 items-center justify-center gap-1.5 rounded-md px-2 text-xs text-zinc-400 transition-colors outline-none",
        "hover:bg-zinc-800 hover:text-zinc-100 focus-visible:ring-2 focus-visible:ring-zinc-500 disabled:pointer-events-none disabled:opacity-40",
        "[&_svg]:size-3.5",
        active && "bg-zinc-800 text-zinc-100",
        className,
      )}
      {...props}
    />
  );
}

export function OutputPanel() {
  const { runs, selectedRun, selectRun, stopAction, clearFinished } = useDevPanelContext();
  const [wrap, setWrap] = useState(readWrapPref);
  const [copied, setCopied] = useState(false);

  const toggleWrap = () => {
    setWrap((prev) => {
      try {
        localStorage.setItem(WRAP_KEY, String(!prev));
      } catch {
        // preference just won't persist
      }
      return !prev;
    });
  };

  const copy = async () => {
    if (!selectedRun) return;
    try {
      await navigator.clipboard.writeText(selectedRun.lines.map((l) => l.line).join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard unavailable (insecure context); nothing sensible to fall back to
    }
  };

  return (
    <section
      aria-label="Output"
      className="flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950 text-zinc-200 [color-scheme:dark]"
    >
      <div className="flex flex-wrap items-center gap-1 border-b border-zinc-800 p-2">
        <div role="tablist" aria-label="Runs" className="flex min-w-0 flex-1 flex-wrap gap-1">
          {runs.length === 0 ? <span className="px-2 text-xs font-medium text-zinc-400">Output</span> : null}
          {runs.map((run) => (
            <TermButton
              key={run.runId}
              role="tab"
              aria-selected={run.runId === selectedRun?.runId}
              active={run.runId === selectedRun?.runId}
              className="px-2.5 font-mono"
              onClick={() => selectRun(run.runId)}
              title={run.command}
            >
              <RunStatusIcon status={run.status} tone="dark" />
              {run.actionId}
            </TermButton>
          ))}
        </div>
        {selectedRun?.status === "running" ? (
          <TermButton className="bg-red-950 text-red-200 hover:bg-red-900 hover:text-red-100" onClick={() => void stopAction()}>
            <Square className="fill-current" aria-hidden="true" />
            Stop
          </TermButton>
        ) : null}
        <TermButton aria-label="Wrap long lines" aria-pressed={wrap} active={wrap} onClick={toggleWrap} title="Wrap long lines">
          <WrapText aria-hidden="true" />
        </TermButton>
        <TermButton aria-label="Copy output" onClick={() => void copy()} disabled={!selectedRun} title="Copy output">
          {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
        </TermButton>
        <TermButton
          aria-label="Clear finished runs"
          onClick={clearFinished}
          disabled={!runs.some((r) => r.status !== "running")}
          title="Clear finished runs"
        >
          <Trash2 aria-hidden="true" />
        </TermButton>
      </div>

      {selectedRun ? (
        // Keyed so each run gets its own scroll position and follow state
        <LogView key={selectedRun.runId} run={selectedRun} wrap={wrap} />
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-6 text-center text-sm text-zinc-400">
          <p>Run an action to see its output here.</p>
          <p className="flex items-center gap-1.5 text-xs">
            Find one with <Kbd className="bg-zinc-800 text-zinc-200">Ctrl K</Kbd>
          </p>
        </div>
      )}
    </section>
  );
}

function LogView({ run, wrap }: { run: Run; wrap: boolean }) {
  const viewportRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  // Following = keep the newest line in view. Any upward scroll by the user turns it off;
  // scrolling back to the bottom (or the jump button) turns it on again.
  const followingRef = useRef(true);
  const [following, setFollowingState] = useState(true);
  // Line count when following stopped, for the "N new lines" button
  const [pausedAt, setPausedAt] = useState(0);
  // Where we last scrolled to ourselves, to tell our scrolls apart from the user's
  const autoScrollTop = useRef(0);
  const linesRef = useRef(run.lines.length);
  linesRef.current = run.lines.length;
  const now = useNow(run.status === "running");

  const setFollowing = (value: boolean) => {
    if (followingRef.current === value) return;
    followingRef.current = value;
    setFollowingState(value);
    if (!value) setPausedAt(linesRef.current);
  };

  const scrollToEnd = () => {
    const viewport = viewportRef.current;
    if (!viewport) return;
    viewport.scrollTop = viewport.scrollHeight;
    autoScrollTop.current = viewport.scrollTop;
  };

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const content = contentRef.current;
    if (!viewport || !content) return;

    // Wheel and keys stop following immediately: Chrome animates wheel scrolling over several
    // frames, and waiting for the position to move would let new output snap it back down
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0) setFollowing(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowUp" || e.key === "PageUp" || e.key === "Home") setFollowing(false);
    };
    // Catches everything else (scrollbar drag, touch) and re-enables following at the bottom
    const onScroll = () => {
      const fromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
      if (fromBottom <= 4) setFollowing(true);
      else if (viewport.scrollTop < autoScrollTop.current - 1) setFollowing(false);
    };
    // New lines, re-wrapping and panel resizes all change these sizes
    const observer = new ResizeObserver(() => {
      if (followingRef.current) scrollToEnd();
    });

    scrollToEnd();
    viewport.addEventListener("wheel", onWheel, { passive: true });
    viewport.addEventListener("keydown", onKeyDown);
    viewport.addEventListener("scroll", onScroll, { passive: true });
    observer.observe(content);
    observer.observe(viewport);
    return () => {
      viewport.removeEventListener("wheel", onWheel);
      viewport.removeEventListener("keydown", onKeyDown);
      viewport.removeEventListener("scroll", onScroll);
      observer.disconnect();
    };
  }, []);

  const jumpToLatest = () => {
    setFollowing(true);
    scrollToEnd();
  };

  const newLines = run.lines.length - pausedAt;
  const elapsed = run.durationMs ?? now - run.startedAt;

  return (
    <>
      <div className="relative min-h-0 flex-1">
        <ScrollArea
          className="h-full"
          horizontal={!wrap}
          thumbClassName="bg-zinc-700"
          viewportRef={viewportRef}
          viewportProps={{
            tabIndex: 0,
            "aria-label": `Output of ${run.command}`,
            className: "py-3 font-mono text-[12.5px] leading-relaxed focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-zinc-600",
          }}
        >
          <div ref={contentRef} className={cn(!wrap && "w-max min-w-full")}>
            {run.lines.length === 0 ? <p className="px-4 text-zinc-400">Waiting for output…</p> : null}
            {run.lines.map((entry, i) => (
              <div key={i} className="flex gap-3.5 px-3.5">
                <span aria-hidden="true" className="w-8 shrink-0 text-right text-zinc-500 select-none">
                  {i + 1}
                </span>
                <LineText entry={entry} wrap={wrap} />
              </div>
            ))}
          </div>
        </ScrollArea>
        {!following && run.lines.length > 0 ? (
          <button
            type="button"
            onClick={jumpToLatest}
            className="absolute bottom-4 left-1/2 inline-flex h-8 -translate-x-1/2 items-center gap-1.5 rounded-full border border-zinc-700 bg-zinc-50 px-3.5 text-[13px] font-medium text-zinc-950 shadow-lg shadow-black/40 hover:bg-white focus-visible:ring-2 focus-visible:ring-zinc-400 focus-visible:outline-none"
          >
            <ArrowDown className="size-3.5" aria-hidden="true" />
            {run.status === "running" && newLines > 0
              ? `${newLines} new line${newLines === 1 ? "" : "s"}`
              : "Jump to latest"}
          </button>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-zinc-800 px-3.5 py-2 text-xs text-zinc-400">
        {run.status === "running" ? <span className="font-medium text-zinc-100">Running</span> : null}
        {run.status === "passed" ? <span className="font-medium text-green-400">exit 0</span> : null}
        {run.status === "failed" ? <span className="font-medium text-red-400">exit {run.exitCode ?? "?"}</span> : null}
        {run.status === "stopped" ? <span className="font-medium text-zinc-200">Stopped</span> : null}
        <span>{run.status === "running" ? formatClock(elapsed) : formatDuration(elapsed)}</span>
        <span>
          {run.lines.length} line{run.lines.length === 1 ? "" : "s"}
        </span>
        <span className="min-w-0 truncate font-mono">{run.command}</span>
        <span className="flex-1" />
        <span className="flex items-center gap-1.5">
          <span aria-hidden="true" className={cn("size-1.5 rounded-full", following ? "bg-green-400" : "bg-amber-400")} />
          {following ? "Following" : "Paused, scroll to the bottom to follow"}
        </span>
      </div>
    </>
  );
}

const STREAM_COLORS: Record<LogLine["stream"], string> = {
  stdout: "text-zinc-200",
  stderr: "text-red-400",
  meta: "text-zinc-400",
};

// Highlights the PASS / FAIL markers test scripts print at the start of a line
function LineText({ entry, wrap }: { entry: LogLine; wrap: boolean }) {
  const marker = /^(\s*)(PASS|FAIL|FAILED|ERROR)\b(.*)$/.exec(entry.line);
  return (
    <span className={cn(STREAM_COLORS[entry.stream], wrap ? "min-w-0 whitespace-pre-wrap [overflow-wrap:anywhere]" : "whitespace-pre")}>
      {marker ? (
        <>
          {marker[1]}
          <span className={marker[2] === "PASS" ? "text-green-400" : "font-medium text-red-400"}>{marker[2]}</span>
          {marker[3]}
        </>
      ) : (
        // keep blank lines from collapsing to zero height
        entry.line || " "
      )}
    </span>
  );
}
