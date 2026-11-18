import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchConfig,
  fetchManifest,
  fetchSystemChecks,
  runAction as apiRunAction,
  stopAction as apiStopAction,
} from "@/lib/api";
import type { Action, DoctorCheck, LogLine, Manifest, PanelConfig } from "@/types";

// Older lines are dropped past this, so a long build can't make the log sluggish
const MAX_LOG_LINES = 10_000;
// Finished runs kept as output tabs
const MAX_RUNS = 8;
const LAST_RESULTS_KEY = "bobr-panel:last-results";

export type RunStatus = "running" | "passed" | "failed" | "stopped";

export type Run = {
  runId: number;
  actionId: string;
  label: string;
  command: string;
  lines: LogLine[];
  status: RunStatus;
  exitCode?: number;
  durationMs?: number;
  startedAt: number;
};

export type LastResult = { status: Exclude<RunStatus, "running">; durationMs: number };

export function commandFor(action: Pick<Action, "bobr">) {
  return `./bobr ${action.bobr.join(" ")}`;
}

function readLastResults(): Record<string, LastResult> {
  try {
    return JSON.parse(localStorage.getItem(LAST_RESULTS_KEY) ?? "{}");
  } catch {
    return {};
  }
}

export function useDevPanel() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [config, setConfig] = useState<PanelConfig | null>(null);
  // null until the first doctor run comes back
  const [checks, setChecks] = useState<DoctorCheck[] | null>(null);
  const [checksError, setChecksError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Newest first
  const [runs, setRuns] = useState<Run[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  // Survives reloads, unlike runs: what each action's row shows as its last result
  const [lastResults, setLastResults] = useState<Record<string, LastResult>>(readLastResults);
  // A destructive action waiting for the user to confirm it
  const [pendingConfirm, setPendingConfirm] = useState<string | null>(null);
  const runningRef = useRef(false);
  const nextRunId = useRef(1);

  const actions = useMemo(() => {
    const byId = new Map<string, Action>();
    for (const g of manifest?.groups ?? []) for (const a of g.actions) byId.set(a.id, a);
    return byId;
  }, [manifest]);

  const updateRun = useCallback((runId: number, update: (run: Run) => Run) => {
    setRuns((prev) => prev.map((r) => (r.runId === runId ? update(r) : r)));
  }, []);

  // Lines arrive far faster than React should re-render; flush them once per frame
  const pendingLines = useRef(new Map<number, LogLine[]>());
  const flushFrame = useRef<number | null>(null);
  const appendLine = useCallback((runId: number, line: LogLine) => {
    const pending = pendingLines.current.get(runId);
    if (pending) pending.push(line);
    else pendingLines.current.set(runId, [line]);
    flushFrame.current ??= requestAnimationFrame(() => {
      flushFrame.current = null;
      const batches = pendingLines.current;
      pendingLines.current = new Map();
      setRuns((prev) =>
        prev.map((r) => {
          const batch = batches.get(r.runId);
          if (!batch) return r;
          const lines = r.lines.concat(batch);
          return { ...r, lines: lines.length > MAX_LOG_LINES ? lines.slice(-MAX_LOG_LINES) : lines };
        }),
      );
    });
  }, []);

  const refreshChecks = useCallback(async () => {
    setChecking(true);
    try {
      const s = await fetchSystemChecks();
      setChecks(s.checks ?? []);
      setChecksError(
        s.checks?.length ? null : `doctor --json gave no checks (exit ${s.exitCode}): ${s.raw ?? ""}`.trim(),
      );
    } catch (e) {
      setChecksError(String(e));
    } finally {
      setChecking(false);
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [m, c] = await Promise.all([fetchManifest(), fetchConfig()]);
      setManifest(m);
      setConfig(c);
      setError(null);
    } catch (e) {
      setError(`Can't reach the panel API: ${String(e)}`);
    }
    await refreshChecks();
  }, [refreshChecks]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const recordResult = (actionId: string, result: LastResult) => {
    setLastResults((prev) => {
      const next = { ...prev, [actionId]: result };
      try {
        localStorage.setItem(LAST_RESULTS_KEY, JSON.stringify(next));
      } catch {
        // last results just won't survive a reload
      }
      return next;
    });
  };

  const runAction = async (actionId: string) => {
    const action = actions.get(actionId);
    if (runningRef.current || !action) return;
    runningRef.current = true;

    const runId = nextRunId.current++;
    const run: Run = {
      runId,
      actionId,
      label: action.label,
      command: commandFor(action),
      lines: [],
      status: "running",
      startedAt: Date.now(),
    };
    setRuns((prev) => {
      // Keep the newest finished runs; a running one is never dropped
      const kept = prev.filter((r) => r.status !== "running").slice(0, MAX_RUNS - 1);
      return [run, ...prev.filter((r) => r.status === "running"), ...kept];
    });
    setSelectedRunId(runId);

    try {
      const result = await apiRunAction(actionId, (line) => appendLine(runId, line));
      const status: RunStatus = result.stopped ? "stopped" : result.exitCode === 0 ? "passed" : "failed";
      updateRun(runId, (r) => ({ ...r, status, exitCode: result.exitCode, durationMs: result.durationMs }));
      recordResult(actionId, { status, durationMs: result.durationMs });
      // Actions can change what doctor reports (installs, builds)
      void refreshChecks();
    } catch (e) {
      appendLine(runId, { stream: "stderr", line: String(e) });
      updateRun(runId, (r) => ({ ...r, status: "failed", durationMs: Date.now() - r.startedAt }));
    } finally {
      runningRef.current = false;
    }
  };

  // Destructive actions go through a confirm dialog first
  const requestRun = (actionId: string) => {
    if (actions.get(actionId)?.variant === "destructive") setPendingConfirm(actionId);
    else void runAction(actionId);
  };

  const confirmPending = () => {
    if (pendingConfirm) void runAction(pendingConfirm);
    setPendingConfirm(null);
  };

  const stopAction = async () => {
    try {
      await apiStopAction();
    } catch (e) {
      setError(String(e));
    }
  };

  const runningRun = runs.find((r) => r.status === "running") ?? null;

  const clearFinished = () => {
    setRuns((prev) => prev.filter((r) => r.status === "running"));
    setSelectedRunId(runningRun?.runId ?? null);
  };
  const selectedRun = runs.find((r) => r.runId === selectedRunId) ?? runs[0] ?? null;

  return {
    manifest,
    actions,
    config,
    checks,
    checksError,
    checking,
    error,
    runs,
    selectedRun,
    runningRun,
    lastResults,
    pendingConfirm: pendingConfirm ? (actions.get(pendingConfirm) ?? null) : null,
    refresh,
    refreshChecks,
    requestRun,
    confirmPending,
    cancelPending: () => setPendingConfirm(null),
    stopAction,
    selectRun: setSelectedRunId,
    clearFinished,
  };
}
