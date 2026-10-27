import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchConfig,
  fetchManifest,
  fetchSystemChecks,
  runAction as apiRunAction,
} from "@/lib/api";
import type { LogLine, Manifest, PanelConfig, RunResult } from "@/types";

export function useDevPanel() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [config, setConfig] = useState<PanelConfig | null>(null);
  const [checks, setChecks] = useState<{ id: string; level: string; message: string; ok: boolean }[]>([]);
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [running, setRunning] = useState(false);
  const [lastResult, setLastResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [m, c, s] = await Promise.all([
        fetchManifest(),
        fetchConfig(),
        fetchSystemChecks(),
      ]);
      setManifest(m);
      setConfig(c);
      setChecks(s.checks ?? []);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const clearLogs = () => setLogs([]);

  const runAction = async (actionId: string) => {
    if (running) return;
    setRunning(true);
    setError(null);
    setLogs((prev) => [
      ...prev,
      { stream: "meta", line: `── ${actionId} ──` },
    ]);
    try {
      const result = await apiRunAction(actionId, (line) => {
        setLogs((prev) => [...prev, line]);
      });
      setLastResult(result);
      if (result.exitCode !== 0) {
        setError(`exit ${result.exitCode}`);
      }
      void refresh();
    } catch (e) {
      setError(String(e));
      setLogs((prev) => [...prev, { stream: "stderr", line: String(e) }]);
    } finally {
      setRunning(false);
    }
  };

  return {
    manifest,
    config,
    checks,
    logs,
    running,
    lastResult,
    error,
    logEndRef,
    refresh,
    clearLogs,
    runAction,
  };
}
