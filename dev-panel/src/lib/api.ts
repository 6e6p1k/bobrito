import type { Action, ActionGroup, LogLine, Manifest, PanelConfig, RunResult, SystemInfo } from "@/types";

const API = "/api";

export async function fetchManifest(): Promise<Manifest> {
  const res = await fetch(`${API}/manifest`);
  if (!res.ok) throw new Error(`manifest: ${res.status}`);
  return res.json();
}

export async function fetchConfig(): Promise<PanelConfig> {
  const res = await fetch(`${API}/config`);
  if (!res.ok) throw new Error(`config: ${res.status}`);
  return res.json();
}

export async function fetchSystemChecks(): Promise<SystemInfo> {
  const res = await fetch(`${API}/system`);
  if (!res.ok && res.status !== 500) throw new Error(`system: ${res.status}`);
  // A 500 still carries { checks: [], raw } describing why doctor couldn't run
  return res.json();
}

export async function stopAction(): Promise<void> {
  const res = await fetch(`${API}/stop`, { method: "POST" });
  // 409 = it already finished; nothing to do
  if (!res.ok && res.status !== 409) throw new Error(`stop: ${res.status}`);
}

export async function runAction(
  actionId: string,
  onLine: (line: LogLine) => void
): Promise<RunResult> {
  const res = await fetch(`${API}/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ actionId }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error ?? res.statusText);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";
  let result: RunResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data: ")) continue;
      let payload: LogLine & Partial<RunResult> & { done?: boolean };
      try {
        payload = JSON.parse(line.slice(6));
      } catch {
        onLine({ stream: "stderr", line: `[panel] unparseable event: ${line.slice(6, 200)}` });
        continue;
      }
      onLine({ stream: payload.stream, line: payload.line });
      if (payload.done) {
        result = {
          exitCode: payload.exitCode ?? -1,
          durationMs: payload.durationMs ?? 0,
          actionId: payload.actionId ?? actionId,
          stopped: payload.stopped ?? false,
        };
      }
    }
  }
  if (!result) throw new Error("Connection to the panel API dropped before the action finished");
  return result;
}

export type { Action, ActionGroup };
