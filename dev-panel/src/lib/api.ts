import type { Action, ActionGroup, LogLine, Manifest, PanelConfig, RunResult } from "@/types";

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

export async function fetchSystemChecks(): Promise<{ checks: { id: string; level: string; message: string; ok: boolean }[] }> {
  const res = await fetch(`${API}/system`);
  if (!res.ok) throw new Error(`system: ${res.status}`);
  return res.json();
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
  let result: RunResult = { exitCode: -1, durationMs: 0, actionId };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data: ")) continue;
      const payload = JSON.parse(line.slice(6)) as LogLine & Partial<RunResult> & { done?: boolean };
      onLine({ stream: payload.stream as LogLine["stream"], line: payload.line });
      if (payload.done) {
        result = {
          exitCode: payload.exitCode ?? -1,
          durationMs: payload.durationMs ?? 0,
          actionId: payload.actionId ?? actionId,
        };
      }
    }
  }
  return result;
}

export type { Action, ActionGroup };
