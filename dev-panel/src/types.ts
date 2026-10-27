export type Action = {
  id: string;
  label: string;
  description?: string;
  bobr: string[];
  variant?: "destructive" | "default";
};

export type ActionGroup = {
  id: string;
  label: string;
  description?: string;
  actions: Action[];
};

export type Manifest = {
  version: number;
  groups: ActionGroup[];
};

export type PanelConfig = {
  projectRoot: string;
  gpuArch: string;
  manifestPath: string;
  configPath: string;
  devPanelRoot: string;
};

export type SystemInfo = {
  checks: { id: string; level: string; message: string; ok: boolean }[];
};

export type LogLine = {
  stream: "stdout" | "stderr" | "meta";
  line: string;
};

export type RunResult = {
  exitCode: number;
  durationMs: number;
  actionId: string;
};
