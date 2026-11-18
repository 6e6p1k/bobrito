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
  branch: string | null;
  platform: string;
  gpuArch: string;
  manifestPath: string;
  configPath: string;
  devPanelRoot: string;
};

export type DoctorCheck = { id: string; level: string; message: string; ok: boolean };

export type SystemInfo = {
  checks: DoctorCheck[];
  // Set when doctor's output couldn't be parsed
  raw?: string;
  exitCode: number | null;
};

export type LogLine = {
  stream: "stdout" | "stderr" | "meta";
  line: string;
};

export type RunResult = {
  exitCode: number;
  durationMs: number;
  actionId: string;
  stopped: boolean;
};
