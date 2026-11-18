import type { DoctorCheck } from "@/types";

export type Severity = "fail" | "warn" | "note" | "ok";

// Purely informational checks: shown, but never counted as a problem
const NOTES = new Set(["nvcc_path"]);

// Which manifest action fixes a failing or warning check, and the button label for it
const FIXES: Record<string, { actionId: string; label: string }> = {
  config_mk: { actionId: "install", label: "Run install" },
  venv: { actionId: "install", label: "Run install" },
  python: { actionId: "install", label: "Run install" },
  torch: { actionId: "install", label: "Run install" },
  cuda_home: { actionId: "install-cuda", label: "Install CUDA" },
  cuda_libs: { actionId: "install-cuda", label: "Install CUDA" },
  bobr_import: { actionId: "build-python", label: "Rebuild" },
  build_stale: { actionId: "build-python", label: "Rebuild" },
  compile_db: { actionId: "configure", label: "Run configure" },
};

const RANK: Record<Severity, number> = { fail: 0, warn: 1, note: 2, ok: 3 };

export function severity(check: DoctorCheck): Severity {
  if (check.level === "fail") return "fail";
  if (check.level === "warn") return "warn";
  if (NOTES.has(check.id)) return "note";
  // doctor reports a day-old compile_commands.json as ok with "(stale …)" in the message
  if (/\bstale\b/i.test(check.message)) return "warn";
  return "ok";
}

export function fixFor(check: DoctorCheck) {
  const s = severity(check);
  return s === "fail" || s === "warn" ? FIXES[check.id] : undefined;
}

// Problems first, keeping doctor's order within each severity
export function sortChecks(checks: DoctorCheck[]) {
  return checks
    .map((check, i) => ({ check, i }))
    .sort((a, b) => RANK[severity(a.check)] - RANK[severity(b.check)] || a.i - b.i)
    .map(({ check }) => check);
}

export function summarize(checks: DoctorCheck[]) {
  const counts = { fail: 0, warn: 0, note: 0, ok: 0 };
  for (const c of checks) counts[severity(c)]++;
  const parts = [];
  if (counts.fail) parts.push(`${counts.fail} failing`);
  if (counts.warn) parts.push(`${counts.warn} to fix`);
  parts.push(`${counts.ok} passing`);
  if (counts.note) parts.push(`${counts.note} note${counts.note === 1 ? "" : "s"}`);
  return { counts, text: parts.join(" · ") };
}

export type EnvTile = { label: string; value: string; sub: string; ok: boolean };

// Doctor's checks are free text; pull the headline numbers out of them, "—" when absent
export function environment(checks: DoctorCheck[]): EnvTile[] {
  const get = (id: string) => checks.find((c) => c.id === id);
  const match = (id: string, re: RegExp) => re.exec(get(id)?.message ?? "")?.[1];

  const gpu = get("gpu");
  const [gpuName, driver] = (gpu?.ok && gpu.level === "ok" ? gpu.message : "").split(",").map((s) => s.trim());

  const cudaHome = get("cuda_home");
  const toolkit = cudaHome?.message.includes("(venv") ? ".venv toolkit" : "system toolkit";
  const arch = match("gpu_arch", /GPU_ARCH=(\S+)/);

  const ext = get("bobr_import");
  const stale = get("build_stale");

  return [
    {
      label: "GPU",
      value: gpuName ? gpuName.replace(/^NVIDIA (GeForce )?/, "") : "—",
      sub: driver ? `driver ${driver}` : "nvidia-smi unavailable",
      ok: Boolean(gpuName),
    },
    {
      label: "CUDA",
      value: match("cuda_home", /release ([\d.]+)/) ?? "—",
      sub: cudaHome?.ok ? [toolkit, arch].filter(Boolean).join(" · ") : "nvcc not found",
      ok: cudaHome?.ok ?? false,
    },
    {
      label: "Python",
      value: match("python", /Python ([\d.]+)/) ?? "—",
      sub: match("torch", /^(\S+)/) ? `torch ${match("torch", /^(\S+)/)}` : "torch not importable",
      ok: (get("python")?.ok ?? false) && (get("torch")?.ok ?? false),
    },
    {
      label: "Extension",
      value: ext?.ok ? "bobrito" : "not built",
      sub: ext?.ok
        ? [match("bobr_import", /(cpython-\d+)/), stale && stale.level !== "ok" ? "rebuild needed" : "up to date"]
            .filter(Boolean)
            .join(" · ")
        : "run Rebuild Python lib",
      ok: ext?.ok ?? false,
    },
  ];
}
