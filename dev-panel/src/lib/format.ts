// 77 ms · 3.4 s · 2:05
export function formatDuration(ms: number) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  return formatClock(ms);
}

// 0:02 · 12:40
export function formatClock(ms: number) {
  const total = Math.floor(ms / 1000);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

// Doctor messages carry absolute paths; inside the panel the repo root is implied
export function relativize(text: string, root: string | undefined) {
  return root ? text.split(`${root}/`).join("").split(root).join(".") : text;
}
