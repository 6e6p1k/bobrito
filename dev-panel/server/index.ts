import express from "express";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { StringDecoder } from "node:string_decoder";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PANEL_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(PANEL_ROOT, "..");
const BOBR = path.join(REPO_ROOT, "bobr");
const MANIFEST = path.join(PANEL_ROOT, "manifest.json");

const app = express();
// No CORS: the UI reaches us through the Vite proxy, so other origins have no business here
app.use(express.json());

type Action = {
  id: string;
  label: string;
  description?: string;
  bobr: string[];
};

type Group = { id: string; label: string; description?: string; actions: Action[] };

// The action currently running, if any. Only one at a time.
let current: { child: ChildProcess; actionId: string; stopRequested: boolean } | null = null;

// The extension's source files, straight from the Makefile so new files show up without
// touching the manifest. Cached until the Makefile changes.
let sourcesCache: { mtimeMs: number; files: string[] } | null = null;

function extensionSources(): string[] {
  const makefile = path.join(REPO_ROOT, "Makefile");
  try {
    const { mtimeMs } = fs.statSync(makefile);
    if (sourcesCache?.mtimeMs !== mtimeMs) {
      const out = execFileSync("make", ["-s", "--no-print-directory", "print-PY_CU_SRCS", "print-PY_CPP_SRCS"], {
        cwd: REPO_ROOT,
        encoding: "utf8",
      });
      sourcesCache = { mtimeMs, files: out.split(/\s+/).filter(Boolean) };
    }
    return sourcesCache.files;
  } catch (e) {
    console.error(`can't list extension sources from the Makefile: ${e}`);
    return [];
  }
}

function buildFilesGroup(): Group {
  return {
    id: "build-files",
    label: "Single files",
    description: "Recompile one source file, then relink with Rebuild Python lib",
    actions: extensionSources().map((src) => {
      const file = path.basename(src);
      const name = file.replace(/\.(cu|cpp)$/, "");
      return {
        id: `build-file-${name}`,
        label: file,
        description: `Rebuilds build/python/${name}.o`,
        bobr: ["build", name],
      };
    }),
  };
}

// manifest.json plus the generated "Single files" group right after "build"
function loadManifest(): { version: number; groups: Group[] } {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST, "utf8")) as { version: number; groups: Group[] };
  const files = buildFilesGroup();
  if (files.actions.length) {
    const at = manifest.groups.findIndex((g) => g.id === "build");
    manifest.groups.splice(at === -1 ? manifest.groups.length : at + 1, 0, files);
  }
  return manifest;
}

function findAction(actionId: string): Action | undefined {
  for (const g of loadManifest().groups) {
    const a = g.actions.find((x) => x.id === actionId);
    if (a) return a;
  }
  return undefined;
}

app.get("/api/manifest", (_req, res) => {
  res.json(loadManifest());
});

function gitBranch(): string | null {
  try {
    return execFileSync("git", ["rev-parse", "--abbrev-ref", "HEAD"], { cwd: REPO_ROOT, encoding: "utf8" }).trim();
  } catch {
    return null;
  }
}

app.get("/api/config", (_req, res) => {
  const configPath = path.join(PANEL_ROOT, "config.local.json");
  const examplePath = path.join(PANEL_ROOT, "config.local.example.json");
  let local = {};
  if (fs.existsSync(configPath)) {
    try {
      local = JSON.parse(fs.readFileSync(configPath, "utf8"));
    } catch (e) {
      console.error(`ignoring unparseable ${configPath}: ${e}`);
    }
  }
  res.json({
    projectRoot: REPO_ROOT,
    branch: gitBranch(),
    platform: os.release().toLowerCase().includes("microsoft") ? "WSL2" : os.platform(),
    gpuArch: process.env.BOBR_GPU_ARCH ?? "(see config.mk)",
    manifestPath: MANIFEST,
    configPath,
    devPanelRoot: PANEL_ROOT,
    local,
    exampleExists: fs.existsSync(examplePath),
  });
});

function spawnBobr(args: string[], opts: { detached?: boolean } = {}) {
  return spawn("bash", [BOBR, ...args], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    // detached = own process group, so stopping kills make/nvcc/python too, not just bash
    detached: opts.detached ?? false,
  });
}

// Signal the child's whole process group; escalate to SIGKILL if it hasn't exited after 5s
function killTree(child: ChildProcess) {
  if (child.pid === undefined || child.exitCode !== null || child.signalCode !== null) return;
  const signal = (sig: NodeJS.Signals) => {
    try {
      process.kill(-child.pid!, sig);
    } catch {
      // group already gone
    }
  };
  signal("SIGTERM");
  const timer = setTimeout(() => signal("SIGKILL"), 5000);
  timer.unref();
  child.once("close", () => clearTimeout(timer));
}

// Splits a byte stream into lines. Decodes UTF-8 across chunk boundaries and keeps
// partial lines until their newline arrives. Of "\r"-overwritten text (progress bars)
// only the final state is kept, as a terminal would show it.
function lineSplitter(onLine: (line: string) => void) {
  const decoder = new StringDecoder("utf8");
  let pending = "";
  const emit = (raw: string) => onLine(raw.replace(/\r$/, "").split("\r").pop() ?? "");
  return {
    push(chunk: Buffer) {
      pending += decoder.write(chunk);
      const parts = pending.split("\n");
      pending = parts.pop() ?? "";
      for (const part of parts) emit(part);
    },
    end() {
      pending += decoder.end();
      if (pending) emit(pending);
      pending = "";
    },
  };
}

// Doctor takes several seconds; concurrent requests share one run
let doctorRun: Promise<{ status: number; body: object }> | null = null;

function runDoctor() {
  doctorRun ??= new Promise<{ status: number; body: object }>((resolve) => {
    const proc = spawnBobr(["doctor", "--json"]);
    let out = "";
    proc.stdout.on("data", (d) => (out += d));
    proc.stderr.on("data", (d) => (out += d));
    proc.on("error", (err) => {
      resolve({ status: 500, body: { checks: [], raw: String(err), exitCode: -1 } });
    });
    proc.on("close", (code) => {
      try {
        resolve({ status: 200, body: { checks: JSON.parse(out), exitCode: code } });
      } catch {
        resolve({ status: 200, body: { checks: [], raw: out, exitCode: code } });
      }
    });
  }).finally(() => {
    doctorRun = null;
  });
  return doctorRun;
}

app.get("/api/system", async (_req, res) => {
  const { status, body } = await runDoctor();
  res.status(status).json(body);
});

app.post("/api/run", (req, res) => {
  const actionId = req.body?.actionId as string | undefined;
  if (!actionId) {
    res.status(400).json({ error: "actionId required" });
    return;
  }
  if (current) {
    res.status(409).json({ error: `Another action is running: ${current.actionId}` });
    return;
  }

  const action = findAction(actionId);
  if (!action?.bobr?.length) {
    res.status(404).json({ error: `Unknown action: ${actionId}` });
    return;
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const send = (payload: object) => {
    if (!res.writableEnded && !res.destroyed) res.write(`data: ${JSON.stringify(payload)}\n\n`);
  };

  send({ stream: "meta", line: `$ ./bobr ${action.bobr.join(" ")}` });

  const started = Date.now();
  const child = spawnBobr(action.bobr, { detached: true });
  const run = { child, actionId, stopRequested: false };
  current = run;

  let gotOutput = false;
  const stdout = lineSplitter((line) => send({ stream: "stdout", line }));
  const stderr = lineSplitter((line) => send({ stream: "stderr", line }));
  child.stdout.on("data", (d: Buffer) => {
    gotOutput = true;
    stdout.push(d);
  });
  child.stderr.on("data", (d: Buffer) => {
    gotOutput = true;
    stderr.push(d);
  });

  const finish = (exitCode: number, detail: string) => {
    const ms = Date.now() - started;
    send({
      stream: exitCode === 0 ? "meta" : "stderr",
      line: `${detail} · ${ms}ms`,
      done: true,
      exitCode,
      durationMs: ms,
      actionId,
      stopped: run.stopRequested,
    });
    res.end();
    if (current === run) current = null;
  };

  child.on("error", (err) => {
    send({ stream: "stderr", line: `spawn failed: ${err.message}` });
    finish(-1, "exit -1");
  });

  // "close" (not "exit") fires once every process holding the pipes is gone, so no output is lost
  child.on("close", (code, signal) => {
    stdout.end();
    stderr.end();
    if (run.stopRequested) {
      send({ stream: "meta", line: "stopped by user" });
    } else if (code !== 0 && !gotOutput) {
      send({ stream: "stderr", line: "No output captured. Try running the same command in a terminal." });
    }
    finish(code ?? -1, signal ? `killed by ${signal}` : `exit ${code}`);
  });

  // Client went away (tab closed / reloaded) before the action finished: stop it
  res.on("close", () => {
    if (current === run) killTree(child);
  });
});

app.post("/api/stop", (_req, res) => {
  if (!current) {
    res.status(409).json({ error: "Nothing is running" });
    return;
  }
  current.stopRequested = true;
  killTree(current.child);
  res.json({ ok: true, actionId: current.actionId });
});

// Don't leave a detached build running when the panel itself is stopped
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"] as const) {
  process.on(sig, () => {
    if (current?.child.pid !== undefined) {
      try {
        process.kill(-current.child.pid, "SIGTERM");
      } catch {
        // already gone
      }
    }
    process.exit(0);
  });
}

// Loopback only by default: anyone who can reach this port can run manifest actions.
// For a remote GPU host prefer an SSH tunnel; BOBR_PANEL_API_HOST=0.0.0.0 opens it up.
const HOST = process.env.BOBR_PANEL_API_HOST ?? "127.0.0.1";
const PORT = Number(process.env.BOBR_PANEL_API_PORT ?? 1421);
app.listen(PORT, HOST, () => {
  console.log(`bobr panel API → http://${HOST}:${PORT} (repo: ${REPO_ROOT})`);
});
