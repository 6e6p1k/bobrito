import express from "express";
import cors from "cors";
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PANEL_ROOT = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(PANEL_ROOT, "..");
const BOBR = path.join(REPO_ROOT, "bobr");
const MANIFEST = path.join(PANEL_ROOT, "manifest.json");

const app = express();
app.use(cors());
app.use(express.json());

let busy = false;

type Action = {
  id: string;
  label: string;
  bobr: string[];
};

type Group = { id: string; label: string; description?: string; actions: Action[] };

function loadManifest(): { version: number; groups: Group[] } {
  return JSON.parse(fs.readFileSync(MANIFEST, "utf8"));
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

app.get("/api/config", (_req, res) => {
  const configPath = path.join(PANEL_ROOT, "config.local.json");
  const examplePath = path.join(PANEL_ROOT, "config.local.example.json");
  let local = {};
  if (fs.existsSync(configPath)) {
    local = JSON.parse(fs.readFileSync(configPath, "utf8"));
  }
  res.json({
    projectRoot: REPO_ROOT,
    gpuArch: process.env.BOBR_GPU_ARCH ?? "(see config.mk)",
    manifestPath: MANIFEST,
    configPath,
    devPanelRoot: PANEL_ROOT,
    local,
    exampleExists: fs.existsSync(examplePath),
  });
});

function spawnBobr(args: string[]) {
  return spawn("bash", [BOBR, ...args], {
    cwd: REPO_ROOT,
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
}

app.get("/api/system", async (_req, res) => {
  const proc = spawnBobr(["doctor", "--json"]);
  let out = "";
  proc.stdout.on("data", (d) => (out += d));
  proc.stderr.on("data", (d) => (out += d));
  proc.on("error", (err) => {
    res.status(500).json({ checks: [], raw: String(err), exitCode: -1 });
  });
  proc.on("close", (code) => {
    try {
      res.json({ checks: JSON.parse(out), exitCode: code });
    } catch {
      res.json({ checks: [], raw: out, exitCode: code });
    }
  });
});

app.post("/api/run", (req, res) => {
  const actionId = req.body?.actionId as string | undefined;
  if (!actionId) {
    res.status(400).json({ error: "actionId required" });
    return;
  }
  if (busy) {
    res.status(409).json({ error: "Another action is running" });
    return;
  }

  const action = findAction(actionId);
  if (!action?.bobr?.length) {
    res.status(404).json({ error: `Unknown action: ${actionId}` });
    return;
  }

  busy = true;
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const cmdDisplay = `./bobr ${action.bobr.join(" ")}`;
  res.write(`data: ${JSON.stringify({ stream: "meta", line: `$ ${cmdDisplay}` })}\n\n`);

  const started = Date.now();
  const child = spawnBobr(action.bobr);

  let outBuf = "";
  let errBuf = "";

  const send = (stream: string, chunk: Buffer) => {
    const text = chunk.toString();
    if (stream === "stdout") outBuf += text;
    else errBuf += text;
    for (const line of text.split(/\r?\n/)) {
      if (line.length) {
        res.write(`data: ${JSON.stringify({ stream, line })}\n\n`);
      }
    }
  };

  child.stdout.on("data", (d) => send("stdout", d));
  child.stderr.on("data", (d) => send("stderr", d));

  child.on("error", (err) => {
    const msg = `spawn failed: ${err.message}`;
    res.write(`data: ${JSON.stringify({ stream: "stderr", line: msg })}\n\n`);
    res.write(
      `data: ${JSON.stringify({ stream: "stderr", line: `exit -1 · ${Date.now() - started}ms`, done: true, exitCode: -1, durationMs: Date.now() - started, actionId })}\n\n`
    );
    res.end();
    busy = false;
  });

  child.on("close", (code) => {
    const ms = Date.now() - started;
    // Flush any trailing partial lines not ending with \n
    const tail = (outBuf + errBuf).split(/\r?\n/).pop();
    if (tail && !outBuf.endsWith("\n") && !errBuf.endsWith("\n") && tail.length > 0) {
      res.write(`data: ${JSON.stringify({ stream: "stdout", line: tail })}\n\n`);
    }
    const exit = code ?? -1;
    const stream = exit === 0 ? "meta" : "stderr";
    if (exit !== 0 && !outBuf && !errBuf) {
      res.write(
        `data: ${JSON.stringify({ stream: "stderr", line: "No output captured. Try running the same command in a terminal." })}\n\n`
      );
    }
    res.write(
      `data: ${JSON.stringify({ stream, line: `exit ${exit} · ${ms}ms`, done: true, exitCode: exit, durationMs: ms, actionId })}\n\n`
    );
    res.end();
    busy = false;
  });

  // Kill child only when the client drops the response stream (not on request end).
  res.on("close", () => {
    if (!child.killed && busy) child.kill("SIGTERM");
  });
});

const PORT = Number(process.env.BOBR_PANEL_API_PORT ?? 1421);
app.listen(PORT, () => {
  console.log(`bobr panel API → http://localhost:${PORT} (repo: ${REPO_ROOT})`);
});
