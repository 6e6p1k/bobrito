import { PageHeader } from "@/components/ActionList";
import { PAGES } from "@/lib/pages";

const PIPELINE = `src/*.cu · *.cpp · headers
  ↓  make  (-MMD tracks header deps)
build/python/*.o
  ↓  nvcc -shared
python/bobrito.cpython-*.so   ← import bobrito`;

const CHECKLIST = [
  { command: "./bobr install", note: "creates .venv with torch, pybind11 and the CUDA 13 toolkit, and config.mk" },
  { command: "./bobr doctor", note: "if GPU_ARCH is wrong for your card, it suggests the value to put in config.mk" },
  { command: "./bobr build python", note: "builds the extension and copies it into python/" },
  { command: "./bobr test smoke", note: "gradient checks against torch" },
  { command: "./bobr configure", note: "optional: compile_commands.json for clangd" },
];

export function DocsPage() {
  const page = PAGES.find((p) => p.to === "/docs")!;

  return (
    <div className="flex max-w-3xl flex-col gap-7 px-8 py-7">
      <PageHeader title={page.label} description={page.description} />

      <section aria-labelledby="pipeline-title" className="flex flex-col gap-2.5">
        <h2 id="pipeline-title" className="text-[15px] font-semibold">
          Build pipeline
        </h2>
        <pre className="overflow-x-auto rounded-lg border bg-muted/40 px-4 py-3.5 font-mono text-xs leading-relaxed">
          {PIPELINE}
        </pre>
        <p className="text-sm text-muted-foreground">
          Rebuilding a single <code className="font-mono text-foreground">.o</code> doesn't reach Python until the
          extension is relinked with <code className="font-mono text-foreground">./bobr build python</code>.
        </p>
      </section>

      <section aria-labelledby="clone-title" className="flex flex-col gap-2.5">
        <h2 id="clone-title" className="text-[15px] font-semibold">
          Fresh clone
        </h2>
        <ol className="divide-y overflow-hidden rounded-lg border">
          {CHECKLIST.map((step, i) => (
            <li key={step.command} className="flex items-baseline gap-3 px-4 py-3">
              <span className="w-4 shrink-0 text-sm text-muted-foreground">{i + 1}</span>
              <div className="flex min-w-0 flex-col gap-0.5">
                <code className="font-mono text-[13px]">{step.command}</code>
                <span className="text-sm text-muted-foreground">{step.note}</span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <p className="text-sm text-muted-foreground">
        More in <code className="font-mono text-foreground">docs/CLI.md</code> and{" "}
        <code className="font-mono text-foreground">docs/DEV.md</code>. Agents use the same{" "}
        <code className="font-mono text-foreground">./bobr</code> commands directly; the panel is optional.
      </p>
    </div>
  );
}
