export function DocsPage() {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold">Pipeline docs</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Build flow and getting started checklist.
        </p>
      </div>

      <div className="rounded-lg border border-border bg-card p-6 text-sm">
        <h3 className="text-base font-medium text-foreground">Build pipeline</h3>
        <pre className="mt-3 text-xs text-muted-foreground">{`src/*.cu / *.cpp / headers
  ↓  make (-MMD tracks header deps)
build/python/*.o
  ↓  nvcc -shared
python/bobr.cpython-*.so  ← import bobr`}</pre>

        <h3 className="mt-6 text-base font-medium text-foreground">Clone checklist</h3>
        <ol className="mt-3 list-decimal space-y-1 pl-5 text-muted-foreground">
          <li>
            <code>./bobr install</code>
          </li>
          <li>
            <code>cp config.mk.example config.mk</code> — set GPU_ARCH
          </li>
          <li>
            <code>./bobr configure</code> — clangd / Cursor
          </li>
          <li>
            <code>./bobr build python</code>
          </li>
          <li>
            <code>./bobr doctor</code>
          </li>
        </ol>

        <p className="mt-4 text-muted-foreground">
          See <code>docs/CLI.md</code> and <code>docs/DEV.md</code> in the repo.
        </p>
      </div>
    </div>
  );
}
