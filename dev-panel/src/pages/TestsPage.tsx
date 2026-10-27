import { ActionGrid } from "@/components/DevPanel";
import { useDevPanelContext } from "@/context/DevPanelContext";

const TEST_GROUP_IDS = new Set(["test", "cpp-tests", "debug"]);

export function TestsPage() {
  const { manifest, running, runAction } = useDevPanelContext();
  const groups = (manifest?.groups ?? []).filter((g) => TEST_GROUP_IDS.has(g.id));

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold">Tests</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Official test suites, legacy C++ targets, and debug scripts.
        </p>
      </div>
      <ActionGrid groups={groups} running={running} onRun={runAction} />
    </div>
  );
}
