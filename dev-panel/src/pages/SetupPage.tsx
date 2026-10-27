import { ActionGrid } from "@/components/DevPanel";
import { useDevPanelContext } from "@/context/DevPanelContext";

export function SetupPage() {
  const { manifest, running, runAction } = useDevPanelContext();
  const groups = (manifest?.groups ?? []).filter((g) => g.id === "setup");

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold">Setup</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          First-time setup and health checks.
        </p>
      </div>
      <ActionGrid groups={groups} running={running} onRun={runAction} />
    </div>
  );
}
