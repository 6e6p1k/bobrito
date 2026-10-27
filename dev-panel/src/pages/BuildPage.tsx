import { ActionGrid } from "@/components/DevPanel";
import { useDevPanelContext } from "@/context/DevPanelContext";

export function BuildPage() {
  const { manifest, running, runAction } = useDevPanelContext();
  const groups = (manifest?.groups ?? []).filter((g) => g.id === "build");

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold">Build</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Compile C++/CUDA — always relink python after .cu changes.
        </p>
      </div>
      <ActionGrid groups={groups} running={running} onRun={runAction} />
    </div>
  );
}
