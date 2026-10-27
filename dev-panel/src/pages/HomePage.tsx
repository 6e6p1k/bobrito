import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { Link } from "react-router-dom";

const quickLinks = [
  { to: "/setup", label: "Setup", description: "Install, configure, doctor" },
  { to: "/build", label: "Build", description: "Compile C++/CUDA targets" },
  { to: "/tests", label: "Tests", description: "Smoke, full, and debug scripts" },
  { to: "/docs", label: "Pipeline docs", description: "Build flow and clone checklist" },
] as const;

export function HomePage() {
  const { manifest, config, checks } = useDevPanelContext();
  const groups = manifest?.groups ?? [];
  const failed = checks.filter((c) => !c.ok && c.level === "fail");

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-base font-semibold">Overview</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Dev panel for the bobr CUDA framework. Run actions via buttons or use{" "}
          <code className="rounded bg-muted px-1 py-0.5 text-xs">./bobr</code> directly.
        </p>
      </div>

      <Card className="border-border">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium">Doctor status</CardTitle>
          <CardDescription className="text-xs">
            {failed.length === 0
              ? "All checks passed"
              : `${failed.length} check${failed.length === 1 ? "" : "s"} failed`}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {checks.length === 0 ? (
              <span className="text-xs text-muted-foreground">Loading checks…</span>
            ) : (
              checks.map((check) => (
                <Badge
                  key={check.id}
                  variant={check.ok ? "outline" : "destructive"}
                  title={check.message}
                >
                  {check.id}
                  {!check.ok ? `: ${check.message}` : ""}
                </Badge>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {config ? (
        <Card className="border-border">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Project</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1 text-xs text-muted-foreground">
            <p>
              <span className="font-medium text-foreground">Root:</span>{" "}
              {config.projectRoot}
            </p>
            <p>
              <span className="font-medium text-foreground">GPU arch:</span>{" "}
              {config.gpuArch || "not set"}
            </p>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        {quickLinks.map((link) => (
          <Link key={link.to} to={link.to} className="group">
            <Card className="border-border transition-colors group-hover:border-primary/40">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium">{link.label}</CardTitle>
                <CardDescription className="text-xs">{link.description}</CardDescription>
              </CardHeader>
              {link.to !== "/docs" ? (
                <CardContent>
                  <span className="text-xs text-primary">
                    {groups.find((g) => g.id === link.to.slice(1))?.actions.length ?? 0}{" "}
                    actions
                  </span>
                </CardContent>
              ) : null}
            </Card>
          </Link>
        ))}
      </div>
    </div>
  );
}
