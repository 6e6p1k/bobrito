import { RunStatusIcon } from "@/components/RunStatusIcon";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { formatDuration } from "@/lib/format";
import { environment } from "@/lib/health";
import { PAGES, groupsFor, pageForGroup } from "@/lib/pages";
import { Cpu, Loader2 } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

export function AppSidebar() {
  const { manifest, config, checks, runs, runningRun, selectRun } = useDevPanelContext();
  const { pathname } = useLocation();
  const groups = manifest?.groups ?? [];
  const runningGroup = groups.find((g) => g.actions.some((a) => a.id === runningRun?.actionId));
  const runningPage = runningGroup ? pageForGroup(runningGroup.id) : undefined;
  const projectName = config?.projectRoot.split("/").filter(Boolean).pop();
  const [gpu, cuda] = checks ? environment(checks) : [];

  return (
    <Sidebar>
      <SidebarHeader className="gap-3">
        <div className="flex items-center gap-2.5 px-2 pt-1.5">
          <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-brand text-brand-foreground">
            <Cpu className="size-4" aria-hidden="true" />
          </div>
          <div className="flex flex-col leading-tight">
            <span className="font-semibold">bobr</span>
            <span className="text-xs text-muted-foreground">dev panel</span>
          </div>
        </div>
        {config ? (
          <div className="flex flex-col gap-0.5 rounded-lg border bg-background px-2.5 py-2">
            <span className="truncate text-[13px] font-medium" title={config.projectRoot}>
              {projectName}
            </span>
            <span className="truncate font-mono text-xs text-muted-foreground">
              {[config.branch, config.platform].filter(Boolean).join(" · ")}
            </span>
          </div>
        ) : null}
      </SidebarHeader>

      <SidebarContent className="overflow-hidden">
        <ScrollArea className="min-h-0 flex-1">
          <SidebarGroup>
            <SidebarGroupLabel>Workspace</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {PAGES.map((page) => {
                  const count = groupsFor(page, groups).reduce((n, g) => n + g.actions.length, 0);
                  const active = page.to === "/" ? pathname === "/" : pathname.startsWith(page.to);
                  return (
                    <SidebarMenuItem key={page.to}>
                      <SidebarMenuButton isActive={active} render={<NavLink to={page.to} end={page.to === "/"} />}>
                        <page.icon aria-hidden="true" />
                        <span>{page.label}</span>
                      </SidebarMenuButton>
                      {runningPage === page ? (
                        <SidebarMenuBadge>
                          <Loader2 className="size-3.5 animate-spin" role="img" aria-label="action running" />
                        </SidebarMenuBadge>
                      ) : count ? (
                        <SidebarMenuBadge className="text-muted-foreground">{count}</SidebarMenuBadge>
                      ) : null}
                    </SidebarMenuItem>
                  );
                })}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>

          {runs.length ? (
            <SidebarGroup>
              <SidebarGroupLabel>Recent runs</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {runs.slice(0, 5).map((run) => (
                    <SidebarMenuItem key={run.runId}>
                      <SidebarMenuButton size="sm" onClick={() => selectRun(run.runId)} title={`Show output of ${run.command}`}>
                        <RunStatusIcon status={run.status} />
                        <span className="font-mono text-xs">{run.actionId}</span>
                      </SidebarMenuButton>
                      {run.durationMs !== undefined ? (
                        <SidebarMenuBadge className="font-normal text-muted-foreground">
                          {formatDuration(run.durationMs)}
                        </SidebarMenuBadge>
                      ) : null}
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          ) : null}
        </ScrollArea>
      </SidebarContent>

      {gpu ? (
        <SidebarFooter>
          <div className="flex flex-col gap-1 rounded-lg border bg-background px-3 py-2.5">
            <div className="flex items-center gap-2">
              <span
                aria-hidden="true"
                className={gpu.ok ? "size-2 rounded-full bg-success" : "size-2 rounded-full bg-destructive"}
              />
              <span className="truncate text-[13px] font-medium">{gpu.value === "—" ? "No GPU found" : gpu.value}</span>
            </div>
            <span className="truncate text-xs text-muted-foreground">
              {[gpu.sub, cuda?.value !== "—" ? `CUDA ${cuda?.value}` : null].filter(Boolean).join(" · ")}
            </span>
          </div>
        </SidebarFooter>
      ) : null}
    </Sidebar>
  );
}
