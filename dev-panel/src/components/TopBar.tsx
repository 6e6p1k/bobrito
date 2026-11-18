import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Kbd } from "@/components/ui/kbd";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { PAGES } from "@/lib/pages";
import { cn } from "@/lib/utils";
import { RefreshCw, Search } from "lucide-react";
import { useLocation } from "react-router-dom";

export function TopBar({ onOpenPalette }: { onOpenPalette: () => void }) {
  const { config, checking, refreshChecks } = useDevPanelContext();
  const { pathname } = useLocation();
  const page = PAGES.find((p) => (p.to === "/" ? pathname === "/" : pathname.startsWith(p.to)));
  const projectName = config?.projectRoot.split("/").filter(Boolean).pop();

  return (
    <header className="flex min-h-14 shrink-0 flex-wrap items-center gap-3 border-b px-4 py-2.5">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="h-4" />
      <Breadcrumb>
        <BreadcrumbList>
          {projectName ? (
            <>
              <BreadcrumbItem className="hidden md:inline-flex">{projectName}</BreadcrumbItem>
              <BreadcrumbSeparator className="hidden md:inline-flex" />
            </>
          ) : null}
          <BreadcrumbItem>
            <BreadcrumbPage>{page?.label}</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>
      <div className="flex-1" />
      <Button
        variant="outline"
        className="w-full justify-start gap-2 font-normal text-muted-foreground sm:w-64"
        onClick={onOpenPalette}
      >
        <Search aria-hidden="true" />
        <span className="flex-1 text-left">Run an action…</span>
        <Kbd>Ctrl K</Kbd>
      </Button>
      <Button variant="outline" onClick={() => void refreshChecks()} disabled={checking}>
        <RefreshCw aria-hidden="true" className={cn(checking && "animate-spin")} />
        {checking ? "Running doctor…" : "Re-run doctor"}
      </Button>
    </header>
  );
}
