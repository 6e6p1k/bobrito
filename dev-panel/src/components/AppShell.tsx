import { AppSidebar } from "@/components/AppSidebar";
import { CommandPalette } from "@/components/CommandPalette";
import { ConfirmRunDialog } from "@/components/ConfirmRunDialog";
import { OutputPanel } from "@/components/OutputPanel";
import { TopBar } from "@/components/TopBar";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ScrollArea } from "@/components/ui/scroll-area";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { useIsMobile } from "@/hooks/use-mobile";
import { AlertCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { useDefaultLayout } from "react-resizable-panels";
import { Outlet } from "react-router-dom";

export function AppShell() {
  const { error } = useDevPanelContext();
  const [paletteOpen, setPaletteOpen] = useState(false);
  // Narrow screens stack the output under the page instead of beside it
  const stacked = useIsMobile();
  const groupId = stacked ? "bobr-panel-main-stacked" : "bobr-panel-main";
  // Remembers where the content/output split was dragged to (localStorage)
  const layout = useDefaultLayout({ id: groupId });

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPaletteOpen((open) => !open);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset className="h-svh min-w-0 overflow-hidden">
        <TopBar onOpenPalette={() => setPaletteOpen(true)} />
        {error ? (
          <div role="alert" className="flex items-center gap-2 border-b bg-destructive/5 px-4 py-2 text-sm text-destructive">
            <AlertCircle className="size-4 shrink-0" aria-hidden="true" />
            {error}
          </div>
        ) : null}

        <ResizablePanelGroup
          key={groupId}
          id={groupId}
          orientation={stacked ? "vertical" : "horizontal"}
          className="min-h-0 flex-1"
          defaultLayout={layout.defaultLayout}
          onLayoutChanged={layout.onLayoutChanged}
        >
          <ResizablePanel id="content" minSize={stacked ? 160 : 360}>
            <ScrollArea className="h-full">
              <Outlet />
            </ScrollArea>
          </ResizablePanel>

          <ResizableHandle withHandle />

          <ResizablePanel id="output" defaultSize={stacked ? "50" : "42"} minSize={stacked ? 160 : 340}>
            <div className="h-full p-3">
              <OutputPanel />
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </SidebarInset>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
      <ConfirmRunDialog />
    </SidebarProvider>
  );
}
