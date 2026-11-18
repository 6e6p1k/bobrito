import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandShortcut,
} from "@/components/ui/command";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { commandFor } from "@/hooks/useDevPanel";
import { PAGES, groupsFor } from "@/lib/pages";
import { ArrowRight, Play, Trash2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

export function CommandPalette({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { manifest, runningRun, requestRun } = useDevPanelContext();
  const navigate = useNavigate();
  const groups = PAGES.flatMap((page) => groupsFor(page, manifest?.groups ?? []));

  const select = (fn: () => void) => {
    onOpenChange(false);
    fn();
  };

  return (
    <CommandDialog
      open={open}
      onOpenChange={onOpenChange}
      title="Run an action"
      description="Search actions and pages"
      // Fixed size, so the dialog doesn't jump around while results filter
      className="sm:max-w-2xl"
    >
      <Command>
        <CommandInput placeholder="Search actions and pages…" />
        <ScrollArea className="h-[min(26rem,60vh)]">
          <CommandList className="max-h-none overflow-visible">
            <CommandEmpty>No matching action.</CommandEmpty>
            {groups.map((group) => (
              <CommandGroup key={group.id} heading={group.label}>
                {group.actions.map((action) => {
                  const command = commandFor(action);
                  const destructive = action.variant === "destructive";
                  return (
                    <CommandItem
                      key={action.id}
                      value={`${action.label} ${command}`}
                      disabled={runningRun !== null}
                      onSelect={() => select(() => requestRun(action.id))}
                      className={destructive ? "text-destructive" : undefined}
                    >
                      {destructive ? <Trash2 aria-hidden="true" /> : <Play aria-hidden="true" />}
                      <span>{action.label}</span>
                      <CommandShortcut className="font-mono tracking-normal">{command}</CommandShortcut>
                    </CommandItem>
                  );
                })}
              </CommandGroup>
            ))}
            <CommandGroup heading="Pages">
              {PAGES.map((page) => (
                <CommandItem key={page.to} value={`Go to ${page.label}`} onSelect={() => select(() => navigate(page.to))}>
                  <ArrowRight aria-hidden="true" />
                  <span>Go to {page.label}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </ScrollArea>
      </Command>
    </CommandDialog>
  );
}
