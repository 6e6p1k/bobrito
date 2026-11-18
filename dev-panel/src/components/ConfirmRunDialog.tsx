import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { commandFor } from "@/hooks/useDevPanel";

// Asks before a manifest action marked "variant": "destructive" runs
export function ConfirmRunDialog() {
  const { pendingConfirm: action, confirmPending, cancelPending } = useDevPanelContext();

  return (
    <AlertDialog open={action !== null} onOpenChange={(open) => !open && cancelPending()}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Run {action?.label}?</AlertDialogTitle>
          <AlertDialogDescription>
            {action?.description ? `${action.description}. ` : null}
            This runs <code className="font-mono text-foreground">{action ? commandFor(action) : null}</code>.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction variant="destructive" onClick={confirmPending}>
            Run {action?.label}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
