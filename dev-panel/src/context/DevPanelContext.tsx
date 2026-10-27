import { useDevPanel } from "@/hooks/useDevPanel";
import { createContext, useContext, type ReactNode } from "react";

type DevPanelContextValue = ReturnType<typeof useDevPanel>;

const DevPanelContext = createContext<DevPanelContextValue | null>(null);

export function DevPanelProvider({ children }: { children: ReactNode }) {
  const value = useDevPanel();
  return <DevPanelContext.Provider value={value}>{children}</DevPanelContext.Provider>;
}

export function useDevPanelContext() {
  const ctx = useContext(DevPanelContext);
  if (!ctx) throw new Error("useDevPanelContext must be used within DevPanelProvider");
  return ctx;
}
