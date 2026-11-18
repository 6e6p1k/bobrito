import { AppShell } from "@/components/AppShell";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DevPanelProvider } from "@/context/DevPanelContext";
import { PAGES } from "@/lib/pages";
import { ActionsPage } from "@/pages/ActionsPage";
import { DocsPage } from "@/pages/DocsPage";
import { OverviewPage } from "@/pages/OverviewPage";
import { Navigate, Route, Routes } from "react-router-dom";

function App() {
  return (
    <TooltipProvider>
      <DevPanelProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<OverviewPage />} />
            {PAGES.filter((p) => p.groupIds.length).map((page) => (
              <Route key={page.to} path={page.to.slice(1)} element={<ActionsPage page={page} />} />
            ))}
            <Route path="docs" element={<DocsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </DevPanelProvider>
    </TooltipProvider>
  );
}

export default App;
