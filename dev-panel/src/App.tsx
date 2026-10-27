import { AppShell } from "@/components/AppShell";
import { DevPanelProvider } from "@/context/DevPanelContext";
import { BuildPage } from "@/pages/BuildPage";
import { DocsPage } from "@/pages/DocsPage";
import { HomePage } from "@/pages/HomePage";
import { SetupPage } from "@/pages/SetupPage";
import { TestsPage } from "@/pages/TestsPage";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Route, Routes } from "react-router-dom";

function App() {
  return (
    <TooltipProvider>
      <DevPanelProvider>
        <Routes>
          <Route element={<AppShell />}>
            <Route index element={<HomePage />} />
            <Route path="setup" element={<SetupPage />} />
            <Route path="build" element={<BuildPage />} />
            <Route path="tests" element={<TestsPage />} />
            <Route path="docs" element={<DocsPage />} />
          </Route>
        </Routes>
      </DevPanelProvider>
    </TooltipProvider>
  );
}

export default App;
