import type { ActionGroup } from "@/types";
import { BookOpen, FlaskConical, Hammer, LayoutDashboard, Wrench, type LucideIcon } from "lucide-react";

export type PageDef = {
  to: string;
  label: string;
  icon: LucideIcon;
  description: string;
  // Manifest groups this page shows, in order
  groupIds: string[];
};

// Single source for the sidebar, routes, the command palette and per-page action counts
export const PAGES: PageDef[] = [
  {
    to: "/",
    label: "Overview",
    icon: LayoutDashboard,
    description: "Environment health and recent activity. Every button runs a ./bobr command; the CLI stays the source of truth.",
    groupIds: [],
  },
  {
    to: "/setup",
    label: "Setup",
    icon: Wrench,
    description: "First-time setup and health checks.",
    groupIds: ["setup"],
  },
  {
    to: "/build",
    label: "Build",
    icon: Hammer,
    description: "Compile C++/CUDA. After changing a .cu file, relink with Rebuild Python lib.",
    groupIds: ["build", "build-files"],
  },
  {
    to: "/tests",
    label: "Tests",
    icon: FlaskConical,
    description: "Official suites, legacy C++ targets and debug scripts. One action runs at a time.",
    groupIds: ["test", "cpp-tests", "debug"],
  },
  {
    to: "/docs",
    label: "Docs",
    icon: BookOpen,
    description: "How the build fits together and how to set up a fresh clone.",
    groupIds: [],
  },
];

export function groupsFor(page: PageDef, groups: ActionGroup[]) {
  return page.groupIds
    .map((id) => groups.find((g) => g.id === id))
    .filter((g): g is ActionGroup => g !== undefined);
}

export function pageForGroup(groupId: string) {
  return PAGES.find((p) => p.groupIds.includes(groupId));
}
