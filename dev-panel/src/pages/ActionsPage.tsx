import { ActionList, PageHeader } from "@/components/ActionList";
import { useDevPanelContext } from "@/context/DevPanelContext";
import { groupsFor, type PageDef } from "@/lib/pages";

// Setup, Build and Tests: a page of manifest action groups
export function ActionsPage({ page }: { page: PageDef }) {
  const { manifest } = useDevPanelContext();

  return (
    <div className="flex flex-col gap-7 px-8 py-7">
      <PageHeader title={page.label} description={page.description} />
      {manifest ? (
        <ActionList groups={groupsFor(page, manifest.groups)} />
      ) : (
        <p className="text-sm text-muted-foreground">Loading actions…</p>
      )}
    </div>
  );
}
