import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import * as AvivsComponents from "./quartz/components/avivr"

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],

  // Use custom footer
  footer: AvivsComponents.Footer({}),
}

const navbarLinks = { Home: "/", About: "/About", Posts: "/Posts", News: "/News" }

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),
    Component.ArticleTitle(),
    Component.TagList(),
    // AvivsComponents.ConditionalComponent(Component.ContentMeta(), { notOnPages: ["index"] }),
    Component.ContentMeta(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    AvivsComponents.MultiComponentContainer([Component.Search(), Component.Darkmode()]),
    AvivsComponents.Navbar({
      links: navbarLinks,
    }),
    // Component.Search(),
    // Component.Darkmode(),
    // Component.DesktopOnly(Component.Explorer({ folderClickBehavior: "link" })),
  ],
  right: [
    Component.DesktopOnly(Component.TableOfContents({ layout: "modern" })),
    // Component.Graph(),
    AvivsComponents.ConditionalComponent(Component.Backlinks(), { notOnPages: ["About-me"] }),
  ],
}

// components for pages that display lists of pages  (e.g. tags or folders)
export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    AvivsComponents.MultiComponentContainer([Component.Search(), Component.Darkmode()]),
    AvivsComponents.Navbar({ links: navbarLinks }),
    // Component.Search(),
    // Component.Darkmode(),
    // Component.DesktopOnly(Component.Explorer({ folderClickBehavior: "link" })),
  ],
  right: [],
}
