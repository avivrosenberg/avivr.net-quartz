import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"
import * as AvivsComponents from "./quartz/components/avivr"

const navbarLinks = {
  //Home: "/",
  "about/": "/About",
  "posts/": "/Posts",
  "news/": "/News",
}
const simplePages = ["index", "About-me"]

// components shared across all pages
export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [],

  // Use custom footer
  footer: AvivsComponents.Footer({}),
}

// components for pages that display a single page (e.g. a single note)
export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.Breadcrumbs(),

    AvivsComponents.ConditionalComponent(Component.ArticleTitle(), {
      notOnPages: simplePages,
    }),
    AvivsComponents.ConditionalComponent(Component.TagList(), { notOnPages: simplePages }),
    AvivsComponents.ConditionalComponent(Component.ContentMeta(), {
      notOnPages: simplePages,
    }),
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
    AvivsComponents.ConditionalComponent(Component.Backlinks(), { notOnPages: simplePages }),
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
