import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"
import * as AvivsPlugins from "./quartz/plugins/avivr"
import { colorschemes } from "./quartz/avivr/colorschemes"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Q.99",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "google",
      tagId: "G-0E7V13KGS6",
    },
    locale: "en-US",
    baseUrl: "avivr.net",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    generateSocialImages: false,
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        // header: "Schibsted Grotesk",
        // body: "Source Sans Pro",
        header: "Roboto Flex",
        body: "Kumbh Sans",
        // body: "Roboto Flex",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: colorschemes["Solarized Light"],
        darkMode: colorschemes["Solarized Dark"],
      },
    },
  },
  plugins: {
    transformers: [
      // Custom plugin to remove the first H1 heading.
      // In Obsidian, I write my notes with a H1 heading at the top, which I
      // don't want to display here because the page title is already displayed
      // by Quartz, and disabling it makes the page look bad.
      AvivsPlugins.RemoveFirstH1(),
      AvivsPlugins.AddCustomResources(),
      //
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "solarized-light",
          dark: "solarized-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents({
        maxDepth: 3,
        minEntries: 1,
        showByDefault: false,
        collapseByDefault: false,
      }),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({
        renderEngine: "mathjax", //"mathjax", "katex"
        customMacros: {
          // I decided to put the macros in each file to make it more flexible to change them over time.
        },
        mathJaxOptions: {
          svg: {
            scale: 1.1,
          },
        },
      }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.NotFoundPage(),
      Plugin.CNAME(),
    ],
  },
}

export default config
