import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4.0 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Aviv A. Rosenberg",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: {
      provider: "plausible",
    },
    locale: "en-US",
    baseUrl: "quartz.jzhao.xyz",
    ignorePatterns: ["private", "templates", ".obsidian"],
    defaultDateType: "created",
    generateSocialImages: false,
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        lightMode: {
          light: "#fdf6e3", // base3
          lightgray: "#eee8d5", // base2
          gray: "#93a1a1", // base1
          darkgray: "#586e75", // base01
          dark: "#073642", // base02
          secondary: "#268bd2", // blue
          tertiary: "#2aa198", // cyan
          highlight: "rgba(38, 139, 210, 0.15)", // blue with 15% opacity
          textHighlight: "#b58900", // yellow
        },
        darkMode: {
          light: "#002b36", // base03
          lightgray: "#073642", // base02
          gray: "#586e75", // base01
          darkgray: "#93a1a1", // base1
          dark: "#fdf6e3", // base3
          secondary: "#268bd2", // blue
          tertiary: "#2aa198", // cyan
          highlight: "rgba(38, 139, 210, 0.15)", // blue with 15% opacity
          textHighlight: "#b58900", // yellow
        },
      },
    },
  },
  plugins: {
    transformers: [
      // Custom plugin to remove the first H1 heading.
      // In Obsidian, I write my notes with a H1 heading at the top, which I
      // don't want to display here because the page title is already displayed
      // by Quartz, and disabling it makes the page look bad.
      Plugin.RemoveFirstH1(),
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
        collapseByDefault: true,
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
    ],
  },
}

export default config
