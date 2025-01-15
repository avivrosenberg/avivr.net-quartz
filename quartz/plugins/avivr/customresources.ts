import rehypeCitation from "rehype-citation"
import { PluggableList } from "unified"
import { visit } from "unist-util-visit"
import { QuartzTransformerPlugin } from "../types"
import { JSResource, CSSResource } from "../../util/resources"

export interface Options {}

const defaultOptions: Options = {}

export const AddCustomResources: QuartzTransformerPlugin<Partial<Options>> = (userOpts) => {
  const opts = { ...defaultOptions, ...userOpts }
  return {
    name: "AddCustomResources",

    // Add custom resources to the page
    externalResources() {
      const js: JSResource[] = []
      const css: CSSResource[] = []

      // Add fontawesome and academicons support.
      // TODO: This is not the right place, but I'm not sure where else to put it.
      css.push({
        content: "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css",
        inline: false,
      })
      css.push({
        content: "https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css",
        inline: false,
      })

      return { js, css }
    },
  }
}
