import { QuartzTransformerPlugin } from "../types"
import { visit } from "unist-util-visit"
import { Root, Heading, Text } from "mdast"

export const RemoveFirstH1: QuartzTransformerPlugin = () => {
  return {
    name: "remove-first-h1",
    markdownPlugins({ cfg }) {
      return [
        () => (tree: Root) => {
          let firstH1Removed = false

          // Use the `visit` function to traverse the Markdown AST
          visit(tree, "heading", (node: Heading, index, parent) => {
            if (!firstH1Removed && node.depth === 1 && parent && typeof index === "number") {
              // Remove the first H1 heading
              parent.children.splice(index, 1)
              firstH1Removed = true
            }
          })
        },
      ]
    },
  }
}
