import { FullSlug } from "../../util/path"
import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"

interface Options {
  onlyOnPages: string[] | undefined
  notOnPages: string[] | undefined
  onlyIfFrontmatter: Record<string, string> | undefined
}

// const defaultOptions: Options = {
//   onlyOnPages: [],
//   notOnPages: [],
//   onlyIfFrontmatter: {},
// }

/*
This is a general purpose component that can be used to wrap another component
and control whther it appears based on a condition like the page slug or frontmatter.
*/
export default ((component?: QuartzComponent, userOpts?: Partial<Options>) => {
  if (component) {
    const opts = { ...userOpts }
    const Component = component

    const ConditionalComponent: QuartzComponent = (props: QuartzComponentProps) => {
      const wrappedComponent = <Component {...props} />

      const pageSlug = (props.fileData.slug ?? "") as string
      const pageFrontmatter = props.fileData.frontmatter ?? ({} as Record<string, string>)

      // If onlyOnPages is not empty, and the page is in the onlyOnPages list, show the component:
      if (opts.onlyOnPages !== undefined && opts.onlyOnPages.includes(pageSlug)) {
        return wrappedComponent
      }

      // If the notOnPages list is not empty page is not in the notOnPages list, show the component:
      if (opts.notOnPages !== undefined && !opts.notOnPages.includes(pageSlug)) {
        return wrappedComponent
      }

      // If the page frontmatter contains one of the keys in onlyIfFrontmatter, and the value matches, show the component
      if (
        opts.onlyIfFrontmatter !== undefined &&
        Object.keys(opts.onlyIfFrontmatter).every(
          (key) => pageFrontmatter[key] === opts.onlyIfFrontmatter[key],
        )
      ) {
        return wrappedComponent
      }
    }

    ConditionalComponent.displayName = component?.displayName
    ConditionalComponent.afterDOMLoaded = component?.afterDOMLoaded
    ConditionalComponent.beforeDOMLoaded = component?.beforeDOMLoaded
    ConditionalComponent.css = component?.css

    return ConditionalComponent
  } else {
    return () => <></>
  }
}) satisfies QuartzComponentConstructor
