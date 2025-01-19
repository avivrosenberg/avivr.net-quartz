import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"

/*
This is a general purpose component that can be used to wrap multiple components in a
container. They will be placed side by side in the order they are passed in.
*/
export default ((components?: QuartzComponent[]) => {
  if (components) {
    function MultiComponentContainer({ displayClass, cfg }: QuartzComponentProps) {
      return (
        <div class="multi-component-container">
          {components.map((Component) => (
            <Component displayClass={displayClass} cfg={cfg} />
          ))}
        </div>
      )
    }

    const css = components.map((Component) => Component.css).join("\n")
    const scriptAfter = components.map((Component) => Component.afterDOMLoaded).join("\n")
    const scriptBefore = components.map((Component) => Component.beforeDOMLoaded).join("\n")
    MultiComponentContainer.css = css
    MultiComponentContainer.afterDOMLoaded = scriptAfter
    MultiComponentContainer.beforeDOMLoaded = scriptBefore

    return MultiComponentContainer
  } else {
    return () => <></>
  }
}) satisfies QuartzComponentConstructor
