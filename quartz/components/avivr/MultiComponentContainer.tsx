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

    return MultiComponentContainer
  } else {
    return () => <></>
  }
}) satisfies QuartzComponentConstructor
