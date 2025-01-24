import { QuartzComponentConstructor, QuartzComponentProps } from "../types"

import styles from "./navbar.scss"

// @ts-ignore
import script from "./navbar.inline"

interface Options {
  links: Record<string, string>
}

export default ((opts?: Options) => {
  function NavBar({ displayClass, cfg }: QuartzComponentProps) {
    const links = opts?.links ?? []

    return (
      <nav className={`navbar ${displayClass}`}>
        <div className="navbar-container">
          <div className="menu-icon" id="menuIcon">
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
          </div>
          <ul className="nav-menu" id="navMenu">
            {Object.entries(links).map(([text, link]) => (
              <li>
                <a href={link}>{text}</a>
              </li>
            ))}
          </ul>
        </div>
      </nav>
    )
  }

  NavBar.css = styles
  NavBar.afterDOMLoaded = script

  return NavBar
}) satisfies QuartzComponentConstructor
