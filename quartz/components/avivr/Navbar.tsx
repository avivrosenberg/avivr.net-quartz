import { QuartzComponentConstructor, QuartzComponentProps } from "../types"
import { Darkmode, Search } from ".."

import styles from "./navbar.scss"

// @ts-ignore
import script from "./navbar.inline"

export default (() => {
  function NavBar({ displayClass, cfg }: QuartzComponentProps) {
    // const DarkmodeInstance = Darkmode()
    // const SearchInstance = Search()

    return (
      <nav className={`navbar ${displayClass}`}>
        <div className="navbar-container">
          <div className="menu-icon" id="menuIcon">
            <span className="bar"></span>
            <span className="bar"></span>
            <span className="bar"></span>
          </div>
          <ul className="nav-menu" id="navMenu">
            <li>
              <a href="/">Home</a>
            </li>
            <li>
              <a href="/About-me">About</a>
            </li>
            <li>
              <a href="/posts">Posts</a>
            </li>
          </ul>
        </div>
        {/* <DarkmodeInstance displayClass={displayClass} cfg={cfg} /> */}
        {/* <SearchInstance displayClass={displayClass} cfg={cfg} /> */}
      </nav>
    )
  }

  NavBar.css = styles
  NavBar.afterDOMLoaded = script

  return NavBar
}) satisfies QuartzComponentConstructor
