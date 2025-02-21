import { QuartzComponent, QuartzComponentConstructor, QuartzComponentProps } from "../types"
import style from "../styles/footer.scss"
// import { version } from "../../../package.json"
// import { i18n } from "../../i18n"

interface Options {
  // links: Record<string, string>
}

export default ((opts?: Options) => {
  const Footer: QuartzComponent = ({ displayClass, cfg }: QuartzComponentProps) => {
    const year = new Date().getFullYear()
    const iconsize = "2x" // lg, 2x, 3x, .., 10x
    // const links = opts?.links ?? []
    return (
      <footer class={`${displayClass ?? ""}`}>
        <ul>
          <li>
            <a href="mailto:aviv@avivr.net">
              <i class={`fa-solid fa-envelope fa-${iconsize}`}></i>
            </a>
          </li>
          <li>
            <a href="https://linkedin.com/in/avivrosenberg">
              <i class={`fa-brands fa-linkedin fa-${iconsize}`}></i>
            </a>
          </li>
          <li>
            <a href="https://github.com/avivrosenberg">
              <i class={`fa-brands fa-github fa-${iconsize}`}></i>
            </a>
          </li>
          <li>
            <a href="https://scholar.google.com/citations?user=L6h2cnsAAAAJ">
              <i class={`ai ai-google-scholar-square ai-${iconsize}`}></i>
            </a>
          </li>
        </ul>
        <p>© Aviv A. Rosenberg 2018-{year}.</p>
        {/* <p>
          Built with
          <a href="https://obsidian.md"> Obsidian</a> and
          <a href="https://quartz.jzhao.xyz"> Quartz</a>.
        </p> */}
      </footer>
    )
  }

  Footer.css = style
  return Footer
}) satisfies QuartzComponentConstructor
