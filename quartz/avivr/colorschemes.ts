const colorschemes = {
  "Nord Light": {
    light: "#eceff4", // Nord Light - Snow Storm 3
    lightgray: "#e5e9f0", // Nord Light - Snow Storm 2
    gray: "#d8dee9", // Nord Light - Snow Storm 1
    darkgray: "#4c566a", // Nord Light - Polar Night 4
    dark: "#2e3440", // Nord Light - Polar Night 1
    secondary: "#81a1c1", // Nord Light - Frost 3
    tertiary: "#88c0d0", // Nord Light - Frost 2
    highlight: "rgba(129, 161, 193, 0.15)", // Nord Light - Frost 3 with 15% opacity
    textHighlight: "#ebcb8b88", // Nord Light - Aurora Yellow with 53% opacity
  },
  "Nord Dark": {
    light: "#2e3440", // Nord Dark - Polar Night 1
    lightgray: "#3b4252", // Nord Dark - Polar Night 2
    gray: "#434c5e", // Nord Dark - Polar Night 3
    darkgray: "#d8dee9", // Nord Dark - Snow Storm 1
    dark: "#eceff4", // Nord Dark - Snow Storm 3
    secondary: "#81a1c1", // Nord Dark - Frost 3
    tertiary: "#88c0d0", // Nord Dark - Frost 2
    highlight: "rgba(129, 161, 193, 0.15)", // Nord Dark - Frost 3 with 15% opacity
    textHighlight: "#ebcb8b88", // Nord Dark - Aurora Yellow with 53% opacity
  },
  "Gruvbox Dark": {
    light: "#282828", // Gruvbox Dark - Background
    lightgray: "#3c3836", // Gruvbox Dark - Dark0
    gray: "#928374", // Gruvbox Dark - Comment
    darkgray: "#ebdbb2", // Gruvbox Dark - Foreground
    dark: "#fbf1c7", // Gruvbox Dark - Light0
    secondary: "#83a598", // Gruvbox Dark - Blue
    tertiary: "#d3869b", // Gruvbox Dark - Purple
    highlight: "rgba(131, 165, 152, 0.15)", // Gruvbox Dark - Blue with 15% opacity
    textHighlight: "#fabd2f88", // Gruvbox Dark - Yellow with 53% opacity
  },
  "Gruvbox Light": {
    light: "#fbf1c7", // Gruvbox Light - Background
    lightgray: "#ebdbb2", // Gruvbox Light - Foreground
    gray: "#928374", // Gruvbox Light - Comment
    darkgray: "#3c3836", // Gruvbox Light - Dark0
    dark: "#282828", // Gruvbox Light - Dark1
    secondary: "#458588", // Gruvbox Light - Blue
    tertiary: "#b16286", // Gruvbox Light - Purple
    highlight: "rgba(69, 133, 136, 0.15)", // Gruvbox Light - Blue with 15% opacity
    textHighlight: "#d7992188", // Gruvbox Light - Yellow with 53% opacity
  },
  Dracula: {
    light: "#282a36", // Dracula - Background
    lightgray: "#44475a", // Dracula - Current Line/Selection
    gray: "#6272a4", // Dracula - Comment
    darkgray: "#f8f8f2", // Dracula - Foreground
    dark: "#bd93f9", // Dracula - Purple
    secondary: "#ff79c6", // Dracula - Pink
    tertiary: "#8be9fd", // Dracula - Cyan
    highlight: "rgba(255, 121, 198, 0.15)", // Dracula - Pink with 15% opacity
    textHighlight: "#ffb86c88", // Dracula - Orange with 53% opacity
  },
  "Catpuccin Mocha": {
    light: "#1e1e2e", // Mocha - Base
    lightgray: "#313244", // Mocha - Surface0
    gray: "#a6adc8", // Mocha - Subtext0
    darkgray: "#cdd6f4", // Mocha - Text
    dark: "#f5e0dc", // Mocha - Rosewater
    secondary: "#89b4fa", // Mocha - Blue
    tertiary: "#94e2d5", // Mocha - Teal
    highlight: "rgba(137, 180, 250, 0.15)", // Mocha - Blue with 15% opacity
    textHighlight: "#fab38788", // Mocha - Peach with 53% opacity
  },
  "Catpuccin Macchiato": {
    light: "#24273a", // Macchiato - Base
    lightgray: "#363a4f", // Macchiato - Surface0
    gray: "#a5adcb", // Macchiato - Subtext0
    darkgray: "#cad3f5", // Macchiato - Text
    dark: "#f4dbd6", // Macchiato - Rosewater
    secondary: "#8aadf4", // Macchiato - Blue
    tertiary: "#7dc4e4", // Macchiato - Sky
    highlight: "rgba(138, 173, 244, 0.15)", // Macchiato - Blue with 15% opacity
    textHighlight: "#f5bde688", // Macchiato - Peach with 53% opacity
  },
  "Catpuccin Frappe": {
    light: "#303446", // Frappé - Base
    lightgray: "#414559", // Frappé - Surface0
    gray: "#a5adce", // Frappé - Subtext0
    darkgray: "#c6d0f5", // Frappé - Text
    dark: "#f2d5cf", // Frappé - Rosewater
    secondary: "#8caaee", // Frappé - Blue
    tertiary: "#81c8be", // Frappé - Teal
    highlight: "rgba(140, 170, 238, 0.15)", // Frappé - Blue with 15% opacity
    textHighlight: "#f4a26188", // Frappé - Peach with 53% opacity
  },
  "Catpuccin Latte": {
    light: "#eff1f5", // Latte - Base
    lightgray: "#ccd0da", // Latte - Surface0
    gray: "#6c6f85", // Latte - Subtext0
    darkgray: "#4c4f69", // Latte - Text
    dark: "#24273a", // Latte - Crust
    secondary: "#1e66f5", // Latte - Blue
    tertiary: "#179299", // Latte - Teal
    highlight: "rgba(30, 102, 245, 0.15)", // Latte - Blue with 15% opacity
    textHighlight: "#df8e1d88", // Latte - Yellow with 53% opacity
  },
  "Solarized Light": {
    light: "#fdf6e3", // Solarized Light - Base3
    lightgray: "#eee8d5", // Solarized Light - Base2
    gray: "#93a1a1", // Solarized Light - Base1
    darkgray: "#586e75", // Solarized Light - Base01
    dark: "#073642", // Solarized Light - Base02
    secondary: "#268bd2", // Solarized Light - Blue
    tertiary: "#2aa198", // Solarized Light - Cyan
    highlight: "rgba(38, 139, 210, 0.15)", // Solarized Light - Blue with 15% opacity
    textHighlight: "#b5890088", // Solarized Light - Yellow with 53% opacity
  },
  "Solarized Dark": {
    light: "#002b36", // Solarized Dark - Base03
    lightgray: "#073642", // Solarized Dark - Base02
    gray: "#586e75", // Solarized Dark - Base01
    darkgray: "#93a1a1", // Solarized Dark - Base1
    dark: "#fdf6e3", // Solarized Dark - Base3
    secondary: "#268bd2", // Solarized Dark - Blue
    tertiary: "#2aa198", // Solarized Dark - Cyan
    highlight: "rgba(38, 139, 210, 0.15)", // Solarized Dark - Blue with 15% opacity
    textHighlight: "#b5890088", // Solarized Dark - Yellow with 53% opacity
  },
}
export { colorschemes }
