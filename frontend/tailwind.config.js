import daisyui from 'daisyui'
import daisyThemes from 'daisyui/src/theming/themes'

const config = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  plugins: [
    daisyui,
  ],
  daisyui: {
    themes: [
      {
        light: {
          ...daisyThemes['light'],
          'primary': '#6C4AFF',
          'secondary': '#BCFF47',
        },
      },
    ],
  },
}

export default config
