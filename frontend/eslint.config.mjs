import js from "@eslint/js";
import globals from "globals";
// eslint-disable-next-line import/no-unresolved
import { defineConfig } from "eslint/config";
import pluginReact from "eslint-plugin-react";
import pluginImport from "eslint-plugin-import";
import r3fConfig from "./r3f.eslint.config.mjs";

export default defineConfig([
  pluginReact.configs.flat.recommended,
  pluginImport.flatConfigs.recommended,
  r3fConfig, // React Three Fiber custom rules
  {
    files: ["**/*.{js,mjs,cjs,jsx}"],
    plugins: { js },
    extends: ["js/recommended"],
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node,
        MAIN_WINDOW_VITE_DEV_SERVER_URL: "readonly",
        MAIN_WINDOW_VITE_NAME: "readonly",
      },
    },
    settings: {
      //
      react: {
        version: "detect",
      },
    },
    rules: {
      "react/no-unknown-property": ["error", { ignore: ["position"] }],
      "import/no-restricted-paths": [
        // cf. https://github.com/alan2207/bulletproof-react/blob/master/docs/project-structure.md
        "error",
        {
          zones: [
            /* --- disables cross-feature imports ---*/
            // eg. src/features/viewer should not import from src/features/something, etc.
            {
              target: "./src/features/viewer",
              from: "./src/features",
              except: ["./viewer"], // allow self-imports
            },

            /* --- enforce unidirectional codebase --- */
            // e.g. src/app can import from src/features but not the other way around
            {
              target: "./src/features",
              from: "./src/app",
            },

            // e.g src/features and src/app can import from these shared modules but not the other way around
            {
              target: [
                "./src/components",
                "./src/hooks",
                "./src/lib",
                "./src/types",
                "./src/utils",
              ],
              from: ["./src/features", "./src/app"],
            },
          ],
        },
      ],
    },
    "import/resolver": {
      node: {
        extensions: [".js", ".jsx"], // this is to allow importing without specifying extensions
      },
    },
  },
]);
