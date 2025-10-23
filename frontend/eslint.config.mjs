import js from "@eslint/js";
import globals from "globals";
// eslint-disable-next-line import/no-unresolved
import { defineConfig } from "eslint/config";
import pluginReact from "eslint-plugin-react";
import pluginImport from "eslint-plugin-import";
import r3fConfig from "./r3f.eslint.config.mjs";

export default defineConfig([
  js.configs.recommended,
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
      ecmaVersion: "latest",
    },
    settings: {
      //
      react: {
        version: "detect",
      },
      "import/resolver": {
        node: {
          paths: ["src"],
          extensions: [".js", ".jsx", ".ts", ".tsx", ".mjs"]
        },
      },
    },
    rules: {
      "react/no-unknown-property": ["error", { ignore: ["position", "rotation", "args", "uniforms"] }],
      "import/no-unresolved": "off", // TODO: Too generic, but I don't want to deal with these errors now
      "import/named": "off", // TODO: Too generic, but I don't want to deal with these errors now
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
  },
]);
