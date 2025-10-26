/**
 * Custom ESLint configuration for React Three Fiber projects.
 * If you get false positives from the `react/no-unknown-property` rule,
 * you can add the relevant properties to the `ignore` array below.
 * If this is not convenient, just disable the rule entirely:
 * ```js
 * "react/no-unknown-property": "off"
 * ```
 */
export default {
  rules: {
    "react/no-unknown-property": [
      "error",
      {
        ignore: ["position"],
      },
    ],
  },
};
