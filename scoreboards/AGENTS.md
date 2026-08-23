# Agent Instructions: Adding a New Sport to Reco Scoreboards

Implement a new sport only by adding or modifying files inside its scoreboard package unless a genuine generic API limitation exists.

DO NOT add American football rules to Rust.
DO NOT add handball rules to Rust.
DO NOT add lacrosse, field hockey, or rugby rules to Rust.
DO NOT add volleyball rules to Rust.
DO NOT hardcode sport names into the GUI.

A new sport normally requires only:

```text
scoreboards/<sport>/
├── manifest.json
├── index.html
├── style.css
├── scoreboard.js
└── assets/
```

Keep the package offline-capable, transparent, accessible at 1920 × 1080, and compatible with JavaScript update API version 1. Add package-specific tests where practical.

## Example task: Add football

When asked to “Add a football scoreboard,” follow these steps.

### Step 1: Read the technical reference

Read `scoreboards/basketball/` only for packaging, manifest structure, JavaScript API, transparent rendering, and file organization. Do not copy basketball rules.

### Step 2: Create the package

Create `scoreboards/football/` with HTML, CSS, JavaScript, and local assets.

### Step 3: Add its manifest

```json
{
  "schemaVersion": 1,
  "id": "football",
  "name": "Football",
  "sport": "football",
  "version": "1.0.0",
  "author": "Reco Community",
  "description": "Football scoreboard",
  "entry": "index.html",
  "viewport": { "width": 1920, "height": 1080 },
  "transparent": true,
  "updateApiVersion": 1
}
```

### Step 4: Interpret football data inside the package

```javascript
{
    version: 1,
    game: { clock: "63:18", period: 2, running: true, status: "live" },
    home: { name: "FC Example", shortName: "FCE", score: 2 },
    away: { name: "United Example", shortName: "UNE", score: 1 },
    sport: { addedTime: 3, homeRedCards: 0, awayRedCards: 1 }
}
```

Football must ignore basketball-specific fields such as `shotClock` and `teamFouls`. Reco Rust code must not interpret added time or red cards.

### Step 5: Add tests

Test manifest discovery, generic state updates, and the rendered football DOM. Test transparency and reference viewport where the browser test environment permits it.

### Step 6: Avoid Rust changes

Do not change Rust unless you discovered a genuine generic limitation. If one exists:

1. Document the limitation.
2. Extend the API in sport-neutral terms.
3. Keep basketball and update API version 1 compatible, or introduce a new explicit API version for an incompatible contract.
4. Test both basketball and football.


## Universal Designer v2

The shared operator surface is scoreboards/designer/index.html. It supports Basketball, Soccer, Handball, Lacrosse, Field Hockey, American Football, and Rugby without changing Rust or the generic SDK. When adding a sport, extend the designer with its default version 1 state, sport-field controls, and preview interpretation, then add a package contract test. Keep common values in game, home, and away; keep sport rules under sport.

A package-specific editor remains valid and may expose controls that are more specialized than the universal designer. Any editor that publishes to Reco must use the authenticated X-Reco-Editor-Token request described in DESIGNER_GUIDE.md.
