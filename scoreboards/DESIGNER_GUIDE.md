# Creating a Reco Scoreboard

This guide is for web designers and developers who know HTML, CSS, and JavaScript. You do not need to understand Reco's Rust code.

## Create a new scoreboard

1. Create `scoreboards/<your-sport>/`.
2. Add and validate `manifest.json`.
3. Create the declared `index.html` entry page.
4. Keep the full page transparent.
5. Include the shared Reco JavaScript SDK.
6. React to generic state updates and interpret your own `sport` object.
7. Test the entry page in Chrome or Chromium.
8. Test discovery and rendering in Reco.
9. Open a pull request containing the package and its tests.

Normally a new sport changes only its package. Do not add sport rules or names to Rust or Slint.

## Manifest

Every package needs these fields:

```json
{
  "schemaVersion": 1,
  "id": "example-sport",
  "name": "Example Sport",
  "sport": "example-sport",
  "version": "1.0.0",
  "author": "Reco Community",
  "description": "Example scoreboard",
  "entry": "index.html",
  "editor": "index.html?debug=1",
  "viewport": { "width": 1920, "height": 1080 },
  "transparent": true,
  "updateApiVersion": 1
}
```

`schemaVersion`, `id`, `name`, `sport`, `version`, `entry`, both viewport dimensions, and `updateApiVersion` are required. The directory name must equal `id`. IDs start with a lowercase ASCII letter and contain only lowercase letters, digits, `-`, or `_`. `entry` must be a relative `.html` path contained by the package. The optional `editor` target follows the same containment rules and may include a query string. When present, Reco shows a generic **Edit scoreboard…** button.

An editor page receives a short-lived `recoEditorToken` query parameter. It can publish complete version 1 state objects to the active overlay with an authenticated loopback request:

```javascript
const token = new URLSearchParams(location.search).get("recoEditorToken");
await fetch("/__reco/editor-state", {
    method: "PUT",
    headers: {
        "Content-Type": "application/json",
        "X-Reco-Editor-Token": token
    },
    body: JSON.stringify(state)
});
```

`GET /__reco/editor-state` with the same token returns the most recently published state or `null`. Keep sport rules and sport-specific controls inside the package; the host forwards the JSON without interpreting it.

Reco snapshots the latest published editor state when an export starts and applies it before the export renderer's first capture, so preview, recording, and exported video use the same values.

`updateApiVersion: 1` is a compatibility contract. Do not reinterpret or remove its fields incompatibly. A future incompatible contract will use a new number.

## Complete minimal page

```html
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        html, body {
            margin: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: transparent;
        }
        .scoreboard {
            position: absolute;
            left: 60px;
            bottom: 60px;
            padding: 24px;
            color: white;
            background: rgba(0, 0, 0, 0.78);
            font: 40px system-ui;
        }
    </style>
</head>
<body>
<div class="scoreboard">
    <span id="home">HOME</span>
    <span id="home-score">0</span>
    <span id="clock">10:00</span>
    <span id="away-score">0</span>
    <span id="away">AWAY</span>
</div>

<script src="../sdk/reco-scoreboard.js"></script>
<script>
Reco.onUpdate(state => {
    document.querySelector("#home").textContent = state.home.shortName;
    document.querySelector("#home-score").textContent = state.home.score;
    document.querySelector("#away").textContent = state.away.shortName;
    document.querySelector("#away-score").textContent = state.away.score;
    document.querySelector("#clock").textContent = state.game.clock;
});
Reco.ready();
</script>
</body>
</html>
```

Adjust the SDK path for nested examples. An installed package under `scoreboards/<id>/` normally uses `../sdk/reco-scoreboard.js`.

## JavaScript interfaces

The host expects this versioned global interface:

```javascript
window.RecoScoreboard = {
    apiVersion: 1,
    init(context) {},       // optional
    update(state) {},       // required
    reset() {},             // optional
    destroy() {}            // optional
};
```

The SDK supplies that interface and exposes the designer-oriented helper:

```javascript
Reco.onUpdate(callback); // returns an unsubscribe function
Reco.getState();
Reco.getContext();
Reco.ready();
Reco.log("message");
```

Reco may later call `scoreboard.update(jsonState)` from a controller, API, OCR source, or other integration. A package may also obtain all state itself using normal browser APIs:

```javascript
const ws = new WebSocket("ws://192.168.1.50:8080/scoreboard");
ws.onmessage = event => RecoScoreboard.update(JSON.parse(event.data));
```

`fetch`, `XMLHttpRequest`, `WebSocket`, `EventSource`, and timers behave as in Chrome, including normal CORS and certificate rules.

## Version 1 state

The shared state deliberately has a small generic area and an unconstrained sport-specific area:

```javascript
{
    version: 1,
    game: { clock: "07:42", period: 2, running: true, status: "live" },
    home: {
        name: "Schapen Sharks", shortName: "SHARKS", score: 37,
        color: "#0057a8", secondaryColor: "#ffffff", logo: null
    },
    away: {
        name: "Braunschweig Lions", shortName: "LIONS", score: 32,
        color: "#cf2027", secondaryColor: "#ffffff", logo: null
    },
    sport: {
        // Owned entirely by this package: fouls, cards, sets, shot clock, etc.
    },
    custom: {}
}
```

Check optional properties before using them. Do not assume another sport has your fields.

## Design rules

### Resolution and scaling

Design for a 1920 × 1080 reference canvas unless the manifest has a good reason to declare another size. Reco scales the complete canvas proportionally. A 3840 × 2160 output uses factor 2; 1280 × 720 uses approximately 0.6667. For different aspect ratios, Reco fits the reference canvas without distortion and centers it.

### Safe area

Keep important content at least about 60 px from each 1080p edge. The HTML page decides whether the graphic is top-left, top-center, bottom-right, or elsewhere; Reco has no sport-specific positioning.

### Transparency

Never set a fixed page background:

```css
html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: transparent;
}
```

Semi-transparent panels, shadows, antialiased text, and transparent PNG/SVG assets are supported.

### Type and contrast

At 1080p, useful starting sizes are team names 30–48 px, scores 44–72 px, clock 36–60 px, and secondary information 20–32 px. Test on both light and dark footage. Prefer a translucent surface, shadow or outline, and high-contrast type.

### Performance

Reco captures HTML at no more than 30 fps and reuses the GPU texture while the DOM is unchanged. Still avoid huge animations, particle systems, unnecessary DOM reconstruction, large video elements, and WebGL without a clear need. Prefer targeted updates:

```javascript
element.textContent = value;
```

CSS animations are detected and captured while running. Canvas-only changes do not mutate the DOM; call `window.__recoMarkDirty?.()` after drawing if you need Reco to capture them.

## Local assets and offline use

Keep fonts, logos, icons, and other assets inside the package:

```text
scoreboards/<id>/
├── manifest.json
├── index.html
├── style.css
├── scoreboard.js
└── assets/
    ├── home.png
    ├── away.png
    ├── icon.svg
    └── fonts/
```

Use relative URLs. Avoid required CDN resources so the scoreboard works offline in a sports hall.

## Browser and Reco testing

Open `index.html?debug=1` in Chrome for standalone designer controls when your package offers them. To test live editing, declare `editor` in the manifest, enable the scoreboard in Reco, and use **Edit scoreboard…** so Reco supplies the authenticated editor URL. Test long and missing names, large scores, light/dark backgrounds, transparent edges, shadows, logos, network reconnects, and 1920 × 1080.

Then place the directory under an installed discovery root, start Reco, enable Scoreboard, and choose it. Verify preview, recording, and stream/export. A missing/invalid manifest, duplicate ID, missing entry, unsupported API, browser crash, or JavaScript exception must result in a visible/logged error while video continues.
