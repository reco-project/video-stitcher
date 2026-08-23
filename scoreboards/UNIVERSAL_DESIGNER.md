# Universal Scoreboard Designer v2

The shared designer is at `scoreboards/designer/index.html`. It configures Basketball, Soccer, Handball, Lacrosse, Field Hockey, American Football, and Rugby with one versioned state model.

Open it directly for a local preview:

    scoreboards/designer/index.html?sport=american-football

Reco can also open it with `sport` and `recoEditorToken` query parameters. The designer supports movable and scalable scoreboard placement, free-positioned text and logos, local demo team logos, and the Reco Cam “created with” branding. Drag an element in the preview; use the mouse wheel over it to scale it.

The designer always publishes a version 1 state object. Common fields stay in `game`, `home`, and `away`. Sport rules stay under `state.sport`; the host forwards that object without interpreting it. With a token, Publish to active scoreboard sends the complete state to `PUT /__reco/editor-state` using `X-Reco-Editor-Token`. Without a token, Download JSON exports the local state.

## Sport profiles

- Basketball: quarters, team fouls, and shot clock; scoring buttons `+1`, `+2`, and `+3`.
- Soccer: halves, added time, and yellow/red cards; goals are `+1` only.
- Handball: halves, timeouts, two-minute suspensions, and seven-metre goals; goals are `+1`.
- Lacrosse: quarters, 60-second shot clock, penalties, man-up goals, and timeouts; goals are `+1`.
- Field Hockey: quarters, penalty corners, and green/yellow/red cards; goals are `+1`.
- American Football: quarters, down, yards to go, ball position, play clock, possession, and timeouts; buttons cover PAT `+1`, safety `+2`, field goal `+3`, and touchdown `+6`.
- Rugby: halves, tries, conversions, cards, and sin bins; buttons cover conversion `+2`, penalty `+3`, and try `+5`.

## Adding another sport

1. Add `scoreboards/<id>/` with a schema version 1 manifest, transparent HTML/CSS, package JavaScript, and local assets.
2. Keep sport-specific fields inside that package’s `state.sport` object.
3. Add the default state, field list, score values, and preview interpretation to `scoreboards/designer/designer.js`.
4. Add the sport ID to the designer selector and the package list in `scoreboards/tests/scoreboard-contract.test.js`.
5. Keep the package offline-capable, transparent, accessible, and designed for its manifest viewport.
6. Validate the package JavaScript, manifest, SDK hooks, and sport-specific scoring rules before committing.

The universal designer is a reference operator surface. A package may still provide a specialized editor through its manifest `editor` field.
