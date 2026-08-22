# Universal Scoreboard Designer v2

The shared designer is at scoreboards/designer/index.html. It supports Basketball (quarters, fouls, and shot clock) and Soccer (halves, added time, and yellow/red cards).

Open it directly for a local preview. Reco can also open it with sport and recoEditorToken query parameters:

    scoreboards/designer/index.html?sport=soccer&recoEditorToken=<token>

The designer always publishes a version 1 state object. Common fields stay in game, home, and away. Sport rules stay under state.sport; the host forwards that object without interpreting it. With a token, Publish to active scoreboard sends the complete state to PUT /__reco/editor-state using X-Reco-Editor-Token. Without a token, Download JSON exports the local state.

To add another sport:

1. Add scoreboards/<id>/ with a schema version 1 manifest and local assets.
2. Keep sport-specific fields inside that package's state.sport object.
3. Add the default state, field list, and preview interpretation to scoreboards/designer/designer.js.
4. Add a package contract case to scoreboards/tests/scoreboard-contract.test.js.
5. Keep the package offline-capable, transparent, accessible, and designed for its manifest viewport.

The universal designer is a reference operator surface. A package may still provide a specialized editor through its manifest editor field.
