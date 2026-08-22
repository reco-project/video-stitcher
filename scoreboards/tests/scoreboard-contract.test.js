const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "..");
const packageIds = ["basketball", "soccer"];
function readPackageFile(packageId, filename) { return fs.readFileSync(path.join(root, packageId, filename), "utf8"); }

test("bundled scoreboards use manifest schema v1", () => {
    for (const packageId of packageIds) {
        const manifest = JSON.parse(readPackageFile(packageId, "manifest.json"));
        assert.equal(manifest.schemaVersion, 1); assert.equal(manifest.id, packageId); assert.equal(manifest.entry, "index.html");
        assert.equal(manifest.viewport.width, 1920); assert.equal(manifest.viewport.height, 1080); assert.equal(manifest.transparent, true); assert.equal(manifest.updateApiVersion, 1);
    }
});
test("bundled packages use the Reco SDK contract", () => {
    for (const packageId of packageIds) {
        const html = readPackageFile(packageId, "index.html"); const script = readPackageFile(packageId, "scoreboard.js");
        assert.match(html, /\.\.\/sdk\/reco-scoreboard\.js/); assert.match(script, /Reco\.onUpdate/); assert.match(script, /RecoScoreboard\.update/); assert.match(script, /Reco\.ready/); assert.doesNotMatch(html, /https?:\/\//);
    }
});
test("soccer keeps its rules in the package", () => {
    const script = readPackageFile("soccer", "scoreboard.js"); assert.match(script, /addedTime/); assert.match(script, /homeYellowCards/); assert.match(script, /homeRedCards/);
    assert.doesNotMatch(script, /shotClock/); assert.doesNotMatch(script, /teamFoul/);
});
test("universal designer supports both sports and publishing", () => {
    const html = fs.readFileSync(path.join(root, "designer", "index.html"), "utf8"); const script = fs.readFileSync(path.join(root, "designer", "designer.js"), "utf8");
    assert.match(html, /value="basketball"/); assert.match(html, /value="soccer"/); assert.match(html, /Download JSON/); assert.match(script, /__reco\/editor-state/); assert.match(script, /RecoScoreboard\.update/);
});
