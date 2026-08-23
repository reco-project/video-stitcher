const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");

const root = path.resolve(__dirname, "..");
const packageIds = ["basketball", "soccer", "handball", "lacrosse", "field-hockey", "american-football", "rugby"];
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
test("sport packages keep their sport-specific scoring models", () => {
    const handball = readPackageFile("handball", "scoreboard.js");
    const lacrosse = readPackageFile("lacrosse", "scoreboard.js");
    const fieldHockey = readPackageFile("field-hockey", "scoreboard.js");
    const americanFootball = readPackageFile("american-football", "scoreboard.js");
    const rugby = readPackageFile("rugby", "scoreboard.js");
    assert.match(handball, /suspensionsHome/); assert.match(handball, /sevenMeterGoalsHome/); assert.match(handball, /scoreAmounts: \[1\]/);
    assert.match(lacrosse, /shotClock/); assert.match(lacrosse, /manUpHome/);
    assert.match(fieldHockey, /penaltyCornersHome/); assert.match(fieldHockey, /greenCardsHome/);
    assert.match(americanFootball, /yardsToGo/); assert.match(americanFootball, /Touchdown/); assert.match(americanFootball, /scoreAmounts: \[1,2,3,6\]/);
    assert.match(rugby, /triesHome/); assert.match(rugby, /sinBinsHome/); assert.match(rugby, /scoreAmounts: \[2,3,5\]/); assert.match(handball, /foulsHome/); assert.match(handball, /suspensionsHome/);
});
test("bundled packages honor template branding and typography", () => { for (const packageId of packageIds) { const script = readPackageFile(packageId, "scoreboard.js"); assert.match(script, /applyTypography/); assert.match(script, /applyTheme/); assert.match(readPackageFile(packageId, "assets/springfield-falcons.svg"), /Springfield Falcons/); assert.match(readPackageFile(packageId, "assets/springfield-lions.svg"), /Springfield Lions/); } });
test("universal designer supports all bundled sports and publishing", () => {
    const html = fs.readFileSync(path.join(root, "designer", "index.html"), "utf8"); const script = fs.readFileSync(path.join(root, "designer", "designer.js"), "utf8"); const style = fs.readFileSync(path.join(root, "designer", "style.css"), "utf8");
    for (const sport of ["basketball", "soccer", "handball", "lacrosse", "field-hockey", "american-football", "rugby"]) assert.match(html, new RegExp("value=\"" + sport + "\"")); assert.match(html, /Download state JSON/); assert.match(script, /__reco\/editor-state/); assert.match(script, /RecoScoreboard\.update/); assert.match(script, /scoreAmounts/); assert.match(html, /toggle-editing/); assert.match(style, /aspect-ratio: 16 \/ 9/); assert.match(script, /sportId/); assert.match(script, /toggleEditing/); assert.match(script, /updateEditorMode/); assert.match(script, /template-mode/); assert.match(html, /mode-template/); assert.match(html, /mode-game/); assert.match(script, /data-foul-team/); assert.match(script, /modeTabs/); assert.match(script, /foulsHome/); assert.match(script, /handleImageUpload/); assert.match(script, /typographyFor/); assert.match(html, /template-free-logo-upload/); assert.match(html, /game-home-logo-upload/); assert.match(html, /ratio-square/); assert.match(html, /design-style/);
});
