(function universalScoreboardDesigner() {
    "use strict";

    const specifications = {
        basketball: {
            title: "Basketball", rules: "Basketball rules", period: (value, count) => "Q" + value + " / " + count,
            detail: (state) => "SHOT " + (state.sport.shotClock ?? "-"),
            fields: [["Quarters", "sport.periodCount"], ["Quarter length (min)", "sport.periodDurationMinutes"], ["Team foul limit", "sport.teamFoulLimit"], ["Home fouls", "sport.teamFoulsHome"], ["Away fouls", "sport.teamFoulsAway"], ["Shot clock", "sport.shotClock"]]
        },
        soccer: {
            title: "Soccer", rules: "Soccer rules", period: (value) => Number(value) > 1 ? "2ND HALF" : "1ST HALF",
            detail: (state) => state.sport.addedTime ? "+" + state.sport.addedTime : "NO ADDED TIME",
            fields: [["Halves", "sport.periodCount"], ["Half length (min)", "sport.periodDurationMinutes"], ["Added time", "sport.addedTime"], ["Home yellow cards", "sport.homeYellowCards"], ["Away yellow cards", "sport.awayYellowCards"], ["Home red cards", "sport.homeRedCards"], ["Away red cards", "sport.awayRedCards"]]
        }
    };
    const defaults = {
        basketball: { version: 1, game: { competition: "Regional League", clock: "07:42", period: 2, running: false, status: "live" }, home: { name: "Schapen Sharks", shortName: "SHARKS", score: 37, color: "#0057a8", secondaryColor: "#fff" }, away: { name: "Braunschweig Lions", shortName: "LIONS", score: 32, color: "#cf2027", secondaryColor: "#fff" }, sport: { periodCount: 4, periodDurationMinutes: 10, teamFoulLimit: 5, teamFoulsHome: 3, teamFoulsAway: 5, shotClock: 18 }, custom: {} },
        soccer: { version: 1, game: { competition: "Regional League", clock: "63:18", period: 2, running: false, status: "live" }, home: { name: "Schapen Sharks", shortName: "SHARKS", score: 2, color: "#087f5b", secondaryColor: "#fff" }, away: { name: "Braunschweig Lions", shortName: "LIONS", score: 1, color: "#9b1c31", secondaryColor: "#fff" }, sport: { periodCount: 2, periodDurationMinutes: 45, addedTime: 3, homeYellowCards: 1, awayYellowCards: 0, homeRedCards: 0, awayRedCards: 1 }, custom: {} }
    };
    const query = new URLSearchParams(location.search);
    const editorToken = query.get("recoEditorToken");
    let sport = query.get("sport") === "soccer" ? "soccer" : "basketball";
    let state = structuredClone(defaults[sport]);
    let publishTimer = null;
    const elements = {
        select: document.querySelector("#sport-select"), connection: document.querySelector("#connection-status"), title: document.querySelector("#preview-title"),
        status: document.querySelector("#preview-status"), competition: document.querySelector("#preview-competition"), homeName: document.querySelector("#preview-home-name"),
        awayName: document.querySelector("#preview-away-name"), homeScore: document.querySelector("#preview-home-score"), awayScore: document.querySelector("#preview-away-score"),
        clock: document.querySelector("#preview-clock"), period: document.querySelector("#preview-period"), detail: document.querySelector("#preview-detail"),
        card: document.querySelector("#preview-card"), legend: document.querySelector("#sport-legend"), fields: document.querySelector("#sport-fields"), scoreButtons: document.querySelector("#score-buttons"),
        form: document.querySelector("#state-form"), message: document.querySelector("#editor-message")
    };
    function pathGet(target, path) { return path.split(".").reduce((value, key) => value?.[key], target); }
    function pathSet(target, path, value) {
        const keys = path.split("."); const leaf = keys.pop(); const parent = keys.reduce((current, key) => current[key] ??= {}, target); parent[leaf] = value;
    }
    function inputValue(input) { return input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) || 0 : input.value; }
    function renderFields() {
        elements.fields.innerHTML = specifications[sport].fields.map(([label, path]) => "<label>" + label + "<input data-bind=\"" + path + "\" type=\"number\" min=\"0\" max=\"999\" value=\"" + pathGet(state, path) + "\"></label>").join("");
    }
    function renderInputs() {
        for (const input of elements.form.querySelectorAll("[data-bind]")) {
            if (document.activeElement === input) continue;
            const value = pathGet(state, input.dataset.bind); if (input.type === "checkbox") input.checked = Boolean(value); else input.value = value ?? "";
        }
    }
    function renderScoreButtons() {
        const amounts = sport === "basketball" ? [1, 2, 3] : [1];
        elements.scoreButtons.innerHTML = ["home", "away"].flatMap((team) => amounts.map((amount) => "<button type=\"button\" class=\"button\" data-team=\"" + team + "\" data-amount=\"" + amount + "\">" + (team === "home" ? "Home" : "Away") + " +" + amount + "</button>")).join("");
    }
    function render() {
        const specification = specifications[sport]; elements.select.value = sport; elements.title.textContent = specification.title; elements.legend.textContent = specification.rules;
        elements.competition.textContent = state.game.competition || ""; elements.homeName.textContent = state.home.shortName || "HOME"; elements.awayName.textContent = state.away.shortName || "AWAY";
        elements.homeScore.textContent = Number(state.home.score) || 0; elements.awayScore.textContent = Number(state.away.score) || 0; elements.clock.textContent = state.game.clock || "00:00";
        elements.period.textContent = specification.period(state.game.period || 1, state.sport.periodCount || 1); elements.detail.textContent = specification.detail(state);
        elements.status.textContent = String(state.game.status || "live").toUpperCase(); elements.status.classList.toggle("error", state.game.status === "paused");
        elements.card.style.setProperty("--home-color", state.home.color || "#0057a8"); elements.card.style.setProperty("--away-color", state.away.color || "#cf2027");
        renderInputs();
    }
    function headers() { return { "Content-Type": "application/json", "X-Reco-Editor-Token": editorToken || "" }; }
    async function publishRemote() {
        if (!editorToken) return;
        try {
            const response = await fetch("/__reco/editor-state", { method: "PUT", headers: headers(), body: JSON.stringify(state) });
            if (!response.ok) throw new Error("HTTP " + response.status);
            elements.connection.textContent = "Published to Reco"; elements.connection.classList.remove("error"); elements.message.textContent = "State published to the active scoreboard.";
        } catch (error) { elements.connection.textContent = "Publish failed"; elements.connection.classList.add("error"); elements.message.textContent = "Could not reach the active scoreboard: " + error.message; }
    }
    function publish() { RecoScoreboard.update(state); window.clearTimeout(publishTimer); publishTimer = window.setTimeout(() => void publishRemote(), 180); }
    function switchSport(nextSport) {
        const previous = state; sport = nextSport === "soccer" ? "soccer" : "basketball"; state = structuredClone(defaults[sport]);
        state.game.competition = previous.game.competition; state.home.shortName = previous.home.shortName; state.home.name = previous.home.name; state.home.score = previous.home.score; state.home.color = previous.home.color;
        state.away.shortName = previous.away.shortName; state.away.name = previous.away.name; state.away.score = previous.away.score; state.away.color = previous.away.color;
        renderScoreButtons(); renderFields(); render(); publish();
    }
    function resetState() { state = structuredClone(defaults[sport]); render(); publish(); }
    function handleInput(event) {
        if (!event.target.dataset.bind) return; pathSet(state, event.target.dataset.bind, inputValue(event.target));
        if (event.target.dataset.bind === "home.shortName") state.home.name = event.target.value;
        if (event.target.dataset.bind === "away.shortName") state.away.name = event.target.value;
        render(); publish();
    }
    async function loadPublished() {
        if (!editorToken) return;
        try {
            const response = await fetch("/__reco/editor-state", { headers: headers() }); if (!response.ok) return;
            const published = await response.json(); if (!published || typeof published !== "object") return;
            sport = published.sport?.shotClock !== undefined ? "basketball" : "soccer"; state = published; renderFields(); render(); elements.connection.textContent = "Loaded from Reco";
        } catch (_) { elements.message.textContent = "No published state was available; using the reference state."; }
    }
    document.querySelector("#sport-select").addEventListener("change", (event) => switchSport(event.target.value));
    elements.form.addEventListener("input", handleInput); document.querySelector("#reset-state").addEventListener("click", resetState);
    document.querySelector("#publish-state").addEventListener("click", () => { publish(); elements.message.textContent = "Publishing state to the active scoreboard…"; });
    document.querySelector("#download-state").addEventListener("click", () => {
        const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(state, null, 2) + "\n"], { type: "application/json" }));
        link.download = sport + "-scoreboard-state.json"; link.click(); URL.revokeObjectURL(link.href);
    });
    elements.scoreButtons.addEventListener("click", (event) => {
        const button = event.target.closest("button[data-team]"); if (!button) return; state[button.dataset.team].score = Math.max(0, Number(state[button.dataset.team].score) + Number(button.dataset.amount)); render(); publish();
    });
    renderScoreButtons(); renderFields(); render(); void loadPublished(); RecoScoreboard.update(state); Reco.ready();
})();
