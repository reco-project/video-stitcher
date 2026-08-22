(function soccerScoreboard() {
    "use strict";

    const elements = {
        homeName: document.querySelector("#home-name"), homeScore: document.querySelector("#home-score"),
        awayName: document.querySelector("#away-name"), awayScore: document.querySelector("#away-score"),
        competition: document.querySelector("#competition"), status: document.querySelector("#status"),
        clock: document.querySelector("#clock"), period: document.querySelector("#period"),
        addedTime: document.querySelector("#added-time"), homeCards: document.querySelector("#home-cards"),
        awayCards: document.querySelector("#away-cards"), board: document.querySelector(".scoreboard")
    };
    const fallbackState = {
        version: 1,
        game: { competition: "Regional League", clock: "63:18", period: 2, running: false, status: "live" },
        home: { name: "Schapen Sharks", shortName: "SHARKS", score: 2, color: "#087f5b", secondaryColor: "#fff", logo: null },
        away: { name: "Braunschweig Lions", shortName: "LIONS", score: 1, color: "#9b1c31", secondaryColor: "#fff", logo: null },
        sport: { periodCount: 2, periodDurationMinutes: 45, addedTime: 3, homeYellowCards: 1, awayYellowCards: 0, homeRedCards: 0, awayRedCards: 1 },
        custom: {}
    };
    const query = new URLSearchParams(location.search);
    const editorMode = query.get("debug") === "1";
    const editorToken = query.get("recoEditorToken");
    let debugState = structuredClone(fallbackState);
    let timer = null;

    function ensureStateShape(state) {
        state.game ??= {}; state.home ??= {}; state.away ??= {}; state.sport ??= {}; state.custom ??= {};
        state.game.period ??= 1; state.sport.periodCount ??= 2; state.sport.periodDurationMinutes ??= 45;
        state.sport.addedTime ??= 0; state.sport.homeYellowCards ??= 0; state.sport.awayYellowCards ??= 0;
        state.sport.homeRedCards ??= 0; state.sport.awayRedCards ??= 0;
        return state;
    }
    function setText(element, value, fallback) { element.textContent = value ?? fallback; }
    function renderCards(container, yellow, red) {
        container.replaceChildren();
        for (let index = 0; index < Math.max(0, Number(yellow) || 0); index += 1) {
            const card = document.createElement("span"); card.className = "card yellow"; card.setAttribute("aria-label", "Yellow card"); container.append(card);
        }
        for (let index = 0; index < Math.max(0, Number(red) || 0); index += 1) {
            const card = document.createElement("span"); card.className = "card red"; card.setAttribute("aria-label", "Red card"); container.append(card);
        }
    }
    function setInput(id, value) {
        const input = document.querySelector("#" + id);
        if (input && document.activeElement !== input) input.value = value ?? "";
    }
    function syncEditor() {
        if (!editorMode) return;
        setInput("competition-input", debugState.game.competition); setInput("home-name-input", debugState.home.shortName || debugState.home.name);
        setInput("away-name-input", debugState.away.shortName || debugState.away.name); setInput("period-duration-input", debugState.sport.periodDurationMinutes);
        setInput("period-input", debugState.game.period); setInput("added-time-input", debugState.sport.addedTime);
        setInput("home-yellow-input", debugState.sport.homeYellowCards); setInput("away-yellow-input", debugState.sport.awayYellowCards);
        setInput("home-red-input", debugState.sport.homeRedCards); setInput("away-red-input", debugState.sport.awayRedCards);
        const button = document.querySelector("#toggle-clock"); if (button) button.textContent = timer ? "Stop clock" : "Start clock";
    }
    Reco.onUpdate((state) => {
        const safeState = ensureStateShape(structuredClone(state)); debugState = safeState;
        setText(elements.homeName, safeState.home.shortName || safeState.home.name, "HOME"); setText(elements.homeScore, safeState.home.score, 0);
        setText(elements.awayName, safeState.away.shortName || safeState.away.name, "AWAY"); setText(elements.awayScore, safeState.away.score, 0);
        setText(elements.competition, safeState.game.competition, ""); elements.competition.hidden = !safeState.game.competition;
        setText(elements.clock, safeState.game.clock, "00:00");
        const period = Number(safeState.game.period) || 1; setText(elements.period, period > 1 ? "2ND HALF" : "1ST HALF", "1ST HALF");
        const addedTime = Math.max(0, Number(safeState.sport.addedTime) || 0); elements.addedTime.hidden = addedTime === 0; setText(elements.addedTime, addedTime ? "+" + addedTime : "", "");
        renderCards(elements.homeCards, safeState.sport.homeYellowCards, safeState.sport.homeRedCards);
        renderCards(elements.awayCards, safeState.sport.awayYellowCards, safeState.sport.awayRedCards);
        const status = String(safeState.game.status || "live").toLowerCase(); setText(elements.status, status.toUpperCase(), "LIVE");
        elements.status.className = "status " + status; elements.board.style.setProperty("--home-color", safeState.home.color || "#087f5b");
        elements.board.style.setProperty("--home-secondary", safeState.home.secondaryColor || "#fff"); elements.board.style.setProperty("--away-color", safeState.away.color || "#9b1c31");
        elements.board.style.setProperty("--away-secondary", safeState.away.secondaryColor || "#fff"); syncEditor();
    });
    function headers() { return { "Content-Type": "application/json", "X-Reco-Editor-Token": editorToken || "" }; }
    async function publishRemote() {
        if (!editorToken) return;
        const indicator = document.querySelector("#editor-status");
        try {
            const response = await fetch("/__reco/editor-state", { method: "PUT", headers: headers(), body: JSON.stringify(debugState) });
            if (!response.ok) throw new Error("HTTP " + response.status);
            if (indicator) { indicator.textContent = "Live"; indicator.classList.remove("error"); }
        } catch (error) {
            if (indicator) { indicator.textContent = "Not connected"; indicator.classList.add("error"); }
            console.error("Cannot publish soccer editor state", error);
        }
    }
    function publish() { RecoScoreboard.update(debugState); void publishRemote(); }
    function clockSeconds() {
        const parts = String(debugState.game.clock || "00:00").split(":").map(Number);
        return Math.max(0, (Number(parts[0]) || 0) * 60 + (Number(parts[1]) || 0));
    }
    function formatClock(total) { return String(Math.floor(total / 60)).padStart(2, "0") + ":" + String(total % 60).padStart(2, "0"); }
    function tickClock() { debugState.game.clock = formatClock(clockSeconds() + 1); publish(); }
    function resetClock() {
        const minutes = Math.max(1, Number(debugState.sport.periodDurationMinutes) || 45);
        debugState.game.clock = String(minutes).padStart(2, "0") + ":00"; debugState.game.running = false; debugState.game.status = "paused";
        if (timer) clearInterval(timer); timer = null; publish();
    }
    function numberValue(id, minimum, maximum) {
        const value = Number(document.querySelector("#" + id).value);
        return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : minimum));
    }
    function bindEditor() {
        const controls = document.querySelector("#debug-controls"); controls.hidden = false;
        controls.addEventListener("click", (event) => {
            const scoreButton = event.target.closest("button[data-score]");
            if (scoreButton) { debugState[scoreButton.dataset.score].score = Number(debugState[scoreButton.dataset.score].score || 0) + 1; publish(); }
        });
        const textBindings = [
            ["competition-input", (value) => { debugState.game.competition = value.trim(); }],
            ["home-name-input", (value) => { debugState.home.name = value.trim(); debugState.home.shortName = value.trim(); }],
            ["away-name-input", (value) => { debugState.away.name = value.trim(); debugState.away.shortName = value.trim(); }]
        ];
        for (const binding of textBindings) document.querySelector("#" + binding[0]).addEventListener("change", (event) => { binding[1](event.target.value); publish(); });
        const numericBindings = [
            ["period-duration-input", "sport", "periodDurationMinutes", 1, 99], ["period-input", "game", "period", 1, 2],
            ["added-time-input", "sport", "addedTime", 0, 30], ["home-yellow-input", "sport", "homeYellowCards", 0, 10],
            ["away-yellow-input", "sport", "awayYellowCards", 0, 10], ["home-red-input", "sport", "homeRedCards", 0, 5],
            ["away-red-input", "sport", "awayRedCards", 0, 5]
        ];
        for (const binding of numericBindings) document.querySelector("#" + binding[0]).addEventListener("change", () => {
            debugState[binding[1]][binding[2]] = numberValue(binding[0], binding[3], binding[4]); publish();
        });
        document.querySelector("#toggle-clock").addEventListener("click", () => {
            if (timer) { clearInterval(timer); timer = null; debugState.game.running = false; debugState.game.status = "paused"; }
            else { debugState.game.running = true; debugState.game.status = "live"; timer = setInterval(tickClock, 1000); }
            publish();
        });
        document.querySelector("#reset-clock").addEventListener("click", resetClock);
        document.querySelector("#next-period").addEventListener("click", () => { debugState.game.period = Math.min(2, Number(debugState.game.period || 1) + 1); resetClock(); });
    }
    async function loadPublished() {
        if (!editorToken) return null;
        try { const response = await fetch("/__reco/editor-state", { headers: headers() }); return response.ok ? await response.json() : null; } catch (_) { return null; }
    }
    async function initialize() {
        if (editorMode) { bindEditor(); const published = await loadPublished(); if (published && typeof published === "object") debugState = ensureStateShape(published); }
        RecoScoreboard.update(debugState); if (editorMode) void publishRemote(); Reco.ready();
    }
    void initialize();
})();
