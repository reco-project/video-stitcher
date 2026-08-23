(function basketballReference() {
    "use strict";

    const elements = {
        homeName: document.querySelector("#home-name"),
        homeScore: document.querySelector("#home-score"),
        homeLogo: document.querySelector("#home-logo"),
        awayLogo: document.querySelector("#away-logo"),
        awayName: document.querySelector("#away-name"),
        awayScore: document.querySelector("#away-score"),
        competition: document.querySelector("#competition"),
        clock: document.querySelector("#clock"),
        period: document.querySelector("#period"),
        homeFouls: document.querySelector("#home-fouls"),
        awayFouls: document.querySelector("#away-fouls"),
        homeFoulsWrap: document.querySelector("#home-fouls-wrap"),
        awayFoulsWrap: document.querySelector("#away-fouls-wrap"),
        shotClock: document.querySelector("#shot-clock"),
        shotClockWrap: document.querySelector("#shot-clock-wrap"),
        customText: document.querySelector("#custom-text"), customLogo: document.querySelector("#custom-logo"), board: document.querySelector(".scoreboard")
    };

    const fallbackState = {
        version: 1,
        game: {
            competition: "Regional League",
            clock: "07:42",
            period: 2,
            running: false,
            status: "live"
        },
        home: {
            name: "Schapen Sharks",
            shortName: "SHARKS",
            score: 37,
            color: "#0057a8",
            secondaryColor: "#ffffff",
            logo: "assets/sharks.svg"
        },
        away: {
            name: "Springfield Lions",
            shortName: "LIONS",
            score: 32,
            color: "#cf2027",
            secondaryColor: "#ffffff",
            logo: "assets/lions.svg"
        },
        sport: {
            periodCount: 4,
            periodDurationMinutes: 10,
            teamFoulLimit: 5,
            teamFoulsHome: 3,
            teamFoulsAway: 5,
            shotClock: 18
        },
        custom: { scoreboard: { x: 430, y: 120, scale: .62 }, freeText: { text: "created with", x: 1640, y: 940, scale: 1, color: "#fff", visible: true }, freeLogo: { src: "assets/reco-logo.png", x: 1800, y: 1005, scale: 0.8, visible: true } }
    };
    const query = new URLSearchParams(location.search);
    const editorMode = query.get("debug") === "1";
    const editorToken = query.get("recoEditorToken");
    let debugState = structuredClone(fallbackState);
    let timer = null;

    function text(element, value, fallback) {
        element.textContent = value ?? fallback;
    }

    function ensureStateShape(state) {
        state.game ??= {};
        state.home ??= {};
        state.away ??= {};
        state.sport ??= {};
        state.custom ??= {};
        state.home.logo ??= "assets/sharks.svg"; state.away.logo ??= "assets/lions.svg";
        state.custom.scoreboard ??= { x: 430, y: 120, scale: .62 };
        state.custom.freeText ??= { text: "created with", x: 1640, y: 940, scale: 1, color: "#fff", visible: true };
        state.custom.freeLogo ??= { src: "assets/reco-logo.png", x: 1800, y: 1005, scale: 0.8, visible: true };
        state.sport.periodCount ??= 4;
        state.sport.periodDurationMinutes ??= 10;
        state.sport.teamFoulLimit ??= 5;
        state.sport.teamFoulsHome ??= 0;
        state.sport.teamFoulsAway ??= 0;
        state.game.period ??= 1;
        return state;
    }

    function setInput(id, value) {
        const input = document.querySelector(`#${id}`);
        if (input && document.activeElement !== input) input.value = value ?? "";
    }

    function syncEditorFields() {
        if (!editorMode) return;
        setInput("competition-input", debugState.game.competition);
        setInput("home-name-input", debugState.home.shortName || debugState.home.name);
        setInput("away-name-input", debugState.away.shortName || debugState.away.name);
        setInput("period-count-input", debugState.sport.periodCount);
        setInput("period-duration-input", debugState.sport.periodDurationMinutes);
        setInput("period-input", debugState.game.period);
        setInput("shot-clock-input", debugState.sport.shotClock);
        setInput("team-foul-limit-input", debugState.sport.teamFoulLimit);
        const clockButton = document.querySelector("#toggle-clock");
        if (clockButton) clockButton.textContent = timer ? "Stop clock" : "Start clock";
    }

    function positionOverlay(element, config, defaultX, defaultY, defaultScale) { const x = Number(config?.x) || defaultX; const y = Number(config?.y) || defaultY; const scale = Math.min(3, Math.max(0.5, Number(config?.scale) || defaultScale)); element.style.left = x + "px"; element.style.top = y + "px"; element.style.transform = "translate(-50%, 0) scale(" + scale + ")"; }
    function renderCustom(state) { positionOverlay(elements.board, state.custom.scoreboard, 430, 120, .62); positionOverlay(elements.customText, state.custom.freeText, 1640, 1005, 1); positionOverlay(elements.customLogo, state.custom.freeLogo, 1800, 940, 0.8); elements.customText.textContent = state.custom.freeText.text || ""; elements.customText.hidden = state.custom.freeText.visible === false || !state.custom.freeText.text; elements.customText.style.color = state.custom.freeText.color || "#fff"; elements.customLogo.src = state.custom.freeLogo.src || ""; elements.customLogo.hidden = state.custom.freeLogo.visible === false || !state.custom.freeLogo.src; }
    function applyTypography(state) {
        const typography = state.custom?.typography || {};
        const defaults = {fontFamily:"Inter",fontSize:18,fontWeight:"700",fontStyle:"normal",color:"#fff"};
        const targets = {
            competition: [elements.competition], teamNames: [elements.homeName, elements.awayName], teamLabels: [...document.querySelectorAll(".team .label")],
            scores: [elements.homeScore, elements.awayScore], clock: [elements.clock], period: [elements.period], detail: [elements.detail], freeText: [elements.customText]
        };
        for (const [target, targetElements] of Object.entries(targets)) {
            const config = {...defaults, ...(typography[target] || {})};
            for (const element of targetElements) if (element) { element.style.fontFamily=config.fontFamily; element.style.fontSize=(Number(config.fontSize)||18)+"px"; element.style.fontWeight=config.fontWeight; element.style.fontStyle=config.fontStyle; element.style.color=config.color; }
        }
    }
    Reco.onUpdate((state) => {
        debugState = ensureStateShape(structuredClone(state));
        text(elements.homeName, state.home?.shortName || state.home?.name, "HOME");
        text(elements.homeScore, state.home?.score, 0);
        text(elements.awayName, state.away?.shortName || state.away?.name, "AWAY");
        text(elements.awayScore, state.away?.score, 0); elements.homeLogo.src = debugState.home.logo || ""; elements.homeLogo.hidden = !debugState.home.logo; elements.awayLogo.src = debugState.away.logo || ""; elements.awayLogo.hidden = !debugState.away.logo; renderCustom(debugState); applyTypography(debugState);
        text(elements.competition, state.game?.competition, "");
        elements.competition.hidden = !state.game?.competition;
        text(elements.clock, state.game?.clock, "00:00");
        const period = state.game?.period ?? 1;
        const periodCount = state.sport?.periodCount ?? 4;
        text(elements.period, `Q${period} / ${periodCount}`, "Q1 / 4");
        const teamFoulLimit = Math.max(1, Number(state.sport?.teamFoulLimit) || 5);
        const homeTeamFouls = Math.max(0, Number(state.sport?.teamFoulsHome) || 0);
        const awayTeamFouls = Math.max(0, Number(state.sport?.teamFoulsAway) || 0);
        text(elements.homeFouls, `${homeTeamFouls} / ${teamFoulLimit}`, `0 / ${teamFoulLimit}`);
        text(elements.awayFouls, `${awayTeamFouls} / ${teamFoulLimit}`, `0 / ${teamFoulLimit}`);
        elements.homeFoulsWrap.classList.toggle("is-at-limit", homeTeamFouls >= teamFoulLimit);
        elements.awayFoulsWrap.classList.toggle("is-at-limit", awayTeamFouls >= teamFoulLimit);
        const shotClock = state.sport?.shotClock;
        elements.shotClockWrap.hidden = shotClock === null || shotClock === undefined;
        text(elements.shotClock, shotClock, "");
        elements.board.style.setProperty("--home-color", state.home?.color || "#0057a8");
        elements.board.style.setProperty("--home-secondary", state.home?.secondaryColor || "#ffffff");
        elements.board.style.setProperty("--away-color", state.away?.color || "#cf2027");
        elements.board.style.setProperty("--away-secondary", state.away?.secondaryColor || "#ffffff");
        syncEditorFields();
    });

    function editorHeaders() {
        return {
            "Content-Type": "application/json",
            "X-Reco-Editor-Token": editorToken || ""
        };
    }

    async function publishRemote() {
        if (!editorToken) return;
        const status = document.querySelector("#editor-status");
        try {
            const response = await fetch("/__reco/editor-state", {
                method: "PUT",
                headers: editorHeaders(),
                body: JSON.stringify(debugState)
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            if (status) {
                status.textContent = "Live";
                status.classList.remove("error");
            }
        } catch (error) {
            if (status) {
                status.textContent = "Not connected";
                status.classList.add("error");
            }
            console.error("Cannot publish scoreboard editor state", error);
        }
    }

    function publish() {
        RecoScoreboard.update(debugState);
        void publishRemote();
    }

    function clockFromDuration() {
        const minutes = Math.max(1, Number(debugState.sport.periodDurationMinutes) || 10);
        debugState.game.clock = `${String(minutes).padStart(2, "0")}:00`;
    }

    function tickClock() {
        const [minutes, seconds] = String(debugState.game.clock || "00:00").split(":").map(Number);
        const total = Math.max(0, (minutes || 0) * 60 + (seconds || 0) - 1);
        debugState.game.clock = `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
        if (total === 0) {
            clearInterval(timer);
            timer = null;
            debugState.game.running = false;
        }
        publish();
    }

    function numberValue(id, minimum, maximum) {
        const value = Number(document.querySelector(`#${id}`).value);
        return Math.min(maximum, Math.max(minimum, Number.isFinite(value) ? value : minimum));
    }

    function bindEditorControls() {
        const controls = document.querySelector("#debug-controls");
        controls.hidden = false;
        controls.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-score]");
            if (!button) return;
            const team = button.dataset.score;
            debugState[team].score = Number(debugState[team].score || 0) + Number(button.dataset.points);
            publish();
        });

        const textBindings = [
            ["competition-input", (value) => { debugState.game.competition = value.trim(); }],
            ["home-name-input", (value) => { debugState.home.name = value.trim(); debugState.home.shortName = value.trim(); }],
            ["away-name-input", (value) => { debugState.away.name = value.trim(); debugState.away.shortName = value.trim(); }]
        ];
        for (const [id, apply] of textBindings) {
            document.querySelector(`#${id}`).addEventListener("change", (event) => {
                apply(event.target.value);
                publish();
            });
        }

        document.querySelector("#period-count-input").addEventListener("change", () => {
            debugState.sport.periodCount = numberValue("period-count-input", 1, 12);
            debugState.game.period = Math.min(debugState.game.period, debugState.sport.periodCount);
            publish();
        });
        document.querySelector("#period-duration-input").addEventListener("change", () => {
            debugState.sport.periodDurationMinutes = numberValue("period-duration-input", 1, 99);
            if (!timer) clockFromDuration();
            publish();
        });
        document.querySelector("#period-input").addEventListener("change", () => {
            debugState.game.period = numberValue("period-input", 1, debugState.sport.periodCount);
            publish();
        });
        document.querySelector("#shot-clock-input").addEventListener("change", () => {
            debugState.sport.shotClock = numberValue("shot-clock-input", 0, 99);
            publish();
        });
        document.querySelector("#team-foul-limit-input").addEventListener("change", () => {
            debugState.sport.teamFoulLimit = numberValue("team-foul-limit-input", 1, 20);
            publish();
        });
        document.querySelector("#home-foul").addEventListener("click", () => {
            debugState.sport.teamFoulsHome = Number(debugState.sport.teamFoulsHome || 0) + 1;
            publish();
        });
        document.querySelector("#away-foul").addEventListener("click", () => {
            debugState.sport.teamFoulsAway = Number(debugState.sport.teamFoulsAway || 0) + 1;
            publish();
        });
        document.querySelector("#toggle-clock").addEventListener("click", () => {
            if (timer) {
                clearInterval(timer);
                timer = null;
                debugState.game.running = false;
            } else {
                debugState.game.running = true;
                timer = setInterval(tickClock, 1000);
            }
            publish();
        });
        document.querySelector("#reset-clock").addEventListener("click", () => {
            clockFromDuration();
            publish();
        });
        document.querySelector("#next-period").addEventListener("click", () => {
            debugState.game.period = Math.min(
                debugState.sport.periodCount,
                Number(debugState.game.period || 1) + 1
            );
            debugState.sport.teamFoulsHome = 0;
            debugState.sport.teamFoulsAway = 0;
            clockFromDuration();
            publish();
        });
    }

    async function loadPublishedState() {
        if (!editorToken) return null;
        try {
            const response = await fetch("/__reco/editor-state", { headers: editorHeaders() });
            if (!response.ok) return null;
            return await response.json();
        } catch (_) {
            return null;
        }
    }

    async function initialize() {
        if (editorMode) {
            bindEditorControls();
            const published = await loadPublishedState();
            if (published && typeof published === "object") debugState = ensureStateShape(published);
        }
        RecoScoreboard.update(debugState);
        if (editorMode) void publishRemote();
        Reco.ready();
    }

    void initialize();
})();
