(function basketballReference() {
    "use strict";

    const elements = {
        homeName: document.querySelector("#home-name"),
        homeScore: document.querySelector("#home-score"),
        awayName: document.querySelector("#away-name"),
        awayScore: document.querySelector("#away-score"),
        competition: document.querySelector("#competition"),
        clock: document.querySelector("#clock"),
        period: document.querySelector("#period"),
        homeFouls: document.querySelector("#home-fouls"),
        awayFouls: document.querySelector("#away-fouls"),
        shotClock: document.querySelector("#shot-clock"),
        shotClockWrap: document.querySelector("#shot-clock-wrap"),
        board: document.querySelector(".scoreboard")
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
            logo: null
        },
        away: {
            name: "Braunschweig Lions",
            shortName: "LIONS",
            score: 32,
            color: "#cf2027",
            secondaryColor: "#ffffff",
            logo: null
        },
        sport: {
            periodCount: 4,
            periodDurationMinutes: 10,
            teamFoulsHome: 3,
            teamFoulsAway: 5,
            shotClock: 18
        },
        custom: {}
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
        state.sport.periodCount ??= 4;
        state.sport.periodDurationMinutes ??= 10;
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
        setInput("home-fouls-input", debugState.sport.teamFoulsHome);
        setInput("away-fouls-input", debugState.sport.teamFoulsAway);
        const clockButton = document.querySelector("#toggle-clock");
        if (clockButton) clockButton.textContent = timer ? "Stop clock" : "Start clock";
    }

    Reco.onUpdate((state) => {
        debugState = ensureStateShape(structuredClone(state));
        text(elements.homeName, state.home?.shortName || state.home?.name, "HOME");
        text(elements.homeScore, state.home?.score, 0);
        text(elements.awayName, state.away?.shortName || state.away?.name, "AWAY");
        text(elements.awayScore, state.away?.score, 0);
        text(elements.competition, state.game?.competition, "");
        elements.competition.hidden = !state.game?.competition;
        text(elements.clock, state.game?.clock, "00:00");
        const period = state.game?.period ?? 1;
        const periodCount = state.sport?.periodCount ?? 4;
        text(elements.period, `Q${period} / ${periodCount}`, "Q1 / 4");
        text(elements.homeFouls, state.sport?.teamFoulsHome, 0);
        text(elements.awayFouls, state.sport?.teamFoulsAway, 0);
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
        document.querySelector("#home-fouls-input").addEventListener("change", () => {
            debugState.sport.teamFoulsHome = numberValue("home-fouls-input", 0, 99);
            publish();
        });
        document.querySelector("#away-fouls-input").addEventListener("change", () => {
            debugState.sport.teamFoulsAway = numberValue("away-fouls-input", 0, 99);
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
