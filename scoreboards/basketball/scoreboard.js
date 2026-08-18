(function basketballReference() {
    "use strict";

    const elements = {
        homeName: document.querySelector("#home-name"),
        homeScore: document.querySelector("#home-score"),
        awayName: document.querySelector("#away-name"),
        awayScore: document.querySelector("#away-score"),
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
        game: { clock: "07:42", period: 2, running: false, status: "live" },
        home: { name: "Schapen Sharks", shortName: "SHARKS", score: 37, color: "#0057a8", secondaryColor: "#ffffff", logo: null },
        away: { name: "Braunschweig Lions", shortName: "LIONS", score: 32, color: "#cf2027", secondaryColor: "#ffffff", logo: null },
        sport: { teamFoulsHome: 3, teamFoulsAway: 5, shotClock: 18 },
        custom: {}
    };
    let debugState = structuredClone(fallbackState);
    let timer = null;

    function text(element, value, fallback) {
        element.textContent = value ?? fallback;
    }

    Reco.onUpdate((state) => {
        debugState = structuredClone(state);
        text(elements.homeName, state.home?.shortName || state.home?.name, "HOME");
        text(elements.homeScore, state.home?.score, 0);
        text(elements.awayName, state.away?.shortName || state.away?.name, "AWAY");
        text(elements.awayScore, state.away?.score, 0);
        text(elements.clock, state.game?.clock, "00:00");
        text(elements.period, `Q${state.game?.period ?? 1}`, "Q1");
        text(elements.homeFouls, state.sport?.teamFoulsHome, 0);
        text(elements.awayFouls, state.sport?.teamFoulsAway, 0);
        const shotClock = state.sport?.shotClock;
        elements.shotClockWrap.hidden = shotClock === null || shotClock === undefined;
        text(elements.shotClock, shotClock, "");
        elements.board.style.setProperty("--home-color", state.home?.color || "#0057a8");
        elements.board.style.setProperty("--home-secondary", state.home?.secondaryColor || "#ffffff");
        elements.board.style.setProperty("--away-color", state.away?.color || "#cf2027");
        elements.board.style.setProperty("--away-secondary", state.away?.secondaryColor || "#ffffff");
    });

    function publish() {
        RecoScoreboard.update(debugState);
    }

    function tickClock() {
        const [minutes, seconds] = debugState.game.clock.split(":").map(Number);
        const total = Math.max(0, minutes * 60 + seconds - 1);
        debugState.game.clock = `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
        if (total === 0) {
            clearInterval(timer);
            timer = null;
            debugState.game.running = false;
        }
        publish();
    }

    if (new URLSearchParams(location.search).get("debug") === "1") {
        const controls = document.querySelector("#debug-controls");
        controls.hidden = false;
        controls.addEventListener("click", (event) => {
            const button = event.target.closest("button[data-score]");
            if (!button) return;
            const team = button.dataset.score;
            debugState[team].score += Number(button.dataset.points);
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
        document.querySelector("#next-period").addEventListener("click", () => { debugState.game.period += 1; publish(); });
        document.querySelector("#home-foul").addEventListener("click", () => { debugState.sport.teamFoulsHome += 1; publish(); });
        document.querySelector("#away-foul").addEventListener("click", () => { debugState.sport.teamFoulsAway += 1; publish(); });
    }

    RecoScoreboard.update(fallbackState);
    Reco.ready();
})();
