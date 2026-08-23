(function universalScoreboardDesigner() {
    "use strict";

    const specifications = {
        "basketball": {
            title: "Basketball", rules: "Basketball rules", period: (value, count) => "Q" + value + " / " + count,
            detail: (state) => "SHOT " + (state.sport.shotClock ?? "-") + " · FOULS " + (Number(state.sport.teamFoulsHome || 0) + Number(state.sport.teamFoulsAway || 0)), scoreAmounts: [1,2,3], scoreLabels: {1:"Free throw",2:"2-pointer",3:"3-pointer"},
            fields: [["Quarters","sport.periodCount","number"],["Quarter length (min)","sport.periodDurationMinutes","number"],["Team foul limit","sport.teamFoulLimit","number"],["Shot clock","sport.shotClock","number"]]
        },
        "soccer": {
            title: "Soccer", rules: "Soccer rules", period: (value) => Number(value) > 1 ? "2ND HALF" : "1ST HALF",
            detail: (state) => state.sport.addedTime ? "+" + state.sport.addedTime : "NO ADDED TIME", scoreAmounts: [1], scoreLabels: {1:"Goal"},
            fields: [["Halves","sport.periodCount","number"],["Half length (min)","sport.periodDurationMinutes","number"],["Added time","sport.addedTime","number"],["Home yellow cards","sport.homeYellowCards","number"],["Away yellow cards","sport.awayYellowCards","number"],["Home red cards","sport.homeRedCards","number"],["Away red cards","sport.awayRedCards","number"]]
        },
        "handball": {
            title: "Handball", rules: "Handball rules", period: (value) => Number(value) > 1 ? "2ND HALF" : "1ST HALF",
            detail: (state) => "FOULS " + (Number(state.sport.foulsHome || 0) + Number(state.sport.foulsAway || 0)) + " · 2-MIN " + (Number(state.sport.suspensionsHome || 0) + Number(state.sport.suspensionsAway || 0)) + " · 7M " + (Number(state.sport.sevenMeterGoalsHome || 0) + Number(state.sport.sevenMeterGoalsAway || 0)), scoreAmounts: [1], scoreLabels: {1:"Goal"},
            fields: [["Halves","sport.periodCount","number"],["Half length (min)","sport.periodDurationMinutes","number"],["Home fouls","sport.foulsHome","number"],["Away fouls","sport.foulsAway","number"],["Home timeouts","sport.timeoutsHome","number"],["Away timeouts","sport.timeoutsAway","number"],["Home 2-min suspensions","sport.suspensionsHome","number"],["Away 2-min suspensions","sport.suspensionsAway","number"],["Home 7m goals","sport.sevenMeterGoalsHome","number"],["Away 7m goals","sport.sevenMeterGoalsAway","number"]]
        },
        "lacrosse": {
            title: "Lacrosse", rules: "Lacrosse rules", period: (value, count) => "Q" + value + " / " + count,
            detail: (state) => "SHOT " + (state.sport.shotClock ?? "-") + " · MAN-UP " + (Number(state.sport.manUpHome || 0) + Number(state.sport.manUpAway || 0)), scoreAmounts: [1], scoreLabels: {1:"Goal"},
            fields: [["Quarters","sport.periodCount","number"],["Quarter length (min)","sport.periodDurationMinutes","number"],["Shot clock","sport.shotClock","number"],["Home penalties","sport.penaltiesHome","number"],["Away penalties","sport.penaltiesAway","number"],["Home man-up goals","sport.manUpHome","number"],["Away man-up goals","sport.manUpAway","number"],["Home timeouts","sport.timeoutsHome","number"],["Away timeouts","sport.timeoutsAway","number"]]
        },
        "field-hockey": {
            title: "Field Hockey", rules: "Field hockey rules", period: (value, count) => "Q" + value + " / " + count,
            detail: (state) => "PC " + (Number(state.sport.penaltyCornersHome || 0) + Number(state.sport.penaltyCornersAway || 0)) + " · CARDS " + (Number(state.sport.yellowCardsHome || 0) + Number(state.sport.yellowCardsAway || 0)), scoreAmounts: [1], scoreLabels: {1:"Goal"},
            fields: [["Quarters","sport.periodCount","number"],["Quarter length (min)","sport.periodDurationMinutes","number"],["Penalty corners home","sport.penaltyCornersHome","number"],["Penalty corners away","sport.penaltyCornersAway","number"],["Green cards home","sport.greenCardsHome","number"],["Green cards away","sport.greenCardsAway","number"],["Yellow cards home","sport.yellowCardsHome","number"],["Yellow cards away","sport.yellowCardsAway","number"],["Red cards home","sport.redCardsHome","number"],["Red cards away","sport.redCardsAway","number"]]
        },
        "american-football": {
            title: "American Football", rules: "American football rules", period: (value, count) => "Q" + value + " / " + count,
            detail: (state) => "DOWN " + (state.sport.down ?? 1) + " · " + (state.sport.yardsToGo ?? 10) + " TO GO · BALL " + (state.sport.ballOn ?? "-"), scoreAmounts: [1,2,3,6], scoreLabels: {1:"PAT",2:"Safety",3:"Field goal",6:"Touchdown"},
            fields: [["Quarters","sport.periodCount","number"],["Quarter length (min)","sport.periodDurationMinutes","number"],["Down","sport.down","number"],["Yards to go","sport.yardsToGo","number"],["Ball on","sport.ballOn","number"],["Play clock","sport.playClock","number"],["Possession","sport.possession","text"],["Home timeouts","sport.timeoutsHome","number"],["Away timeouts","sport.timeoutsAway","number"]]
        },
        "rugby": {
            title: "Rugby", rules: "Rugby rules", period: (value) => Number(value) > 1 ? "2ND HALF" : "1ST HALF",
            detail: (state) => "TRY " + (Number(state.sport.triesHome || 0) + Number(state.sport.triesAway || 0)) + " · YC " + (Number(state.sport.yellowCardsHome || 0) + Number(state.sport.yellowCardsAway || 0)), scoreAmounts: [2,3,5], scoreLabels: {2:"Conversion",3:"Penalty",5:"Try"},
            fields: [["Halves","sport.periodCount","number"],["Half length (min)","sport.periodDurationMinutes","number"],["Home tries","sport.triesHome","number"],["Away tries","sport.triesAway","number"],["Home conversions","sport.conversionsHome","number"],["Away conversions","sport.conversionsAway","number"],["Home yellow cards","sport.yellowCardsHome","number"],["Away yellow cards","sport.yellowCardsAway","number"],["Home red cards","sport.redCardsHome","number"],["Away red cards","sport.redCardsAway","number"],["Home sin bins","sport.sinBinsHome","number"],["Away sin bins","sport.sinBinsAway","number"]]
        }
    };
    const defaults = {
        "basketball": {version:1,game:{competition:"Regional League",clock:"07:42",period:2,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:37,color:"#0057a8",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:32,color:"#cf2027",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"basketball",periodCount:4,periodDurationMinutes:10,teamFoulLimit:5,teamFoulsHome:3,teamFoulsAway:5,shotClock:18},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "soccer": {version:1,game:{competition:"Regional League",clock:"63:18",period:2,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:2,color:"#087f5b",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:1,color:"#9b1c31",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"soccer",periodCount:2,periodDurationMinutes:45,addedTime:3,homeYellowCards:1,awayYellowCards:0,homeRedCards:0,awayRedCards:1},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "handball": {version:1,game:{competition:"Regional Handball League",clock:"18:24",period:1,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:14,color:"#0057a8",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:12,color:"#cf2027",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"handball",periodCount:2,periodDurationMinutes:30,foulsHome:2,foulsAway:3,timeoutsHome:1,timeoutsAway:2,suspensionsHome:1,suspensionsAway:0,sevenMeterGoalsHome:2,sevenMeterGoalsAway:1},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "lacrosse": {version:1,game:{competition:"Regional Lacrosse League",clock:"08:42",period:3,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:7,color:"#0057a8",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:6,color:"#cf2027",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"lacrosse",periodCount:4,periodDurationMinutes:15,shotClock:60,penaltiesHome:2,penaltiesAway:1,manUpHome:1,manUpAway:0,timeoutsHome:1,timeoutsAway:1},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "field-hockey": {version:1,game:{competition:"Regional Field Hockey League",clock:"11:36",period:2,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:3,color:"#0b7285",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:2,color:"#d9480f",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"field-hockey",periodCount:4,periodDurationMinutes:15,penaltyCornersHome:3,penaltyCornersAway:2,greenCardsHome:0,greenCardsAway:1,yellowCardsHome:1,yellowCardsAway:0,redCardsHome:0,redCardsAway:0},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "american-football": {version:1,game:{competition:"Regional American Football League",clock:"04:18",period:2,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:21,color:"#0057a8",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:17,color:"#cf2027",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"american-football",periodCount:4,periodDurationMinutes:15,down:2,yardsToGo:7,ballOn:42,playClock:18,possession:"SHARKS",timeoutsHome:2,timeoutsAway:1},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}},
        "rugby": {version:1,game:{competition:"Regional Rugby League",clock:"31:12",period:1,running:false,status:"live"},home:{name:"Schapen Sharks",shortName:"SHARKS",score:10,color:"#087f5b",secondaryColor:"#fff",logo:"assets/sharks.svg"},away:{name:"Springfield Lions",shortName:"LIONS",score:8,color:"#9b1c31",secondaryColor:"#fff",logo:"assets/lions.svg"},sport:{sportId:"rugby",periodCount:2,periodDurationMinutes:40,triesHome:1,triesAway:1,conversionsHome:1,conversionsAway:0,yellowCardsHome:1,yellowCardsAway:0,redCardsHome:0,redCardsAway:0,sinBinsHome:0,sinBinsAway:1},custom:{scoreboard:{x:960,y:280,scale:1},freeText:{text:"created with",x:1640,y:940,scale:1,color:"#fff",visible:true},freeLogo:{src:"assets/reco-logo.png",x:1800,y:940,scale:0.8,visible:true}}}
    };
    const query = new URLSearchParams(location.search);
    const editorToken = query.get("recoEditorToken");
    let sport = specifications[query.get("sport")] ? query.get("sport") : "basketball";
    let state = structuredClone(defaults[sport]);
    let publishTimer = null;
    const elements = {
        select: document.querySelector("#sport-select"), connection: document.querySelector("#connection-status"), title: document.querySelector("#preview-title"),
        status: document.querySelector("#preview-status"), competition: document.querySelector("#preview-competition"), homeName: document.querySelector("#preview-home-name"),
        awayName: document.querySelector("#preview-away-name"), homeScore: document.querySelector("#preview-home-score"), awayScore: document.querySelector("#preview-away-score"),
        clock: document.querySelector("#preview-clock"), period: document.querySelector("#preview-period"), detail: document.querySelector("#preview-detail"),
        card: document.querySelector("#preview-card"), canvas: document.querySelector("#preview-canvas"), homeLogo: document.querySelector("#preview-home-logo"), awayLogo: document.querySelector("#preview-away-logo"), freeText: document.querySelector("#preview-free-text"), freeLogo: document.querySelector("#preview-free-logo"), legend: document.querySelector("#sport-legend"), fields: document.querySelector("#sport-fields"), scoreButtons: document.querySelector("#score-buttons"),
        form: document.querySelector("#state-form"), message: document.querySelector("#editor-message"), toggleEditing: document.querySelector("#toggle-editing"), toggleSettings: document.querySelector("#toggle-settings"), settingsPanel: document.querySelector("#settings-panel"), canvasToolbar: document.querySelector("#canvas-toolbar")
    };
    function pathGet(target, path) { return path.split(".").reduce((value, key) => value?.[key], target); }
    function pathSet(target, path, value) {
        const keys = path.split("."); const leaf = keys.pop(); const parent = keys.reduce((current, key) => current[key] ??= {}, target); parent[leaf] = value;
    }
    function inputValue(input) { return input.type === "checkbox" ? input.checked : input.type === "number" ? Number(input.value) || 0 : input.value; }
    function ensureCustomState() {
        state.custom ??= {};
        state.custom.scoreboard ??= { x: 960, y: sport === "soccer" ? 120 : 280, scale: 1 };
        state.custom.freeText ??= { text: "created with", x: 1640, y: 940, scale: 1, color: "#fff", visible: true };
        state.custom.freeLogo ??= { src: "assets/reco-logo.png", x: 1800, y: 940, scale: 0.8, visible: true };
    }
    function previewAsset(asset) { return asset && asset.startsWith("data:") ? asset : asset ? "../" + sport + "/" + asset : ""; }
    function setPreviewPosition(element, config, defaultX, defaultY, defaultScale) {
        const x = Number(config?.x) || defaultX; const y = Number(config?.y) || defaultY;
        const scale = Math.min(3, Math.max(0.5, Number(config?.scale) || defaultScale));
        element.style.left = x / 1920 * 100 + "%"; element.style.top = y / 1080 * 100 + "%";
        element.style.transform = "translate(-50%, 0) scale(" + scale + ")";
    }
    function renderPlacement() {
        ensureCustomState();
        setPreviewPosition(elements.card, state.custom.scoreboard, 960, sport === "soccer" ? 120 : 280, 1);
        setPreviewPosition(elements.freeText, state.custom.freeText, 1640, 940, 1);
        setPreviewPosition(elements.freeLogo, state.custom.freeLogo, 1800, 940, 0.8);
        elements.freeText.textContent = state.custom.freeText.text || ""; elements.freeText.hidden = state.custom.freeText.visible === false || !state.custom.freeText.text;
        elements.freeLogo.src = previewAsset(state.custom.freeLogo.src); elements.freeLogo.hidden = state.custom.freeLogo.visible === false || !state.custom.freeLogo.src;
    }
    function renderTeamLogos() {
        elements.homeLogo.src = previewAsset(state.home.logo); elements.homeLogo.hidden = !state.home.logo;
        elements.awayLogo.src = previewAsset(state.away.logo); elements.awayLogo.hidden = !state.away.logo;
    }
    
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
        ensureCustomState();
        const specification = specifications[sport]; elements.select.value = sport; elements.title.textContent = specification.title; elements.legend.textContent = specification.rules;
        elements.competition.textContent = state.game.competition || ""; elements.homeName.textContent = state.home.shortName || "HOME"; elements.awayName.textContent = state.away.shortName || "AWAY";
        elements.homeScore.textContent = Number(state.home.score) || 0; elements.awayScore.textContent = Number(state.away.score) || 0; elements.clock.textContent = state.game.clock || "00:00";
        elements.period.textContent = specification.period(state.game.period || 1, state.sport.periodCount || 1); elements.detail.textContent = specification.detail(state);
        renderPlacement(); renderTeamLogos();
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
        const previous = state; sport = specifications[nextSport] ? nextSport : "basketball"; state = structuredClone(defaults[sport]);
        state.game.competition = previous.game.competition; state.home.shortName = previous.home.shortName; state.home.name = previous.home.name; state.home.score = previous.home.score; state.home.color = previous.home.color;
        state.away.shortName = previous.away.shortName; state.away.name = previous.away.name; state.away.score = previous.away.score; state.away.color = previous.away.color;
        renderScoreButtons(); renderFields(); render(); publish();
    }
    function resetState() { state = structuredClone(defaults[sport]); render(); publish(); }
    function placementConfig(target) { ensureCustomState(); return target === "scoreboard" ? state.custom.scoreboard : target === "freeText" ? state.custom.freeText : state.custom.freeLogo; }
    function clamp(value, minimum, maximum) { return Math.min(maximum, Math.max(minimum, value)); }
    let dragState = null; let editingEnabled = true; let selectedTarget = "scoreboard"; let settingsVisible = true;
    function updateSettingsUi() { document.body.classList.toggle("settings-hidden", !settingsVisible); elements.toggleSettings.textContent = settingsVisible ? "Hide settings" : "Show settings"; }
    function updateEditingUi() { elements.canvas.classList.toggle("editing-hidden", !editingEnabled); elements.toggleEditing.textContent = editingEnabled ? "Hide editing guides" : "Show editing guides"; for (const button of elements.canvasToolbar.querySelectorAll("[data-select-target]")) button.classList.toggle("selected", button.dataset.selectTarget === selectedTarget); for (const element of [elements.card, elements.freeText, elements.freeLogo]) element.classList.toggle("is-selected", element.dataset.dragTarget === selectedTarget); }
    elements.toggleSettings.addEventListener("click", () => { settingsVisible = !settingsVisible; updateSettingsUi(); }); updateSettingsUi();
    elements.toggleEditing.addEventListener("click", () => { editingEnabled = !editingEnabled; updateEditingUi(); }); elements.canvasToolbar.addEventListener("click", (event) => { const button = event.target.closest("[data-select-target]"); if (!button) return; selectedTarget = button.dataset.selectTarget; updateEditingUi(); });
    elements.canvas.addEventListener("pointerdown", (event) => { const target = event.target.closest("[data-drag-target]") || (event.target.closest("#preview-card") ? event.target.closest("#preview-card") : null); if (!target || !editingEnabled) return; event.preventDefault(); selectedTarget = target.dataset.dragTarget || "scoreboard"; updateEditingUi(); dragState = { target: selectedTarget, rect: elements.canvas.getBoundingClientRect() }; });
    elements.canvas.addEventListener("pointermove", (event) => { if (!dragState || !editingEnabled) return; const config = placementConfig(dragState.target); config.x = Math.round(clamp((event.clientX - dragState.rect.left) / dragState.rect.width * 1920, 0, 1920)); config.y = Math.round(clamp((event.clientY - dragState.rect.top) / dragState.rect.height * 1080, 0, 1080)); renderPlacement(); renderInputs(); publish(); });
    elements.canvas.addEventListener("pointerup", () => { dragState = null; }); elements.canvas.addEventListener("pointercancel", () => { dragState = null; });
    elements.canvas.addEventListener("wheel", (event) => { const target = event.target.closest("[data-drag-target]") || (event.target.closest("#preview-card") ? event.target.closest("#preview-card") : null); if (!target || !editingEnabled) return; event.preventDefault(); const config = placementConfig(target.dataset.dragTarget || "scoreboard"); config.scale = Math.round(clamp((Number(config.scale) || 1) + (event.deltaY < 0 ? 0.05 : -0.05), 0.5, 3) * 100) / 100; renderPlacement(); renderInputs(); publish(); });
    
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
            sport = specifications[published.sport?.sportId] ? published.sport.sportId : published.sport?.shotClock !== undefined ? "basketball" : "soccer"; state = published; renderFields(); render(); elements.connection.textContent = "Loaded from Reco";
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
        const foulButton = event.target.closest("button[data-foul-team]"); if (foulButton) { const key = foulButton.dataset.foulTeam === "home" ? "teamFoulsHome" : "teamFoulsAway"; state.sport[key] = Math.max(0, Number(state.sport[key] || 0) + 1); render(); publish(); return; }
        const button = event.target.closest("button[data-team]"); if (!button) return; state[button.dataset.team].score = Math.max(0, Number(state[button.dataset.team].score) + Number(button.dataset.amount)); render(); publish();
    });
    renderScoreButtons(); renderFields(); render(); updateEditingUi(); void loadPublished(); RecoScoreboard.update(state); Reco.ready();
})();
