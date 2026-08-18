(function installRecoScoreboardSdk(global) {
    "use strict";

    const listeners = new Set();
    let state = null;
    let context = null;

    function markDirty() {
        if (typeof global.__recoMarkDirty === "function") {
            global.__recoMarkDirty();
        }
    }

    global.RecoScoreboard = {
        apiVersion: 1,

        init(nextContext) {
            context = nextContext;
            markDirty();
        },

        update(nextState) {
            state = nextState;
            for (const listener of listeners) {
                try {
                    listener(nextState);
                } catch (error) {
                    console.error("[Reco scoreboard] update listener failed", error);
                    global.__recoLastError = error instanceof Error ? error.message : String(error);
                }
            }
            markDirty();
        },

        reset() {
            state = null;
            markDirty();
        },

        destroy() {
            listeners.clear();
            state = null;
            context = null;
        }
    };

    global.Reco = Object.freeze({
        version: 1,

        onUpdate(callback) {
            if (typeof callback !== "function") {
                throw new TypeError("Reco.onUpdate requires a function");
            }
            listeners.add(callback);
            if (state !== null) {
                callback(state);
            }
            return () => listeners.delete(callback);
        },

        getState() {
            return state;
        },

        getContext() {
            return context;
        },

        ready() {
            global.dispatchEvent(new CustomEvent("reco-scoreboard-ready"));
            markDirty();
        },

        log(message) {
            console.log("[Reco scoreboard]", message);
        }
    });
})(window);
