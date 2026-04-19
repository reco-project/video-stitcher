//! WebSocket transport scaffold.
//!
//! Placeholder for future WebSocket-based remote control. Intended
//! use cases:
//!
//! - Browser-based operator dashboard (panel UI separate from the
//!   Reco binary, hosted on a laptop at the broadcast truck).
//! - Cloud orchestration (auto-director calling back through a
//!   WebSocket to drive pose intents based on analytics).
//! - Dev-tools harness for replaying recorded intent streams.
//!
//! Intended shape:
//!
//! - `tokio-tungstenite` server embedded in the Reco binary when
//!   the feature is enabled.
//! - JSON wire format matching the `ControlIntent` enum
//!   (serde-derived) — sticks to one format for simplicity.
//! - Authentication via pre-shared secret or OAuth; decided at
//!   first-impl time.
//!
//! Today the module is empty — the feature gate exists so the
//! feature-combo CI matrix exercises it and future work has an
//! obvious target path.

use crate::{ControlIntent, ControlTransport};

/// Placeholder WebSocket transport. Returns no intents and logs a
/// warning on instantiation.
pub struct WebSocketTransport {
    _private: (),
}

impl WebSocketTransport {
    /// Start listening on the given bind address. Not yet
    /// implemented — returns a stub transport that never emits
    /// intents.
    pub fn listen(_bind_addr: &str) -> Self {
        log::warn!(
            "reco-control: WebSocketTransport::listen is a stub; \
             tokio-tungstenite support lands in a future tranche"
        );
        Self { _private: () }
    }
}

impl ControlTransport for WebSocketTransport {
    fn name(&self) -> &'static str {
        "websocket-stub"
    }

    fn poll(&mut self, _out: &mut Vec<ControlIntent>) -> usize {
        0
    }
}
