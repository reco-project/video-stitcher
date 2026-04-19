//! Mobile-app transport scaffold.
//!
//! Placeholder for future Android / iOS companion-app control
//! support. The intent: a mobile app pairs with a running Reco
//! instance over the local network (mDNS discovery + WebRTC data
//! channel) and lets operators drive pose / quality / capture
//! intents from a phone or tablet while standing on the field.
//!
//! Intended shape:
//!
//! - mDNS service advertisement (`_reco-control._tcp.local`).
//! - WebRTC data channel for low-latency intent delivery.
//! - Protobuf or JSON on the wire — decided at first-impl time.
//! - The app itself is out of scope for this repo; a separate
//!   mobile project talks to this transport.
//!
//! Today the module is empty — the feature gate exists so the
//! feature-combo CI matrix exercises it and future work has an
//! obvious target path.

use crate::{ControlIntent, ControlTransport};

/// Placeholder mobile transport. Returns no intents and logs a
/// warning on instantiation.
pub struct MobileTransport {
    _private: (),
}

impl MobileTransport {
    /// Begin advertising on the local network. Not yet
    /// implemented — returns a stub transport that never emits
    /// intents.
    pub fn advertise() -> Self {
        log::warn!(
            "reco-control: MobileTransport::advertise is a stub; \
             mDNS + WebRTC support lands in a future tranche"
        );
        Self { _private: () }
    }
}

impl ControlTransport for MobileTransport {
    fn name(&self) -> &'static str {
        "mobile-stub"
    }

    fn poll(&mut self, _out: &mut Vec<ControlIntent>) -> usize {
        0
    }
}
