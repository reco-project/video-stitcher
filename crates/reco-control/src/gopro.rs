//! GoPro transport scaffold.
//!
//! Placeholder for future GoPro USB-HID / OpenGoPro BLE / Wi-Fi
//! control support. When implemented, this module exposes a
//! [`GoProTransport`] that reads camera state (pan/tilt from a
//! gimbal, button events from the front panel, model-select
//! commands from the smartphone companion app) and emits
//! [`crate::ControlIntent`] values.
//!
//! Intended integration points:
//!
//! - OpenGoPro BLE protocol (discovery, command, query, streaming
//!   status).
//! - USB-HID device interface for wired control sessions.
//! - Wi-Fi HTTP REST API for older cameras without OpenGoPro
//!   support.
//!
//! Today the module is empty — feature gate exists so the
//! feature-combo CI matrix exercises it and so future work has an
//! obvious target path without restructuring the crate.

use crate::{ControlIntent, ControlTransport};

/// Placeholder GoPro transport. Returns no intents and panics on
/// any attempt to initialize the real device connection.
pub struct GoProTransport {
    _private: (),
}

impl GoProTransport {
    /// Connect to a GoPro. Not yet implemented — returns a stub
    /// transport that never emits intents.
    pub fn connect() -> Self {
        log::warn!(
            "reco-control: GoProTransport::connect is a stub; \
             OpenGoPro BLE / USB-HID / REST support lands in a future tranche"
        );
        Self { _private: () }
    }
}

impl ControlTransport for GoProTransport {
    fn name(&self) -> &'static str {
        "gopro-stub"
    }

    fn poll(&mut self, _out: &mut Vec<ControlIntent>) -> usize {
        0
    }
}
