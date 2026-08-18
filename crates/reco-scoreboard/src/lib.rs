//! Generic HTML scoreboard packages and asynchronous rendering.
//!
//! This crate owns filesystem discovery, manifest validation, and the browser
//! process. Sport rules and fields remain inside each package's HTML/JS.

mod discovery;
mod local_server;
mod manifest;
mod runtime;

pub use discovery::{
    DiscoveryIssue, DiscoveryReport, default_scoreboard_roots, discover_installed,
};
pub use manifest::{ScoreboardManifest, ScoreboardPackage, ScoreboardViewport};
pub use runtime::{RuntimeError, ScoreboardRuntime};
