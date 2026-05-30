//! Filter building blocks for online trackers.
//!
//! Each filter is a self-contained component that a [`Tracker`]
//! implementation composes into its update loop. Filters are stateful
//! but stateless between independent tracker instances — you can run
//! multiple trackers sharing no state.
//!
//! - [`Coaster`] holds a last-known position for up to N frames
//!   after detection is lost, then declares the track as lost.
//!
//! [`Tracker`]: reco_core::detect::tracker::Tracker

mod coaster;

pub use coaster::{CoastStatus, Coaster};
