//! Toast notification state for the GUI.
//!
//! Maintains the set of currently-visible toast cards and their
//! expiration timestamps. The main playback timer calls `expire` each
//! cycle to drop expired entries and push the updated list into Slint.
//!
//! Toasts are keyed by a monotonic `id` so the UI can dismiss one by
//! its handle without the Rust state and Slint model drifting on
//! insertion order.

use std::time::{Duration, Instant};

use slint::{ModelRc, SharedString, VecModel};

use crate::Toast;

/// Severity categorizes the toast's visual accent and default lifetime.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warn,
    Error,
}

impl Severity {
    /// Lowercase string form matching the Slint `toast.severity` field.
    fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }

    /// How long a toast of this severity stays visible by default.
    /// Errors linger the longest so the user can actually read them.
    fn default_ttl(&self) -> Duration {
        match self {
            Self::Info => Duration::from_secs(4),
            Self::Warn => Duration::from_secs(7),
            Self::Error => Duration::from_secs(10),
        }
    }
}

/// One live toast with its expiration timestamp.
#[derive(Debug, Clone)]
struct Entry {
    id: i32,
    severity: Severity,
    title: String,
    body: String,
    expires_at: Instant,
}

/// Toast stack state. Owned by `AppState`.
pub struct ToastManager {
    entries: Vec<Entry>,
    next_id: i32,
    /// Hard cap on visible toasts. Oldest gets evicted when we exceed.
    max_visible: usize,
}

impl Default for ToastManager {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            next_id: 1,
            max_visible: 4,
        }
    }
}

impl ToastManager {
    /// Push a toast with the default lifetime for its severity.
    /// Returns the assigned id so the caller can dismiss programmatically.
    pub fn push(
        &mut self,
        severity: Severity,
        title: impl Into<String>,
        body: impl Into<String>,
    ) -> i32 {
        self.push_with_ttl(severity, title, body, severity.default_ttl())
    }

    /// Push a toast with an explicit lifetime.
    pub fn push_with_ttl(
        &mut self,
        severity: Severity,
        title: impl Into<String>,
        body: impl Into<String>,
        ttl: Duration,
    ) -> i32 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        if self.next_id == 0 {
            // 0 would collide with "uninitialized" sentinels in a
            // future redesign; skip it defensively.
            self.next_id = 1;
        }

        let entry = Entry {
            id,
            severity,
            title: title.into(),
            body: body.into(),
            expires_at: Instant::now() + ttl,
        };
        self.entries.push(entry);

        // Evict oldest if we exceed the visible cap.
        if self.entries.len() > self.max_visible {
            let overflow = self.entries.len() - self.max_visible;
            self.entries.drain(..overflow);
        }
        id
    }

    /// Remove a toast by id (no-op if it isn't in the list).
    pub fn dismiss(&mut self, id: i32) {
        self.entries.retain(|e| e.id != id);
    }

    /// Drop any entries past their expiration. Returns `true` if the
    /// list changed, so the caller knows whether to push the new model
    /// back into Slint.
    pub fn expire(&mut self, now: Instant) -> bool {
        let before = self.entries.len();
        self.entries.retain(|e| e.expires_at > now);
        self.entries.len() != before
    }

    /// Build a fresh Slint model reflecting current state. Cheap - we
    /// only call this when the list actually changed (see `expire` and
    /// the push/dismiss callers).
    pub fn to_model(&self) -> ModelRc<Toast> {
        let toasts: Vec<Toast> = self
            .entries
            .iter()
            .map(|e| Toast {
                id: e.id,
                severity: SharedString::from(e.severity.as_str()),
                title: SharedString::from(e.title.as_str()),
                body: SharedString::from(e.body.as_str()),
            })
            .collect();
        ModelRc::new(VecModel::from(toasts))
    }

    /// Whether there are any live toasts. Useful to skip the periodic
    /// expiration check when the stack is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// For test + debug: number of currently-live toasts.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Push the current toast list into the Slint model property. Called
/// after any operation that changed the list.
pub fn sync_to_ui(manager: &ToastManager, app: &crate::RecoApp) {
    app.set_toasts(manager.to_model());
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::Model;

    #[test]
    fn push_increments_id_monotonically() {
        let mut m = ToastManager::default();
        let a = m.push(Severity::Info, "A", "");
        let b = m.push(Severity::Info, "B", "");
        let c = m.push(Severity::Info, "C", "");
        assert_eq!((a, b, c), (1, 2, 3));
        assert_eq!(m.len(), 3);
    }

    #[test]
    fn dismiss_removes_by_id() {
        let mut m = ToastManager::default();
        let a = m.push(Severity::Info, "A", "");
        let _b = m.push(Severity::Info, "B", "");
        m.dismiss(a);
        assert_eq!(m.len(), 1);
        // Dismissing something not in the list is fine.
        m.dismiss(999);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn expire_drops_past_ttl() {
        let mut m = ToastManager::default();
        let _a = m.push_with_ttl(Severity::Info, "quick", "", Duration::from_millis(1));
        let _b = m.push_with_ttl(Severity::Info, "slow", "", Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(20));
        let changed = m.expire(Instant::now());
        assert!(changed);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn cap_evicts_oldest() {
        let mut m = ToastManager {
            max_visible: 2,
            ..Default::default()
        };
        m.push(Severity::Info, "1", "");
        m.push(Severity::Info, "2", "");
        m.push(Severity::Info, "3", "");
        assert_eq!(m.len(), 2);
        let model = m.to_model();
        assert_eq!(model.row_count(), 2);
        assert_eq!(model.row_data(0).unwrap().title.as_str(), "2");
        assert_eq!(model.row_data(1).unwrap().title.as_str(), "3");
    }
}
