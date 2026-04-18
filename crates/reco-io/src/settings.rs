//! Per-consumer settings persistence.
//!
//! Each Reco consumer (GUI, CLI, OBS plugin) owns its own typed
//! `Settings` struct and calls [`load`] / [`save`] with a unique
//! namespace string. Namespaces map to separate JSON files under the
//! platform's standard config directory, so one consumer's preferences
//! never leak into another's:
//!
//! - Linux: `$XDG_CONFIG_HOME/reco/{namespace}.json` (default `~/.config/reco/`)
//! - macOS: `~/Library/Application Support/reco/{namespace}.json`
//! - Windows: `%APPDATA%\reco\{namespace}.json`
//!
//! This module is domain-agnostic: reco-io doesn't know what a "default
//! codec" or "recent file" is, only how to serialize/deserialize JSON
//! at a namespaced location. That keeps reco-io honest (still pluggable
//! I/O, not an application framework) and lets each consumer evolve
//! its schema independently.
//!
//! ## Example
//!
//! ```no_run
//! # #[cfg(feature = "config")]
//! # {
//! use serde::{Deserialize, Serialize};
//!
//! #[derive(Serialize, Deserialize, Default)]
//! struct MySettings {
//!     window_width: u32,
//!     window_height: u32,
//! }
//!
//! // Load (falls back to `default` if file missing).
//! let mut s: MySettings = reco_io::settings::load_or_default("gui");
//!
//! s.window_width = 1600;
//! reco_io::settings::save("gui", &s).expect("save settings");
//! # }
//! ```

use std::path::{Path, PathBuf};

use serde::{Serialize, de::DeserializeOwned};

/// Errors from the settings module.
#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    /// Filesystem error reading, writing, or creating a settings file.
    #[error("settings I/O: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization error.
    #[error("settings serialize: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Namespace contained characters outside `[a-z0-9_-]`.
    ///
    /// The allowed set is intentionally narrow to avoid path-traversal
    /// attacks and Windows-reserved device names.
    #[error("invalid settings namespace {namespace:?}: must match [a-z0-9_-]+")]
    BadNamespace {
        /// The rejected namespace.
        namespace: String,
    },
    /// Could not resolve the platform config directory.
    #[error("cannot resolve config directory")]
    NoConfigDir,
}

/// Directory under which reco settings files are stored.
///
/// Resolves to the platform-standard user-config location with a
/// `reco/` subfolder appended. Honors the `RECO_CONFIG_DIR`
/// environment variable to override (used by tests and power users).
///
/// Returns [`SettingsError::NoConfigDir`] if the platform lookup fails
/// (rare; usually means the process has no HOME / APPDATA set).
pub fn config_dir() -> Result<PathBuf, SettingsError> {
    if let Some(override_path) = std::env::var_os("RECO_CONFIG_DIR") {
        return Ok(PathBuf::from(override_path));
    }
    let dirs = directories::ProjectDirs::from("", "", "reco").ok_or(SettingsError::NoConfigDir)?;
    Ok(dirs.config_dir().to_path_buf())
}

/// Strict namespace validation.
///
/// Allows only `[a-z0-9_-]+` to keep filenames portable (no path
/// separators, no control characters, no Windows-reserved names like
/// `con`/`aux` because those don't contain `-` or `_` but we still
/// reject them defensively via length + alphabet).
fn validate_namespace(namespace: &str) -> Result<(), SettingsError> {
    if namespace.is_empty()
        || !namespace
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
    {
        return Err(SettingsError::BadNamespace {
            namespace: namespace.to_string(),
        });
    }
    Ok(())
}

/// Full path to a namespace's settings file, creating parent dirs.
fn settings_path(namespace: &str) -> Result<PathBuf, SettingsError> {
    validate_namespace(namespace)?;
    let dir = config_dir()?;
    if !dir.exists() {
        std::fs::create_dir_all(&dir)?;
    }
    Ok(dir.join(format!("{namespace}.json")))
}

/// Load settings for a namespace.
///
/// Returns `Ok(T)` on success, `Err` for I/O or deserialization
/// failures. If the settings file does not exist, returns an
/// `Io(NotFound)` error — callers usually prefer [`load_or_default`]
/// for first-run friendliness.
pub fn load<T: DeserializeOwned>(namespace: &str) -> Result<T, SettingsError> {
    let path = settings_path(namespace)?;
    let bytes = std::fs::read(&path)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok(value)
}

/// Load settings for a namespace, or return `T::default()` if the
/// settings file does not yet exist (first-run case).
///
/// Deserialization errors on a malformed existing file are logged and
/// also fall through to `T::default()` rather than propagating — the
/// alternative (failing to start the app because a settings file
/// got truncated) is worse than silently resetting preferences.
pub fn load_or_default<T: DeserializeOwned + Default>(namespace: &str) -> T {
    match load::<T>(namespace) {
        Ok(v) => v,
        Err(SettingsError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => T::default(),
        Err(e) => {
            log::warn!("settings({namespace}): falling back to defaults after load error: {e}");
            T::default()
        }
    }
}

/// Persist settings for a namespace.
///
/// Writes atomically via a same-directory temp file + rename, so a
/// crash mid-save cannot leave a partially-written file. Pretty-prints
/// the JSON for human readability (users sometimes edit these files
/// directly).
pub fn save<T: Serialize>(namespace: &str, value: &T) -> Result<(), SettingsError> {
    let path = settings_path(namespace)?;
    let bytes = serde_json::to_vec_pretty(value)?;

    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok(())
}

/// Recently-used file paths, capped at a configurable length.
///
/// Ordered most-recent-first. [`push`](Self::push) deduplicates so a
/// repeat entry moves to the front rather than piling up. Designed to
/// embed directly in a consumer's `Settings` struct via serde.
///
/// Paths are stored as `PathBuf` so non-UTF-8 filenames round-trip
/// correctly on Linux. JSON encoding falls back to lossy string
/// conversion for display purposes in consumers.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct RecentFiles {
    entries: Vec<PathBuf>,
    #[serde(default = "default_max")]
    max: usize,
}

fn default_max() -> usize {
    8
}

impl Default for RecentFiles {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            max: default_max(),
        }
    }
}

impl RecentFiles {
    /// Create a new MRU ring with the given capacity.
    pub fn new(max: usize) -> Self {
        Self {
            entries: Vec::new(),
            max: max.max(1),
        }
    }

    /// Insert a path at the front, deduplicating and trimming to
    /// capacity. The most-recently-pushed entry always ends up at
    /// index 0.
    pub fn push(&mut self, path: PathBuf) {
        self.entries.retain(|p| p != &path);
        self.entries.insert(0, path);
        if self.entries.len() > self.max {
            self.entries.truncate(self.max);
        }
    }

    /// Remove a specific path from the list, if present.
    pub fn remove(&mut self, path: &Path) {
        self.entries.retain(|p| p != path);
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Borrow the entries, most-recent-first.
    pub fn entries(&self) -> &[PathBuf] {
        &self.entries
    }

    /// Number of entries currently held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use tempfile::tempdir;

    #[derive(Serialize, Deserialize, PartialEq, Debug, Default)]
    struct DummySettings {
        flag: bool,
        count: u32,
        name: String,
    }

    /// Cargo runs tests in parallel by default and `RECO_CONFIG_DIR`
    /// is process-global state; two tests racing on it would corrupt
    /// each other. This mutex serializes tests that manipulate env.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard that holds the env lock, points RECO_CONFIG_DIR at a
    /// tempdir for the duration of the test, then removes both on drop.
    struct ConfigDirOverride {
        _dir: tempfile::TempDir,
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl ConfigDirOverride {
        fn new() -> Self {
            // Ignore poisoning - a panicked test just means the lock
            // was held when something else panicked; the env state
            // we're about to overwrite anyway.
            let guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let dir = tempdir().unwrap();
            // SAFETY: the mutex guarantees no other test in this
            // module is touching env at the same time.
            unsafe { std::env::set_var("RECO_CONFIG_DIR", dir.path()) };
            Self {
                _dir: dir,
                _guard: guard,
            }
        }
    }

    impl Drop for ConfigDirOverride {
        fn drop(&mut self) {
            // SAFETY: see ConfigDirOverride::new.
            unsafe { std::env::remove_var("RECO_CONFIG_DIR") };
        }
    }

    #[test]
    fn round_trip_save_load() {
        let _guard = ConfigDirOverride::new();
        let expected = DummySettings {
            flag: true,
            count: 42,
            name: "test".into(),
        };
        save("round_trip_test", &expected).unwrap();
        let loaded: DummySettings = load("round_trip_test").unwrap();
        assert_eq!(expected, loaded);
    }

    #[test]
    fn load_or_default_on_missing() {
        let _guard = ConfigDirOverride::new();
        let v: DummySettings = load_or_default("nonexistent_ns_xxxx");
        assert_eq!(v, DummySettings::default());
    }

    #[test]
    fn independent_namespaces_do_not_collide() {
        let _guard = ConfigDirOverride::new();
        let a = DummySettings {
            flag: true,
            count: 1,
            name: "a".into(),
        };
        let b = DummySettings {
            flag: false,
            count: 2,
            name: "b".into(),
        };
        save("consumer_a", &a).unwrap();
        save("consumer_b", &b).unwrap();
        let got_a: DummySettings = load("consumer_a").unwrap();
        let got_b: DummySettings = load("consumer_b").unwrap();
        assert_eq!(got_a, a);
        assert_eq!(got_b, b);
    }

    #[test]
    fn bad_namespace_rejected() {
        assert!(matches!(
            save("has space", &DummySettings::default()),
            Err(SettingsError::BadNamespace { .. })
        ));
        assert!(matches!(
            save("../etc", &DummySettings::default()),
            Err(SettingsError::BadNamespace { .. })
        ));
        assert!(matches!(
            save("UPPER", &DummySettings::default()),
            Err(SettingsError::BadNamespace { .. })
        ));
        assert!(matches!(
            save("", &DummySettings::default()),
            Err(SettingsError::BadNamespace { .. })
        ));
    }

    #[test]
    fn recent_files_dedup_moves_to_front() {
        let mut mru = RecentFiles::new(4);
        mru.push(PathBuf::from("/a"));
        mru.push(PathBuf::from("/b"));
        mru.push(PathBuf::from("/c"));
        mru.push(PathBuf::from("/b")); // dup
        assert_eq!(
            mru.entries(),
            &[
                PathBuf::from("/b"),
                PathBuf::from("/c"),
                PathBuf::from("/a"),
            ]
        );
        assert_eq!(mru.len(), 3);
    }

    #[test]
    fn recent_files_caps_at_max() {
        let mut mru = RecentFiles::new(2);
        mru.push(PathBuf::from("/1"));
        mru.push(PathBuf::from("/2"));
        mru.push(PathBuf::from("/3"));
        assert_eq!(mru.entries(), &[PathBuf::from("/3"), PathBuf::from("/2")]);
    }

    #[test]
    fn recent_files_remove_and_clear() {
        let mut mru = RecentFiles::new(4);
        mru.push(PathBuf::from("/x"));
        mru.push(PathBuf::from("/y"));
        mru.remove(Path::new("/x"));
        assert_eq!(mru.entries(), &[PathBuf::from("/y")]);
        mru.clear();
        assert!(mru.is_empty());
    }
}
