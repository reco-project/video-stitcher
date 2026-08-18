//! Package discovery independent of the current working directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::manifest::ScoreboardPackage;

/// One package or directory that could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscoveryIssue {
    /// Path that caused the issue.
    pub path: PathBuf,
    /// User-readable reason.
    pub message: String,
}

/// Valid packages plus all skipped-package diagnostics.
#[derive(Clone, Debug, Default)]
pub struct DiscoveryReport {
    /// Valid packages sorted by display name.
    pub packages: Vec<ScoreboardPackage>,
    /// Invalid packages and discovery problems.
    pub issues: Vec<DiscoveryIssue>,
}

/// Candidate installation locations ordered from explicit override to bundled
/// application data. Existing paths are de-duplicated by discovery.
pub fn default_scoreboard_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(value) = std::env::var_os("RECO_SCOREBOARDS_DIR") {
        roots.extend(std::env::split_paths(&value));
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(bin_dir) = executable.parent()
    {
        roots.push(bin_dir.join("scoreboards"));
        // Conventional macOS .app layout: Contents/MacOS -> Contents/Resources.
        roots.push(bin_dir.join("../Resources/scoreboards"));
    }
    if let Some(data_dir) = platform_data_dir() {
        roots.push(data_dir.join("reco/scoreboards"));
    }
    #[cfg(debug_assertions)]
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scoreboards"));
    roots
}

/// Discover packages in all installed locations.
pub fn discover_installed() -> DiscoveryReport {
    discover(default_scoreboard_roots())
}

/// Discover direct child packages under the provided roots.
pub fn discover(roots: impl IntoIterator<Item = PathBuf>) -> DiscoveryReport {
    let roots: Vec<PathBuf> = roots.into_iter().collect();
    let existing: Vec<PathBuf> = roots.iter().filter(|path| path.is_dir()).cloned().collect();
    if existing.is_empty() {
        return DiscoveryReport {
            packages: Vec::new(),
            issues: vec![DiscoveryIssue {
                path: PathBuf::new(),
                message: format!(
                    "scoreboards directory not found; searched {}",
                    roots
                        .iter()
                        .map(|path| path.display().to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            }],
        };
    }

    let mut report = DiscoveryReport::default();
    let mut ids: HashMap<String, PathBuf> = HashMap::new();
    let mut seen_roots = Vec::new();
    for root in existing {
        let canonical = root.canonicalize().unwrap_or(root.clone());
        if seen_roots.contains(&canonical) {
            continue;
        }
        seen_roots.push(canonical);
        discover_root(&root, &mut ids, &mut report);
    }
    report
        .packages
        .sort_by(|left, right| left.manifest.name.cmp(&right.manifest.name));
    report
}

fn discover_root(root: &Path, ids: &mut HashMap<String, PathBuf>, report: &mut DiscoveryReport) {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) => {
            report.issues.push(DiscoveryIssue {
                path: root.to_path_buf(),
                message: format!("cannot read scoreboards directory: {error}"),
            });
            return;
        }
    };
    let mut directories: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_dir()
                && !matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some("sdk" | "examples")
                )
        })
        .collect();
    directories.sort();

    for directory in directories {
        match ScoreboardPackage::load(&directory) {
            Ok(package) => {
                if let Some(first_path) = ids.get(&package.manifest.id) {
                    report.issues.push(DiscoveryIssue {
                        path: directory,
                        message: format!(
                            "duplicate scoreboard id {:?}; already provided by {}",
                            package.manifest.id,
                            first_path.display()
                        ),
                    });
                    continue;
                }
                ids.insert(package.manifest.id.clone(), package.directory.clone());
                report.packages.push(package);
            }
            Err(error) => report.issues.push(DiscoveryIssue {
                path: directory,
                message: error.to_string(),
            }),
        }
    }
}

#[cfg(target_os = "windows")]
fn platform_data_dir() -> Option<PathBuf> {
    std::env::var_os("APPDATA").map(PathBuf::from)
}

#[cfg(target_os = "macos")]
fn platform_data_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library/Application Support"))
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_data_dir() -> Option<PathBuf> {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|home| home.join(".local/share"))
        })
}

#[cfg(not(any(unix, target_os = "windows")))]
fn platform_data_dir() -> Option<PathBuf> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_package(root: &Path, id: &str, extra: &str) {
        let directory = root.join(id);
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("index.html"), "<!doctype html>").unwrap();
        std::fs::write(
            directory.join("manifest.json"),
            format!(
                r#"{{
                    "schemaVersion": 1,
                    "id": "{id}",
                    "name": "{id}",
                    "sport": "{id}",
                    "version": "1.0.0",
                    "entry": "index.html",
                    "viewport": {{"width": 1920, "height": 1080}},
                    "updateApiVersion": 1
                    {extra}
                }}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn valid_manifest_and_unknown_optional_field_are_accepted() {
        let temp = tempfile::tempdir().unwrap();
        write_package(temp.path(), "basketball", r#", "futureOption": true"#);
        let report = discover([temp.path().to_path_buf()]);
        assert_eq!(report.packages.len(), 1);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn missing_required_property_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        let directory = temp.path().join("broken");
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("index.html"), "").unwrap();
        std::fs::write(directory.join("manifest.json"), r#"{"schemaVersion":1}"#).unwrap();
        let report = discover([temp.path().to_path_buf()]);
        assert!(report.packages.is_empty());
        assert_eq!(report.issues.len(), 1);
    }

    #[test]
    fn missing_manifest_is_reported() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("broken")).unwrap();
        let report = discover([temp.path().to_path_buf()]);
        assert!(report.packages.is_empty());
        assert_eq!(report.issues.len(), 1);
        assert!(report.issues[0].message.contains("cannot read manifest"));
    }

    #[test]
    fn missing_entry_and_unsupported_schema_are_rejected() {
        let temp = tempfile::tempdir().unwrap();
        write_package(temp.path(), "missing", "");
        std::fs::remove_file(temp.path().join("missing/index.html")).unwrap();
        write_package(temp.path(), "future", "");
        let manifest_path = temp.path().join("future/manifest.json");
        let json = std::fs::read_to_string(&manifest_path)
            .unwrap()
            .replace("\"schemaVersion\": 1", "\"schemaVersion\": 2");
        std::fs::write(manifest_path, json).unwrap();
        let report = discover([temp.path().to_path_buf()]);
        assert!(report.packages.is_empty());
        assert_eq!(report.issues.len(), 2);
    }

    #[test]
    fn duplicate_id_is_skipped_and_new_package_is_automatic() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write_package(first.path(), "basketball", "");
        write_package(second.path(), "basketball", "");
        write_package(second.path(), "football", "");
        let report = discover([first.path().to_path_buf(), second.path().to_path_buf()]);
        assert_eq!(report.packages.len(), 2);
        assert_eq!(report.issues.len(), 1);
        assert!(report.packages.iter().any(|p| p.manifest.id == "football"));
    }

    #[test]
    fn missing_directory_is_reported() {
        let report = discover([PathBuf::from("/definitely/not/a/reco/scoreboards/path")]);
        assert!(report.packages.is_empty());
        assert_eq!(report.issues.len(), 1);
        assert!(report.issues[0].message.contains("not found"));
    }

    #[test]
    fn bundled_basketball_is_discovered() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scoreboards");
        let report = discover([root]);
        assert!(report.issues.is_empty(), "{:?}", report.issues);
        assert!(
            report
                .packages
                .iter()
                .any(|package| package.manifest.id == "basketball")
        );
    }
}
