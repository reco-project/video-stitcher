//! Scoreboard manifest schema and package validation.

use std::path::{Component, Path, PathBuf};

use serde::Deserialize;
use thiserror::Error;

/// Current package manifest schema understood by Reco.
pub const SUPPORTED_SCHEMA_VERSION: u32 = 1;
/// Current JavaScript update contract understood by Reco.
pub const SUPPORTED_UPDATE_API_VERSION: u32 = 1;

/// Reference canvas dimensions declared by a package.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct ScoreboardViewport {
    /// Reference width in CSS/device pixels.
    pub width: u32,
    /// Reference height in CSS/device pixels.
    pub height: u32,
}

/// `manifest.json` contract for a community scoreboard package.
///
/// Unknown fields are intentionally accepted for forward-compatible optional
/// metadata. Required fields have no serde defaults, so omissions are errors.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ScoreboardManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Unique filesystem-safe package identifier.
    pub id: String,
    /// User-facing display name.
    pub name: String,
    /// Sport identifier. Reco does not interpret this value.
    pub sport: String,
    /// Package semantic version.
    pub version: String,
    /// Optional package author.
    pub author: Option<String>,
    /// Optional user-facing description.
    pub description: Option<String>,
    /// Relative path to the HTML entry point.
    pub entry: String,
    /// Optional relative HTML editor entry point, including a query string.
    pub editor: Option<String>,
    /// Reference HTML viewport.
    pub viewport: ScoreboardViewport,
    /// Whether the package expects a transparent canvas.
    #[serde(default)]
    pub transparent: bool,
    /// JavaScript update contract version.
    pub update_api_version: u32,
}

/// A validated, loadable package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreboardPackage {
    /// Validated package metadata.
    pub manifest: ScoreboardManifest,
    /// Absolute package directory.
    pub directory: PathBuf,
    /// Absolute, contained HTML entry point.
    pub entry_path: PathBuf,
    /// Absolute, contained HTML editor entry point when declared.
    pub editor_path: Option<PathBuf>,
}

impl ScoreboardPackage {
    /// Load and validate one package directory.
    pub fn load(directory: &Path) -> Result<Self, ManifestError> {
        let manifest_path = directory.join("manifest.json");
        let json = std::fs::read_to_string(&manifest_path).map_err(|source| {
            ManifestError::ReadManifest {
                path: manifest_path.clone(),
                source,
            }
        })?;
        let manifest: ScoreboardManifest =
            serde_json::from_str(&json).map_err(|source| ManifestError::InvalidJson {
                path: manifest_path,
                source,
            })?;
        manifest.validate()?;

        let directory_name = directory
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if directory_name != manifest.id {
            return Err(ManifestError::DirectoryIdMismatch {
                directory: directory_name.to_string(),
                id: manifest.id,
            });
        }

        let canonical_directory =
            directory
                .canonicalize()
                .map_err(|source| ManifestError::ReadPackage {
                    path: directory.to_path_buf(),
                    source,
                })?;
        let canonical_entry = validate_html_target(
            directory,
            &canonical_directory,
            &manifest.entry,
            TargetKind::Entry,
        )?;
        let editor_path = manifest
            .editor
            .as_deref()
            .map(|target| {
                validate_html_target(directory, &canonical_directory, target, TargetKind::Editor)
            })
            .transpose()?;

        Ok(Self {
            manifest,
            directory: canonical_directory,
            entry_path: canonical_entry,
            editor_path,
        })
    }
}

#[derive(Clone, Copy)]
enum TargetKind {
    Entry,
    Editor,
}

fn validate_html_target(
    directory: &Path,
    canonical_directory: &Path,
    target: &str,
    kind: TargetKind,
) -> Result<PathBuf, ManifestError> {
    let path_text = target.split(['?', '#']).next().unwrap_or_default();
    let relative = Path::new(path_text);
    if relative.as_os_str().is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
        || relative.extension().and_then(|ext| ext.to_str()) != Some("html")
    {
        return Err(match kind {
            TargetKind::Entry => ManifestError::UnsafeEntry(target.to_owned()),
            TargetKind::Editor => ManifestError::UnsafeEditor(target.to_owned()),
        });
    }

    let candidate = directory.join(relative);
    let canonical = candidate.canonicalize().map_err(|source| match kind {
        TargetKind::Entry => ManifestError::MissingEntry {
            path: candidate.clone(),
            source,
        },
        TargetKind::Editor => ManifestError::MissingEditor {
            path: candidate.clone(),
            source,
        },
    })?;
    if !canonical.starts_with(canonical_directory) || !canonical.is_file() {
        return Err(match kind {
            TargetKind::Entry => ManifestError::EntryOutsidePackage(canonical),
            TargetKind::Editor => ManifestError::EditorOutsidePackage(canonical),
        });
    }
    Ok(canonical)
}

impl ScoreboardManifest {
    fn validate(&self) -> Result<(), ManifestError> {
        if self.schema_version != SUPPORTED_SCHEMA_VERSION {
            return Err(ManifestError::UnsupportedSchema(self.schema_version));
        }
        if self.update_api_version != SUPPORTED_UPDATE_API_VERSION {
            return Err(ManifestError::UnsupportedUpdateApi(self.update_api_version));
        }
        if !is_safe_id(&self.id) {
            return Err(ManifestError::UnsafeId(self.id.clone()));
        }
        if self.name.trim().is_empty() || self.sport.trim().is_empty() {
            return Err(ManifestError::EmptyRequiredText);
        }
        semver::Version::parse(&self.version)
            .map_err(|_| ManifestError::InvalidVersion(self.version.clone()))?;
        if self.viewport.width == 0
            || self.viewport.height == 0
            || self.viewport.width > 8192
            || self.viewport.height > 8192
        {
            return Err(ManifestError::InvalidViewport(
                self.viewport.width,
                self.viewport.height,
            ));
        }
        Ok(())
    }
}

fn is_safe_id(id: &str) -> bool {
    let mut chars = id.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_lowercase()
        && chars.all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-' || ch == '_')
}

/// Validation failures for one package.
#[derive(Debug, Error)]
pub enum ManifestError {
    /// Manifest could not be read.
    #[error("cannot read manifest {path}: {source}")]
    ReadManifest {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Package directory could not be resolved.
    #[error("cannot access package directory {path}: {source}")]
    ReadPackage {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Manifest JSON is malformed or missing a required property.
    #[error("invalid manifest JSON {path}: {source}")]
    InvalidJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    /// Schema is newer or otherwise unsupported.
    #[error("unsupported schemaVersion {0}; supported version is 1")]
    UnsupportedSchema(u32),
    /// JavaScript update API is unsupported.
    #[error("unsupported updateApiVersion {0}; supported version is 1")]
    UnsupportedUpdateApi(u32),
    /// Identifier is not filesystem-safe.
    #[error("id {0:?} must start with a-z and contain only a-z, 0-9, '-' or '_'")]
    UnsafeId(String),
    /// Required display/sport text is empty.
    #[error("name and sport must not be empty")]
    EmptyRequiredText,
    /// Package version is not semantic versioning.
    #[error("version {0:?} is not valid semantic versioning")]
    InvalidVersion(String),
    /// Viewport is outside supported bounds.
    #[error("viewport {0}x{1} must be between 1x1 and 8192x8192")]
    InvalidViewport(u32, u32),
    /// Directory name and package identifier disagree.
    #[error("package directory {directory:?} must match manifest id {id:?}")]
    DirectoryIdMismatch { directory: String, id: String },
    /// Entry path is absolute, traverses parents, or is not HTML.
    #[error("entry {0:?} must be a contained relative .html path")]
    UnsafeEntry(String),
    /// Editor path is absolute, traverses parents, or is not HTML.
    #[error("editor {0:?} must be a contained relative .html path")]
    UnsafeEditor(String),
    /// Entry file is missing.
    #[error("entry HTML is missing at {path}: {source}")]
    MissingEntry {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Editor file is missing.
    #[error("editor HTML is missing at {path}: {source}")]
    MissingEditor {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    /// Symlink or path escaped the package directory.
    #[error("entry HTML resolves outside its package: {0}")]
    EntryOutsidePackage(PathBuf),
    /// Editor symlink or path escaped the package directory.
    #[error("editor HTML resolves outside its package: {0}")]
    EditorOutsidePackage(PathBuf),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ids_are_portable() {
        assert!(is_safe_id("basketball"));
        assert!(is_safe_id("ice-hockey_3x3"));
        assert!(!is_safe_id("Football"));
        assert!(!is_safe_id("../football"));
        assert!(!is_safe_id(""));
    }
}
