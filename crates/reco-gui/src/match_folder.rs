//! Match-folder auto-detection for the "Select Match Folder" picker.
//!
//! Users record each match with a fixed convention: one top-level folder
//! per match (usually named after the two teams and a date), containing
//! `Left/` and `Right/` subfolders with that camera's recording(s). This
//! module locates those subfolders and their video files, and derives
//! where the per-match calibration file should live, so `main.rs` can
//! fill in the left video, right video, and calibration pickers from a
//! single folder pick instead of three separate file dialogs.

use std::path::{Path, PathBuf};

/// Video extensions recognized by both the manual file pickers and the
/// match-folder scanner. Comparison is case-insensitive.
pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "avi", "mkv"];

/// Why a match folder could not be auto-loaded.
#[derive(Debug, PartialEq, Eq)]
pub enum MatchFolderError {
    /// No subdirectory matching `name` (case-insensitive) was found
    /// directly inside the selected match folder.
    MissingCameraFolder { name: &'static str },
    /// The camera folder exists but contains no recognized video files.
    NoVideoFiles { name: &'static str },
}

impl std::fmt::Display for MatchFolderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingCameraFolder { name } => {
                write!(f, "No \"{name}\" folder found in the selected match folder")
            }
            Self::NoVideoFiles { name } => {
                write!(f, "The \"{name}\" folder has no video files")
            }
        }
    }
}

/// Result of scanning a match folder: video segments for each camera
/// (found in `<base>/Left` and `<base>/Right`, case-insensitive), plus
/// where this match's calibration file belongs.
#[derive(Debug)]
pub struct MatchFolderScan {
    /// Left camera's video segment(s), sorted so multi-segment
    /// recordings (e.g. DJI 4GB splits) chain in recording order.
    pub left_videos: Vec<PathBuf>,
    /// Right camera's video segment(s), same ordering guarantee.
    pub right_videos: Vec<PathBuf>,
    /// Where this match's calibration file lives (or should be created).
    /// Callers decide whether to load an existing file at this path, copy
    /// the user's Default Calibration there, or leave calibration unset.
    pub calibration_path: PathBuf,
}

/// Scan `base` for `Left`/`Right` camera subfolders and their video
/// files, and compute this match's calibration file path.
pub fn scan_match_folder(base: &Path) -> Result<MatchFolderScan, MatchFolderError> {
    let left_videos = find_camera_videos(base, "Left")?;
    let right_videos = find_camera_videos(base, "Right")?;
    let folder_name = base
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let calibration_path = base.join(sanitize_calibration_filename(&folder_name));
    Ok(MatchFolderScan {
        left_videos,
        right_videos,
        calibration_path,
    })
}

fn find_camera_videos(base: &Path, name: &'static str) -> Result<Vec<PathBuf>, MatchFolderError> {
    let dir =
        find_camera_subdir(base, name).ok_or(MatchFolderError::MissingCameraFolder { name })?;
    let videos = collect_video_files(&dir);
    if videos.is_empty() {
        return Err(MatchFolderError::NoVideoFiles { name });
    }
    Ok(videos)
}

/// Find a direct subdirectory of `base` whose name matches `name`
/// case-insensitively (e.g. "left", "LEFT" and "Left" all match "Left").
fn find_camera_subdir(base: &Path, name: &str) -> Option<PathBuf> {
    let entries = std::fs::read_dir(base).ok()?;
    entries.filter_map(|e| e.ok()).map(|e| e.path()).find(|p| {
        p.is_dir()
            && p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.eq_ignore_ascii_case(name))
    })
}

/// All recognized video files directly inside `dir`, sorted alphabetically
/// so a multi-segment recording chains in the same order the manual
/// multi-select picker would produce (see `on_pick_left_video` in `main.rs`).
fn collect_video_files(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut videos: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|ext| VIDEO_EXTENSIONS.iter().any(|v| v.eq_ignore_ascii_case(ext)))
        })
        .collect();
    videos.sort();
    videos
}

/// Derive the per-match calibration filename from the match folder's
/// name, e.g. "TeamA - TeamB 2026-08-22" becomes
/// "TeamA - TeamB 2026-08-22_calibration.json". Characters invalid in
/// Windows/macOS/Linux filenames are replaced with "-" so the copy step
/// never fails on the folder name alone.
fn sanitize_calibration_filename(folder_name: &str) -> String {
    let cleaned: String = folder_name
        .chars()
        .map(|c| if is_invalid_filename_char(c) { '-' } else { c })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "match_calibration.json".to_string()
    } else {
        format!("{trimmed}_calibration.json")
    }
}

fn is_invalid_filename_char(c: char) -> bool {
    matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') || c.is_control()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_match_dir(subdirs: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in subdirs {
            std::fs::create_dir(dir.path().join(name)).unwrap();
        }
        dir
    }

    #[test]
    fn finds_left_right_case_insensitively() {
        let dir = make_match_dir(&["left", "RIGHT"]);
        std::fs::write(dir.path().join("left/clip1.mp4"), b"").unwrap();
        std::fs::write(dir.path().join("RIGHT/clip1.MP4"), b"").unwrap();

        let scan = scan_match_folder(dir.path()).unwrap();
        assert_eq!(scan.left_videos.len(), 1);
        assert_eq!(scan.right_videos.len(), 1);
    }

    #[test]
    fn missing_left_folder_errors() {
        let dir = make_match_dir(&["Right"]);
        std::fs::write(dir.path().join("Right/clip1.mp4"), b"").unwrap();

        let err = scan_match_folder(dir.path()).unwrap_err();
        assert_eq!(err, MatchFolderError::MissingCameraFolder { name: "Left" });
    }

    #[test]
    fn empty_camera_folder_errors() {
        let dir = make_match_dir(&["Left", "Right"]);
        std::fs::write(dir.path().join("Right/clip1.mp4"), b"").unwrap();

        let err = scan_match_folder(dir.path()).unwrap_err();
        assert_eq!(err, MatchFolderError::NoVideoFiles { name: "Left" });
    }

    #[test]
    fn multiple_videos_sort_for_chaining() {
        let dir = make_match_dir(&["Left", "Right"]);
        std::fs::write(dir.path().join("Left/DJI_0002.mp4"), b"").unwrap();
        std::fs::write(dir.path().join("Left/DJI_0001.mp4"), b"").unwrap();
        std::fs::write(dir.path().join("Right/clip1.mp4"), b"").unwrap();

        let scan = scan_match_folder(dir.path()).unwrap();
        assert_eq!(scan.left_videos[0].file_name().unwrap(), "DJI_0001.mp4");
        assert_eq!(scan.left_videos[1].file_name().unwrap(), "DJI_0002.mp4");
    }

    #[test]
    fn non_video_files_ignored() {
        let dir = make_match_dir(&["Left", "Right"]);
        std::fs::write(dir.path().join("Left/clip1.mp4"), b"").unwrap();
        std::fs::write(dir.path().join("Left/notes.txt"), b"").unwrap();
        std::fs::write(dir.path().join("Right/clip1.mp4"), b"").unwrap();

        let scan = scan_match_folder(dir.path()).unwrap();
        assert_eq!(scan.left_videos.len(), 1);
    }

    #[test]
    fn calibration_filename_derived_from_folder_name() {
        let dir = make_match_dir(&["Left", "Right"]);
        std::fs::write(dir.path().join("Left/clip1.mp4"), b"").unwrap();
        std::fs::write(dir.path().join("Right/clip1.mp4"), b"").unwrap();

        let scan = scan_match_folder(dir.path()).unwrap();
        let folder_name = dir
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            scan.calibration_path,
            dir.path().join(format!("{folder_name}_calibration.json"))
        );
    }

    #[test]
    fn sanitizes_invalid_filename_characters() {
        assert_eq!(
            sanitize_calibration_filename("Team A : Team B"),
            "Team A - Team B_calibration.json"
        );
    }

    #[test]
    fn empty_folder_name_falls_back() {
        assert_eq!(sanitize_calibration_filename(""), "match_calibration.json");
    }
}
