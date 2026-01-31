#!/usr/bin/env python3
"""
Script to convert Gyroflow lens profiles to video-stitcher JSON format.
Downloads profiles from GitHub and converts them without cloning the repo.
"""

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Base URL for raw GitHub content
GYROFLOW_RAW_URL = "https://raw.githubusercontent.com/gyroflow/lens_profiles/main"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "data" / "lens_profiles"


def slugify(text: str) -> str:
    """Convert text to slug format (lowercase, hyphens)."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def normalize_camera_name(name: str) -> str:
    """
    Normalize camera model names to avoid duplicates due to casing.
    
    Common patterns:
    - "HERO3 Black" vs "Hero3 Black" -> "HERO3 Black"
    - "hero3 Silver" -> "HERO3 Silver"
    - "Hero3+Black" -> "HERO3+ Black"
    """
    # Normalize to title case first
    name = name.strip()
    
    # GoPro HERO series normalization
    # Match HERO followed by optional + and number
    name = re.sub(r'\bhero(\+?)(\d+)', r'HERO\1\2', name, flags=re.IGNORECASE)
    
    # Normalize common color/edition names to Title Case
    name = re.sub(r'\b(black|silver|white|session)\b', lambda m: m.group(1).capitalize(), name, flags=re.IGNORECASE)
    
    # Clean up spacing around +
    name = re.sub(r'\s*\+\s*', '+', name)
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()


def parse_gyroflow_json(data: Dict[str, Any], filename: str = "") -> Optional[Dict[str, Any]]:
    """
    Convert Gyroflow JSON format to video-stitcher format.
    
    Gyroflow uses fisheye_params with camera_matrix and distortion_coeffs.
    The distortion model is typically fisheye (4 coefficients: k1, k2, k3, k4).
    """
    try:
        # Get fisheye parameters
        fisheye = data.get("fisheye_params", {})
        if not fisheye:
            return None
        
        # Extract camera info
        camera_brand = data.get("camera_brand", "Unknown")
        camera_model = data.get("camera_model", "Unknown")
        lens_model = data.get("lens_model", "Unknown")
        
        # Normalize camera model to avoid duplicates
        camera_model = normalize_camera_name(camera_model)
        
        # Get camera matrix
        cam_matrix = fisheye.get("camera_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        fx = cam_matrix[0][0]
        fy = cam_matrix[1][1]
        cx = cam_matrix[0][2]
        cy = cam_matrix[1][2]
        
        # Get distortion coefficients (k1, k2, k3, k4 for fisheye)
        dist_coeffs = fisheye.get("distortion_coeffs", [])
        
        # Gyroflow profiles use fisheye model with 4 coefficients
        distortion_model = "fisheye_kb4"
        
        # Get resolution
        calib_dim = data.get("calib_dimension", {})
        width = calib_dim.get("w", 0)
        height = calib_dim.get("h", 0)
        
        # Get calibration metadata
        calibrated_by = data.get("calibrated_by", "Gyroflow Official")
        fps = data.get("fps", "unknown")
        official = data.get("official", False)
        
        # Create ID from brand, model, lens
        profile_id = f"{slugify(camera_brand)}-{slugify(camera_model)}-{slugify(lens_model)}-{width}x{height}"
        
        # Build our format
        profile = {
            "id": profile_id,
            "camera_brand": camera_brand,
            "camera_model": camera_model,
            "lens_model": lens_model,
            "resolution": {
                "width": width,
                "height": height
            },
            "distortion_model": distortion_model,
            "camera_matrix": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy
            },
            "distortion_coeffs": dist_coeffs,
            "metadata": {
                "source": "Gyroflow lens_profiles",
                "source_file": data.get("name", ""),
                "calibrated_by": calibrated_by,
                "original_resolution": f"{width}x{height}",
                "calibrated_fps": str(fps),
                "official": official,
                "notes": f"{'Official ' if official else ''}Gyroflow calibration. Fisheye distortion model.",
                "license": "CC0 1.0 Universal"
            }
        }
        
        return profile
        
    except Exception as e:
        return None


def count_json_files_recursive(path: str) -> int:
    """Recursively count JSON files in a path."""
    count = 0
    api_url = f"https://api.github.com/repos/gyroflow/lens_profiles/contents/{path}"
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req) as response:
            items = json.loads(response.read())
            for item in items:
                if item["type"] == "file" and item["name"].endswith(".json"):
                    count += 1
                elif item["type"] == "dir":
                    count += count_json_files_recursive(item["path"])
    except:
        pass
    return count


def count_json_files(brands: List[str]) -> int:
    """Count total JSON files to process across all brands."""
    total = 0
    for brand in brands:
        total += count_json_files_recursive(brand)
    return total


def collect_all_json_files(source_path: str) -> List[Dict[str, str]]:
    """Recursively collect all JSON file metadata."""
    files = []
    api_url = f"https://api.github.com/repos/gyroflow/lens_profiles/contents/{source_path}"
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        
        with urllib.request.urlopen(req) as response:
            items = json.loads(response.read())
        
        for item in items:
            if item["type"] == "dir":
                files.extend(collect_all_json_files(item["path"]))
            elif item["type"] == "file" and item["name"].endswith(".json"):
                files.append({
                    "name": item["name"],
                    "path": item["path"],
                    "download_url": item["download_url"]
                })
    except:
        pass
    
    return files


def fetch_directory_listing(url: str) -> list:
    """Fetch GitHub directory listing via API."""
    api_url = url.replace("https://github.com/", "https://api.github.com/repos/")
    api_url = api_url.replace("/tree/main/", "/contents/")
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error fetching directory: {e}")
        return []


def download_and_convert(source_path: str, output_dir: Path, dry_run: bool = False, stats: Dict = None, total_files: int = 0):
    """
    Download JSON files from Gyroflow repo and convert them.
    
    Args:
        source_path: Path relative to repo root (e.g., "GoPro")
        output_dir: Output directory for converted profiles
        dry_run: If True, only print what would be done
        stats: Dictionary to track conversion statistics
        total_files: Total number of files to process (for progress bar)
    """
    if stats is None:
        stats = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    api_url = f"https://api.github.com/repos/gyroflow/lens_profiles/contents/{source_path}"
    
    try:
        req = urllib.request.Request(api_url)
        req.add_header("Accept", "application/vnd.github.v3+json")
        
        with urllib.request.urlopen(req) as response:
            items = json.loads(response.read())
        
        for item in items:
            if item["type"] == "dir":
                # Recursively process subdirectories
                download_and_convert(item["path"], output_dir, dry_run, stats, total_files)
            
            elif item["type"] == "file" and item["name"].endswith(".json"):
                processed = stats['success'] + stats['failed'] + stats['skipped']
                
                # Progress bar
                if total_files > 0:
                    percent = int((processed / total_files) * 100)
                    bar_length = 30
                    filled = int((processed / total_files) * bar_length)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\r[{bar}] {percent}% ({processed}/{total_files})", end="", flush=True)
                else:
                    print(f"\r[{processed}] Processing...", end="", flush=True)
                
                try:
                    # Download the JSON file
                    raw_url = item["download_url"]
                    with urllib.request.urlopen(raw_url) as response:
                        gyroflow_data = json.loads(response.read())
                except Exception as e:
                    stats["failed"] += 1
                    continue
                
                # Convert to our format
                converted = parse_gyroflow_json(gyroflow_data, item["name"])
                
                if converted:
                    # Determine output path based on brand
                    brand_slug = slugify(converted["camera_brand"])
                    model_slug = slugify(converted["camera_model"])
                    
                    out_dir = output_dir / brand_slug / model_slug
                    out_file = out_dir / f"{converted['id']}.json"
                    
                    if not dry_run:
                        out_dir.mkdir(parents=True, exist_ok=True)
                        with open(out_file, 'w') as f:
                            json.dump(converted, f, indent='\t')
                    
                    stats["success"] += 1
                else:
                    stats["skipped"] += 1
    
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"\r❌ HTTP Error {e.code} for {source_path}" + " " * 30)
    except Exception as e:
        print(f"\r❌ Error processing {source_path}: {e}" + " " * 30)
    
    return stats


def process_single_file(file_info: Dict[str, str], output_dir: Path, dry_run: bool, update_mode: bool) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single JSON file.
    
    Returns:
        Tuple of (status, success, error_message)
        status: 'success', 'skipped', 'failed', or 'unchanged'
    """
    try:
        # Download the JSON file
        with urllib.request.urlopen(file_info["download_url"]) as response:
            gyroflow_data = json.loads(response.read())
    except Exception as e:
        return ('failed', False, str(e))
    
    # Convert to our format
    converted = parse_gyroflow_json(gyroflow_data, file_info["name"])
    
    if not converted:
        return ('skipped', False, 'No fisheye_params found')
    
    # Determine output path
    brand_slug = slugify(converted["camera_brand"])
    model_slug = slugify(converted["camera_model"])
    out_dir = output_dir / brand_slug / model_slug
    out_file = out_dir / f"{converted['id']}.json"
    
    # Check if file exists and is unchanged (update mode)
    if update_mode and out_file.exists():
        try:
            with open(out_file, 'r') as f:
                existing = json.load(f)
            # Compare camera_matrix and distortion_coeffs
            if (existing.get("camera_matrix") == converted["camera_matrix"] and
                existing.get("distortion_coeffs") == converted["distortion_coeffs"]):
                return ('unchanged', True, None)
        except:
            pass  # If comparison fails, update the file
    
    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_file, 'w') as f:
            json.dump(converted, f, indent='\t')
    
    return ('success', True, None)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Gyroflow lens profiles")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without actually converting")
    parser.add_argument("--brands", nargs="+", help="Specific brands to convert (e.g., GoPro DJI)")
    parser.add_argument("--update", action="store_true", help="Update mode: skip unchanged files")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers (default: 10)")
    
    args = parser.parse_args()
    
    print("🔄 Gyroflow Lens Profile Converter")
    print("=" * 50)
    
    if args.dry_run:
        print("⚠️  DRY RUN MODE - No files will be created")
    if args.update:
        print("🔄 UPDATE MODE - Skipping unchanged files")
    print()
    
    # List of top-level directories in Gyroflow repo
    brands_to_convert = args.brands or [
        "AKASO",
        "ARRI",
        "Blackmagic",
        "Caddx",
        "Canon",
        "DJI",
        "Eken",
        "Foxeer",
        "Freefly",
        "Fujifilm",
        "GoPro",
        "Hawkeye",
        "Insta360",
        "Kinefinity",
        "Mobile phones",
        "Mobius",
        "Morecam",
        "Nikon",
        "Olympus",
        "Other",
        "Panasonic",
        "RED",
        "RunCam",
        "SJCAM",
        "Sigma",
        "Sony",
        "ThiEYE",
        "Walksnail",
        "XTU",
        "Xiaomi",
        "Z CAM",
        "apeman"
    ]
    
    # Collect all files
    print("📊 Collecting files...", end="", flush=True)
    all_files = []
    for brand in brands_to_convert:
        all_files.extend(collect_all_json_files(brand))
    print(f"\r📊 Found {len(all_files)} profiles to process")
    
    # Process files in parallel
    stats = {"success": 0, "failed": 0, "skipped": 0, "unchanged": 0}
    stats_lock = Lock()
    
    def update_progress():
        with stats_lock:
            processed = stats['success'] + stats['failed'] + stats['skipped'] + stats['unchanged']
            percent = int((processed / len(all_files)) * 100)
            bar_length = 30
            filled = int((processed / len(all_files)) * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r[{bar}] {percent}% ({processed}/{len(all_files)})", end="", flush=True)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_single_file, file_info, OUTPUT_DIR, args.dry_run, args.update): file_info
            for file_info in all_files
        }
        
        for future in as_completed(futures):
            status, success, error = future.result()
            
            with stats_lock:
                stats[status] += 1
            
            update_progress()
    
    # Clear progress line
    print("\r" + " " * 80 + "\r", end="")
    
    print("=" * 50)
    print(f"✅ Complete: {stats['success']} converted", end="")
    if args.update:
        print(f", {stats['unchanged']} unchanged", end="")
    print(f", {stats['skipped']} skipped, {stats['failed']} failed")
    if args.dry_run:
        print("   Run without --dry-run to actually convert files")


if __name__ == "__main__":
    main()
