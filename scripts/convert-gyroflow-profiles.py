#!/usr/bin/env python3
"""
Script to convert Gyroflow lens profiles to video-stitcher JSON format.
Downloads the repository as a zip file and converts profiles locally.
"""

import json
import os
import re
import urllib.request
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# GitHub repository zip download URL
GYROFLOW_ZIP_URL = "https://github.com/gyroflow/lens_profiles/archive/refs/heads/main.zip"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "backend" / "data" / "lens_profiles"

# Global lock for file writing to prevent race conditions
_file_locks: Dict[str, Lock] = {}
_file_locks_lock = Lock()

def get_file_lock(path: str) -> Lock:
    """Get or create a lock for a specific file path."""
    with _file_locks_lock:
        if path not in _file_locks:
            _file_locks[path] = Lock()
        return _file_locks[path]


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


def normalize_camera_name(name: str) -> str:
    """Normalize camera model names to avoid duplicates."""
    # Remove common prefixes/suffixes
    name = re.sub(r'\s+(camera|cam)\s*$', '', name, flags=re.IGNORECASE)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name


def normalize_brand_name(brand: str) -> str:
    """Normalize brand names to Title Case for consistency.
    
    Handles common variations like 'GOPRO' -> 'GoPro', 'DJI' stays 'DJI', etc.
    """
    # Known brand name mappings (uppercase key -> proper casing)
    BRAND_MAPPINGS = {
        # Action cameras
        "GOPRO": "GoPro",
        "DJI": "DJI",
        "SJCAM": "SJCAM",
        "AKASO": "Akaso",
        "EKEN": "Eken",
        "COOAU": "Cooau",
        "COTUO": "Cotuo",
        "FIMI": "Fimi",
        "INSTA360": "Insta360",
        "INSTA TITAN": "Insta360",
        "ACTIVEON": "Activeon",
        "AIKUCAM": "Aikucam",
        "AXNEN": "Axnen",
        "RUNCAM": "RunCam",
        "WALKSNAIL": "Walksnail",
        "WALKSNAIL AVATAR V2": "Walksnail",
        "WALKSNAIL AVATAR V2 PRO": "Walksnail",
        "THIEYE": "ThiEYE",
        
        # FPV drones
        "BETAFPV": "BetaFPV",
        "HDZERO": "HDZero",
        "CADDX": "Caddx",
        "FOXEER": "Foxeer",
        "HAWKEYE": "Hawkeye",
        
        # Smartphones - Xiaomi ecosystem
        "XIAOMI": "Xiaomi",
        "MI": "Xiaomi",
        "REDMI": "Xiaomi",
        "POCO": "Xiaomi",
        
        # Smartphones - Other Chinese
        "ONEPLUS": "OnePlus",
        "OPPO": "Oppo",
        "VIVO": "Vivo",
        "REALME": "Realme",
        "HONOR": "Honor",
        "HONOR 80": "Honor",
        "HUAWEI": "Huawei",
        "MATE40PRO": "Huawei",
        "NOVA7": "Huawei",
        "IQOO": "iQOO",
        "IQOO 9": "iQOO",
        "IQOO Z3": "iQOO",
        "ZTE": "ZTE",
        "NUBIA": "ZTE",
        "MEIZU": "Meizu",
        "TECNO": "Tecno",
        "TECHNO SPARK": "Tecno",
        "INFINIX": "Infinix",
        "INFINIX HOT 30": "Infinix",
        
        # Smartphones - Korean/Japanese
        "SAMSUNG": "Samsung",
        "LG": "LG",
        "LGE": "LG",
        "LG V30": "LG",
        "SONY": "Sony",
        
        # Smartphones - Western
        "APPLE": "Apple",
        "GOOGLE": "Google",
        "MOTOROLA": "Motorola",
        "NOKIA": "Nokia",
        "BLACKBERRY": "Blackberry",
        
        # Professional cameras
        "BLACKMAGIC": "Blackmagic",
        "BMD": "Blackmagic",
        "RED": "RED",
        "ARRI": "Arri",
        "CANON": "Canon",
        "NIKON": "Nikon",
        "SONY": "Sony",
        "PANASONIC": "Panasonic",
        "FUJIFILM": "Fujifilm",
        "FUFIFILM": "Fujifilm",  # Common typo
        "OLYMPUS": "Olympus",
        "PENTAX": "Pentax",
        "LEICA": "Leica",
        "HASSELBLAD": "Hasselblad",
        "SIGMA": "Sigma",
        "RICOH": "Ricoh",
        "Z CAM": "Z-Cam",
        "KINEFINITY": "Kinefinity",
        
        # Other
        "AEE": "AEE",
        "HP": "HP",
        "HTC": "HTC",
        "JVC": "JVC",
        "TCL": "TCL",
        "DDPAI": "DDPAI",
        "YI": "YI",
        "HOLY STONE": "Holystone",
        "HOLYSTONE": "Holystone",
        "FEIYU TECH": "Feiyu Tech",
        "FEIYU-TECH": "Feiyu Tech",
        "GARMIN": "Garmin",
        "TOMTOM": "TomTom",
        "RASPBERRY PI": "Raspberry Pi",
        "SKYDIO": "Skydio",
        "DRIFT": "Drift",
        "MOBIUS": "Mobius",
        "GITUP": "GitUp",
        "ROLLEI": "Rollei",
        "ROLLEI (禄来)": "Rollei",
    }
    
    # Major brands to keep separate (all others go to "Others")
    # These are brands with 10+ profiles or well-known camera brands
    MAJOR_BRANDS = {
        "GoPro", "DJI", "SJCAM", "Akaso", "Insta360", "RunCam", "Walksnail",
        "BetaFPV", "HDZero", "Caddx", "Foxeer", "Hawkeye",  # Action/FPV
        "Xiaomi", "OnePlus", "Oppo", "Vivo", "Realme", "Honor", "Huawei",
        "Samsung", "LG", "Sony", "Apple", "Google", "Motorola",  # Smartphones
        "Blackmagic", "RED", "Arri", "Canon", "Nikon", "Panasonic",
        "Fujifilm", "Olympus", "Sigma", "Z-Cam", "Kinefinity",  # Pro cameras
        "XTU", "Apeman", "Generic",  # Other major
    }
    
    # Check for known mapping first (case-insensitive)
    brand_upper = brand.upper().strip()
    if brand_upper in BRAND_MAPPINGS:
        normalized = BRAND_MAPPINGS[brand_upper]
    elif len(brand) <= 3 and brand.isupper():
        normalized = brand
    else:
        normalized = brand.title()
    
    # If not a major brand, put in "Others"
    if normalized not in MAJOR_BRANDS:
        return "Others"
    
    return normalized


def parse_gyroflow_json(data: Dict[str, Any], file_path: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse Gyroflow JSON and extract brand, model, preset info.
    Returns: (brand_slug, model_slug, preset_name) or None if invalid
    """
    try:
        # Get camera info from various possible locations
        camera_brand = data.get("camera_brand") or data.get("brand") or ""
        camera_model = data.get("camera_model") or data.get("model") or ""
        note = data.get("note", "")
        
        # Extract brand from path if not in data
        path_parts = file_path.split('/')
        if not camera_brand and len(path_parts) > 0:
            camera_brand = path_parts[0]
        
        if not camera_brand or not camera_model:
            return None
        
        # Create slugs
        brand_slug = slugify(camera_brand)
        model_slug = slugify(normalize_camera_name(camera_model))
        
        # Get resolution info for preset name
        fisheye = data.get("fisheye_params", {})
        calib_dim = data.get("calib_dimension", {})
        width = calib_dim.get("w", 0)
        height = calib_dim.get("h", 0)
        
        # Build preset name with brand and always include resolution to ensure uniqueness
        if note:
            # Include both note and resolution to avoid collisions
            preset_name = f"{brand_slug}-{model_slug}--{slugify(note)}--{width}x{height}"
        else:
            preset_name = f"{brand_slug}-{model_slug}--{width}x{height}"
        
        return (brand_slug, model_slug, preset_name)
    
    except Exception as e:
        return None


def convert_to_our_format(data: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """Convert Gyroflow profile format to our format.
    
    Our format expects:
    - resolution: {width, height}
    - camera_matrix: {fx, fy, cx, cy}
    - distortion_coeffs: [k1, k2, k3, k4]
    - metadata: optional dict with calibrated_by, notes, etc.
    """
    try:
        fisheye = data.get("fisheye_params", {})
        
        # Get camera info and normalize
        camera_brand = normalize_brand_name(data.get("camera_brand", "Unknown"))
        camera_model = data.get("camera_model", "Unknown")
        
        # Normalize camera model to avoid duplicates
        camera_model = normalize_camera_name(camera_model)
        
        # Get camera matrix from fisheye_params and convert to our format
        cam_matrix = fisheye.get("camera_matrix", [[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        fx = cam_matrix[0][0]
        fy = cam_matrix[1][1]
        cx = cam_matrix[0][2]
        cy = cam_matrix[1][2]
        
        # Get distortion coefficients (k1, k2, k3, k4 for fisheye)
        dist_coeffs = fisheye.get("distortion_coeffs", [])
        
        # Ensure we have exactly 4 coefficients for fisheye_kb4
        if len(dist_coeffs) < 4:
            dist_coeffs = dist_coeffs + [0.0] * (4 - len(dist_coeffs))
        elif len(dist_coeffs) > 4:
            dist_coeffs = dist_coeffs[:4]
        
        # Gyroflow profiles use fisheye model with 4 coefficients
        distortion_model = "fisheye_kb4"
        
        # Get resolution from calib_dimension
        calib_dim = data.get("calib_dimension", {})
        width = calib_dim.get("w", 0)
        height = calib_dim.get("h", 0)
        
        # Skip profiles with invalid resolution or camera matrix
        if width <= 0 or height <= 0:
            raise Exception("Invalid resolution")
        if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
            raise Exception("Invalid camera matrix")
        
        # Build metadata from optional fields
        metadata = {}
        if data.get("calibrated_by"):
            metadata["calibrated_by"] = data.get("calibrated_by")
        if data.get("note"):
            metadata["notes"] = data.get("note")
        metadata["source"] = "gyroflow"
        metadata["official"] = True
        
        # Build our format matching LensProfileModel
        result = {
            "id": preset_name,
            "camera_brand": camera_brand,
            "camera_model": camera_model,
            "lens_model": data.get("lens_model") or None,
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
            "metadata": metadata if metadata else None
        }
        
        return result
    
    except Exception as e:
        raise Exception(f"Conversion error: {e}")


def download_and_extract_repo(temp_dir: Path) -> Path:
    """Download the Gyroflow repository as a zip and extract it."""
    print("📥 Downloading Gyroflow lens profiles repository...")
    
    zip_path = temp_dir / "gyroflow-profiles.zip"
    
    # Download the zip file
    try:
        with urllib.request.urlopen(GYROFLOW_ZIP_URL) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(zip_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        bar_length = 30
                        filled = int((downloaded / total_size) * bar_length)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r[{bar}] {percent}% ({downloaded}/{total_size} bytes)", end="", flush=True)
        
        print("\n✅ Download complete")
    except Exception as e:
        print(f"\n❌ Error downloading repository: {e}")
        raise
    
    # Extract the zip file
    print("📦 Extracting repository...")
    extract_dir = temp_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("✅ Extraction complete")
    except Exception as e:
        print(f"❌ Error extracting zip: {e}")
        raise
    
    # The extracted folder will be named "lens_profiles-main"
    repo_dir = extract_dir / "lens_profiles-main"
    if not repo_dir.exists():
        raise Exception(f"Expected directory not found: {repo_dir}")
    
    return repo_dir


def collect_json_files_from_disk(repo_dir: Path, brands: List[str]) -> List[Tuple[Path, str]]:
    """Collect all JSON files from the extracted repository."""
    json_files = []
    
    for brand in brands:
        brand_dir = repo_dir / brand
        if not brand_dir.exists():
            continue
        
        # Recursively find all JSON files
        for json_file in brand_dir.rglob("*.json"):
            # Get relative path from repo root for parsing
            rel_path = json_file.relative_to(repo_dir)
            json_files.append((json_file, str(rel_path)))
    
    return json_files


def process_single_file(file_path: Path, rel_path: str, output_dir: Path, dry_run: bool, update_mode: bool) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single JSON file.
    
    Returns: (status, success, error_message)
        status: 'success', 'failed', 'skipped', 'unchanged'
    """
    try:
        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse the path to extract brand, model, and preset
        parsed = parse_gyroflow_json(data, rel_path)
        if not parsed:
            return ('skipped', False, f"Could not parse: {rel_path}")
        
        brand_slug, model_slug, preset_name = parsed
        
        # Create output path
        out_dir = output_dir / brand_slug / model_slug
        out_file = out_dir / f"{preset_name}.json"
        
        # Convert to our format
        converted = convert_to_our_format(data, preset_name)
        
        if not dry_run:
            # Use a file-specific lock to prevent race conditions when multiple
            # source files generate the same output filename
            file_lock = get_file_lock(str(out_file))
            with file_lock:
                # Check if file exists and is unchanged (update mode)
                if update_mode and out_file.exists():
                    with open(out_file, 'r', encoding='utf-8') as f:
                        try:
                            existing = json.load(f)
                            if existing == converted:
                                return ('unchanged', True, None)
                        except json.JSONDecodeError:
                            pass  # File is corrupted, overwrite it
                
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_file, 'w', encoding='utf-8') as f:
                    json.dump(converted, f, indent='\t')
        
        return ('success', True, None)
    
    except Exception as e:
        return ('failed', False, str(e))


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
    
    # Create temporary directory for download and extraction
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        
        try:
            # Download and extract repository
            repo_dir = download_and_extract_repo(temp_dir)
            
            # Collect all JSON files
            print("\n📊 Collecting files...", end="", flush=True)
            all_files = collect_json_files_from_disk(repo_dir, brands_to_convert)
            print(f"\r📊 Found {len(all_files)} profiles to process")
            
            if len(all_files) == 0:
                print("⚠️  No profiles found to convert")
                return
            
            print("\n🔄 Processing profiles...")
            print("=" * 50)
            
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
                    executor.submit(process_single_file, file_path, rel_path, OUTPUT_DIR, args.dry_run, args.update): (file_path, rel_path)
                    for file_path, rel_path in all_files
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
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()
