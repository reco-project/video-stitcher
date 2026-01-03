"""
Prototype modifications for rVFC dual-video support.

This shows how the backend would need to change to support
separate video files instead of stacked videos.
"""

# BEFORE (Current approach):
# 1. Extract audio from both videos
# 2. Compute audio offset
# 3. Stack videos vertically into one file (SLOW!)
# 4. Return stacked video path


# AFTER (rVFC approach):
# 1. Extract audio from both videos
# 2. Compute audio offset
# 3. Return BOTH original video paths + offset
# 4. Frontend handles sync with rVFC


# Example match data modification:

# OLD format:
old_match = {
    "id": "match-123",
    "name": "Test Match",
    "src": "videos/match-123-stacked.mp4",  # Single stacked file
    "params": {...},
    "left_uniforms": {...},
    "right_uniforms": {...}
}

# NEW format (rVFC):
new_match = {
    "id": "match-123",
    "name": "Test Match",
    # Separate video paths:
    "left_video_path": "videos/match-123-left.mp4",
    "right_video_path": "videos/match-123-right.mp4",
    # Audio offset computed once:
    "audio_offset": 0.147,  # seconds (right = left + offset)
    # Rest stays the same:
    "params": {...},
    "left_uniforms": {...},
    "right_uniforms": {...},
    # Optional: Keep stacked for fallback
    "src": "videos/match-123-stacked.mp4"  # Only if already generated
}


# Backend endpoint modification example:

# OLD: POST /api/matches/{match_id}/process
# - Transcodes videos (slow!)
# - Returns stacked video path

# NEW: POST /api/matches/{match_id}/process-rvfc
# - Only computes audio offset
# - Skips transcoding entirely
# - Returns offset + original paths

def process_match_rvfc(match_id: str, left_video: str, right_video: str):
    """
    Process match for rVFC mode (no transcoding).
    
    Returns:
        {
            "left_video_path": "videos/left.mp4",
            "right_video_path": "videos/right.mp4", 
            "audio_offset": 0.147,
            "params": {...},
            "left_uniforms": {...},
            "right_uniforms": {...}
        }
    """
    # 1. Extract audio (reuse existing code)
    audio1_path = extract_audio(left_video)
    audio2_path = extract_audio(right_video)
    
    # 2. Compute offset (reuse existing code)
    offset = compute_audio_offset(audio1_path, audio2_path)
    
    # 3. Extract frames for calibration (reuse existing code)
    left_frame = extract_frame(left_video, time=3.33)
    right_frame = extract_frame(right_video, time=3.33 + offset)
    
    # 4. Run feature matching (reuse existing code)
    params = compute_match_params(left_frame, right_frame)
    
    # 5. Return paths + offset (NO STACKING!)
    return {
        "left_video_path": left_video,
        "right_video_path": right_video,
        "audio_offset": offset,
        "params": params,
        "left_uniforms": {...},
        "right_uniforms": {...}
    }


# Time savings estimate:
# - Current: ~30-60 seconds for transcoding (depending on video length)
# - rVFC: ~2-5 seconds (audio extraction + offset only)
# - Speedup: 10-30x faster! 🚀

