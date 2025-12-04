"""
Video preprocessing and keyframe sampling module.

Provides shot detection, frame extraction, deduplication, and temporal sampling
for efficient video understanding.
"""

import cv2
import os
import numpy as np
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from typing import List, Dict, Tuple, Optional
import hashlib
from pathlib import Path
import json


class VideoSampler:
    """
    Video preprocessing and keyframe extraction.
    
    Features:
    - Shot detection using PySceneDetect
    - Keyframe sampling (uniform, middle, adaptive)
    - Frame deduplication using perceptual hashing
    - Audio presence detection (placeholder for ASR)
    """
    
    def __init__(self, cache_dir: str = "./cache", deduplicate: bool = True):
        """
        Initialize the video sampler.
        
        Args:
            cache_dir: Directory to cache extracted frames and metadata
            deduplicate: Whether to remove duplicate frames
        """
        self.cache_dir = cache_dir
        self.deduplicate = deduplicate
        os.makedirs(cache_dir, exist_ok=True)
        self._frame_hashes = {}  # For deduplication

    def get_video_info(self, video_path: str) -> Dict:
        """
        Extract video metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dict containing fps, frame_count, duration, resolution, has_audio
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Check for audio (placeholder - requires ffprobe or similar)
        has_audio = self._check_audio_presence(video_path)
        
        return {
            "video_path": video_path,
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "resolution": (width, height),
            "has_audio": has_audio
        }

    def _check_audio_presence(self, video_path: str) -> bool:
        """
        Check if video has audio track.
        
        TODO: Implement using ffprobe or librosa for actual audio detection.
        For now, returns True as placeholder.
        """
        # Placeholder implementation
        return True
    
    def _compute_frame_hash(self, frame: np.ndarray, hash_size: int = 8) -> str:
        """
        Compute perceptual hash of frame for deduplication.
        
        Args:
            frame: Input frame (BGR)
            hash_size: Size of hash grid
            
        Returns:
            Hexadecimal hash string
        """
        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Compute difference hash
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert to hex string
        hash_value = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        return f"{hash_value:016x}"
    
    def _is_duplicate_frame(self, frame: np.ndarray, threshold: int = 5) -> bool:
        """
        Check if frame is duplicate based on hash similarity.
        
        Args:
            frame: Input frame
            threshold: Hamming distance threshold for duplicates
            
        Returns:
            True if frame is likely a duplicate
        """
        if not self.deduplicate:
            return False
            
        frame_hash = self._compute_frame_hash(frame)
        
        # Check against existing hashes
        for existing_hash in self._frame_hashes.keys():
            # Compute Hamming distance
            distance = bin(int(frame_hash, 16) ^ int(existing_hash, 16)).count('1')
            if distance <= threshold:
                return True
        
        self._frame_hashes[frame_hash] = True
        return False

    def detect_shots(self, video_path: str, threshold: float = 27.0) -> List[Tuple[float, float]]:
        """
        Detect shot boundaries using content-based detection.
        
        Args:
            video_path: Path to video file
            threshold: Detection sensitivity (lower = more sensitive)
            
        Returns:
            List of (start_time, end_time) tuples for each shot
        """
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        # Convert to seconds
        shots = []
        for scene in scene_list:
            start, end = scene
            shots.append((start.get_seconds(), end.get_seconds()))
        
        return shots

    def sample_keyframes(
        self, 
        video_path: str, 
        num_frames_per_shot: int = 1,
        save_frames: bool = True
    ) -> List[Dict]:
        """
        Detect shots and sample keyframes from each shot.
        
        Args:
            video_path: Path to video file
            num_frames_per_shot: Number of frames to sample per shot
            save_frames: Whether to save frames to disk
            
        Returns:
            List of dicts with frame_id, timestamp, shot_id, path, and frame data
        """
        video_name = os.path.basename(video_path).split('.')[0]
        frames_dir = os.path.join(self.cache_dir, video_name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Reset deduplication hashes for new video
        self._frame_hashes = {}
        
        shots = self.detect_shots(video_path)
        if not shots:
            # If no scenes detected, treat whole video as one scene
            info = self.get_video_info(video_path)
            shots = [(0.0, info['duration'])]

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        sampled_frames = []
        frames_saved = 0
        frames_skipped = 0
        
        for i, (start_time, end_time) in enumerate(shots):
            # Sample frames uniformly or at middle
            duration = end_time - start_time
            
            timestamps = []
            if num_frames_per_shot == 1:
                # Sample middle frame
                timestamps = [start_time + duration / 2]
            else:
                # Sample uniformly, excluding start/end boundaries
                timestamps = np.linspace(start_time, end_time, num_frames_per_shot + 2)[1:-1]
            
            for ts in timestamps:
                frame_idx = int(ts * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Check for duplicates
                    if self._is_duplicate_frame(frame):
                        frames_skipped += 1
                        continue
                    
                    frame_filename = f"shot_{i:04d}_time_{ts:.2f}_frame_{frame_idx:08d}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    
                    if save_frames:
                        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    sampled_frames.append({
                        "frame_id": frame_filename,
                        "timestamp": ts,
                        "shot_id": i,
                        "frame_index": frame_idx,
                        "path": frame_path if save_frames else None,
                        "frame": frame if not save_frames else None  # Keep in memory if not saving
                    })
                    frames_saved += 1
        
        cap.release()
        
        # Save manifest
        manifest_path = os.path.join(self.cache_dir, video_name, "manifest.json")
        manifest = {
            "video_info": self.get_video_info(video_path),
            "shots": [{"start": s, "end": e} for s, e in shots],
            "num_frames": len(sampled_frames),
            "frames_saved": frames_saved,
            "frames_skipped_duplicates": frames_skipped,
        }
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return sampled_frames
    
    def sample_uniform(
        self,
        video_path: str,
        num_frames: int = 32,
        save_frames: bool = True
    ) -> List[Dict]:
        """
        Sample frames uniformly across entire video (ignoring shots).
        
        Args:
            video_path: Path to video file
            num_frames: Total number of frames to sample
            save_frames: Whether to save frames to disk
            
        Returns:
            List of frame dicts
        """
        video_name = os.path.basename(video_path).split('.')[0]
        frames_dir = os.path.join(self.cache_dir, video_name, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Reset deduplication
        self._frame_hashes = {}
        
        info = self.get_video_info(video_path)
        duration = info['duration']
        fps = info['fps']
        
        # Generate uniform timestamps
        timestamps = np.linspace(0, duration, num_frames + 2)[1:-1]  # Exclude start/end
        
        cap = cv2.VideoCapture(video_path)
        sampled_frames = []
        
        for i, ts in enumerate(timestamps):
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and not self._is_duplicate_frame(frame):
                frame_filename = f"uniform_{i:04d}_time_{ts:.2f}_frame_{frame_idx:08d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                
                if save_frames:
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                sampled_frames.append({
                    "frame_id": frame_filename,
                    "timestamp": ts,
                    "frame_index": frame_idx,
                    "path": frame_path if save_frames else None,
                    "frame": frame if not save_frames else None
                })
        
        cap.release()
        return sampled_frames


if __name__ == "__main__":
    # Simple test
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        sampler = VideoSampler(cache_dir="./cache/test", deduplicate=True)
        
        print("=" * 80)
        print("VIDEO SAMPLER TEST")
        print("=" * 80)
        
        # Get video info
        info = sampler.get_video_info(video_path)
        print(f"\nVideo Info:")
        print(f"  Duration: {info['duration']:.2f}s")
        print(f"  FPS: {info['fps']:.2f}")
        print(f"  Resolution: {info['resolution']}")
        print(f"  Has Audio: {info['has_audio']}")
        
        # Detect shots
        shots = sampler.detect_shots(video_path)
        print(f"\nDetected {len(shots)} shots:")
        for i, (start, end) in enumerate(shots[:5]):  # Show first 5
            print(f"  Shot {i}: {start:.2f}s - {end:.2f}s ({end-start:.2f}s)")
        if len(shots) > 5:
            print(f"  ... and {len(shots) - 5} more")
        
        # Sample keyframes
        print(f"\nSampling keyframes (1 per shot)...")
        frames = sampler.sample_keyframes(video_path, num_frames_per_shot=1)
        print(f"Sampled {len(frames)} frames")
        print(f"Saved to: {Path(frames[0]['path']).parent if frames else 'N/A'}")
    else:
        print("Usage: python sampler.py <video_path>")
