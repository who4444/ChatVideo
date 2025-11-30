import imagehash
import subprocess
from dataclasses import dataclass
# import torch
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm

class FrameExtractor:

    def __init__(self,
                #  device = 'cuda',
                 output_dir = '',
                 max_interval = 15,
                 min_interval = 5,
                 save_frames = True):
        # self.device = torch.device(device if torch.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.save_frames = save_frames
        # self.device = device
        if save_frames and output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    def extract_frames(self, video_path, output_dir):
        """Extract I and P frames using ffmpeg 
        Args: 
        1. Video path
        2. Output dir
        """    
        video_path  = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vf", "select='eq(pict_type\\,I)+eq(pict_type\\,P)',showinfo",
            "-vsync", "vfr", "-qscale:v", "2",
            str(Path(output_dir) / "%04d.jpg")
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_dir
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Frame extraction failed: {e.stderr}")
    
    def remove_dupes(self, output_dir, threshold=5, window=10, delete_files=True):
        """
        Remove near-duplicate frames using perceptual hashing with a sliding window.
        
        Args:
            output_dir (str | Path): Directory containing extracted frames (.jpg)
            threshold (int): Max Hamming distance between considered duplicates
            window (int): Number of recent hashes to compare against
            delete_files (bool): Whether to delete duplicate files
            
        Returns:
            list[str]: Filenames of kept frames
            
        Raises:
            RuntimeError: If image processing fails
        """
        output_dir = Path(output_dir)
        img_files = sorted([f for f in os.listdir(output_dir) if f.lower().endswith((".jpg", ".jpeg"))])
        last_hashes = []
        kept = []
        removed_count = 0
        errors = []

        for fname in tqdm(img_files, desc="Removing dupes"):
            path = output_dir / fname
            try:
                with Image.open(path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    h = imagehash.phash(img)
                    
                # Compare with recent hashes
                if not any(abs(h - prev) <= threshold for prev in last_hashes):
                    kept.append(fname)
                else:
                    removed_count += 1
                    if delete_files:
                        try:
                            os.remove(path)
                        except OSError as e:
                            errors.append(f"Failed to delete {fname}: {e}")
                            
                # Slide window
                last_hashes = (last_hashes + [h])[-window:]
                    
            except Exception as e:
                errors.append(f"Failed to process {fname}: {e}")
                continue  # Skip this file on error

            # compare with recent hashes
            if not any(abs(h - prev) <= threshold for prev in last_hashes):
                kept.append(fname)
            else:
                removed_count += 1
                if delete_files:
                    os.remove(path)
        print(f"Removed {removed_count} near-duplicates, kept {len(kept)} frames.")
        if errors:
            print("Warnings during processing:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"- {error}")
            if len(errors) > 5:
                print(f"...and {len(errors) - 5} more errors")
                
        return kept
    
    def process_video(self, video_path, dedup_thr, dedup_window):
        # Complete pipeline

        video_name  = Path(video_path).stem
        output_dir  = self.output_dir/video_name
        self.extract_frames(video_path, output_dir)
        kept_frames = self.remove_dupes(output_dir, dedup_thr, dedup_window)
        return kept_frames
    
