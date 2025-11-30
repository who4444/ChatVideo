import subprocess
from pathlib import Path
from faster_whisper import WhisperModel
import torch

class AudioProcessor:
    def __init__(self, output_dir='',
                 device = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_size='large-v3'):
        self.output_dir = Path(output_dir)
        self.device = device
        if output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # Load model once during init
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type="float16")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")

    def extract_audio(self, video_path, output_audio_path):
        """Extract audio from video using ffmpeg 
        Args: 
        1. Video path: Path to input video file
        2. Output audio path: Where to save the extracted audio
        
        Returns:
            Path: Path to the extracted audio file
        """    
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        output_audio_path = Path(output_audio_path)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-q:a", "0", "-map", "a",
            str(output_audio_path)
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return output_audio_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio: {e.stderr}")

    def transcribe_audio(self, audio_path, return_segments=True, language=None): 
        """Transcribe audio using Whisper model with timestamp-aligned segments
        
        Args:
            audio_path: Path to the audio file
            return_segments: If True, return full result with timestamps. If False, return only text.
            language: Optional language code to use (e.g. 'en'). If None, Whisper will detect.

        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            segments, info = self.model.transcribe(
                str(audio_path),
                beam_size = 5  
            )
            results = []
            for segment in segments:
                results.append({
                    "text": segment.text.strip(),
                    "start": segment.start,
                    "end": segment.end
        })
            return results
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")
        
    def process_video(self, video_path, 
                      language=None):
        video_path = Path(video_path)
        
        # 1. Get the base filename without extension
        video_stem = video_path.stem
        
        # 2. Define output paths
        audio_path = self.output_dir / f"{video_stem}.mp3"
        
        try:
            self.extract_audio(video_path, audio_path)

            result = self.transcribe_audio(audio_path, return_segments=True, language=language)
            srt_path = self.output_dir / f"{video_stem}.srt"

            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(result["segments"], start=1):
                    f.write(f"{i}\n")
                    f.write(f"{seg['start']:.2f} --> {seg['end']:.2f}\n")
                    f.write(f"{seg['text'].strip()}\n\n")
                

            return result
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
