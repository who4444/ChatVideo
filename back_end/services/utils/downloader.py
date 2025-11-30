import yt_dlp
import re
import os

def extract_video_id(url):
    """Extract video ID from a YouTube URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")
def download_video(url, output_path):
    video_id = extract_video_id(url)
    output_path = os.path.join(output_path, f"{video_id}.mp4")
    yt_dlp_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': output_path,
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(yt_dlp_opts) as ytdl:
        ytdl.download([url])
    return output_path