"""
YouTube Transcript Fetcher Module

Handles fetching transcripts from YouTube videos:
1. YouTube Transcript API (primary)
2. Whisper fallback (if enabled)
"""

import os
import re
import logging
from typing import Optional
from pathlib import Path
import urllib3

# Disable SSL warnings (for Streamlit Cloud)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["YOUTUBE_TRANSCRIPT_API_FORCE_HTTP"] = "1"

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
except ImportError:
    YouTubeTranscriptApi = None
    TextFormatter = None

try:
    import whisper
    import yt_dlp
except ImportError:
    whisper = None
    yt_dlp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_transcript_api(video_id: str) -> Optional[str]:
    import traceback
    if not YouTubeTranscriptApi:
        logger.warning("youtube-transcript-api not installed.")
        return None

    try:
        languages = ['en', 'en-US', 'en-GB']
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        logger.info(f"Transcript options: {[t.language_code for t in transcript_list]}")

        for lang in languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                data = transcript.fetch()
                return TextFormatter().format_transcript(data)
            except Exception as e:
                logger.warning(f"Manual transcript not found for lang {lang}: {e}")

        for lang in languages:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                data = transcript.fetch()
                return TextFormatter().format_transcript(data)
            except Exception as e:
                logger.warning(f"Generated transcript not found for lang {lang}: {e}")

    except Exception as e:
        logger.error(f"[YouTubeTranscriptApi] Error: {e}")
        logger.error(traceback.format_exc())
    return None


def fetch_transcript_whisper(video_url: str) -> Optional[str]:
    import traceback
    import torch
    if not whisper or not yt_dlp:
        logger.warning("Whisper or yt-dlp not available.")
        return None

    try:
        temp_dir = Path("temp_audio")
        temp_dir.mkdir(exist_ok=True)

        output_path = temp_dir / "audio.%(ext)s"

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(output_path),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': False,
            'noplaylist': True,
        }

        logger.info("📥 Downloading audio using yt_dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        audio_file = temp_dir / "audio.mp3"
        if not audio_file.exists():
            logger.error("❌ Audio file not created at expected path: %s", audio_file)
            return None
        else:
            logger.info("✅ Audio file created at: %s", audio_file)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🎧 Loading Whisper model on device: {device}...")
        model = whisper.load_model("base", device=device)  # Consider 'tiny' if RAM is tight

        logger.info("🎧 Transcribing audio with Whisper...")
        result = model.transcribe(str(audio_file))
        logger.info("✅ Whisper transcription result keys: %s", list(result.keys()) if result else "None")
        logger.info("✅ Whisper transcription text length: %d", len(result.get("text", "")) if result and "text" in result else 0)

        audio_file.unlink(missing_ok=True)  # Clean up
        return result.get("text") if result else None

    except Exception as e:
        logger.error(f"[Whisper] Transcription error: {e}")
        logger.error(traceback.format_exc())
        return None


def fetch_transcript(video_url: str) -> str:
    import traceback
    logger.info(f"🔎 Fetching transcript for: {video_url}")
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("❌ Invalid YouTube URL: Could not extract video ID")

    logger.info(f"🎯 Video ID: {video_id}")

    transcript = fetch_transcript_api(video_id)
    if transcript:
        logger.info("✅ Fetched via YouTubeTranscriptApi")
        return transcript

    logger.warning("⚠️ Falling back to Whisper method...")
    transcript = fetch_transcript_whisper(video_url)
    if transcript:
        logger.info("✅ Fetched via Whisper")
        return transcript

    logger.error("❌ Could not fetch transcript using any method for video: %s", video_url)
    logger.error(traceback.format_exc())
    raise ValueError("❌ Could not fetch transcript using any method. Ensure the video has captions or allow audio processing.")


def validate_transcript(transcript: str) -> bool:
    if not transcript or len(transcript.strip()) < 100:
        return False
    words = transcript.split()
    if len(words) < 50:
        return False
    unique_chars = len(set(transcript.lower().replace(' ', '')))
    return unique_chars >= 10


if __name__ == "__main__":
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    try:
        transcript = fetch_transcript(test_url)
        print(f"Transcript length: {len(transcript)} characters")
        print(f"First 200 chars:\n{transcript[:200]}")
        print("✅ Valid" if validate_transcript(transcript) else "❌ Invalid")
    except Exception as e:
        print(f"❌ Error: {e}")
