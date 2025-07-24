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

# Disable warnings and set HTTP fallback (useful for Streamlit Cloud)
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
    if not YouTubeTranscriptApi:
        logger.warning("youtube-transcript-api not installed.")
        return None

    try:
        languages = ['en', 'en-US', 'en-GB']
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        logger.info(f"Transcript options available: {[t.language_code for t in transcript_list]}")

        # Manual transcripts
        for lang in languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([lang])
                data = transcript.fetch()
                return TextFormatter().format_transcript(data)
            except Exception as e:
                logger.debug(f"No manual transcript for {lang}: {e}")

        # Auto-generated transcripts
        for lang in languages:
            try:
                transcript = transcript_list.find_generated_transcript([lang])
                data = transcript.fetch()
                return TextFormatter().format_transcript(data)
            except Exception as e:
                logger.debug(f"No auto transcript for {lang}: {e}")

        return None
    except Exception as e:
        logger.error(f"[Transcript API] Error: {e}")
        return None


def fetch_transcript_whisper(video_url: str) -> Optional[str]:
    if not whisper or not yt_dlp:
        logger.warning("Whisper or yt-dlp not available.")
        return None

    try:
        temp_dir = Path("youtube_summarizer/temp")
        temp_dir.mkdir(exist_ok=True)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
            'extractaudio': True,
            'audioformat': 'wav',
            'quiet': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            audio_file = ydl.prepare_filename(info)
            if not audio_file.endswith('.wav'):
                audio_file = audio_file.rsplit('.', 1)[0] + '.wav'

        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")

        logger.info("Transcribing audio...")
        result = model.transcribe(audio_file)

        try:
            os.remove(audio_file)
        except:
            pass

        return result["text"]

    except Exception as e:
        logger.error(f"[Whisper] Error: {e}")
        return None


def fetch_transcript(video_url: str) -> str:
    logger.info(f"🔎 Fetching transcript for: {video_url}")
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL: Could not extract video ID")

    logger.info(f"🎯 Video ID: {video_id}")

    # Try YouTube API
    transcript = fetch_transcript_api(video_id)
    if transcript:
        logger.info("✅ Transcript fetched via YouTube API")
        return transcript

    # Whisper fallback
    logger.warning("⚠️ YouTube API failed. Trying Whisper fallback...")
    transcript = fetch_transcript_whisper(video_url)
    if transcript:
        logger.info("✅ Transcript fetched via Whisper fallback")
        return transcript

    raise ValueError(
        "❌ Could not fetch transcript using any method. "
        "Please check if the video has captions available or try uploading a transcript manually."
    )


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
        print(f"First 200 characters: {transcript[:200]}...")
        print("✅ Valid" if validate_transcript(transcript) else "❌ Invalid")
    except Exception as e:
        print(f"❌ Error: {e}")
