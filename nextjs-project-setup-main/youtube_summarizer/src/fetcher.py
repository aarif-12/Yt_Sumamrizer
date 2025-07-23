"""
YouTube Transcript Fetcher Module

This module handles fetching transcripts from YouTube videos using multiple methods:
1. YouTube Transcript API (primary)
2. Local Whisper model (fallback)
"""

import os
import re
import logging
from typing import Optional, Dict, Any
from pathlib import Path

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_video_id(url: str) -> Optional[str]:
    """
    Extract video ID from various YouTube URL formats
    
    Args:
        url: YouTube URL in various formats
        
    Returns:
        Video ID string or None if not found
    """
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
    """
    Fetch transcript using YouTube Transcript API
    
    Args:
        video_id: YouTube video ID
        
    Returns:
        Raw transcript text or None if failed
    """
    if not YouTubeTranscriptApi:
        logger.warning("youtube-transcript-api not available")
        return None
    
    try:
        # Try to get transcript in preferred languages
        languages = ['en', 'en-US', 'en-GB']
        
        # Get available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try manual transcripts first (more accurate)
        for language in languages:
            try:
                transcript = transcript_list.find_manually_created_transcript([language])
                transcript_data = transcript.fetch()
                
                # Format transcript
                formatter = TextFormatter()
                text = formatter.format_transcript(transcript_data)
                
                logger.info(f"Successfully fetched manual transcript in {language}")
                return text
                
            except Exception as e:
                logger.debug(f"Manual transcript not available in {language}: {e}")
                continue
        
        # Try auto-generated transcripts
        for language in languages:
            try:
                transcript = transcript_list.find_generated_transcript([language])
                transcript_data = transcript.fetch()
                
                # Format transcript
                formatter = TextFormatter()
                text = formatter.format_transcript(transcript_data)
                
                logger.info(f"Successfully fetched auto-generated transcript in {language}")
                return text
                
            except Exception as e:
                logger.debug(f"Auto-generated transcript not available in {language}: {e}")
                continue
        
        logger.warning("No suitable transcript found")
        return None
        
    except Exception as e:
        logger.error(f"Error fetching transcript via API: {e}")
        return None

def fetch_transcript_whisper(video_url: str) -> Optional[str]:
    """
    Fetch transcript using local Whisper model (fallback method)
    
    Args:
        video_url: Full YouTube URL
        
    Returns:
        Transcript text or None if failed
    """
    if not whisper or not yt_dlp:
        logger.warning("Whisper or yt-dlp not available for fallback transcription")
        return None
    
    try:
        # Create temporary directory for audio
        temp_dir = Path("youtube_summarizer/temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Download audio using yt-dlp
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
            
            # Convert to wav if needed
            if not audio_file.endswith('.wav'):
                audio_file = audio_file.rsplit('.', 1)[0] + '.wav'
        
        # Load Whisper model (base model for 8GB RAM compatibility)
        logger.info("Loading Whisper model...")
        model = whisper.load_model("base")
        
        # Transcribe audio
        logger.info("Transcribing audio...")
        result = model.transcribe(audio_file)
        
        # Clean up temporary files
        try:
            os.remove(audio_file)
        except:
            pass
        
        logger.info("Successfully transcribed using Whisper")
        return result["text"]
        
    except Exception as e:
        logger.error(f"Error with Whisper transcription: {e}")
        return None

def fetch_transcript(video_url: str) -> str:
    """
    Main function to fetch transcript from YouTube video
    
    Args:
        video_url: YouTube video URL
        
    Returns:
        Raw transcript text
        
    Raises:
        ValueError: If no transcript could be fetched
    """
    logger.info(f"Fetching transcript for: {video_url}")
    
    # Extract video ID
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL - could not extract video ID")
    
    logger.info(f"Extracted video ID: {video_id}")
    
    # Try YouTube Transcript API first
    transcript = fetch_transcript_api(video_id)
    if transcript:
        logger.info("Successfully fetched transcript via YouTube API")
        return transcript
    
    # Fallback to Whisper
    logger.info("Falling back to Whisper transcription...")
    transcript = fetch_transcript_whisper(video_url)
    if transcript:
        logger.info("Successfully fetched transcript via Whisper")
        return transcript
    
    # If all methods fail
    raise ValueError(
        "Could not fetch transcript using any method. "
        "Please check if the video has captions available or try uploading a transcript file manually."
    )

def validate_transcript(transcript: str) -> bool:
    """
    Validate that the transcript contains meaningful content
    
    Args:
        transcript: Raw transcript text
        
    Returns:
        True if transcript appears valid
    """
    if not transcript or len(transcript.strip()) < 100:
        return False
    
    # Check for reasonable word count
    words = transcript.split()
    if len(words) < 50:
        return False
    
    # Check for reasonable character variety (not just repeated characters)
    unique_chars = len(set(transcript.lower().replace(' ', '')))
    if unique_chars < 10:
        return False
    
    return True

if __name__ == "__main__":
    # Test the fetcher
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing
    
    try:
        transcript = fetch_transcript(test_url)
        print(f"Transcript length: {len(transcript)} characters")
        print(f"First 200 characters: {transcript[:200]}...")
        
        if validate_transcript(transcript):
            print("✅ Transcript validation passed")
        else:
            print("❌ Transcript validation failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
