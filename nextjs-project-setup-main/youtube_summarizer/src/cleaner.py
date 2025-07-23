"""
Transcript Cleaner Module

This module handles cleaning and formatting raw YouTube transcripts:
1. Remove timestamps and formatting artifacts
2. Fix punctuation and capitalization
3. Insert logical timestamps every ~75 seconds
4. Group into coherent paragraphs
"""

import re
import logging
from typing import List, Tuple
from pathlib import Path
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def remove_artifacts(text: str) -> str:
    """
    Remove common transcript artifacts and formatting issues
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text
    """
    # Remove common YouTube transcript artifacts
    artifacts = [
        r'\[Music\]',
        r'\[Applause\]',
        r'\[Laughter\]',
        r'\[Inaudible\]',
        r'\[Background noise\]',
        r'\[.*?\]',  # Any text in brackets
        r'\(Music\)',
        r'\(Applause\)',
        r'\(Laughter\)',
        r'\(.*?\)',  # Any text in parentheses that looks like sound effects
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def fix_punctuation(text: str) -> str:
    """
    Fix common punctuation issues in transcripts
    
    Args:
        text: Text with potential punctuation issues
        
    Returns:
        Text with improved punctuation
    """
    # Add periods at the end of sentences that don't have punctuation
    # Look for word followed by capital letter or end of text
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    
    # Fix spacing around punctuation
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
    
    # Fix multiple punctuation marks
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    
    # Ensure sentences end with punctuation
    if text and not text[-1] in '.!?':
        text += '.'
    
    return text

def fix_capitalization(text: str) -> str:
    """
    Fix capitalization issues common in auto-generated transcripts
    
    Args:
        text: Text with potential capitalization issues
        
    Returns:
        Text with improved capitalization
    """
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    fixed_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            # Capitalize first letter of sentence
            sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
            
            # Fix common proper nouns (basic list)
            proper_nouns = [
                'YouTube', 'Google', 'Facebook', 'Twitter', 'Instagram', 'LinkedIn',
                'Microsoft', 'Apple', 'Amazon', 'Netflix', 'Tesla', 'SpaceX',
                'AI', 'API', 'CEO', 'CTO', 'USA', 'UK', 'EU', 'NASA', 'FBI', 'CIA'
            ]
            
            for noun in proper_nouns:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(noun.lower()) + r'\b'
                sentence = re.sub(pattern, noun, sentence, flags=re.IGNORECASE)
            
            fixed_sentences.append(sentence)
    
    return '. '.join(fixed_sentences)

def insert_timestamps(text: str, interval_seconds: int = 75) -> str:
    """
    Insert logical timestamps every ~75 seconds based on estimated speaking pace
    
    Args:
        text: Cleaned transcript text
        interval_seconds: Interval for timestamp insertion
        
    Returns:
        Text with timestamps inserted
    """
    # Estimate words per minute (average is ~150-200 WPM for presentations)
    words_per_minute = 175
    words_per_second = words_per_minute / 60
    
    words = text.split()
    total_words = len(words)
    
    # Calculate approximate positions for timestamps
    words_per_interval = int(words_per_second * interval_seconds)
    
    result_parts = []
    current_time = 0
    
    for i in range(0, total_words, words_per_interval):
        # Add timestamp
        time_str = str(timedelta(seconds=current_time)).split('.')[0]  # Remove microseconds
        result_parts.append(f"\n[{time_str}]")
        
        # Add words for this interval
        end_idx = min(i + words_per_interval, total_words)
        interval_text = ' '.join(words[i:end_idx])
        result_parts.append(interval_text)
        
        current_time += interval_seconds
    
    return ' '.join(result_parts)

def create_paragraphs(text: str) -> str:
    """
    Group sentences into logical paragraphs
    
    Args:
        text: Text with timestamps
        
    Returns:
        Text organized into paragraphs
    """
    # Split by timestamps
    sections = re.split(r'\n\[[^\]]+\]', text)
    timestamps = re.findall(r'\n\[[^\]]+\]', text)
    
    paragraphs = []
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # Add timestamp if available
        if i > 0 and i-1 < len(timestamps):
            paragraphs.append(timestamps[i-1].strip())
        
        # Split section into sentences
        sentences = re.split(r'[.!?]+', section.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Group sentences into paragraphs (3-5 sentences per paragraph)
        current_paragraph = []
        
        for sentence in sentences:
            current_paragraph.append(sentence)
            
            # Start new paragraph after 3-5 sentences or at natural breaks
            if (len(current_paragraph) >= 3 and 
                (len(current_paragraph) >= 5 or 
                 any(keyword in sentence.lower() for keyword in 
                     ['however', 'therefore', 'furthermore', 'moreover', 'in conclusion', 
                      'first', 'second', 'third', 'finally', 'next', 'then']))):
                
                paragraph_text = '. '.join(current_paragraph) + '.'
                paragraphs.append(paragraph_text)
                current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph) + '.'
            paragraphs.append(paragraph_text)
    
    return '\n\n'.join(paragraphs)

def clean_transcript(raw_transcript: str, save_to_file: bool = True) -> str:
    """
    Main function to clean and format a raw transcript
    
    Args:
        raw_transcript: Raw transcript text
        save_to_file: Whether to save the cleaned transcript to file
        
    Returns:
        Path to cleaned transcript file or cleaned text
    """
    logger.info("Starting transcript cleaning process...")
    
    # Step 1: Remove artifacts
    logger.info("Removing artifacts and formatting issues...")
    text = remove_artifacts(raw_transcript)
    
    # Step 2: Fix punctuation
    logger.info("Fixing punctuation...")
    text = fix_punctuation(text)
    
    # Step 3: Fix capitalization
    logger.info("Fixing capitalization...")
    text = fix_capitalization(text)
    
    # Step 4: Insert timestamps
    logger.info("Inserting logical timestamps...")
    text = insert_timestamps(text)
    
    # Step 5: Create paragraphs
    logger.info("Organizing into paragraphs...")
    text = create_paragraphs(text)
    
    # Add header
    cleaned_text = f"""CLEANED YOUTUBE TRANSCRIPT
{'=' * 50}

{text}

{'=' * 50}
Transcript processed and cleaned automatically.
Timestamps are estimated based on average speaking pace.
"""
    
    if save_to_file:
        # Save to file
        output_path = Path("youtube_summarizer/outputs/transcripts/cleaned_transcript.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"Cleaned transcript saved to: {output_path}")
        return str(output_path)
    else:
        return cleaned_text

def validate_cleaned_transcript(text: str) -> Tuple[bool, List[str]]:
    """
    Validate the quality of the cleaned transcript
    
    Args:
        text: Cleaned transcript text
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check minimum length
    if len(text) < 500:
        issues.append("Transcript is too short (< 500 characters)")
    
    # Check for reasonable sentence structure
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s for s in sentences if len(s.strip().split()) >= 3]
    
    if len(valid_sentences) < 5:
        issues.append("Too few valid sentences found")
    
    # Check for timestamps
    timestamps = re.findall(r'\[[^\]]+\]', text)
    if len(timestamps) < 2:
        issues.append("Insufficient timestamps found")
    
    # Check for paragraph structure
    paragraphs = text.split('\n\n')
    valid_paragraphs = [p for p in paragraphs if len(p.strip()) > 50]
    
    if len(valid_paragraphs) < 3:
        issues.append("Insufficient paragraph structure")
    
    # Check for excessive repetition
    words = text.lower().split()
    unique_words = set(words)
    if len(words) > 0 and len(unique_words) / len(words) < 0.3:
        issues.append("High word repetition detected")
    
    is_valid = len(issues) == 0
    return is_valid, issues

if __name__ == "__main__":
    # Test the cleaner with sample text
    sample_raw = """
    hello everyone welcome to this video today we're going to talk about artificial intelligence 
    and machine learning [Music] so first let's start with the basics what is AI 
    artificial intelligence is the simulation of human intelligence in machines 
    [Applause] that are programmed to think and learn like humans
    """
    
    print("Testing transcript cleaner...")
    print(f"Raw text: {sample_raw}")
    print("\n" + "="*50 + "\n")
    
    cleaned = clean_transcript(sample_raw, save_to_file=False)
    print(f"Cleaned text:\n{cleaned}")
    
    is_valid, issues = validate_cleaned_transcript(cleaned)
    print(f"\nValidation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
