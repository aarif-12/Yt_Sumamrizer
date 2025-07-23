"""
Semantic Text Splitter Module

This module handles intelligent chunking of cleaned transcripts:
1. Semantic splitting using sentence transformers
2. Syntactic splitting using NLTK/spaCy
3. Preserves full ideas and context
4. Avoids arbitrary token count splits
"""

import re
import logging
from typing import List, Tuple, Dict, Any
from pathlib import Path
import math

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    np = None
    cosine_similarity = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
except ImportError:
    nltk = None
    sent_tokenize = None

try:
    import spacy
    # Try to load English model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = None
except ImportError:
    spacy = None
    nlp = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSplitter:
    """Handles semantic-based text splitting using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic splitter
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model = None
        self.model_name = model_name
        
        if SentenceTransformer:
            try:
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("sentence-transformers not available")
    
    def get_sentence_embeddings(self, sentences: List[str]):
        """
        Get embeddings for a list of sentences
        
        Args:
            sentences: List of sentence strings
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model or not sentences or not SENTENCE_TRANSFORMERS_AVAILABLE:
            return []
        
        try:
            embeddings = self.model.encode(sentences)
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            return []
    
    def find_semantic_boundaries(self, sentences: List[str], threshold: float = 0.3) -> List[int]:
        """
        Find semantic boundaries between sentences
        
        Args:
            sentences: List of sentences
            threshold: Similarity threshold for boundary detection
            
        Returns:
            List of indices where semantic boundaries occur
        """
        if not self.model or len(sentences) < 2:
            return []
        
        embeddings = self.get_sentence_embeddings(sentences)
        if not SENTENCE_TRANSFORMERS_AVAILABLE or len(embeddings) == 0:
            return []
        
        boundaries = [0]  # Always start with first sentence
        
        for i in range(1, len(sentences)):
            # Calculate similarity between consecutive sentences
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            # If similarity is below threshold, it's a boundary
            if similarity < threshold:
                boundaries.append(i)
        
        return boundaries

class SyntacticSplitter:
    """Handles syntactic-based text splitting using NLTK/spaCy"""
    
    def __init__(self):
        """Initialize the syntactic splitter"""
        self.use_spacy = nlp is not None
        self.use_nltk = nltk is not None
        
        if self.use_spacy:
            logger.info("Using spaCy for syntactic analysis")
        elif self.use_nltk:
            logger.info("Using NLTK for syntactic analysis")
        else:
            logger.warning("Neither spaCy nor NLTK available for syntactic analysis")
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text using available tools
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if self.use_spacy:
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        elif self.use_nltk:
            sentences = sent_tokenize(text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            # Fallback: simple regex-based sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def find_topic_boundaries(self, sentences: List[str]) -> List[int]:
        """
        Find topic boundaries using syntactic cues
        
        Args:
            sentences: List of sentences
            
        Returns:
            List of boundary indices
        """
        boundaries = [0]
        
        # Topic transition indicators
        transition_words = {
            'strong': ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 
                      'nevertheless', 'meanwhile', 'in contrast', 'on the other hand'],
            'medium': ['next', 'then', 'also', 'additionally', 'similarly', 'likewise',
                      'first', 'second', 'third', 'finally', 'lastly'],
            'weak': ['now', 'so', 'well', 'okay', 'right', 'anyway']
        }
        
        for i, sentence in enumerate(sentences[1:], 1):
            sentence_lower = sentence.lower()
            
            # Check for strong topic transitions
            for word in transition_words['strong']:
                if sentence_lower.startswith(word + ' ') or f' {word} ' in sentence_lower:
                    boundaries.append(i)
                    break
            
            # Check for paragraph markers (if preserved from cleaning)
            if sentence.startswith('[') and ']' in sentence:
                boundaries.append(i)
        
        return boundaries

def combine_boundaries(semantic_boundaries: List[int], syntactic_boundaries: List[int], 
                      total_sentences: int) -> List[int]:
    """
    Combine semantic and syntactic boundaries intelligently
    
    Args:
        semantic_boundaries: Boundaries from semantic analysis
        syntactic_boundaries: Boundaries from syntactic analysis
        total_sentences: Total number of sentences
        
    Returns:
        Combined list of boundary indices
    """
    # Combine and sort boundaries
    all_boundaries = set(semantic_boundaries + syntactic_boundaries)
    all_boundaries.add(0)  # Always include start
    all_boundaries.add(total_sentences)  # Always include end
    
    boundaries = sorted(list(all_boundaries))
    
    # Remove boundaries that are too close together (less than 3 sentences)
    filtered_boundaries = [boundaries[0]]
    
    for boundary in boundaries[1:]:
        if boundary - filtered_boundaries[-1] >= 3:
            filtered_boundaries.append(boundary)
    
    # Ensure we don't have too many small chunks
    if len(filtered_boundaries) > total_sentences // 5:
        # Keep only the most significant boundaries
        step = len(filtered_boundaries) // (total_sentences // 5)
        filtered_boundaries = filtered_boundaries[::step]
        if filtered_boundaries[-1] != total_sentences:
            filtered_boundaries.append(total_sentences)
    
    return filtered_boundaries

def create_chunks_from_boundaries(sentences: List[str], boundaries: List[int]) -> List[str]:
    """
    Create text chunks based on boundary indices
    
    Args:
        sentences: List of sentences
        boundaries: List of boundary indices
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]
        
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = ' '.join(chunk_sentences)
        
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
    
    return chunks

def validate_chunks(chunks: List[str], min_words: int = 50, max_words: int = 500) -> Tuple[List[str], List[str]]:
    """
    Validate and filter chunks based on size criteria
    
    Args:
        chunks: List of text chunks
        min_words: Minimum words per chunk
        max_words: Maximum words per chunk
        
    Returns:
        Tuple of (valid_chunks, issues)
    """
    valid_chunks = []
    issues = []
    
    for i, chunk in enumerate(chunks):
        word_count = len(chunk.split())
        
        if word_count < min_words:
            issues.append(f"Chunk {i+1} too short ({word_count} words)")
            # Try to merge with next chunk if available
            if i + 1 < len(chunks):
                merged = chunk + " " + chunks[i + 1]
                if len(merged.split()) <= max_words:
                    valid_chunks.append(merged)
                    chunks[i + 1] = ""  # Mark as processed
                    continue
        elif word_count > max_words:
            issues.append(f"Chunk {i+1} too long ({word_count} words)")
            # Split the chunk
            sentences = re.split(r'[.!?]+', chunk)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            current_chunk = ""
            for sentence in sentences:
                test_chunk = current_chunk + " " + sentence if current_chunk else sentence
                if len(test_chunk.split()) <= max_words:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        valid_chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                valid_chunks.append(current_chunk.strip())
        else:
            if chunk:  # Only add non-empty chunks
                valid_chunks.append(chunk)
    
    return valid_chunks, issues

def split_into_chunks(transcript_path: str, target_chunk_size: int = 10, 
                     save_chunks: bool = True) -> List[str]:
    """
    Main function to split transcript into semantic chunks
    
    Args:
        transcript_path: Path to cleaned transcript file
        target_chunk_size: Target number of sentences per chunk (guidance only)
        save_chunks: Whether to save chunks to individual files
        
    Returns:
        List of text chunks
    """
    logger.info(f"Starting text splitting for: {transcript_path}")
    
    # Read the transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Error reading transcript file: {e}")
        raise
    
    # Remove header/footer if present
    text = re.sub(r'CLEANED YOUTUBE TRANSCRIPT\n=+\n', '', text)
    text = re.sub(r'\n=+\nTranscript processed.*', '', text, flags=re.DOTALL)
    
    # Initialize splitters
    semantic_splitter = SemanticSplitter()
    syntactic_splitter = SyntacticSplitter()
    
    # Extract sentences
    logger.info("Extracting sentences...")
    sentences = syntactic_splitter.extract_sentences(text)
    logger.info(f"Found {len(sentences)} sentences")
    
    if len(sentences) < 5:
        logger.warning("Too few sentences for meaningful chunking")
        return [text]  # Return entire text as single chunk
    
    # Find semantic boundaries
    logger.info("Finding semantic boundaries...")
    semantic_boundaries = semantic_splitter.find_semantic_boundaries(sentences)
    
    # Find syntactic boundaries
    logger.info("Finding syntactic boundaries...")
    syntactic_boundaries = syntactic_splitter.find_topic_boundaries(sentences)
    
    # Combine boundaries
    logger.info("Combining boundaries...")
    final_boundaries = combine_boundaries(semantic_boundaries, syntactic_boundaries, len(sentences))
    
    # Create chunks
    logger.info("Creating chunks...")
    chunks = create_chunks_from_boundaries(sentences, final_boundaries)
    
    # Validate chunks
    logger.info("Validating chunks...")
    valid_chunks, issues = validate_chunks(chunks)
    
    if issues:
        logger.warning("Chunk validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    logger.info(f"Created {len(valid_chunks)} valid chunks")
    
    # Save chunks to files if requested
    if save_chunks:
        chunks_dir = Path("youtube_summarizer/outputs/chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing chunk files
        for existing_file in chunks_dir.glob("chunk_*.txt"):
            existing_file.unlink()
        
        for i, chunk in enumerate(valid_chunks, 1):
            chunk_path = chunks_dir / f"chunk_{i:03d}.txt"
            with open(chunk_path, 'w', encoding='utf-8') as f:
                f.write(f"CHUNK {i}\n{'=' * 20}\n\n{chunk}\n")
            
            logger.info(f"Saved chunk {i} to {chunk_path}")
    
    return valid_chunks

if __name__ == "__main__":
    # Test the splitter
    sample_text = """
    Hello everyone and welcome to this presentation about artificial intelligence. 
    Today we're going to explore the fascinating world of machine learning and its applications.
    
    First, let's start with the basics. What exactly is artificial intelligence? 
    Artificial intelligence, or AI, is the simulation of human intelligence in machines.
    These machines are programmed to think and learn like humans.
    
    However, there are different types of AI that we should understand. 
    The first type is narrow AI, which is designed to perform specific tasks.
    Examples include voice assistants like Siri and Alexa.
    
    On the other hand, general AI would have the ability to understand and learn any intellectual task.
    This type of AI doesn't exist yet, but it's what many researchers are working towards.
    
    Furthermore, machine learning is a subset of AI that focuses on algorithms.
    These algorithms can learn and make decisions from data without being explicitly programmed.
    
    In conclusion, AI and machine learning are transforming our world in incredible ways.
    Thank you for your attention, and I hope you found this presentation informative.
    """
    
    # Save sample text to test file
    test_path = Path("test_transcript.txt")
    with open(test_path, 'w', encoding='utf-8') as f:
        f.write(sample_text)
    
    try:
        print("Testing text splitter...")
        chunks = split_into_chunks(str(test_path), save_chunks=False)
        
        print(f"\nCreated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n--- Chunk {i} ---")
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print(f"Word count: {len(chunk.split())}")
    
    finally:
        # Clean up test file
        if test_path.exists():
            test_path.unlink()
