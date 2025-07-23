"""
Chunk Summarization Module

This module handles summarizing individual chunks using facebook/bart-large-cnn:
1. Load and configure BART model for summarization
2. Process chunks with optimal parameters
3. Ensure comprehensive summaries (no one-liners)
4. Handle memory efficiently for 8GB RAM systems
"""

import logging
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
import gc

try:
    from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pipeline = None
    BartTokenizer = None
    BartForConditionalGeneration = None
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BARTSummarizer:
    """BART-based summarization with memory optimization for 8GB RAM"""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: str = "auto"):
        """
        Initialize the BART summarizer
        
        Args:
            model_name: Name of the BART model to use
            device: Device to run on ('auto', 'cpu', 'cuda', or specific device)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = self._determine_device(device)
        
        logger.info(f"Initializing BART summarizer with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            if torch.cuda.is_available():
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory >= 4:  # At least 4GB GPU memory
                    return "cuda"
                else:
                    logger.info(f"GPU has only {gpu_memory:.1f}GB memory, using CPU")
                    return "cpu"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the BART model and tokenizer"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library not available")
            
            # Load with memory optimization
            logger.info("Loading BART model...")
            
            if self.device == "cpu":
                # CPU optimization
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=-1,  # CPU
                    torch_dtype=torch.float32
                )
            else:
                # GPU optimization
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0,  # First GPU
                    torch_dtype=torch.float16  # Use half precision to save memory
                )
            
            logger.info("BART model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load BART model: {e}")
            self.summarizer = None
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for optimal summarization"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Ensure text ends with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        return text
    
    def _postprocess_summary(self, summary: str) -> str:
        """Postprocess summary to ensure quality"""
        # Remove leading/trailing whitespace
        summary = summary.strip()
        
        # Ensure proper capitalization
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending punctuation
        if summary and not summary[-1] in '.!?':
            summary += '.'
        
        return summary
    
    def summarize_chunk(self, text: str, min_length: int = 64, max_length: int = 512, 
                       do_sample: bool = False) -> str:
        """
        Summarize a single text chunk
        
        Args:
            text: Input text to summarize
            min_length: Minimum summary length in tokens
            max_length: Maximum summary length in tokens
            do_sample: Whether to use sampling for generation
            
        Returns:
            Summary text
        """
        if not self.summarizer:
            raise RuntimeError("BART model not loaded")
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Check text length
        words = processed_text.split()
        if len(words) < 20:
            logger.warning("Text too short for meaningful summarization")
            return processed_text
        
        try:
            # Adjust parameters based on input length
            input_length = len(words)
            
            # Dynamic length adjustment
            if input_length < 100:
                adjusted_min = max(20, min_length // 2)
                adjusted_max = max(adjusted_min + 20, max_length // 2)
            elif input_length > 500:
                adjusted_min = min_length
                adjusted_max = max_length
            else:
                # Scale based on input length
                scale_factor = input_length / 300
                adjusted_min = int(min_length * scale_factor)
                adjusted_max = int(max_length * scale_factor)
            
            # Ensure reasonable bounds
            adjusted_min = max(20, min(adjusted_min, 150))
            adjusted_max = max(adjusted_min + 20, min(adjusted_max, 800))
            
            logger.debug(f"Summarizing {input_length} words -> {adjusted_min}-{adjusted_max} tokens")
            
            # Generate summary
            summary_result = self.summarizer(
                processed_text,
                min_length=adjusted_min,
                max_length=adjusted_max,
                do_sample=do_sample,
                num_beams=4,  # Use beam search for better quality
                length_penalty=1.0,
                early_stopping=True
            )
            
            # Extract summary text
            summary = summary_result[0]['summary_text']
            
            # Postprocess
            summary = self._postprocess_summary(summary)
            
            # Validate summary quality
            if len(summary.split()) < 10:
                logger.warning("Generated summary is very short, using longer parameters")
                # Retry with longer parameters
                summary_result = self.summarizer(
                    processed_text,
                    min_length=max(30, adjusted_min),
                    max_length=max(100, adjusted_max),
                    do_sample=True,
                    num_beams=2
                )
                summary = self._postprocess_summary(summary_result[0]['summary_text'])
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            # Fallback: return first few sentences
            sentences = processed_text.split('.')[:3]
            return '. '.join(sentences) + '.'
    
    def batch_summarize(self, chunks: List[str], **kwargs) -> List[str]:
        """
        Summarize multiple chunks efficiently
        
        Args:
            chunks: List of text chunks
            **kwargs: Arguments passed to summarize_chunk
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            
            try:
                summary = self.summarize_chunk(chunk, **kwargs)
                summaries.append(summary)
                
                # Memory cleanup every few chunks
                if (i + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i+1}: {e}")
                # Use fallback summary
                sentences = chunk.split('.')[:2]
                fallback = '. '.join(sentences) + '.'
                summaries.append(fallback)
        
        return summaries
    
    def cleanup(self):
        """Clean up model resources"""
        if hasattr(self, 'summarizer') and self.summarizer:
            del self.summarizer
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model resources cleaned up")

# Global summarizer instance (singleton pattern for memory efficiency)
_global_summarizer = None

def get_summarizer(model_name: str = "facebook/bart-large-cnn") -> BARTSummarizer:
    """Get or create global summarizer instance"""
    global _global_summarizer
    
    if _global_summarizer is None or _global_summarizer.model_name != model_name:
        if _global_summarizer:
            _global_summarizer.cleanup()
        
        _global_summarizer = BARTSummarizer(model_name)
    
    return _global_summarizer

def summarize_chunk(text: str, model_name: str = "facebook/bart-large-cnn", 
                   min_length: int = 64, max_length: int = 512) -> str:
    """
    Main function to summarize a single chunk
    
    Args:
        text: Text chunk to summarize
        model_name: BART model to use
        min_length: Minimum summary length
        max_length: Maximum summary length
        
    Returns:
        Summary text
    """
    summarizer = get_summarizer(model_name)
    return summarizer.summarize_chunk(text, min_length, max_length)

def summarize_chunks_batch(chunks: List[str], model_name: str = "facebook/bart-large-cnn",
                          min_length: int = 64, max_length: int = 512,
                          save_summaries: bool = True) -> List[str]:
    """
    Summarize multiple chunks and optionally save to files
    
    Args:
        chunks: List of text chunks
        model_name: BART model to use
        min_length: Minimum summary length
        max_length: Maximum summary length
        save_summaries: Whether to save individual summaries
        
    Returns:
        List of summaries
    """
    logger.info(f"Starting batch summarization of {len(chunks)} chunks")
    
    summarizer = get_summarizer(model_name)
    summaries = summarizer.batch_summarize(
        chunks, 
        min_length=min_length, 
        max_length=max_length
    )
    
    if save_summaries:
        # Save individual summaries
        summaries_dir = Path("youtube_summarizer/outputs/summaries")
        summaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing summary files
        for existing_file in summaries_dir.glob("summary_*.txt"):
            existing_file.unlink()
        
        for i, summary in enumerate(summaries, 1):
            summary_path = summaries_dir / f"summary_{i:03d}.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"CHUNK {i} SUMMARY\n{'=' * 30}\n\n{summary}\n")
            
            logger.info(f"Saved summary {i} to {summary_path}")
    
    logger.info(f"Completed batch summarization: {len(summaries)} summaries generated")
    return summaries

def validate_summary_quality(summary: str, original_text: str) -> Dict[str, Any]:
    """
    Validate the quality of a generated summary
    
    Args:
        summary: Generated summary
        original_text: Original text
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {
        'length_words': len(summary.split()),
        'length_chars': len(summary),
        'compression_ratio': len(original_text) / len(summary) if summary else 0,
        'sentence_count': len([s for s in summary.split('.') if s.strip()]),
        'has_proper_ending': summary.endswith(('.', '!', '?')) if summary else False,
        'is_too_short': len(summary.split()) < 10,
        'is_too_long': len(summary.split()) > 200,
        'quality_score': 0.0
    }
    
    # Calculate quality score
    score = 0.0
    
    # Length appropriateness (30 points)
    if 15 <= metrics['length_words'] <= 100:
        score += 30
    elif 10 <= metrics['length_words'] <= 150:
        score += 20
    else:
        score += 10
    
    # Compression ratio (25 points)
    if 3 <= metrics['compression_ratio'] <= 10:
        score += 25
    elif 2 <= metrics['compression_ratio'] <= 15:
        score += 15
    else:
        score += 5
    
    # Proper formatting (20 points)
    if metrics['has_proper_ending']:
        score += 10
    if metrics['sentence_count'] >= 2:
        score += 10
    
    # Content preservation (25 points)
    # Simple keyword overlap check
    original_words = set(original_text.lower().split())
    summary_words = set(summary.lower().split())
    overlap = len(original_words & summary_words) / len(original_words) if original_words else 0
    score += overlap * 25
    
    metrics['quality_score'] = score
    return metrics

if __name__ == "__main__":
    # Test the summarizer
    sample_text = """
    Artificial intelligence is a rapidly growing field that has the potential to transform many aspects of our lives. 
    Machine learning, a subset of AI, involves training algorithms on large datasets to recognize patterns and make predictions. 
    Deep learning, which uses neural networks with multiple layers, has been particularly successful in areas like image recognition and natural language processing. 
    However, there are also concerns about the ethical implications of AI, including issues around bias, privacy, and job displacement. 
    As AI continues to advance, it will be important to develop frameworks for ensuring that these technologies are used responsibly and for the benefit of society as a whole.
    """
    
    print("Testing BART summarizer...")
    print(f"Original text ({len(sample_text.split())} words):")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    try:
        summary = summarize_chunk(sample_text)
        print(f"Summary ({len(summary.split())} words):")
        print(summary)
        
        # Validate quality
        quality = validate_summary_quality(summary, sample_text)
        print(f"\nQuality metrics:")
        for key, value in quality.items():
            print(f"  {key}: {value}")
        
        print(f"\nOverall quality score: {quality['quality_score']:.1f}/100")
        
    except Exception as e:
        print(f"❌ Error: {e}")
