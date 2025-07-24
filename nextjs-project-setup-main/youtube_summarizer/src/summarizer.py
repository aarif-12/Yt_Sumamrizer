"""
Chunk Summarization Module

1. Use GPT-4o via OpenRouter API for high-quality summaries
2. Fall back to facebook/bart-large-cnn if GPT fails
3. Output Markdown summaries
4. Efficient memory handling for 8GB RAM machines
"""

import logging
import torch
from typing import List, Optional
from pathlib import Path
import gc
import os

# ------------------ OpenRouter / OpenAI v1 client ------------------
import openai

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or "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",
)

# ------------------ GPT-4o Summarization ------------------
def gpt_summarize(
    text: str,
    model: str = "openai/gpt-4o",
    max_tokens: int = 1024
) -> Optional[str]:
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "LocalSummarizer"
            },
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional summarizer. "
                        "Summarize the user's content in clean, well-structured Markdown "
                        "using headings and bullet points. Only include code blocks "
                        "if the original input contains code."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize this in markdown:\n\n{text.strip()}"
                }
            ],
            max_tokens=max_tokens,
            temperature=0.5
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[GPT-4o] Summarization failed: {e}")
        return None

# ------------------ Transformers Setup ------------------
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    pipeline = None

# ------------------ Logger ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------ BART Summarizer ------------------
class BARTSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", device="auto"):
        self.model_name = model_name
        self.device = self._select_device(device)
        logger.info(f"[BART] Using device: {self.device}")
        self._load_model()

    def _select_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_model(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed.")
        try:
            logger.info("[BART] Loading model...")
            self.summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info("[BART] Loaded.")
        except Exception as e:
            logger.error(f"[BART] Load error: {e}")
            self.summarizer = None
            raise

    def summarize_chunk(
        self,
        text: str,
        min_length: int = 64,
        max_length: int = 512,
        do_sample: bool = False,
        prefer_gpt: bool = True
    ) -> str:
        clean = " ".join(text.strip().split())
        if len(clean.split()) < 20:
            return clean

        if prefer_gpt:
            logger.info("[Hybrid] Attempting GPT-4o summary...")
            gpt_summary = gpt_summarize(clean)
            if gpt_summary:
                return gpt_summary
            logger.warning("[Hybrid] GPT-4o failed, falling back to BART")

        try:
            out = self.summarizer(
                clean,
                min_length=min_length,
                max_length=max_length,
                do_sample=do_sample,
                num_beams=4,
                early_stopping=True
            )
            return out[0]["summary_text"].strip()
        except Exception as e:
            logger.error(f"[BART] Summarization error: {e}")
            # Simple fallback: first two sentences
            return ". ".join(clean.split(".")[:2]).strip() + "."

    def batch_summarize(self, chunks: List[str], **kwargs) -> List[str]:
        summaries = []
        for idx, chunk in enumerate(chunks, start=1):
            logger.info(f"[Hybrid] Summarizing chunk {idx}/{len(chunks)}")
            try:
                summaries.append(self.summarize_chunk(chunk, **kwargs))
            except Exception as e:
                logger.error(f"[Hybrid] Chunk {idx} failed: {e}")
                summaries.append(". ".join(chunk.split(".")[:2]).strip() + ".")
            if idx % 5 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return summaries

    def cleanup(self):
        if hasattr(self, "summarizer"):
            del self.summarizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[BART] Resources cleaned")

# ------------------ Singleton Wrapper ------------------
_global_summarizer: Optional[BARTSummarizer] = None

def get_summarizer(model_name="facebook/bart-large-cnn") -> BARTSummarizer:
    global _global_summarizer
    if _global_summarizer is None or _global_summarizer.model_name != model_name:
        if _global_summarizer:
            _global_summarizer.cleanup()
        _global_summarizer = BARTSummarizer(model_name)
    return _global_summarizer

# ------------------ Public Interface ------------------
def summarize_chunk(
    text: str,
    model_name: str = "facebook/bart-large-cnn",
    min_length: int = 64,
    max_length: int = 512,
    prefer_gpt: bool = True
) -> str:
    return get_summarizer(model_name).summarize_chunk(
        text,
        min_length=min_length,
        max_length=max_length,
        prefer_gpt=prefer_gpt
    )

def summarize_chunks_batch(
    chunks: List[str],
    model_name: str = "facebook/bart-large-cnn",
    min_length: int = 64,
    max_length: int = 512,
    save_summaries: bool = True,
    prefer_gpt: bool = True
) -> List[str]:
    logger.info(f"[Hybrid] Batch summarizing {len(chunks)} chunks")
    summarizer = get_summarizer(model_name)
    out = summarizer.batch_summarize(
        chunks,
        min_length=min_length,
        max_length=max_length,
        prefer_gpt=prefer_gpt
    )

    if save_summaries:
        od = Path("youtube_summarizer/outputs/summaries")
        od.mkdir(parents=True, exist_ok=True)
        for f in od.glob("summary_*.txt"):
            f.unlink()
        for idx, sm in enumerate(out, start=1):
            path = od / f"summary_{idx:03d}.txt"
            path.write_text(
                f"CHUNK {idx} SUMMARY\n{'='*30}\n\n{sm}\n",
                encoding="utf-8"
            )
            logger.info(f"[Hybrid] Saved summary {idx}")
    return out
