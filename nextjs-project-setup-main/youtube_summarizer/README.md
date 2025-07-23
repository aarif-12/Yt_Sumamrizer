# 🎬 Merlin AI Video Assistant

A classic, professional offline video analysis assistant that transforms YouTube videos into structured, comprehensive summaries using state-of-the-art AI models.

## ✨ Features

- **🔌 Fully Offline Operation** - No external API calls required
- **🧠 Advanced AI Summarization** - Uses facebook/bart-large-cnn for high-quality summaries
- **📊 Structured Output** - Abstract, Key Insights, FAQs, and Hidden Gems
- **💾 Memory Optimized** - Designed for 8GB RAM systems
- **🎨 Professional UI** - Clean, modern Streamlit interface
- **📁 Multiple Export Formats** - Text and PDF output options
- **🔄 Semantic Chunking** - Intelligent content splitting preserves context

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB RAM (4GB GPU memory optional)
- Internet connection for initial model downloads

### Installation

1. **Clone or download the project**
   ```bash
   cd youtube_summarizer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models** (first run only)
   ```bash
   python -c "
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('all-MiniLM-L6-v2')
   
   from transformers import pipeline
   pipeline('summarization', model='facebook/bart-large-cnn')
   "
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 📋 Usage

### Basic Workflow

1. **Input**: Paste a YouTube URL or upload a transcript file
2. **Configure**: Adjust summarization parameters in the sidebar
3. **Process**: Click "🚀 Start Analysis" to begin
4. **Review**: Examine the structured summary with multiple sections
5. **Export**: Download results in text or PDF format

### Input Options

- **YouTube URL**: Any valid YouTube video URL
- **Transcript Upload**: Pre-existing transcript files (.txt format)

### Output Sections

- **📄 Abstract Overview**: 3-5 paragraph comprehensive summary
- **🔍 Key Insights**: 6-12 bullet points of main takeaways
- **❓ FAQs**: 5-8 question-answer pairs from content
- **💎 Hidden Gems**: 3-5 actionable tips and valuable insights

## 🏗️ Project Structure

```
youtube_summarizer/
├── app.py                # Streamlit UI application
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── src/                 # Core processing modules
│   ├── fetcher.py       # YouTube transcript extraction
│   ├── cleaner.py       # Transcript cleaning and formatting
│   ├── splitter.py      # Semantic text chunking
│   ├── summarizer.py    # BART-based summarization
│   └── aggregator.py    # Final summary assembly
│
└── outputs/             # Generated files (auto-created)
    ├── transcripts/     # Cleaned transcript files
    ├── chunks/          # Text chunks
    ├── summaries/       # Individual chunk summaries
    └── final/           # Final summary outputs
```

## ⚙️ Configuration

### Model Settings

- **Summarization Model**: facebook/bart-large-cnn (default)
- **Chunk Size**: 5-20 sentences per chunk
- **Summary Length**: 50-800 tokens per chunk
- **Device**: Auto-detection (GPU if available, CPU fallback)

### Processing Options

- **Save Intermediates**: Keep chunk and summary files
- **Generate PDF**: Create PDF version of final summary
- **Semantic Chunking**: Use AI-powered content splitting

## 🔧 Technical Details

### AI Models Used

1. **facebook/bart-large-cnn**: Primary summarization model
2. **all-MiniLM-L6-v2**: Sentence embeddings for semantic chunking
3. **youtube-transcript-api**: Transcript extraction
4. **whisper (fallback)**: Local audio transcription

### Memory Optimization

- **Model Loading**: Singleton pattern for memory efficiency
- **Batch Processing**: Chunked processing with garbage collection
- **Device Management**: Automatic GPU/CPU selection
- **Half Precision**: FP16 on GPU to reduce memory usage

### Processing Pipeline

1. **Transcript Extraction**: YouTube API → Whisper fallback
2. **Text Cleaning**: Remove artifacts, fix punctuation, add timestamps
3. **Semantic Chunking**: Sentence transformers + syntactic analysis
4. **Chunk Summarization**: BART model with optimized parameters
5. **Summary Aggregation**: Structured assembly with theme analysis

## 🛠️ Troubleshooting

### Common Issues

**"Model not found" error**
```bash
# Re-download models
python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"
```

**Memory issues**
- Reduce chunk size in sidebar
- Use CPU-only mode
- Close other applications

**Transcript extraction fails**
- Check video has captions available
- Try uploading transcript manually
- Ensure stable internet connection

**Poor summary quality**
- Increase minimum summary length
- Try different chunk sizes
- Check input transcript quality

### Performance Tips

- **GPU Usage**: Ensure CUDA is installed for GPU acceleration
- **Batch Size**: Process fewer chunks simultaneously if memory limited
- **Model Cache**: Models are cached after first download
- **Intermediate Files**: Disable saving if disk space limited

## 📊 System Requirements

### Minimum Requirements
- **RAM**: 8GB system memory
- **Storage**: 5GB free space (for models)
- **CPU**: Multi-core processor recommended
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 16GB system memory
- **GPU**: 4GB+ VRAM (NVIDIA with CUDA)
- **Storage**: 10GB free space
- **CPU**: 8+ cores

## 🔒 Privacy & Security

- **Fully Offline**: No data sent to external services after model download
- **Local Processing**: All analysis happens on your machine
- **No Tracking**: No analytics or usage tracking
- **Open Source**: All code is transparent and auditable

## 📈 Performance Benchmarks

### Typical Processing Times (8GB RAM, CPU-only)

- **5-minute video**: ~2-3 minutes processing
- **15-minute video**: ~5-7 minutes processing
- **30-minute video**: ~10-15 minutes processing
- **1-hour video**: ~20-30 minutes processing

### With GPU Acceleration (4GB+ VRAM)

- **Processing time**: ~50% faster than CPU-only
- **Memory usage**: More efficient with FP16 precision

## 🤝 Contributing

This project follows a modular architecture for easy extension:

1. **fetcher.py**: Add new transcript sources
2. **cleaner.py**: Improve text preprocessing
3. **splitter.py**: Enhance chunking algorithms
4. **summarizer.py**: Integrate new summarization models
5. **aggregator.py**: Add new output formats

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers library and BART models
- **Streamlit**: For the excellent web framework
- **YouTube Transcript API**: For transcript extraction
- **OpenAI Whisper**: For fallback transcription

## 📞 Support

For issues, questions, or contributions:

1. Check the troubleshooting section above
2. Review the project structure and code comments
3. Test with different videos and settings
4. Ensure all dependencies are properly installed

---

**Built with ❤️ for the AI community**

*Transform any YouTube video into actionable insights with the power of offline AI.*
