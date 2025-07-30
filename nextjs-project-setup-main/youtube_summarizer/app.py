import streamlit as st
import os
from pathlib import Path
import sys


# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.fetcher import extract_video_id, fetch_transcript
from src.cleaner import clean_transcript
from src.splitter import split_into_chunks
from src.summarizer import summarize_chunks_batch
from src.aggregator import build_final_summary

# Configure Streamlit page
st.set_page_config(
    page_title="AI - Video Analysis Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Merlin AI-like styling
st.markdown("""
<style>
  /* GLOBAL RESET */
  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    background-color: #0a66c2;
    color: var(--color-text);
    font-family: var(--font-family);
    font-size: var(--fs-base);
    line-height: 1.5;
  }

  .main {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1rem;
  }

  h1, h2, h3 {
    color: #0a66c2;
  }




  /* BUTTON STYLING */
  .stButton>button,
  .btn, button {
    background-color: #0a66c2;
    color: white;
    border-radius: 6px;
    padding: 8px 16px;
    border: none;
    font-weight: bold;
    box-shadow: var(--shadow-sm);
    cursor: pointer;
    transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.2s ease;
  }

  .stButton>button:hover,
  .btn:hover, button:hover {
    background-color: #004182;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
  }

  /* INPUT STYLING */
  .stTextInput>div>div>input,
  input[type="text"] {
    width: 100%;
    padding: 0.75rem 1rem;
    border: 1px solid #0a66c2;
    border-radius: 6px;
    background-color: #fff;
    color: var(--color-text);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
  }

  .stTextInput>div>div>input:focus,
  input[type="text"]:focus {
    border-color: #004182;
    box-shadow: 0 0 0 3px rgba(10,102,194,0.3);
    outline: none;
  }

  input[type="text"]::placeholder {
    color: var(--color-text-secondary);
  }

  /* REMOVE RED FOCUS RING */
  input:focus,
  textarea:focus,
  select:focus {
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(10,102,194,0.3) !important;
    border-color: #004182 !important;
  }
    

  /* URL Column Styling */
  a.url-link {
    color: #0a66c2;
    text-decoration: none;
    font-weight: 500;
    transition: color 0.2s ease, text-decoration 0.2s ease;
  }

  a.url-link:hover,
  a.url-link:active,
  a.url-link:focus {
    text-decoration: underline;
    color: #004182;
  }

  /* HEADER & SUBTITLE */
  .main-header {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin: 2.5rem 1rem 0.75rem;
    color: #0a66c2;
    position: relative;
    padding-bottom: 0.5rem;
  }

  .main-header::after {
    content: '';
    display: block;
    width: 60px;
    height: 4px;
    margin: 0.5rem auto 0;
    background-color: #0a66c2;
    border-radius: var(--radius);
    transition: width 0.3s ease, background-color 0.3s ease;
  }

  .main-header:hover::after {
    width: 100px;
    background-color: #004182;
  }

  .subtitle {
    font-size: 1rem;
    font-weight: 500;
    color: var(--color-text-secondary);
    text-align: center;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
  }

  /* PANELS / CARDS */
  .panel {
    background-color: var(--color-panel);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 1.5rem auto;
    max-width: 800px;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease, transform 0.3s ease;
  }

  .panel:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-2px);
  }

  /* STEP HEADERS */
  .step-header {
    font-size: 1.375rem;
    font-weight: 500;
    color: #0a66c2;
    margin: 2rem 0 0.75rem;
    padding-bottom: 0.25rem;
    border-bottom: 2px solid #0a66c2;
    transition: border-color 0.3s ease, color 0.3s ease;
  }

  .step-header:hover {
    color: #004182;
    border-color: #004182;
  }

  /* INFO BOXES */
  .box {
    background-color: var(--color-panel);
    border-left: 4px solid var(--color-accent);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.3s ease;
  }

  .box:hover {
    box-shadow: var(--shadow-md);
  }

  .box p {
    color: var(--color-text-secondary);
    margin-top: 0.5rem;
  }

  /* LOADER */
  .loader {
    margin: 2rem auto;
    width: 48px;
    height: 48px;
    border: 5px solid #e0e0e0;
    border-top: 5px solid #0a66c2;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</>


""", unsafe_allow_html=True)



def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 Aarifs AI Video Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Transform YouTube videos into structured, comprehensive summaries</p>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "Summarization Model",
            ["facebook/bart-large-cnn", "facebook/bart-large-xsum"],
            help="Choose the model for summarization. BART-large-cnn is recommended for most use cases."
        )
        
        # Processing options
        st.subheader("Processing Options")
        chunk_size = st.slider("Chunk Size (sentences)", 5, 20, 10)
        min_summary_length = st.slider("Min Summary Length", 50, 150, 64)
        max_summary_length = st.slider("Max Summary Length", 200, 800, 512)
        
        # Output options
        st.subheader("Output Options")
        save_intermediates = st.checkbox("Save intermediate files", value=True)
        generate_pdf = st.checkbox("Generate PDF output", value=False)
        
        # System info
        st.subheader("📊 System Info")
        st.info("💾 Designed for 8GB RAM\n🔌 Runs fully offline\n🚀 No external API calls")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.markdown('<h2 class="step-header">📥 Input</h2>', unsafe_allow_html=True)
        
        # URL input
        video_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Enter a valid YouTube video URL"
        )
        if video_url:
            vid = extract_video_id(video_url)
            if vid:
                st.markdown('<h3 class="step-header">▶️ Video Preview</h3>', unsafe_allow_html=True)
                video_width =750 #djust to 300 if you want even smaller
                video_height = int(video_width * 9 / 16)

                st.markdown(
            f'''
            <div style="text-align:center; border-radius:15px;">
                <iframe width="{video_width}" height="{video_height}"
                src="https://www.youtube.com/embed/{vid}"
                frameborder="1px"
                allow="accelerometer; encrypted-media; gyroscope; picture-in-picture"
                allowfullscreen>
                </iframe>
            </div>
            ''',
            unsafe_allow_html=True
        )
            else:
                st.warning("⚠️ Unable to extract video ID for preview.")
        
        # Optional transcript upload
        st.subheader("Or upload existing transcript")
        uploaded_file = st.file_uploader(
            "Upload transcript file (.txt)",
            type=['txt'],
            help="Upload a pre-existing transcript file to skip the extraction step"
        )
        
        
        # Process button
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if video_url or uploaded_file:
                process_video(video_url, uploaded_file, model_choice, chunk_size, 
                            min_summary_length, max_summary_length, save_intermediates, generate_pdf)
            else:
                st.error("Please provide either a YouTube URL or upload a transcript file.")
    
    with col2:
        # Status and progress
        st.markdown('<h2 class="step-header">📊 Status</h2>', unsafe_allow_html=True)
        
        # Create status placeholders
        status_container = st.container()
        
        # Recent outputs
        st.subheader("📁 Recent Outputs")
        display_recent_outputs()

def process_video(url, uploaded_file, model, chunk_size, min_len, max_len, save_files, gen_pdf):
    """Process the video through the entire pipeline"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    import traceback
    try:
        # Step 1: Fetch transcript
        status_text.text("🔍 Fetching transcript...")
        progress_bar.progress(10)
        
        if uploaded_file:
            raw_transcript = uploaded_file.read().decode('utf-8')
            st.success("✅ Transcript uploaded successfully")
        else:
            try:
                raw_transcript = fetch_transcript(url)
                st.success("✅ Transcript fetched successfully")
            except Exception as e:
                st.error(f"❌ Error fetching transcript: {str(e)}")
                st.error("Please ensure the video has captions or allow audio processing.")
                st.exception(e)
                return
        
        # Step 2: Clean transcript
        status_text.text("🧹 Cleaning transcript...")
        progress_bar.progress(25)
        
        cleaned_path = clean_transcript(raw_transcript, save_files)
        st.success("✅ Transcript cleaned and formatted")
        
        # Step 3: Split into chunks
        status_text.text("✂️ Splitting into chunks...")
        progress_bar.progress(40)
        
        chunks = split_into_chunks(cleaned_path, chunk_size, save_files)
        st.success(f"✅ Split into {len(chunks)} chunks")
        
        # Step 4: Summarize chunks
        status_text.text("🤖 Generating summaries...")
        progress_bar.progress(60)
        
        summaries = summarize_chunks_batch(
            chunks, 
            model_name=model, 
            min_length=min_len, 
            max_length=max_len,
            save_summaries=save_files
        )
        
        st.success(f"✅ Generated {len(summaries)} chunk summaries")
        
        # Step 5: Build final summary
        status_text.text("📝 Building final summary...")
        progress_bar.progress(80)
        
        final_summary = build_final_summary(summaries, save_files, gen_pdf)
        
        # Step 6: Display results
        status_text.text("✨ Analysis complete!")
        progress_bar.progress(100)
        
        display_results(final_summary, cleaned_path if save_files else None)
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
        st.exception(e)

def display_results(final_summary, transcript_path):
    """Display the final results"""
    
    st.markdown('<h2 class="step-header">📋 Results</h2>', unsafe_allow_html=True)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "🔍 Key Insights", "❓ FAQs", "💎 Hidden Gems"])
    
    with tab1:
        st.subheader("Abstract-Style Overview")
        if "abstract" in final_summary:
            st.write(final_summary["abstract"])
        else:
            st.write(final_summary.get("overview", "No overview available"))
    
    with tab2:
        st.subheader("Key Insights")
        insights = final_summary.get("insights", [])
        for i, insight in enumerate(insights, 1):
            st.write(f"**{i}.** {insight}")
    
    with tab3:
        st.subheader("Flash cards")
        faqs = final_summary.get("faqs", [])
        for faq in faqs:
            with st.expander(faq.get("question", "Question")):
                st.write(faq.get("answer", "Answer"))
    
    with tab4:
        st.subheader("Hidden Gems & Actionable Tips")
        gems = final_summary.get("gems", [])
        for i, gem in enumerate(gems, 1):
            st.info(f"💡 **Tip {i}:** {gem}")
    
    # Download section
    st.subheader("📥 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if transcript_path and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as f:
                st.download_button(
                    "📄 Download Transcript",
                    f.read(),
                    file_name="cleaned_transcript.txt",
                    mime="text/plain"
                )
    
    with col2:
        # Convert final summary to text
        summary_text = format_summary_for_download(final_summary)
        st.download_button(
            "📋 Download Summary",
            summary_text,
            file_name="final_summary.txt",
            mime="text/plain"
        )
    
    with col3:
        if os.path.exists("youtube_summarizer/outputs/final/final_summary.pdf"):
            with open("youtube_summarizer/outputs/final/final_summary.pdf", 'rb') as f:
                st.download_button(
                    "📑 Download PDF",
                    f.read(),
                    file_name="final_summary.pdf",
                    mime="application/pdf"
                )

def format_summary_for_download(summary_dict):
    """Format the summary dictionary into a readable text format"""
    text = "YOUTUBE VIDEO ANALYSIS SUMMARY\n"
    text += "=" * 50 + "\n\n"
    
    if "abstract" in summary_dict:
        text += "ABSTRACT OVERVIEW\n"
        text += "-" * 20 + "\n"
        text += summary_dict["abstract"] + "\n\n"
    
    if "insights" in summary_dict:
        text += "KEY INSIGHTS\n"
        text += "-" * 20 + "\n"
        for i, insight in enumerate(summary_dict["insights"], 1):
            text += f"{i}. {insight}\n"
        text += "\n"
    
    if "faqs" in summary_dict:
        text += "FREQUENTLY ASKED QUESTIONS\n"
        text += "-" * 30 + "\n"
        for faq in summary_dict["faqs"]:
            text += f"Q: {faq.get('question', 'Question')}\n"
            text += f"A: {faq.get('answer', 'Answer')}\n\n"
    
    if "gems" in summary_dict:
        text += "HIDDEN GEMS & ACTIONABLE TIPS\n"
        text += "-" * 35 + "\n"
        for i, gem in enumerate(summary_dict["gems"], 1):
            text += f"{i}. {gem}\n"
    
    return text

def display_recent_outputs():
    """Display recent output files"""
    outputs_dir = Path("youtube_summarizer/outputs")
    
    if outputs_dir.exists():
        # Get recent files
        recent_files = []
        for subdir in ["final", "summaries", "transcripts"]:
            subdir_path = outputs_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.glob("*"):
                    if file.is_file():
                        recent_files.append(file)
        
        # Sort by modification time
        recent_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Display recent files
        for file in recent_files[:5]:  # Show last 5 files
            rel_path = file.relative_to(outputs_dir)
            st.text(f"📄 {rel_path}")
    else:
        st.text("No outputs yet")

if __name__ == "__main__":
    # Ensure output directories exist
    output_dirs = [
        "youtube_summarizer/outputs/transcripts",
        "youtube_summarizer/outputs/chunks", 
        "youtube_summarizer/outputs/summaries",
        "youtube_summarizer/outputs/final"
    ]
    
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    main()
