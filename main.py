import streamlit as st
from ai_logic import AudioTranscriber, ContentAnalyzer, get_api_key
import tempfile
import os
import logging

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Lecture Analyzer", layout="wide", initial_sidebar_state="expanded")

# Ensure API keys are available before proceeding
get_api_key('GEMINI_API_KEY')
get_api_key('OPENAI_API_KEY')

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stTextInput input {
        border-radius: 5px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .user-message {
        background-color: rgba(240, 242, 246, 0.1);
    }
    .assistant-message {
        background-color: rgba(230, 243, 255, 0.1);
    }
    /* Updated tab styling for larger text */
    div[role="radiogroup"] {
        display: flex;
        justify-content: center;
        gap: 3rem;  /* Increased gap between tabs */
        margin: 2rem 0;  /* Increased vertical margin */
    }
    .stRadio > label {
        font-size: 1.8rem !important;  /* Much larger font size */
        font-weight: 600 !important;  /* Slightly bolder */
        padding: 0.8rem 1.5rem !important;  /* Larger padding */
        border-radius: 10px !important;
        transition: all 0.2s ease;
    }
    .stRadio > label:hover {
        background-color: rgba(255, 75, 75, 0.1) !important;
        transform: scale(1.05);  /* Slight grow effect on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = ContentAnalyzer()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "📝 Transcription"  # Default to Transcription tab

def process_audio(audio_file):
    logger.info(f"Starting to process audio file: {audio_file.name}")
    logger.info(f"File size: {audio_file.size / (1024*1024):.2f} MB")
    
    # Initialize variables
    tmp_file_path = None
    wav_path = None
    chunks = []
    
    with st.spinner('Processing audio file...'):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name
                logger.info("Temporary file created successfully")

            # Convert to WAV
            logger.info("Starting WAV conversion")
            wav_path = AudioTranscriber.convert_to_wav(tmp_file_path)
            
            # Chunk and transcribe
            logger.info("Starting audio chunking")
            chunks = AudioTranscriber.chunk_audio(wav_path)
            transcriptions = []
            
            # Create a single progress bar
            progress_bar = st.progress(0)
            
            logger.info(f"Beginning transcription of {len(chunks)} chunks")
            for i, chunk in enumerate(chunks, 1):
                progress_bar.progress(i / len(chunks))  # Update the same progress bar
                logger.info(f"Processing chunk {i}/{len(chunks)}")
                transcription = AudioTranscriber.transcribe_chunk(chunk)
                transcriptions.append(transcription)
                logger.info(f"Chunk {i}/{len(chunks)} transcribed successfully")
            
            # Clear the progress bar when done
            progress_bar.empty()
            
            # Combine transcriptions
            logger.info("Combining all transcriptions")
            st.session_state.transcription = " ".join(transcriptions)
            
            # Generate summary
            logger.info("Starting summary generation")
            st.session_state.summary = st.session_state.analyzer.generate_summary(
                st.session_state.transcription
            )
            logger.info("Summary generated successfully")
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            st.error("An error occurred while processing the audio file. Please try again.")
            
        finally:
            # Cleanup temporary files
            logger.info("Cleaning up temporary files")
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            for chunk in chunks:
                if os.path.exists(chunk):
                    os.unlink(chunk)
            logger.info("Temporary files cleaned up successfully")
            logger.info("Audio processing completed successfully")

# Add this before the UI layout
def process_input():
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_question = st.session_state.user_input
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        with st.spinner('Generating response...'):
            response = st.session_state.analyzer.chat_with_context(
                st.session_state.transcription,
                user_question
            )
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Set active tab to Chat only when a message is sent
        st.session_state.active_tab = "💭 Chat"
        # Clear the input
        st.session_state.user_input = ""

# UI Layout
st.title("📚 Lecture Analyzer")
st.markdown("Transform your audio lectures into interactive learning experiences.")

# File upload with better styling
with st.container():
    st.markdown("### 🎯 Get Started")
    uploaded_file = st.file_uploader(
        "Upload your lecture audio (MP3, WAV, M4A)",
        accept_multiple_files=False,
        # type=['mp3', 'wav', 'm4a']
    )

# Process audio only if a new file is uploaded and transcription hasn't been done yet
if uploaded_file and st.session_state.transcription is None:
    process_audio(uploaded_file)

# Display results in tabs
if st.session_state.transcription:
    st.markdown("---")
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  # Add some spacing
    tabs = ["📝 Transcription", "📋 Summary", "💭 Chat"]
    active_tab = st.radio(
        "",
        tabs,
        horizontal=True,
        key="tab_select",
        index=tabs.index(st.session_state.active_tab)
    )
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)  # Add some spacing
    st.markdown("---")
    st.session_state.active_tab = active_tab
    
    if active_tab == "📝 Transcription":
        st.markdown("""
            ### Full Lecture Transcription
            ---
        """)
        st.markdown(st.session_state.transcription)
    
    elif active_tab == "📋 Summary":
        if st.session_state.summary:
            st.markdown("""
                ### Key Points and Summary
                ---
            """)
            st.markdown(st.session_state.summary)
    
    elif active_tab == "💭 Chat":
        st.markdown("### Interactive Q&A")
        st.markdown("Ask questions about the lecture content")
        
        # Chat interface with better styling
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                    <div class="chat-message user-message">
                        <b>You:</b><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <b>Assistant:</b><br>{message["content"]}
                    </div>
                """, unsafe_allow_html=True)
        
        # Move clear chat button and input to a container
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text_input(
                    "Ask a question about the lecture:",
                    key="user_input",
                    on_change=process_input
                )
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()