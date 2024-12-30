import streamlit as st
from ai_logic import AudioTranscriber, ContentAnalyzer
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

st.set_page_config(page_title="Lecture Analyzer", layout="wide")

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = ContentAnalyzer()

def process_audio(audio_file):
    logger.info(f"Starting to process audio file: {audio_file.name}")
    logger.info(f"File size: {audio_file.size / (1024*1024):.2f} MB")
    
    with st.spinner('Processing audio file...'):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
            logger.info("Temporary file created successfully")

        try:
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
            
        finally:
            # Cleanup temporary files
            logger.info("Cleaning up temporary files")
            os.unlink(tmp_file_path)
            os.unlink(wav_path)
            for chunk in chunks:
                os.unlink(chunk)
            logger.info("Temporary files cleaned up successfully")
            logger.info("Audio processing completed successfully")

# UI Layout
st.title("📚 Lecture Analyzer")

# File upload
uploaded_file = st.file_uploader("Upload your lecture audio", accept_multiple_files=False)

# Process audio only if a new file is uploaded and transcription hasn't been done yet
if uploaded_file and st.session_state.transcription is None:
    process_audio(uploaded_file)

# Display transcription
if st.session_state.transcription:
    st.subheader("📝 Transcription")
    st.text_area("", st.session_state.transcription, height=200)

# Display summary
if st.session_state.summary:
    st.subheader("📋 Summary")
    st.markdown(st.session_state.summary)

# Chat interface
if st.session_state.transcription:
    st.subheader("💭 Ask Questions")
    user_question = st.text_input("Ask a question about the lecture:")
    if user_question:
        with st.spinner('Generating response...'):
            response = st.session_state.analyzer.chat_with_context(
                st.session_state.transcription,
                user_question
            )
            st.markdown(response)