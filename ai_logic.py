import os
import google.generativeai as genai
from openai import OpenAI
from pydub import AudioSegment
import tempfile
from typing import List
import logging
from datetime import datetime
import streamlit as st

# Configure logging with timestamp
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_api_key(key_name: str) -> str:
    """
    Get API key from different sources in order:
    1. Environment variable
    2. Streamlit secrets
    3. User input (stored in session state)
    """
    # First check environment variables (including .env)
    api_key = os.getenv(key_name)
    
    # Then check Streamlit secrets
    if not api_key and hasattr(st, 'secrets'):
        api_key = st.secrets.get(key_name)
    
    # Finally, check session state or prompt user
    if not api_key:
        if key_name not in st.session_state:
            st.session_state[key_name] = st.text_input(
                f"Please enter your {key_name}:",
                type="password",
                key=f"input_{key_name}"
            )
        api_key = st.session_state[key_name]
        
    if not api_key:
        st.error(f"{key_name} is required to proceed!")
        st.stop()
    
    return api_key

# Initialize AI models
def initialize_ai():
    gemini_key = get_api_key('GEMINI_API_KEY')
    openai_key = get_api_key('OPENAI_API_KEY')
    
    genai.configure(api_key=gemini_key)
    return OpenAI(api_key=openai_key)

# Global OpenAI client
openai_client = initialize_ai()

class AudioTranscriber:
    CHUNK_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds
    
    @staticmethod
    def convert_to_wav(audio_file) -> str:
        """Convert uploaded audio to WAV format."""
        logger.info("Starting audio conversion to WAV format")
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                audio = AudioSegment.from_file(audio_file)
                audio.export(temp_wav.name, format='wav')
                logger.info(f"Audio successfully converted to WAV format: {temp_wav.name}")
                return temp_wav.name
        except Exception as e:
            logger.error(f"Error converting audio to WAV: {str(e)}")
            raise

    @staticmethod
    def chunk_audio(wav_path: str) -> List[str]:
        """Split audio into 5-minute chunks."""
        logger.info("Starting audio chunking process")
        try:
            audio = AudioSegment.from_wav(wav_path)
            total_duration_minutes = len(audio) / (60 * 1000)  # Convert to minutes
            chunks = []
            
            logger.info(f"Total audio duration: {total_duration_minutes:.2f} minutes")
            expected_chunks = (len(audio) + AudioTranscriber.CHUNK_DURATION - 1) // AudioTranscriber.CHUNK_DURATION
            logger.info(f"Splitting audio into {expected_chunks} chunks")
            
            for i in range(0, len(audio), AudioTranscriber.CHUNK_DURATION):
                chunk = audio[i:i + AudioTranscriber.CHUNK_DURATION]
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_chunk:
                    chunk.export(temp_chunk.name, format='wav')
                    chunks.append(temp_chunk.name)
                    chunk_duration = len(chunk) / (60 * 1000)  # Convert to minutes
                    logger.info(f"Chunk {len(chunks)} created: {chunk_duration:.2f} minutes")
            
            logger.info(f"Audio successfully split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking audio: {str(e)}")
            raise

    @staticmethod
    def transcribe_chunk(chunk_path: str) -> str:
        """Transcribe a single audio chunk using Gemini."""
        chunk_number = chunk_path.split('/')[-1]  # Extract filename for logging
        logger.info(f"Starting transcription of chunk: {chunk_number}")
        try:
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            uploaded_file = genai.upload_file(chunk_path)
            logger.info(f"Successfully uploaded chunk {chunk_number} to Gemini")
            
            response = model.generate_content([
                "Provide a complete and detailed transcript of the audio without any summarization.",
                uploaded_file
            ])
            
            logger.info(f"Successfully transcribed chunk: {chunk_number}")
            return response.text
        except Exception as e:
            logger.error(f"Error transcribing chunk {chunk_number}: {str(e)}")
            raise

class ContentAnalyzer:
    def __init__(self):
        self.conversation_history = []
        logger.info("Content Analyzer initialized")

    def generate_summary(self, transcription: str) -> str:
        """Generate a detailed summary using GPT-4."""
        logger.info("Starting summary generation")
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert at creating detailed, intuitive summaries of academic lectures. Break down complex topics into clear explanations. "},
                    {"role": "user", "content": f"Please provide a very detailed and a very intuitive engaging summary of this lecture transcription: {transcription}"}
                ]
            )
            logger.info("Summary successfully generated")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            raise

    def chat_with_context(self, transcription: str, user_question: str) -> str:
        """Interactive QA with the transcription context."""
        logger.info(f"Processing user question: {user_question[:50]}...")
        try:
            self.conversation_history.append({"role": "user", "content": user_question})
            
            messages = [
                {"role": "system", "content": f"You are a helpful assistant answering questions about this lecture. Here's the lecture transcription for context: {transcription}"},
                *self.conversation_history
            ]
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            assistant_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            logger.info("Successfully generated response to user question")
            return assistant_response
        except Exception as e:
            logger.error(f"Error in chat interaction: {str(e)}")
            raise
