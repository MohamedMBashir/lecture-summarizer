# Lecture Analyzer

An AI-powered web application that transforms audio lectures into interactive learning experiences.

## Features

- 🎙️ Audio Transcription: Convert lecture recordings into text
- 📝 Smart Summarization: Get detailed summaries of lecture content
- 💬 Interactive Q&A: Ask questions about the lecture material
- 🎯 User-Friendly Interface: Simple upload and navigation

## Technology Stack

- **Frontend**: Streamlit
- **AI Models**:
  - Gemini 1.5 Flash (Audio transcription)
  - GPT-4 (Summarization)
  - GPT-4-mini (Interactive Q&A)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys:
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`

3. Run the application:
```bash
streamlit run main.py
```


## Usage

1. Upload your lecture audio file (supports MP3, WAV, M4A)
2. Wait for transcription to complete
3. Navigate between:
   - Full transcription
   - Lecture summary
   - Interactive Q&A chat

## License

MIT