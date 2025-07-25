# YouTube Video Chatbot

An application that allows you to chat with YouTube videos using AI. Load any YouTube video and ask questions about its content!


## Features

-  Load YouTube videos by URL or video ID
-  Chat with video content using Google's Gemini AI
-  Semantic search through video transcripts
-  Modern, responsive React interface
-  Fast API built with FastAPI
-  Real-time chat interface

## Tech Stack

### Frontend
- HTML 

### Backend
- FastAPI (Python)
- LangChain for AI processing
- Google Generative AI (Gemini)
- FAISS for vector storage
- HuggingFace embeddings
- YouTube Transcript API

## Setup Instructions

### Prerequisites
- Python 3.8+
- Google API key for Gemini

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file and add your Google API key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

4. Start the backend server:
```bash
python app.py
```

The backend will run on `http://localhost:8000`

## Environment Variables

### Backend (.env)
- `GOOGLE_API_KEY` - Your Google API key for Gemini AI

## How It Works

1. **Video Loading**: The app fetches the transcript from YouTube using the YouTube Transcript API
2. **Text Processing**: The transcript is split into chunks using LangChain's text splitter
3. **Embeddings**: Text chunks are converted to embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in FAISS for fast similarity search
5. **Chat**: User questions are processed through a RAG (Retrieval-Augmented Generation) pipeline using Google's Gemini AI

## Troubleshooting

- **No transcript available**: Some videos don't have transcripts. Try a different video.
- **Backend offline**: Make sure the Python server is running on the specified port
- **API errors**: Check that your LLM API key is valid and has access

