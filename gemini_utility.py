import os
import json
from PIL import Image
import google.generativeai as genai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# --- Setup ---

working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

try:
    with open(config_file_path, 'r') as f:
        config_data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"Config file not found at {config_file_path}")
except json.JSONDecodeError:
    raise ValueError("Config file is not valid JSON")

GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing in config.json")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Gemini Models ---

def load_text_bison_model():
    # Updated model name from "gemini-pro" to "models/text-bison-001"
    return genai.GenerativeModel("models/text-bison-001")

def text_bison_response(prompt):
    """
    Generate text response from text-bison model.
    """
    model = load_text_bison_model()
    response = model.send_message(prompt)  # use send_message for chat style
    return response.text

# Vision model usage (if available) — placeholder example:
# def gemini_vision_response(prompt, image):
#     # NOTE: Replace 'models/vision-bison-001' with the correct model if available
#     model = genai.GenerativeModel("models/vision-bison-001")
#     if isinstance(image, str):
#         image = Image.open(image)
#     response = model.generate_content([prompt, image])
#     return response.text

def embeddings_model_response(input_text):
    """
    Generate embeddings from Gemini embeddings model.
    """
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document"
    )
    return embedding["embedding"]

# --- Sentiment Analysis ---

nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
    return sid.polarity_scores(text)

# --- YouTube Transcript Handling ---

def transcribe_video(video_url, language='en', paragraph_length=300):
    try:
        if "v=" in video_url:
            video_id = video_url.split('v=')[-1].split('&')[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split('?')[0]
        else:
            raise ValueError("Invalid YouTube URL format.")

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = ' '.join(line['text'] for line in transcript_list)
        paragraphs = [transcript[i:i+paragraph_length] for i in range(0, len(transcript), paragraph_length)]
        return paragraphs, transcript

    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None, None

def display_paragraphs(paragraphs):
    for i, paragraph in enumerate(paragraphs):
        st.markdown(f"**Paragraph {i+1}:**")
        st.write(paragraph)
