import os
import json
from PIL import Image
import google.generativeai as genai
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi

# --- Setup ---

# Use absolute path based on the script location for config file
working_dir = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(working_dir, "config.json")

# Load config safely with error handling
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

def load_gemini_pro_model():
    return genai.GenerativeModel("gemini-pro")

def gemini_pro_vision_response(prompt, image):
    """
    Generate response from Gemini Pro Vision model.

    Args:
        prompt (str): Text prompt
        image (PIL.Image or str): Image or image path

    Returns:
        str: Generated response text
    """
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")

    # If image is a path, load it as PIL Image
    if isinstance(image, str):
        image = Image.open(image)

    response = gemini_pro_vision_model.generate_content([prompt, image])
    return response.text

def embeddings_model_response(input_text):
    """
    Generate embeddings from Gemini embeddings model.

    Args:
        input_text (str): Text to embed

    Returns:
        list[float]: Embedding vector
    """
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document"
    )
    return embedding["embedding"]

def gemini_pro_response(user_prompt):
    """
    Generate text response from Gemini Pro.

    Args:
        user_prompt (str): Prompt text

    Returns:
        str: Generated text response
    """
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    return response.text

# --- Sentiment Analysis ---

# Download VADER lexicon only once
nltk.download('vader_lexicon', quiet=True)
sid = SentimentIntensityAnalyzer()

def sentiment_analysis(text):
    """
    Analyze sentiment of input text.

    Args:
        text (str): Input text

    Returns:
        dict: Sentiment scores with keys 'neg', 'neu', 'pos', 'compound'
    """
    return sid.polarity_scores(text)

# --- YouTube Transcript Handling ---

def transcribe_video(video_url, language='en', paragraph_length=300):
    """
    Fetch and split YouTube video transcript into paragraphs.

    Args:
        video_url (str): Full YouTube video URL
        language (str): Language code (default 'en')
        paragraph_length (int): Max chars per paragraph

    Returns:
        tuple[list[str], str] or (None, None): List of paragraphs and full transcript text
    """
    try:
        # Extract video ID (handle both full URLs and short URLs)
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
    """
    Display transcript paragraphs in Streamlit.

    Args:
        paragraphs (list[str]): List of transcript paragraphs
    """
    for i, paragraph in enumerate(paragraphs):
        st.markdown(f"**Paragraph {i+1}:**")
        st.write(paragraph)
