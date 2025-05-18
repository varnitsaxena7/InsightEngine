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
    st.error(f"Config file not found at {config_file_path}")
    st.stop()
except json.JSONDecodeError:
    st.error("Config file is not valid JSON")
    st.stop()

GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY missing in config.json")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Models ---

def load_gemini_pro_model():
    return genai.GenerativeModel("gemini-pro")

def gemini_pro_response(prompt):
    model = load_gemini_pro_model()
    response = model.generate_content(prompt)
    return response.text

def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    if isinstance(image, str):
        image = Image.open(image)
    response = gemini_pro_vision_model.generate_content([prompt, image])
    return response.text

def load_text_bison_model():
    # Alternative Gemini text model
    return genai.GenerativeModel("models/text-bison-001")

def text_bison_response(prompt):
    model = load_text_bison_model()
    response = model.send_message(prompt)
    return response.text

def embeddings_model_response(input_text):
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

# --- Streamlit App ---

st.title("Google Gemini + Sentiment + YouTube Transcript Demo")

option = st.selectbox("Choose functionality:", 
                      ["Generate Text (Gemini Pro)", 
                       "Generate Text (Text Bison)",
                       "Generate Vision Response (Gemini Pro Vision)",
                       "Get Embeddings",
                       "Sentiment Analysis",
                       "YouTube Transcript"])

if option == "Generate Text (Gemini Pro)":
    prompt = st.text_area("Enter prompt for Gemini Pro:")
    if st.button("Generate Gemini Pro Text"):
        if prompt.strip():
            response = gemini_pro_response(prompt)
            st.success(response)
        else:
            st.warning("Please enter a prompt.")

elif option == "Generate Text (Text Bison)":
    prompt = st.text_area("Enter prompt for Text Bison model:")
    if st.button("Generate Text Bison Response"):
        if prompt.strip():
            response = text_bison_response(prompt)
            st.success(response)
        else:
            st.warning("Please enter a prompt.")

elif option == "Generate Vision Response (Gemini Pro Vision)":
    prompt = st.text_area("Enter prompt for vision model:")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if st.button("Generate Vision Response"):
        if prompt.strip() and uploaded_file:
            img = Image.open(uploaded_file)
            response = gemini_pro_vision_response(prompt, img)
            st.success(response)
        else:
            st.warning("Please enter a prompt and upload an image.")

elif option == "Get Embeddings":
    text = st.text_area("Enter text to get embeddings:")
    if st.button("Get Embeddings"):
        if text.strip():
            embedding = embeddings_model_response(text)
            st.write("Embedding vector (truncated):")
            st.write(embedding[:10])  # Show first 10 dims only
        else:
            st.warning("Please enter text.")

elif option == "Sentiment Analysis":
    text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        if text.strip():
            scores = sentiment_analysis(text)
            st.json(scores)
        else:
            st.warning("Please enter text.")

elif option == "YouTube Transcript":
    url = st.text_input("Enter YouTube video URL:")
    if st.button("Fetch Transcript"):
        if url.strip():
            paragraphs, full_text = transcribe_video(url)
            if paragraphs:
                display_paragraphs(paragraphs)
        else:
            st.warning("Please enter a YouTube URL.")
