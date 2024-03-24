import os
import json
from PIL import Image
import google.generativeai as genai

working_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = f"{working_dir}/config.json"
config_data = json.load(open("config.json"))

GOOGLE_API_KEY = config_data["GOOGLE_API_KEY"]

genai.configure(api_key=GOOGLE_API_KEY)

def load_gemini_pro_model():
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    return gemini_pro_model

def gemini_pro_vision_response(prompt, image):
    gemini_pro_vision_model = genai.GenerativeModel("gemini-pro-vision")
    response = gemini_pro_vision_model.generate_content([prompt, image])
    result = response.text
    return result

def embeddings_model_response(input_text):
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(model=embedding_model,
                                    content=input_text,
                                    task_type="retrieval_document")
    embedding_list = embedding["embedding"]
    return embedding_list

def gemini_pro_response(user_prompt):
    gemini_pro_model = genai.GenerativeModel("gemini-pro")
    response = gemini_pro_model.generate_content(user_prompt)
    result = response.text
    return result

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
def sentiment_analysis(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

#import speech_recognition as sr
#import streamlit as st
#def recognize_speech():
   # recognizer = sr.Recognizer()
   # with sr.Microphone() as source:
    #    st.write("Listening...")
     #   recognizer.adjust_for_ambient_noise(source, duration=1)
     #   audio_data = recognizer.listen(source)
  #  try:
     #   st.write("Processing...")
     #   text = recognizer.recognize_google(audio_data)
     #   return text
   # except sr.UnknownValueError:
   #     st.write("Sorry, I could not understand what you said.")
   # except sr.RequestError:
       # st.write("Sorry, the service is unavailable.")


import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import base64

def transcribe_video(video_url, language='en', paragraph_length=300):
    try:
        video_id = video_url.split('v=')[-1]
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        transcript = ' '.join(line['text'] for line in transcript_list)
        paragraphs = [transcript[i:i+paragraph_length] for i in range(0, len(transcript), paragraph_length)]
        return paragraphs, transcript
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def display_paragraphs(paragraphs):
    for i, paragraph in enumerate(paragraphs):
        st.markdown(f"**Paragraph {i+1}:**")
        st.write(paragraph)
