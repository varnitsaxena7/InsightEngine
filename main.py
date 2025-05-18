import os
from PIL import Image
import streamlit as st
import time
import json
import base64
import random
from googletrans import Translator

from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    embeddings_model_response,
    sentiment_analysis
)

working_dir = os.path.dirname(os.path.abspath(__file__))

def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

st.set_page_config(
    page_title="InsightEngine",
    page_icon="💡",
    layout="centered",
)

st.title("InsightEngine 💡")

selected = st.sidebar.selectbox(
    'Select Feature',
    ['ChatBot', 'Image Captioning', 'Embed text', 'Ask me anything', 'Sentiment Analysis', 'Language Translation', 'Youtube Transcriber', 'Document Summarizer'],
    index=0
)

# ChatBot
if selected == 'ChatBot':
    model = load_gemini_pro_model()
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        gemini_response = st.session_state.chat_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

# Image Captioning
elif selected == "Image Captioning":
    st.title("📷 Image Teller")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        caption_prompt = st.text_input("Caption Prompt", "Write a short caption for this image")

        if st.button("Generate Caption"):
            with st.spinner('Generating Caption...'):
                image = Image.open(uploaded_image)
                col1, col2 = st.columns([3, 1])
                with col1:
                    resized_img = image.resize((800, 500))
                    st.image(resized_img, caption="Original Image", use_column_width=True)
                with col2:
                    caption = gemini_pro_vision_response(caption_prompt, image)
                    st.info(caption)

# Embed Text
elif selected == "Embed text":
    st.title("🔡 Embed Text")
    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        with st.spinner('Generating Embeddings...'):
            response = embeddings_model_response(user_prompt)
            st.markdown(response)

# Ask Me Anything
elif selected == "Ask me anything":
    st.title("❓ Ask me a question")
    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        with st.spinner('Thinking...'):
            response = gemini_pro_response(user_prompt)
            st.markdown(response)

# Sentiment Analysis
elif selected == 'Sentiment Analysis':
    st.title("💬 Mood Mapper")

    def get_mood_phrases():
        with open('mood_phrases.json') as file:
            mood_phrases = json.load(file)
            return mood_phrases

    def get_suggestions(sentiment):
        mood_phrases = get_mood_phrases()
        return random.choice(mood_phrases[sentiment])

    user_text = st.text_area(label='', placeholder="Enter the text for sentiment analysis")

    if st.button("Analyze Sentiment"):
        with st.spinner('Analyzing Sentiment...'):
            sentiment_scores = sentiment_analysis(user_text)
            p = sentiment_scores['pos']
            n = sentiment_scores['neg']
            ne = sentiment_scores['neu']
            c = sentiment_scores['compound']
            st.write("Sentiment Analysis Results:")

            if c > 0:
                sentiment = 'positive'
                st.write("Positive 😄")
            elif c < 0:
                sentiment = 'negative'
                st.write("Negative 😔")
            else:
                sentiment = 'neutral'
                st.write("Neutral 😐")

            st.subheader("Sentiment Distribution:")
            st.write(f"Positive: {round(p * 100, 2)}%")
            st.write(f"Negative: {round(n * 100, 2)}%")
            st.write(f"Neutral: {round(ne * 100, 2)}%")

            st.subheader("Suggestions Based on Your Mood:")
            suggestions = get_suggestions(sentiment)
            st.write(suggestions)

# Language Translation
elif selected == 'Language Translation':
    st.title("Linguify 🗣️")
    st.write("Translate text from one language to another using Google Translate.")

    translator = Translator()
    user_text = st.text_area("Enter text to translate")

    languages = {
        "English": "en", "Hindi": "hi", "Spanish": "es", "French": "fr",
        "German": "de", "Chinese (Simplified)": "zh-CN", "Japanese": "ja",
        "Russian": "ru", "Arabic": "ar", "Portuguese": "pt", "Bengali": "bn",
        "Tamil": "ta", "Telugu": "te", "Korean": "ko", "Urdu": "ur"
    }

    source_lang = st.selectbox("Select source language", list(languages.keys()))
    target_lang = st.selectbox("Select target language", list(languages.keys()), index=1)

    if st.button("Translate"):
        if user_text:
            with st.spinner("Translating..."):
                translated = translator.translate(user_text, src=languages[source_lang], dest=languages[target_lang])
                st.write(f"Translated Text ({target_lang}):")
                st.success(translated.text)
        else:
            st.warning("Please enter text to translate.")
