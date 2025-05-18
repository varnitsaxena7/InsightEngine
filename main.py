import os
from PIL import Image
import streamlit as st
import time
import json
import base64
from gemini_utility import (load_gemini_pro_model, gemini_pro_response, gemini_pro_vision_response, embeddings_model_response)
working_dir = os.path.dirname(os.path.abspath(__file__))
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

st.set_page_config(
    page_title="InsightEngine",
    page_icon="💡",
    layout="centered",  
)

# Features dropdown menu
st.title("InsightEngine 💡")
selected = st.sidebar.selectbox(
    'Select Feature',
    ['ChatBot', 'Image Captioning', 'Embed text', 'Ask me anything','Sentiment Analysis','Language Translation','Topic Modeling','Youtube Transcriber','Document Summarizer'],
    index=0

)

if selected == 'ChatBot':

    if "chat_session" not in st.session_state:
        model = load_gemini_pro_model()
        st.session_state.chat_session = model.start_chat(history=[])

    st.title("🤖 ChatBot")

    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    user_prompt = st.chat_input("Ask Gemini-Pro...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)

        gemini_response = st.session_state.chat_session.send_message(user_prompt)

        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

if selected == "Image Captioning":
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
                        # Generate caption using Gemini-Pro
                        caption = gemini_pro_vision_response(caption_prompt, image)
                        st.info(caption)


# Text embedding model
if selected == "Embed text":
    st.title("🔡 Embed Text")

    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if st.button("Get Response"):
        with st.spinner('Generating Embeddings...'):
            response = embeddings_model_response(user_prompt)
            st.markdown(response)


# Ask me anything page
if selected == "Ask me anything":
    st.title("❓ Ask me a question")

    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if st.button("Get Response"):
        with st.spinner('Thinking...'):
            response = gemini_pro_response(user_prompt)
            st.markdown(response)



from gemini_utility import sentiment_analysis
import random
# Inside the if-else block for the selected feature
if selected == 'Sentiment Analysis':  # Add this option in the dropdown menu
    st.title("💬 Mood Mapper")
    def get_mood_phrases():
        with open('mood_phrases.json') as file:
            mood_phrases = json.load(file)
            return mood_phrases

# Function to generate mood-based suggestions
    def get_suggestions(sentiment):
         mood_phrases = get_mood_phrases()
         return random.choice(mood_phrases[sentiment])

    # Text input for user's message
    user_text = st.text_area(label='', placeholder="Enter the text for sentiment analysis")

    # Button to trigger sentiment analysis
    if st.button("Analyze Sentiment"):
        with st.spinner('Analyzing Sentiment...'):
            # Perform sentiment analysis
            sentiment_scores = sentiment_analysis(user_text)

            # Display sentiment analysis results
            p=sentiment_scores['pos']
            n=sentiment_scores['neg']
            ne=sentiment_scores['neu']
            c=sentiment_scores['compound']
            st.write("Sentiment Analysis Results:")
            if(c>0):
                 sentiment = 'positive'
                 st.write("Positive 😄")
            elif(c<0):
                 sentiment = 'negative'
                 st.write("Negative 😔")
            else:
                sentiment = 'neutral'
                st.write("Neutral 😐")
            st.subheader("Sentiment Distribution:")
            st.write("Positive: {}%".format(round(p * 100, 2)))
            st.write("Negative: {}%".format(round(n * 100, 2)))
            st.write("Neutral: {}%".format(round(ne * 100, 2)))
            st.subheader("Suggestions Based on Your Mood:")
            suggestions = get_suggestions(sentiment)
            st.write(suggestions)

from googletrans import Translator
import base64

if selected =='Language Translation':
                import streamlit as st
                from googletrans import Translator

                # Initialize the translator
                translator = Translator()

                # Streamlit app title and description
                st.title("Linguify 🗣️")
                st.write("Translate text from one language to another using Google Translate.")

                # Input field for user's text
                user_text = st.text_area("Enter text to translate")

                # List of supported languages
                languages = {
                    "Afrikaans": "af", "Albanian": "sq", "Arabic": "ar", "Armenian": "hy", "Azerbaijani": "az", 
                    "Basque": "eu", "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs", "Bulgarian": "bg", 
                    "Catalan": "ca", "Cebuano": "ceb", "Chinese (Simplified)": "zh-CN", "Chinese (Traditional)": "zh-TW", 
                    "Corsican": "co", "Croatian": "hr", "Czech": "cs", "Danish": "da", "Dutch": "nl", "English": "en", 
                    "Esperanto": "eo", "Estonian": "et", "Filipino": "tl", "Finnish": "fi", "French": "fr", "Frisian": "fy", 
                    "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu", "Haitian Creole": "ht", 
                    "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "iw", "Hindi": "hi", "Hmong": "hmn", "Hungarian": "hu", 
                    "Icelandic": "is", "Igbo": "ig", "Indonesian": "id", "Irish": "ga", "Italian": "it", "Japanese": "ja", 
                    "Javanese": "jw", "Kannada": "kn", "Kazakh": "kk", "Khmer": "km", "Korean": "ko", "Kurdish": "ku", 
                    "Kyrgyz": "ky", "Lao": "lo", "Latin": "la", "Latvian": "lv", "Lithuanian": "lt", "Luxembourgish": "lb", 
                    "Macedonian": "mk", "Malagasy": "mg", "Malay": "ms", "Malayalam": "ml", "Maltese": "mt", "Maori": "mi", 
                    "Marathi": "mr", "Mongolian": "mn", "Myanmar (Burmese)": "my", "Nepali": "ne", "Norwegian": "no", 
                    "Nyanja (Chichewa)": "ny", "Pashto": "ps", "Persian": "fa", "Polish": "pl", "Portuguese": "pt", 
                    "Punjabi": "pa", "Romanian": "ro", "Russian": "ru", "Samoan": "sm", "Scots Gaelic": "gd", 
                    "Serbian": "sr", "Sesotho": "st", "Shona": "sn", "Sindhi": "sd", "Sinhala (Sinhalese)": "si", 
                    "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su", "Swahili": "sw", 
                    "Swedish": "sv", "Tagalog (Filipino)": "tl", "Tajik": "tg", "Tamil": "ta", "Telugu": "te", "Thai": "th", 
                    "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Uzbek": "uz", "Vietnamese": "vi", "Welsh": "cy", 
                    "Xhosa": "xh", "Yiddish": "yi", "Yoruba": "yo", "Zulu": "zu"
                }

                # Convert language dictionary to list for selectbox
                language_options = list(languages.keys())

                # Input field for choosing the source language
                source_language = st.selectbox("Select source language", language_options)

                # Input field for choosing the target language
                target_language = st.selectbox("Select target language", language_options)

                # Button to trigger translation
                if st.button("Translate"):
                    with st.spinner('Translating...'):
                        # Perform translation
                        translation = translator.translate(user_text, src=languages[source_language], dest=languages[target_language])
                        translated_text = translation.text

                        # Display translated text
                        st.write("Translated Text:")
                        st.write(translated_text)

if selected=='Topic Modeling':
        import streamlit as st
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        import numpy as np
        import pandas as pd

        def perform_topic_modeling(documents, num_topics=5, max_df=0.95, min_df=2, stop_words='english'):
            vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
            dtm = vectorizer.fit_transform(documents)

            lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda_model.fit(dtm)

            return lda_model, vectorizer

        def display_topics(model, vectorizer, n_top_words=10):
            feature_names = np.array(vectorizer.get_feature_names_out())
            topics = []
            for topic_idx, topic in enumerate(model.components_):
                top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
                top_words = feature_names[top_words_idx]
                topics.append(top_words)
            return topics

        st.title("Topic Modeling 📊")
        st.write("Explore the top words for each topic to gain insights and understand the key themes within your text data.")
        st.write("Customization Options")
        num_topics = st.number_input("Number of Topics", min_value=1, max_value=10, value=5, step=1)
        max_df = st.slider("Max Document Frequency", min_value=0.1, max_value=1.0, value=0.95, step=0.05)
        min_df = st.number_input("Min Document Frequency", min_value=1, max_value=10, value=2, step=1)
        stop_words = "english"

        option = st.radio("Select Input Option", ("Upload Documents", "Write Text"))

        if option == "Upload Documents":
            uploaded_file = st.file_uploader("Upload your documents", type=["txt", "csv"])
            if uploaded_file is not None:
                
                if uploaded_file.type == "txt":
                    documents = uploaded_file.read().decode("utf-8").splitlines()
                elif uploaded_file.type == "csv":
                    df = pd.read_csv(uploaded_file)
                    documents = df[df.columns[0]].astype(str).tolist()
                
                if len(documents) > 0:
                    lda_model, vectorizer = perform_topic_modeling(documents, num_topics=num_topics, max_df=max_df, min_df=min_df, stop_words=stop_words)
                    topics = display_topics(lda_model, vectorizer)
                    st.subheader("Top Words for Each Topic")
                    for idx, topic_words in enumerate(topics):
                        st.write(f"Topic {idx + 1}: {', '.join(topic_words)}")
                else:
                    st.warning("No documents found in the uploaded file.")
            else:
                st.warning("Please upload a text file to perform topic modeling.")
        else:
            text_input = st.text_area("Enter your text here", height=200)
            if st.button("Perform Topic Modeling"):
                if text_input:
                    documents = text_input.split("\n")
                    lda_model, vectorizer = perform_topic_modeling(documents, num_topics=num_topics, max_df=max_df, min_df=min_df, stop_words=stop_words)
                    topics = display_topics(lda_model, vectorizer)
                    st.subheader("Top Words for Each Topic")
                    for idx, topic_words in enumerate(topics):
                        st.write(f"Topic {idx + 1}: {', '.join(topic_words)}")
                else:
                    st.warning("Please enter some text to perform topic modeling.")

if selected=="Youtube Transcriber":
            from gemini_utility import transcribe_video,display_paragraphs
            st.title("YouTube Video Transcriber")

            video_url = st.text_input("Enter YouTube video URL:")
            language = st.radio("Select Language:", options=['English', 'Hindi'])
            paragraph_length = st.slider("Paragraph Length", min_value=50, max_value=1000, step=50, value=300)

            if st.button("Transcribe"):
                if video_url:
                    st.write("Transcribing...")
                    if language == 'English':
                        paragraphs, transcript = transcribe_video(video_url, 'en', paragraph_length)
                    elif language == 'Hindi':
                        paragraphs, transcript = transcribe_video(video_url, 'hi', paragraph_length)
                    if paragraphs:
                        st.header("Transcript:")
                        display_paragraphs(paragraphs)
                        
                        if transcript:
                            b64_text = base64.b64encode(transcript.encode()).decode()
                            href = f'<a href="data:file/txt;base64,{b64_text}" download="Transcript.txt">Download Transcript File</a>'
                            st.markdown(href, unsafe_allow_html=True)
                        else:
                            st.warning("No transcript available to download.")
                else:
                    st.warning("Please enter a YouTube video URL.")
                    
if selected=="Document Summarizer":
     st.title("Document Summarizer 📄")
     st.write("Summarize your document quickly and efficiently!")
     import spacy
        # Text input for user to enter document
     document = st.text_area("Enter your document here")

        # Button to trigger summarization
     if st.button("Summarize"):
            # Process the document using spaCy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(document)

            # Calculate sentence importance scores based on TextRank algorithm
            # Summarization by extracting the most important sentences
            sentence_scores = {}
            for sent in doc.sents:
                sentence_scores[sent] = sum([token.rank for token in sent if not token.is_stop])

            # Sort sentences by importance score in descending order
            sorted_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)

            # Number of sentences in the summarized version
            num_sentences_summary = min(len(sorted_sentences), 5)  # Set the number of sentences in the summary

            # Generate the summary
            summary = " ".join([str(sent) for sent in sorted_sentences[:num_sentences_summary]])

            # Display summarized text
            st.subheader("Summary:")
            st.write(summary)
