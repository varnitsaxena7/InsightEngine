# InsightEngine

InsightEngine is a versatile web application built with Streamlit, providing various functionalities such as chatbot, image captioning, sentiment analysis, language translation, topic modeling, YouTube transcriber, and document summarizer.

## Live Demo

Check out the live demo [here](https://insightengine.onrender.com/).

## Features

- **ChatBot**: Interactive chat interface powered by Gemini Pro models for natural language understanding and generation.
- **Image Captioning**: Generate captions for uploaded images using Gemini Pro Vision models.
- **Sentiment Analysis**: Analyze sentiment of provided text with Vader Sentiment Analysis from NLTK.
- **Language Translation**: Translate text between various languages using Google Translate API.
- **Topic Modeling**: Discover key themes within text data using Latent Dirichlet Allocation (LDA).
- **YouTube Transcriber**: Transcribe YouTube videos into text with language selection and paragraph segmentation.
- **Document Summarizer**: Summarize text documents to extract key information efficiently.

## Setup Instructions

To run this project locally, follow these steps:

Clone the repository: git clone https://github.com/varnitsaxena7/InsightEngine.git
                      
Install dependencies: pip install -r requirements.txt

Configure API Keys: Obtain a Google API key and add it to config.json.

Run the Streamlit app: streamlit run app.py

Open your browser and navigate to http://localhost:8501 to view the application.

Technologies Used
Streamlit: Front-end framework for building interactive web applications.
NLTK: Natural Language Toolkit for sentiment analysis.
Googletrans: Google Translate API for language translation.
YouTube Transcript API: For fetching transcripts of YouTube videos.
PIL: Python Imaging Library for image processing.
Scikit-learn: For topic modeling using Latent Dirichlet Allocation (LDA).
Chardet: For automatic detection of text file encoding.
