# Retail Demand Forecasting Chatbot

A Streamlit chatbot that explains retail demand forecasting concepts in a simple conversational format. The bot can answer questions about demand forecasting, seasonality, trends, promotions, accuracy metrics, inventory planning, overforecasting, underforecasting, and forecast interpretation.

## Live Demo

Try the deployed app here:

https://retail-demand-chatbot.streamlit.app/

## Project Overview

This project uses a lightweight rule-based NLP approach:

- User messages are cleaned and preprocessed with NLTK.
- Example questions from `intents.json` are converted into TF-IDF vectors.
- The chatbot compares the user message with known intent patterns using cosine similarity.
- If the confidence score is high enough, the bot returns a response from the matched intent.
- If the message is outside the chatbot's knowledge area, it returns a fallback response.

## Features

- Streamlit chat interface
- Retail demand forecasting concept explanations
- Intent-based response matching
- TF-IDF and cosine similarity matching
- Debug option to show matched intent and confidence score
- Easy-to-edit `intents.json` knowledge base

## Topics Covered

- What demand forecasting means
- Retail forecasting use cases
- Historical sales data
- Seasonality and trends
- Promotions, holidays, and external factors
- Forecasting model types
- MAE, RMSE, and MAPE
- Overforecasting vs underforecasting
- Inventory and stockout impact
- Interpreting predicted demand
- Forecasting best practices and limitations

## Files

- `app.py` - Streamlit front-end for the chatbot
- `chatbot_engine.py` - Core TF-IDF and cosine similarity matching logic
- `nlp_utils.py` - Text cleaning, tokenization, stopword removal, and lemmatization
- `intents.json` - Chatbot knowledge base with patterns and responses
- `requirements.txt` - Python dependencies

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Locally

Run the Streamlit app:

```bash
streamlit run app.py
```

Or test the chatbot logic in the terminal:

```bash
python chatbot_engine.py
```

## Deployment

This app is deployed on Streamlit Community Cloud. To deploy your own version:

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io.
3. Click **Create app**.
4. Select your GitHub repository and branch.
5. Set the main file path to `app.py`.
6. Click **Deploy**.

## Limitations

This chatbot is not a fine-tuned AI model. It uses predefined intents and TF-IDF similarity, so it works best when user questions are close to the examples in `intents.json`. For advanced use cases, the chatbot could be extended with embeddings, retrieval, or an LLM-based backend.
