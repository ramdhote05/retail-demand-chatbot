"""
app.py
------
Streamlit front-end for the chatbot. The actual NLP logic lives in
chatbot_engine.py / nlp_utils.py - this file is purely UI.

Run with:
    streamlit run app.py
"""

import streamlit as st

from chatbot_engine import ChatbotEngine

st.set_page_config(page_title="Retail Demand Forecasting Chatbot", page_icon="chart_with_upwards_trend")

st.title("Retail Demand Forecasting Chatbot")
st.caption(
    "Rule-based chatbot using NLTK for text preprocessing and "
    "TF-IDF + cosine similarity for intent matching."
)


@st.cache_resource
def load_engine():
    return ChatbotEngine(intents_path="intents.json")


engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm your retail demand forecasting assistant. Ask me "
                "about seasonality, forecast accuracy, promotions, inventory "
                "planning, or how to interpret predicted demand."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    reply = engine.get_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.write(reply)

with st.sidebar:
    st.header("About this bot")
    st.write(
        "Your message is normalized with NLTK (tokenized, stripped of "
        "stopwords, and lemmatized), converted to a TF-IDF vector, and "
        "compared via cosine similarity against a set of predefined "
        "intents in `intents.json`."
    )
    st.write("**Try asking about:**")
    st.markdown(
        "- What demand forecasting means\n"
        "- Seasonality and demand trends\n"
        "- Promotions, holidays, and external factors\n"
        "- MAE, RMSE, and MAPE\n"
        "- Overforecasting vs underforecasting\n"
        "- Interpreting predicted demand"
    )

    if st.session_state.get("messages") and st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    if st.checkbox("Show debug info (matched intent + confidence)"):
        if engine.history:
            last = engine.history[-1]
            st.write(f"Last matched intent: `{last[2]}`")
            st.write(f"Confidence score: `{last[3]}`")
        else:
            st.write("No messages yet.")
