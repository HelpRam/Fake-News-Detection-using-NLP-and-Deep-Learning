# app_bert.py
# Streamlit app for BERT Fake News Classifier

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

# ------------------------ Configuration ------------------------
MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "outputs\\models\\bert_model.pth"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Load model ------------------------
st.set_page_config(page_title="BERT Fake News Classifier", layout="centered")
st.title("Fake News Detection Using NLP and Deep Learning")
st.caption("Fine-tuned BERT model")

@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ------------------------ Predict Function ------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LEN)
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = round(probs[0][pred_class].item(), 4)
        label = "REAL" if pred_class == 1 else "FAKE"
    return label, confidence

# ------------------------ App Interface ------------------------
st.subheader("Enter news text below:")
news_input = st.text_area("Paste the news article or headline:", height=200)

if st.button("Classify"):
    if not news_input.strip():
        st.warning("Please enter some text.")
    else:
        label, confidence = predict(news_input.strip())
        if label == "REAL":
            st.success(f" Prediction: REAL news (Confidence: {confidence})")
        else:
            st.error(f" Prediction: FAKE news (Confidence: {confidence})")
