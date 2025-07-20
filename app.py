import streamlit as st
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, BertModel

# ------------------ Configuration ------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Hybrid Model ------------------
class HybridClassifier(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.conv1 = nn.Conv1d(768, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=768, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 + 128 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

        x_cnn = hidden.permute(0, 2, 1)
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = torch.max_pool1d(x_cnn, kernel_size=x_cnn.size(2)).squeeze(2)

        x_lstm, _ = self.lstm(hidden)
        x_lstm = x_lstm[:, -1, :]

        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x)).squeeze(1)

# ------------------ Load Models ------------------
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Load Fine-tuned BERT
    bert_model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    bert_model.load_state_dict(torch.load("outputs\\models\\bert_model.pth", map_location=DEVICE))
    bert_model.to(DEVICE)
    bert_model.eval()

    # Load Hybrid
    bert_backbone = BertModel.from_pretrained(MODEL_NAME)
    hybrid_model = HybridClassifier(bert_backbone)
    hybrid_model.load_state_dict(torch.load("outputs\\models\\hybrid_model.pth", map_location=DEVICE))
    hybrid_model.to(DEVICE)
    hybrid_model.eval()

    return tokenizer, bert_model, hybrid_model

# ------------------ Prediction ------------------
def classify(text, model_type, tokenizer, bert_model, hybrid_model):
    encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LEN)
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    if model_type == "Fine-tuned BERT":
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
    else:
        with torch.no_grad():
            outputs = hybrid_model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = int(outputs.item() > 0.5)
            confidence = outputs.item()

    return prediction, confidence

# ------------------ Streamlit UI ------------------
st.title("📰 Fake News Detection System")
st.markdown("Detect whether a news statement is **Real** or **Fake** using NLP models.")

tokenizer, bert_model, hybrid_model = load_models()

text_input = st.text_area("Enter news headline or text:", height=150)

model_type = st.selectbox("Choose model for prediction:", ["Fine-tuned BERT", "Hybrid (BERT+CNN+LSTM)"])

if st.button("Classify"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        label, conf = classify(text_input, model_type, tokenizer, bert_model, hybrid_model)
        if label == 1:
            st.success(f"✅ Prediction: **REAL** (Confidence: {conf:.2f})")
        else:
            st.error(f"❌ Prediction: **FAKE** (Confidence: {conf:.2f})")
