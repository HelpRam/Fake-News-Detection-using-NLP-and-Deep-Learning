# scripts/test/test_hybrid.py
# Purpose: Evaluate Hybrid BERT + CNN + LSTM model on WELFake test set and FakeNewsNet

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ------------------------ Configuration ------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------ Dataset ------------------------
class BERTNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# ------------------------ Hybrid Model ------------------------
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
            last_hidden_state = outputs.last_hidden_state

        x_cnn = last_hidden_state.permute(0, 2, 1)
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = torch.max_pool1d(x_cnn, kernel_size=x_cnn.size(2)).squeeze(2)

        x_lstm, _ = self.lstm(last_hidden_state)
        x_lstm = x_lstm[:, -1, :]

        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x)).squeeze(1)

# ------------------------ Evaluation ------------------------
def evaluate_model(model, data_loader, name="WELFake"):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            y = batch['labels'].to(DEVICE)
            output = model(input_ids, attention_mask)
            pred = (output > 0.5).int().cpu().tolist()
            preds.extend(pred)
            labels.extend(y.cpu().tolist())

    print(f"\n{name} Classification Report:")
    print(classification_report(labels, preds, digits=4))

    # Save confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig(f"outputs/plots/conf_matrix_{name.lower()}.png")
    plt.close()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    print("\nLoading test data...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME)
    model = HybridClassifier(bert)
    model.load_state_dict(torch.load("/kaggle/working/hybrid_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    df_test = pd.read_csv("/kaggle/input/welfake-datasets/test.csv")
    df_gen = pd.read_csv("/kaggle/input/fakenewsnet-minimal/test_external.csv")

    test_ds = BERTNewsDataset(df_test['clean_text'].tolist(), df_test['label'].tolist(), tokenizer, MAX_LEN)
    gen_ds = BERTNewsDataset(df_gen['clean_text'].tolist(), df_gen['label'].tolist(), tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    gen_loader = DataLoader(gen_ds, batch_size=BATCH_SIZE)

    evaluate_model(model, test_loader, name="WELFake")
    evaluate_model(model, gen_loader, name="FakeNewsNet")
