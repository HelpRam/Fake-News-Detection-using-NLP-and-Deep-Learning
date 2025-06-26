
# Purpose: Evaluate trained LSTM model on WELFake test set and FakeNewsNet

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from collections import Counter

# --------------------------- Configuration ---------------------------
BATCH_SIZE = 64
MAX_LEN = 500
EMBED_DIM = 100
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------- Dataset & Tokenization ---------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])[:MAX_LEN]
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        return torch.tensor(indices), torch.tensor(self.labels[idx], dtype=torch.float)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_seqs, torch.tensor(labels)

def basic_tokenizer(text):
    return text.lower().strip().split()

# --------------------------- Model Definition ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out).squeeze(1)

# --------------------------- Utility Functions ---------------------------
def evaluate(model, loader, set_name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            output = model(x_batch)
            preds = (output > 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())

    report = classification_report(all_labels, all_preds, digits=4)
    print(f"\n{set_name} Classification Report:\n{report}")

    # Save classification report
    os.makedirs("outputs/reports", exist_ok=True)
    with open(f"outputs/reports/lstm_{set_name.lower().replace(' ', '_')}_report.txt", "w") as f:
        f.write(report)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    labels = [0, 1]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"LSTM - {set_name} Confusion Matrix")
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrices/lstm_{set_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()

# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    # Load saved vocab
    with open("outputs/lstm_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    tokenizer = basic_tokenizer

    # Initialize model and load weights
    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load("outputs/models/lstm_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # ---------------- Evaluate on WELFake test ----------------
    df_test = pd.read_csv("data/wel_fake/test.csv")
    test_dataset = NewsDataset(df_test['clean_text'].tolist(), df_test['label'].tolist(), vocab, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    evaluate(model, test_loader, set_name="WELFake Test")

    # ---------------- Evaluate on FakeNewsNet ----------------
    df_fakenewsnet = pd.read_csv("data/fakenewsnet/test_external.csv")
    fnn_dataset = NewsDataset(df_fakenewsnet['clean_text'].tolist(), df_fakenewsnet['label'].tolist(), vocab, tokenizer)
    fnn_loader = DataLoader(fnn_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    evaluate(model, fnn_loader, set_name="FakeNewsNet")
