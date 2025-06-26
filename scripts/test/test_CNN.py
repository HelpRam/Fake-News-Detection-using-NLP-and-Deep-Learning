# Purpose: Evaluate trained CNN model on WELFake test set and FakeNewsNet

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F

# --------------------------- Configuration ---------------------------
BATCH_SIZE = 64
MAX_LEN = 500
EMBED_DIM = 100
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
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

# --------------------------- CNN Model ---------------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_x]
        cat = torch.cat(pooled, 1)
        out = self.dropout(cat)
        return self.sigmoid(self.fc(out)).squeeze(1)

# --------------------------- Evaluation Function ---------------------------
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

    # Print and save classification report
    report = classification_report(all_labels, all_preds, digits=4)
    print(f"\n{set_name} Classification Report:\n{report}")
    os.makedirs("outputs/reports", exist_ok=True)
    with open(f"outputs/reports/cnn_{set_name.lower().replace(' ', '_')}_report.txt", "w") as f:
        f.write(report)

    # Save confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"CNN - {set_name} Confusion Matrix")
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrices/cnn_{set_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    with open("outputs/cnn_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    tokenizer = basic_tokenizer

    model = CNNClassifier(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES)
    model.load_state_dict(torch.load("outputs/models/cnn_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # WELFake Test Set
    df_test = pd.read_csv("data/wel_fake/test.csv")
    test_dataset = NewsDataset(df_test['clean_text'].tolist(), df_test['label'].tolist(), vocab, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    evaluate(model, test_loader, set_name="WELFake Test")

    # FakeNewsNet Cross-Domain
    df_fakenewsnet = pd.read_csv("data/fakenewsnet/test_external.csv")
    fnn_dataset = NewsDataset(df_fakenewsnet['clean_text'].tolist(), df_fakenewsnet['label'].tolist(), vocab, tokenizer)
    fnn_loader = DataLoader(fnn_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    evaluate(model, fnn_loader, set_name="FakeNewsNet")
