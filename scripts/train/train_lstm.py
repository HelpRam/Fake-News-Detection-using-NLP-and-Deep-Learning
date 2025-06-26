
# Purpose: Train an LSTM model on WELFake dataset using PyTorch with GloVe embeddings (manual loading)

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
from collections import Counter
import pickle
from torchviz import make_dot  # For model graph visualization

# --------------------------- Configuration ---------------------------
BATCH_SIZE = 64               # Number of samples per batch
MAX_LEN = 500                 # Maximum tokens per input sample
EMBED_DIM = 100               # Dimension of GloVe word embeddings
HIDDEN_DIM = 128              # Size of LSTM hidden layer
NUM_EPOCHS = 10               # Number of training epochs (updated from 5 to 10)
PATIENCE = 3                  # Early stopping patience
LEARNING_RATE = 0.001         # Optimizer learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
GLOVE_PATH = "data/glove/glove.6B.100d.txt"  # Path to GloVe vectors

# --------------------------- Custom Dataset ---------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])[:MAX_LEN]  # Tokenize and truncate
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]  # Convert to vocab indices
        return torch.tensor(indices), torch.tensor(self.labels[idx], dtype=torch.float)

# Pad sequences and collate batch
def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_seqs, torch.tensor(labels)

# Basic whitespace tokenizer
def basic_tokenizer(text):
    return text.lower().strip().split()

# Build vocabulary from training data
def build_vocab(texts, tokenizer, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    # Only keep tokens with freq >= min_freq
    tokens = [token for token, freq in counter.items() if freq >= min_freq]
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({token: idx+2 for idx, token in enumerate(tokens)})
    return vocab

# Load pre-trained GloVe vectors into embedding matrix
def load_glove_embeddings(vocab, glove_path=GLOVE_PATH, dim=100):
    embeddings = torch.randn(len(vocab), dim)
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            if word in vocab:
                embeddings[vocab[word]] = vec
    return embeddings

# --------------------------- LSTM Model ---------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)              # Convert indices to vectors
        _, (hn, _) = self.lstm(x)          # Extract final hidden state
        out = self.fc(hn[-1])              # Apply linear transformation
        return self.sigmoid(out).squeeze(1)  # Return probabilities

# --------------------------- Training & Evaluation ---------------------------
def train_model(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for x_batch, y_batch in tqdm(loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluate on validation set using F1-score
def evaluate_model(model, loader):
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
    report = classification_report(all_labels, all_preds, output_dict=True)
    return report['accuracy'], report['0.0']['f1-score'], report['1.0']['f1-score']

# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    print("\nLoading and preprocessing data...")
    df_train = pd.read_csv("data/wel_fake/train.csv")
    df_val = pd.read_csv("data/wel_fake/val.csv")

    tokenizer = basic_tokenizer
    vocab = build_vocab(df_train['clean_text'], tokenizer)
    with open("outputs/lstm_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    print("\nLoading GloVe vectors from file...")
    weights_matrix = load_glove_embeddings(vocab)

    print("\nPreparing datasets...")
    train_dataset = NewsDataset(df_train['clean_text'].tolist(), df_train['label'].tolist(), vocab, tokenizer)
    val_dataset = NewsDataset(df_val['clean_text'].tolist(), df_val['label'].tolist(), vocab, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\nBuilding model...")
    model = LSTMClassifier(vocab_size=len(vocab), embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, pretrained_weights=weights_matrix)
    model.to(DEVICE)

    # Visualize model graph on one batch
    sample_x, _ = next(iter(train_loader))
    sample_x = sample_x.to(DEVICE)
    sample_out = model(sample_x)
    make_dot(sample_out, params=dict(model.named_parameters())).render("outputs/lstm_graph", format="png")

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model with early stopping...")
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion)
        acc, f1_fake, f1_real = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Val Acc: {acc:.4f}, F1 Fake: {f1_fake:.4f}, F1 Real: {f1_real:.4f}")

        if (f1_fake + f1_real) / 2 > best_val_f1:
            best_val_f1 = (f1_fake + f1_real) / 2
            torch.save(model.state_dict(), "outputs/models/lstm_model.pth")
            patience_counter = 0  # reset
            print("Model improved. Saving checkpoint.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete. Best model saved.")