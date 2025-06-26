# scripts/train/train_cnn.py
# Purpose: Train a CNN model on WELFake dataset using GloVe embeddings in PyTorch

# Import necessary libraries
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
import torch.nn.functional as F

# --------------------------- Configuration ---------------------------
BATCH_SIZE = 64  # Number of samples per batch
MAX_LEN = 500  # Maximum sequence length
EMBED_DIM = 100  # Dimension of word embeddings
NUM_FILTERS = 100  # Number of filters per kernel size
FILTER_SIZES = [3, 4, 5]  # Kernel sizes to capture different n-gram patterns
NUM_EPOCHS = 10  # Maximum number of epochs
PATIENCE = 3  # Early stopping patience
LEARNING_RATE = 0.001  # Learning rate for the optimizer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
GLOVE_PATH = "data/glove/glove.6B.100d.txt"  # Path to GloVe embeddings

# --------------------------- Dataset & Tokenization ---------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        self.texts = texts  # List of text samples
        self.labels = labels  # Corresponding labels
        self.vocab = vocab  # Vocabulary dictionary
        self.tokenizer = tokenizer  # Tokenization function

    def __len__(self):
        return len(self.labels)  # Total number of samples

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])[:MAX_LEN]  # Tokenize and truncate
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]  # Convert tokens to indices
        return torch.tensor(indices), torch.tensor(self.labels[idx], dtype=torch.float)  # Return as tensors

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Unpack sequences and labels
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)  # Pad sequences
    return padded_seqs, torch.tensor(labels)  # Return padded batch

def basic_tokenizer(text):
    return text.lower().strip().split()  # Simple whitespace-based tokenizer

def build_vocab(texts, tokenizer, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))  # Count token frequencies
    vocab = {token: i+2 for i, (token, freq) in enumerate(counter.items()) if freq >= min_freq}  # Build vocab
    vocab['<pad>'] = 0  # Padding index
    vocab['<unk>'] = 1  # Unknown token index
    return vocab

def load_glove_embeddings(vocab, glove_path=GLOVE_PATH, dim=100):
    embeddings = torch.randn(len(vocab), dim)  # Initialize with random values
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = torch.tensor([float(val) for val in values[1:]], dtype=torch.float)
            if word in vocab:
                embeddings[vocab[word]] = vec  # Replace with GloVe vector
    return embeddings

# --------------------------- CNN Model ---------------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, pretrained_weights=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Embedding layer
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(pretrained_weights)  # Load pre-trained embeddings
            self.embedding.weight.requires_grad = False  # Freeze embeddings

        # Create convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embed_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)  # Dropout layer to reduce overfitting
        self.fc = nn.Linear(num_filters * len(filter_sizes), 1)  # Fully connected output
        self.sigmoid = nn.Sigmoid()  # Sigmoid for binary classification

    def forward(self, x):
        x = self.embedding(x)  # Convert input to embeddings
        x = x.unsqueeze(1)  # Reshape to fit Conv2D: (B, 1, L, D)
        conv_x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply conv + ReLU
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_x]  # Max-pool over time
        cat = torch.cat(pooled, 1)  # Concatenate all pooled outputs
        out = self.dropout(cat)  # Apply dropout
        return self.sigmoid(self.fc(out)).squeeze(1)  # Output probabilities

# --------------------------- Training & Evaluation ---------------------------
def train_model(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0
    for x_batch, y_batch in tqdm(loader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()  # Clear gradients
        output = model(x_batch)  # Forward pass
        loss = criterion(output, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()
    return running_loss / len(loader)

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            output = model(x_batch)  # Forward pass
            preds = (output > 0.5).int().cpu().tolist()  # Convert probabilities to class labels
            all_preds.extend(preds)
            all_labels.extend(y_batch.tolist())
    report = classification_report(all_labels, all_preds, output_dict=True)  # Generate report
    return report['accuracy'], report['0.0']['f1-score'], report['1.0']['f1-score']  # Return key metrics

# --------------------------- Main ---------------------------
if __name__ == "__main__":
    print("\nLoading data...")
    df_train = pd.read_csv("data/wel_fake/train.csv")  # Load training data
    df_val = pd.read_csv("data/wel_fake/val.csv")  # Load validation data

    tokenizer = basic_tokenizer  # Set tokenizer
    vocab = build_vocab(df_train['clean_text'], tokenizer)  # Build vocab from training data
    with open("outputs/cnn_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)  # Save vocab

    weights_matrix = load_glove_embeddings(vocab)  # Load GloVe embeddings

    # Create datasets and loaders
    train_dataset = NewsDataset(df_train['clean_text'].tolist(), df_train['label'].tolist(), vocab, tokenizer)
    val_dataset = NewsDataset(df_val['clean_text'].tolist(), df_val['label'].tolist(), vocab, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\nBuilding model...")
    model = CNNClassifier(len(vocab), EMBED_DIM, NUM_FILTERS, FILTER_SIZES, weights_matrix)  # Instantiate model
    model.to(DEVICE)  # Move model to device

    criterion = nn.BCELoss()  # Define binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Define Adam optimizer

    best_val_f1 = 0  # Initialize best validation F1 score
    patience_counter = 0  # Initialize patience counter

    print("\nTraining with early stopping...")
    for epoch in range(NUM_EPOCHS):
        loss = train_model(model, train_loader, optimizer, criterion)  # Train for one epoch
        acc, f1_fake, f1_real = evaluate_model(model, val_loader)  # Evaluate on validation set
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {loss:.4f}, Acc: {acc:.4f}, F1 Fake: {f1_fake:.4f}, F1 Real: {f1_real:.4f}")

        avg_f1 = (f1_fake + f1_real) / 2  # Compute average F1
        if avg_f1 > best_val_f1:
            best_val_f1 = avg_f1  # Update best F1
            torch.save(model.state_dict(), "outputs/models/cnn_model.pth")  # Save model
            patience_counter = 0
            print("Model improved. Saving checkpoint.")
        else:
            patience_counter += 1  # Increase patience
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break  # Stop training early

    print("\nTraining complete. Best model saved.")
