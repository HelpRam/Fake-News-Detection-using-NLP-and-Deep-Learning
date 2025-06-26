# Purpose: Train a Hybrid BERT + CNN + LSTM model for Fake News Detection on WELFake

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

# ------------------------ Configuration ------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5
PATIENCE = 3
HIDDEN_DIM = 128
NUM_FILTERS = 64
FILTER_SIZE = 3
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
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=NUM_FILTERS, kernel_size=FILTER_SIZE, padding=1)
        self.lstm = nn.LSTM(input_size=768, hidden_size=HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(NUM_FILTERS + HIDDEN_DIM * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():  # freeze BERT weights
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)

        # CNN branch
        x_cnn = last_hidden_state.permute(0, 2, 1)  # (batch, 768, seq_len)
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = torch.max_pool1d(x_cnn, kernel_size=x_cnn.size(2)).squeeze(2)  # (batch, NUM_FILTERS)

        # LSTM branch
        x_lstm, _ = self.lstm(last_hidden_state)  # (batch, seq_len, hidden*2)
        x_lstm = x_lstm[:, -1, :]  # use final hidden state (batch, hidden*2)

        # Combine CNN + LSTM
        x = torch.cat([x_cnn, x_lstm], dim=1)
        x = self.dropout(x)
        return self.sigmoid(self.fc(x)).squeeze(1)

# ------------------------ Evaluation ------------------------
def evaluate(model, data_loader):
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
    report = classification_report(labels, preds, output_dict=True)
    return report['accuracy'], report['0.0']['f1-score'], report['1.0']['f1-score']

# ------------------------ Main ------------------------
if __name__ == "__main__":
    print("\nLoading data...")
    df_train = pd.read_csv("/kaggle/input/welfake-datasets/train.csv")
    df_val = pd.read_csv("/kaggle/input/welfake-datasets/val.csv")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_ds = BERTNewsDataset(df_train['clean_text'].tolist(), df_train['label'].tolist(), tokenizer, MAX_LEN)
    val_ds = BERTNewsDataset(df_val['clean_text'].tolist(), df_val['label'].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("\nBuilding model...")
    bert = BertModel.from_pretrained(MODEL_NAME)
    model = HybridClassifier(bert)
    model.to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    criterion = nn.BCELoss()

    best_f1 = 0
    patience_counter = 0
    save_path = "/kaggle/working/hybrid_model.pth"

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        acc, f1_fake, f1_real = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss:.4f}, Acc: {acc:.4f}, F1 Fake: {f1_fake:.4f}, F1 Real: {f1_real:.4f}")

        avg_f1 = (f1_fake + f1_real) / 2
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), save_path)
            print(f"Model improved. Saved to {save_path}.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete. Best model saved.")
