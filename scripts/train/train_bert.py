# Purpose: Fine-tune a BERT model for fake news classification on WELFake

# Step 1: Import libraries
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np

# Step 2: Set configuration and hyperparameters
MODEL_NAME = "bert-base-uncased"  # Pre-trained BERT model
MAX_LEN = 256  # Max token length for input sequences
BATCH_SIZE = 16  # Mini-batch size
NUM_EPOCHS = 10  # Total training epochs
LEARNING_RATE = 2e-5  # Learning rate for fine-tuning
PATIENCE = 2  # Early stopping patience
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Step 3: Define a Dataset class for BERT inputs
class BERTNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts  # List of texts
        self.labels = labels  # List of labels (0 or 1)
        self.tokenizer = tokenizer  # BERT tokenizer
        self.max_len = max_len  # Truncation/padding length

    def __len__(self):
        return len(self.texts)  # Number of samples

    def __getitem__(self, idx):
        text = self.texts[idx]  # Get text sample
        label = self.labels[idx]  # Get label
        # Tokenize and encode text
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Input IDs
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Attention mask
            'labels': torch.tensor(label, dtype=torch.long)  # Label tensor
        }

# Step 4: Create DataLoader from DataFrame
def create_data_loader(df, tokenizer):
    dataset = BERTNewsDataset(df['clean_text'].tolist(), df['label'].tolist(), tokenizer, MAX_LEN)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Step 5: Evaluation helper to compute accuracy and F1

def evaluate(model, data_loader):
    model.eval()  # Set model to eval mode
    preds = []
    true = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)  # Get class prediction

            preds.extend(predictions.cpu().numpy())
            true.extend(labels.cpu().numpy())

    # Use sklearn to get metrics
    report = classification_report(true, preds, output_dict=True)
    return report['accuracy'], report['0']['f1-score'], report['1']['f1-score']

# Step 6: Main function to load data and fine-tune BERT
if __name__ == "__main__":
    print("\nLoading data...")
    df_train = pd.read_csv("data/wel_fake/train.csv")  # Load training set
    df_val = pd.read_csv("data/wel_fake/val.csv")  # Load validation set

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)  # Load tokenizer
    train_loader = create_data_loader(df_train, tokenizer)  # Train loader
    val_loader = create_data_loader(df_val, tokenizer)  # Val loader

    print("\nInitializing model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Load pre-trained BERT
    model.to(DEVICE)

    # Step 7: Set up optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = NUM_EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_f1 = 0  # Track best validation F1
    patience_counter = 0  # Count epochs without improvement

    print("\nTraining...")
    for epoch in range(NUM_EPOCHS):
        model.train()  # Enable training mode
        running_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # Forward pass and compute loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()  # Backpropagation

            optimizer.step()  # Optimizer step
            lr_scheduler.step()  # Update learning rate
            optimizer.zero_grad()  # Reset gradients
            running_loss += loss.item()

        # Step 8: Validation after each epoch
        avg_loss = running_loss / len(train_loader)
        acc, f1_fake, f1_real = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1 Fake: {f1_fake:.4f}, F1 Real: {f1_real:.4f}")

        avg_f1 = (f1_fake + f1_real) / 2  # Average F1
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), "outputs/models/bert_model.pth")  # Save best model
            print("Model improved. Saving checkpoint.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print("\nTraining complete. Best model saved.")
