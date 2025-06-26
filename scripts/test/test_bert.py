# Purpose: Evaluate fine-tuned BERT model on WELFake test set and FakeNewsNet

import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

# ------------------------ Config ------------------------
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
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ------------------------ Evaluation Function ------------------------
def evaluate(model, data_loader, set_name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    report = classification_report(all_labels, all_preds, digits=4)
    print(f"\n{set_name} Classification Report:\n{report}")
    os.makedirs("outputs/reports", exist_ok=True)
    with open(f"outputs/reports/bert_{set_name.lower().replace(' ', '_')}_report.txt", "w") as f:
        f.write(report)

    # Plot and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"BERT - {set_name} Confusion Matrix")
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrices/bert_{set_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load("outputs/models/bert_model.pth", map_location=DEVICE))
    model.to(DEVICE)

    # WELFake Test Set
    df_test = pd.read_csv("data/wel_fake/test.csv")
    test_dataset = BERTNewsDataset(df_test['clean_text'].tolist(), df_test['label'].tolist(), tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    evaluate(model, test_loader, set_name="WELFake Test")

    # FakeNewsNet Generalization Test
    df_fnn = pd.read_csv("data/fakenewsnet/test_external.csv")
    fnn_dataset = BERTNewsDataset(df_fnn['clean_text'].tolist(), df_fnn['label'].tolist(), tokenizer, MAX_LEN)
    fnn_loader = DataLoader(fnn_dataset, batch_size=BATCH_SIZE, shuffle=False)
    evaluate(model, fnn_loader, set_name="FakeNewsNet")
# ------------------------ test the finetune model  test bert.py--------------------3