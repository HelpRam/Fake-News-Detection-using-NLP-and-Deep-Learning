# scripts/train/train_baselines.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


# ---------- Load Data ----------
def load_dataset(path):
    return pd.read_csv(path)


# ---------- Save Confusion Matrix Plot ----------
def save_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrices/{filename}.png")
    plt.close()


# ---------- Save Classification Report ----------
def save_classification_report(report_str, filename):
    os.makedirs("outputs/reports", exist_ok=True)
    with open(f"outputs/reports/{filename}.txt", "w") as f:
        f.write(report_str)


# ---------- Train and Evaluate Model ----------
def train_and_evaluate_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test):
    print(f"\nTraining {name}...")
    clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', model)
    ])
    clf.fit(X_train, y_train)

    print(f"\n{name} Evaluation on Validation Set:")
    y_val_pred = clf.predict(X_val)
    print(classification_report(y_val, y_val_pred))

    print(f"\n{name} Evaluation on Internal Test Set:")
    y_test_pred = clf.predict(X_test)
    report = classification_report(y_test, y_test_pred)
    print(report)

    # Save confusion matrix and report
    save_confusion_matrix(y_test, y_test_pred, labels=[0, 1],
                          title=f"{name} - WELFake Test Set",
                          filename=f"{name.lower().replace(' ', '_')}_welfake_confusion")
    save_classification_report(report, f"{name.lower().replace(' ', '_')}_welfake_report")

    return clf


# ---------- Run All Baseline Models ----------
def run_all_models():
    # Load preprocessed WELFake splits
    train_df = load_dataset("data/wel_fake/train.csv")
    val_df = load_dataset("data/wel_fake/val.csv")
    test_df = load_dataset("data/wel_fake/test.csv")

    X_train, y_train = train_df["clean_text"], train_df["label"]
    X_val, y_val = val_df["clean_text"], val_df["label"]
    X_test, y_test = test_df["clean_text"], test_df["label"]

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Support Vector Machine": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    trained_models = {}
    for name, model in models.items():
        trained_models[name] = train_and_evaluate_model(
            name, model, X_train, y_train, X_val, y_val, X_test, y_test
        )

    return trained_models


# ---------- Evaluate on External Dataset ----------
def evaluate_on_fakenewsnet(model_dict):
    df_fnn = load_dataset("data/fakenewsnet/test_external.csv")
    X_ext, y_ext = df_fnn["clean_text"], df_fnn["label"]

    print("\n--- Generalization Test on FakeNewsNet ---")
    for name, model in model_dict.items():
        print(f"\n{name} on FakeNewsNet:")
        y_pred = model.predict(X_ext)
        report = classification_report(y_ext, y_pred)
        print(report)

        # Save confusion matrix and report
        save_confusion_matrix(y_ext, y_pred, labels=[0, 1],
                              title=f"{name} - FakeNewsNet Test Set",
                              filename=f"{name.lower().replace(' ', '_')}_fakenewsnet_confusion")
        save_classification_report(report, f"{name.lower().replace(' ', '_')}_fakenewsnet_report")


# ---------- Main Entry ----------
if __name__ == "__main__":
    models = run_all_models()
    evaluate_on_fakenewsnet(models)
