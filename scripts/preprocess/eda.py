# scripts/preprocess/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os


# ---------- Label Distribution ----------
def plot_label_distribution(df, name="Dataset", output_dir="outputs/eda"):
    sns.countplot(x="label", data=df)
    plt.title(f"{name} Label Distribution (0 = Fake, 1 = Real)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name.lower()}_label_distribution.png")
    plt.savefig(path)
    plt.show()


# ---------- Text Length Distribution ----------
def plot_text_length(df, name="Dataset", output_dir="outputs/eda"):
    df["text_len"] = df["clean_text"].apply(lambda x: len(x.split()))
    sns.histplot(df["text_len"], bins=50, kde=True)
    plt.title(f"{name} Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name.lower()}_text_length.png")
    plt.savefig(path)
    plt.show()


# ---------- WordCloud Visualization ----------
def plot_wordcloud(df, label, name="Dataset", output_dir="outputs/eda"):
    text = " ".join(df[df["label"] == label]["clean_text"].tolist())
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    label_str = "Fake" if label == 0 else "Real"
    plt.title(f"{name}: Common Words in {label_str} News")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name.lower()}_wordcloud_{label_str.lower()}.png")
    plt.savefig(path)
    plt.show()


# ---------- TF-IDF Word Importance ----------
def show_top_tfidf_words(df, label, name="Dataset", top_n=20, output_dir="outputs/eda"):
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    subset = df[df["label"] == label]
    X = vectorizer.fit_transform(subset["clean_text"])
    mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
    top_indices = mean_tfidf.argsort()[::-1][:top_n]

    words = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    scores = mean_tfidf[top_indices]

    plt.figure(figsize=(10, 4))
    sns.barplot(x=scores, y=words)
    label_str = "Fake" if label == 0 else "Real"
    plt.title(f"Top {top_n} TF-IDF Words in {label_str} News")
    plt.xlabel("Mean TF-IDF Score")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{name.lower()}_top_tfidf_{label_str.lower()}.png")
    plt.savefig(path)
    plt.show()


# ---------- Run EDA on a Given Dataset ----------
def run_eda(df, dataset_name):
    output_dir = "outputs/eda"
    os.makedirs(output_dir, exist_ok=True)
    plot_label_distribution(df, dataset_name, output_dir)
    plot_text_length(df, dataset_name, output_dir)
    plot_wordcloud(df, label=0, name=dataset_name, output_dir=output_dir)
    plot_wordcloud(df, label=1, name=dataset_name, output_dir=output_dir)
    show_top_tfidf_words(df, label=0, name=dataset_name, output_dir=output_dir)
    show_top_tfidf_words(df, label=1, name=dataset_name, output_dir=output_dir)


# ---------- Entry Point ----------
if __name__ == "__main__":
    # Load preprocessed CSVs
    df_wel = pd.read_csv("data/wel_fake/train.csv")
    df_fnn = pd.read_csv("data/fakenewsnet/test_external.csv")

    print("\nRunning EDA on WELFake Dataset")
    run_eda(df_wel, "WELFake")

    print("\nRunning EDA on FakeNewsNet Dataset")
    run_eda(df_fnn, "FakeNewsNet")
