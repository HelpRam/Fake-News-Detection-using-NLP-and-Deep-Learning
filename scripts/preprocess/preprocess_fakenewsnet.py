import pandas as pd
import os
import re

def load_fakenewsnet_dataset(base_path="data/fakenewsnet"):
    """
    Loads and combines FakeNewsNet dataset CSVs from GossipCop and PolitiFact.
    
    Returns:
        pd.DataFrame: Combined DataFrame with columns [title, label]
    """
    def load_and_label(filename, label):
        path = os.path.join(base_path, filename)
        df = pd.read_csv(path)
        df['label'] = label
        return df[['title', 'label']]  # Keep only title and label for now

    # Load each subset
    dfs = [
        load_and_label("gossipcop_fake.csv", 0),
        load_and_label("gossipcop_real.csv", 1),
        load_and_label("politifact_fake.csv", 0),
        load_and_label("politifact_real.csv", 1)
    ]

    # Combine into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f" FakeNewsNet loaded: {combined_df.shape[0]} rows")
    print(combined_df.head(5))

    return combined_df


# Function to clean text data
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)       # remove numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_fakenewsnet(df):
    """
    Cleans the title text from FakeNewsNet.
    Returns a DataFrame with clean_text and label columns.
    """
    df["title"] = df["title"].fillna("")
    df["clean_text"] = df["title"].apply(clean_text)

    # Drop if text is too short
    df = df[df["clean_text"].str.len() > 10]

    print(f" Preprocessed FakeNewsNet: {df.shape[0]} samples, {df.shape[1]} columns")
    print(df.columns)
    return df[["clean_text", "label"]]

# Functio to save the preprocessed FakeNewsNet dataset
def save_fakenewsnet(df, output_path="data/fakenewsnet/test_external.csv"):
    """
    Saves full preprocessed FakeNewsNet as external test set.
    """
    df.to_csv(output_path, index=False)
    print(f" Saved FakeNewsNet test set: {output_path}")
