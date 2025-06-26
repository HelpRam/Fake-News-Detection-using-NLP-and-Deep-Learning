import pandas as pd
import re
from sklearn.model_selection import train_test_split

def load_welfake_dataset(path="C:\\Users\\ramme\OneDrive\\Desktop\\Fake News Detection\\data\\wel_fake\\train-00000-of-00001-290868f0a36350c5.parquet"):
    """
    Loads the WELFake dataset from a local Parquet file.

    Args:
        path (str): Path to the .parquet file.

    Returns:
        pd.DataFrame: Loaded DataFrame with cleaned column names.
    """
    df = pd.read_parquet(path)

    # Optional: Clean/rename columns if needed
    if 'Label' in df.columns:
        df = df.rename(columns={
            "Title": "title",
            "Text": "text",
            "Label": "label"
        })

    print(f" WELFake loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head(5))

    return df



# Functio to clean text data
def clean_text(text):
    if pd.isna(text):
        return ""
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", "", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to preprocess WELFake dataset
def preprocess_welfake(df):
    """
    Cleans WELFake text and prepares input column.
    Combines 'title' and 'text' into a new 'clean_text' field.
    """
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["full_text"] = df["title"] + " " + df["text"]
    df["clean_text"] = df["full_text"].apply(clean_text)

    # Drop if text is too short
    df = df[df["clean_text"].str.len() > 20]

    print(f" Preprocessed WELFake: {df.shape[0]} samples,{df.shape[1]} columns")
    #  Show column name
    print(df.columns)
    print(df.head(5))
    return df[["clean_text", "label"]]


# Functio to split and save WELFAKE dataset into train/val/test

def split_and_save_welfake(df, output_dir="data/wel_fake"):
    """
    Splits preprocessed WELFake into train/val/test and saves as CSV.
    """
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(" Saved WELFake splits: train/val/test")

