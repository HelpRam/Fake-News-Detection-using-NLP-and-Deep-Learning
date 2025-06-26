from scripts.preprocess.preprocess_welfake import load_welfake_dataset, preprocess_welfake,split_and_save_welfake
from scripts.preprocess.preprocess_fakenewsnet import load_fakenewsnet_dataset, preprocess_fakenewsnet,save_fakenewsnet

if __name__ == "__main__":
    # Load WELfake dataset
    df_wel = load_welfake_dataset()
    # Load WELfake preprocessed dataset 
    df_wel_clean = preprocess_welfake(df_wel)
    split_and_save_welfake(df_wel_clean)
    #-----------------------------------------------------#
     # Load FakeNewsNet
    df_fnn = load_fakenewsnet_dataset()
    # Load WELfake preprocessed dataset 
    df_wel_clean = preprocess_fakenewsnet(df_fnn)
    save_fakenewsnet(df_wel_clean)
