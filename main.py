if __name__ == "__main__":
    import pandas as pd
    from data.preprocess_data import PreprocessData

    __import__("warnings").filterwarnings("ignore")

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    preprocess_data = PreprocessData(train_df, "text")
    cleaned_train_df = preprocess_data.process()

    for i in range(5):
        print(f"Original: {train_df['text'].iloc[i]}")
        print(f"Cleaned: {cleaned_train_df['cleaned_text'].iloc[i]}")
        print()

    train_df.to_csv("data/processed/train_cleaned.csv", index=False)
    # test_df.to_csv("data/processed/test_cleaned.csv", index=False)
