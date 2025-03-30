if __name__ == "__main__":
    import pandas as pd
    from data.preprocess import TextPreprocessor
    from transformers import BertTokenizer

    __import__("warnings").filterwarnings("ignore")

    MODEL_NAME = "bert-base-uncased"

    train_df = pd.read_csv("data/raw/train.csv")
    test_df = pd.read_csv("data/raw/test.csv")

    preprocessor = TextPreprocessor(text_column="text")
    train_df = preprocessor.process(train_df)
    test_df = preprocessor.process(test_df)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_dataset(df):
        return tokenizer(df["claned_text"].tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    train_encodings = tokenize_dataset(train_df)
    test_encodings = tokenize_dataset(test_df)

    train_labels = pd.factorize(train_df["sentiment"])[0]
    test_labels = pd.factorize(test_df["sentiment"])[0]
