if __name__ == "__main__":
    import torch
    import logging
    import pandas as pd
    from model.train import Trainer
    from model.predict import SentimentPredictor
    from data.preprocess import TextPreprocessor
    from data.dataset import SentimentDataset, create_dataloader
    from transformers import BertTokenizer, BertForSequenceClassification
    from transformers.utils.logging import set_verbosity_error

    __import__("warnings").filterwarnings("ignore")
    set_verbosity_error()

    MODEL_NAME = "bert-base-uncased"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv("data/raw/train.csv")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Loading data...")
    preprocessor = TextPreprocessor(text_column="text")
    train_df = preprocessor.process(train_df)
    train_df, val_df = preprocessor.split_data(train_df, test_size=0.1)

    logger.info("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(text_series):
        return tokenizer(text_series.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    logger.info("Tokenizing data...")
    train_encodings = tokenize(train_df["cleaned_text"])
    val_encodings = tokenize(val_df["cleaned_text"])

    train_labels = pd.factorize(train_df["sentiment"])[0]
    val_labels = pd.factorize(val_df["sentiment"])[0]

    logger.info("Creating Dataset...")
    train_dataset = SentimentDataset(train_encodings, DEVICE, train_labels)
    val_dataset = SentimentDataset(val_encodings, DEVICE, val_labels)

    logger.info("Creating DataLoader...")
    train_loader = create_dataloader(train_dataset, batch_size=32)
    val_loader = create_dataloader(val_dataset, batch_size=32, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3, id2label={0: "negative", 1: "neutral", 2: "positive"})

    trainer = Trainer(model=model, device="cuda" if torch.cuda.is_available() else "cpu", epochs=5, lr=3e-5)
    trainer.train(train_loader, val_loader)

    final_loss, final_acc, final_report = trainer.evaluate(val_loader)
    logger.info(f"\nFinal Result:")
    logger.info(f"Modelo: {MODEL_NAME}")
    logger.info(f"Loss: {final_loss:.4f} | Accuracy: {final_acc:.4f}")
    logger.info(f"Classification Report:\n{final_report}")

    predictor = SentimentPredictor(model_path="models/best_model.pt", device=DEVICE)
    test_predictions = predictor.predict("data/raw/test.csv")
    test_predictions.to_csv("data/predictions/final_predictions.csv", index=False)
    logger.info("Predictions saved in data/predictions/final_predictions.csv")
