import torch
import pandas as pd
from typing import Tuple, Dict
from data.dataset import SentimentDataset, create_dataloader
from transformers import BertTokenizer, BertForSequenceClassification
from data.preprocess import TextPreprocessor


class SentimentPredictor:
    def __init__(self, model_path: str, device, tokenizer_path: str = "bert-base-uncased"):
        self.model = self._load_model(model_path)
        self.tokenizer = BertTokenizer(tokenizer_path)
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}
        self.device = device

    def _load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device)
        model = BertForSequenceClassification("bert-base-uncased", num_labels=3, id2label=self.label_map)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _preprocess_data(self, raw_data: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        preprocessor = TextPreprocessor(text_column="text")
        return preprocessor.process(raw_data)

    def _tokenize_data(self, text_series: pd.Series) -> Dict:
        return self.tokenizer(text_series.tolist(), padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    def predict(self, test_csv_path: str, batch_size: int = 32) -> pd.DataFrame:
        raw_data = pd.read_csv(test_csv_path)
        cleaned_data = self._preprocess_data(raw_data)

        encodings = self._tokenize_data(cleaned_data["cleaned_text"])

        dataset = SentimentDataset(encodings, device=self.device)
        loader = create_dataloader(dataset, batch_size)

        predictions = []
        with torch.no_grad():
            for batch in loader:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(batch_preds)

        cleaned_data["sentiment"] = [self.label_map[pred] for pred in predictions]

        return cleaned_data[["textID", "text", "cleaned_text", "predicted_sentiment"]]
