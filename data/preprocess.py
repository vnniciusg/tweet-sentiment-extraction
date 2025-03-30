from typing import Tuple

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nlpprepkit.pipeline import Pipeline
from nlpprepkit import functions as F
from sklearn.model_selection import train_test_split


class TextPreprocessor:
    """A class to preprocess text data in a DataFrame."""

    def __init__(self, text_column: str = "text", random_state: int = 42) -> None:
        self.text_column = text_column
        self.random_state = random_state
        self.lemmatizer = WordNetLemmatizer()
        self._verify_nltk_resources()
        self._initialize_pipeline()

    def _verify_nltk_resources(self):
        required = set(["stopwords"])
        for resource in required:
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                nltk.download(resource, quiet=True)

    def _initialize_pipeline(self):
        self.pipeline = Pipeline()
        self.stop_words = set(stopwords.words("english"))

        processing_steps = [
            lambda x: x.lower(),
            F.expand_contractions,
            F.remove_social_tags,
            F.remove_urls,
            F.remove_html_tags,
            F.remove_emojis,
            F.remove_special_characters,
            F.remove_extra_whitespace,
            F.remove_numbers,
            self._custom_lemma_stopword_removal,
        ]

        for step in processing_steps:
            self.pipeline.add_step(step)

    def _custom_lemma_stopword_removal(self, text: str) -> str:
        tokens = word_tokenize(text)
        return " ".join([self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and word.isalpha()])

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.text_column not in df.columns:
            raise ValueError(f"Column '{self.text_column}' not found in the df")

        df = df.dropna(subset=[self.text_column])
        df = df[df[self.text_column].str.strip().astype(bool)]

        return df.reset_index(drop=True)

    def split_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple:
        X, y = df["cleaned_text"], pd.factorize(df["sentiment"])[0]
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=self.random_state)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self.validate_data(df)
        df["cleaned_text"] = df[self.text_column].apply(self.pipeline)
        return df
