import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nlpprepkit.pipeline import Pipeline
from nlpprepkit import functions as F


class PreprocessData:
    """A class to preprocess text data in a DataFrame."""

    def __init__(self, df: pd.DataFrame, text_column: str) -> None:
        """
        Preprocess the text data in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the text data.
            text_column (str): Name of the column containing the text data.

        Raises:
            ValueError: If the specified text column is not found in the DataFrame.
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

        self.text_column = text_column

        self.df = df.copy()
        self.df = self.df.dropna(subset=[self.text_column])

        self.pipeline = Pipeline()
        self._stop_words = self._load_stopwords()
        self._self_build_pipeline()

        self.lemmatizer = WordNetLemmatizer()

    def _self_build_pipeline(self):
        """Build a pipeline for text preprocessing."""
        self.pipeline.add_step(lambda text: text.lower())
        self.pipeline.add_step(F.expand_contractions)
        self.pipeline.add_step(F.remove_social_tags)
        self.pipeline.add_step(F.remove_urls)
        self.pipeline.add_step(F.remove_html_tags)
        self.pipeline.add_step(F.remove_emojis)
        self.pipeline.add_step(F.remove_special_characters)
        self.pipeline.add_step(F.remove_extra_whitespace)
        self.pipeline.add_step(F.remove_numbers)
        self.pipeline.add_step(self._remove_stopwords)

    def _load_stopwords(self) -> str:
        """
        Load the stopwords set from NLTK.

        Returns:
            set: A set of english stopwords.
        """
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        return set(stopwords.words("english"))

    def _remove_stopwords(self, text: str) -> str:
        """Helper function to remove stopwords from text"""
        word_tokens = word_tokenize(text)
        return " ".join([
            self.lemmatizer.lemmatize(word)
            for word in word_tokens
            if word not in self._stop_words and word.isalpha()
        ])

    def process(self) -> pd.DataFrame:
        """
        Preprocess the text data in the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with preprocessed text data.
        """
        self.df[f"cleaned_{self.text_column}"] = self.df[self.text_column].apply(self.pipeline)
        return self.df
