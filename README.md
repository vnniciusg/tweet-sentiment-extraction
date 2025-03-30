# Tweet Sentiment Extraction

This project focuses on fine-tuning a BERT-based model for sentiment extraction from text. The pipeline includes data preprocessing, model training, and prediction.

## Overview

The goal is to classify the sentiment of text into three categories: **negative**, **neutral**, and **positive**. The project leverages the `bert-base-uncased` model from the Hugging Face Transformers library, fine-tuned on a custom dataset.

### Key Components

1. **Data Preprocessing**:

   - Text is cleaned using the [`TextPreprocessor`](data/preprocess.py) class, which removes noise such as URLs, special characters, and stopwords.
   - The cleaned text is tokenized using the BERT tokenizer with padding and truncation to a maximum sequence length of 128.

2. **Model Fine-Tuning**:

   - The BERT model is fine-tuned using the [`Trainer`](model/train.py) class.
   - Training includes:
     - Gradient clipping for stability.
     - Learning rate scheduling with warm-up steps.
     - Evaluation on a validation set after each epoch.

3. **Prediction**:
   - The fine-tuned model is used for inference with the [`SentimentPredictor`](model/predict.py) class.
   - Predictions are made on new data, and the sentiment is mapped to the corresponding label.

## Workflow

1. **Preprocessing**:

   - Raw text data is processed using the `TextPreprocessor` to generate a cleaned dataset.
   - The dataset is split into training and validation sets.

2. **Fine-Tuning**:

   - The BERT model is initialized with three output labels.
   - Training is performed using the AdamW optimizer and a linear learning rate scheduler.
   - The best model is saved based on validation loss.

3. **Prediction**:
   - The trained model is loaded, and predictions are made on test data.
   - Results include the original text, cleaned text, and predicted sentiment.

## How It Works

- **Fine-Tuning**:

  - The `Trainer` class handles the training loop, including loss computation, backpropagation, and evaluation.
  - Metrics such as accuracy and a classification report are logged for each epoch.

- **Inference**:
  - The `SentimentPredictor` class preprocesses input text, tokenizes it, and feeds it into the trained model for prediction.
  - The output logits are converted to sentiment labels using `argmax`.

## Dependencies

- `transformers` for BERT model and tokenizer.
- `torch` for model training and inference.
- `nltk` and `nlpprepkit` for text preprocessing.
- `pandas` for data manipulation.

## Results

TODO
