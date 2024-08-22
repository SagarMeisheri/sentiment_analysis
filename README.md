## sentiment_analysis

This repository implements a sentiment classifier using TensorFlow and Sentence Transformers.

## Requirements

- Python 3.6 or later
- TensorFlow (Installation)
- Sentence Transformers (Installation)
- scikit-learn (Installation)

## Installation

Create a virtual environment (recommended) to isolate dependencies:

Bash
```python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat
```
Install required packages:

Bash
```pip install -r requirements.txt
```
Training the Model

Prepare your training data in a CSV format with two columns: text (sentence) and label (sentiment, e.g., positive, negative).

Train the model using the following command:

Bash
```python sentiment_analysis.py \
--mode train \
--train_file training-sample.csv \
--model_path sentiment_model.keras \
--epochs 10 \
--batch_size 64 \
--sample_size 0.1  # Optional: Use a portion of the training data
```
--train_file: Path to your training data CSV file.
--model_path: Path to save the trained model (default: sentiment_model.keras).
--epochs: Number of training epochs (default: 10).
--batch_size: Batch size for training (default: 64).
--sample_size: Optional. Percentage of training data to use (default: all data).
Evaluating the Model

After training, you can evaluate the model's performance on a separate test set or individual sentences.

Evaluate on a new sentence:

Bash
```python sentiment_analysis.py \
--mode evaluate \
--model_path sentiment_model.keras \
--sentence "This movie was a masterpiece!"
```
--sentence: The sentence to classify.
The output will be a JSON string indicating the predicted sentiment and score (e.g., {"sentiment": "positive", "sentiment_score": "0.98"}).