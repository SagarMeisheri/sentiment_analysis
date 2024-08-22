## sentiment_analysis

This repository implements a sentiment classifier using TensorFlow and Sentence Transformers.

## Requirements

- Python 3.6 or later
- TensorFlow (Installation)
- Sentence Transformers (Installation)
- scikit-learn (Installation)

## Installation

Create a virtual environment (recommended) to isolate dependencies:


```python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat
```
Install required packages:


```
pip install -r requirements.txt
```
## Training the Model

Prepare your training data in a CSV format with two columns: text (sentence) and label (sentiment, e.g., positive, negative).

Train the model using the following command:


```
python sentiment_analysis.py \
--mode train \
--train_file training-sample.csv \
--model_path sentiment_model.keras \
--epochs 10 \
--batch_size 64 \
--sample_size 0.1  # Optional: Use a portion of the training data
```

- train_file: Path to your training data CSV file.
- model_path: Path to save the trained model (default: sentiment_model.keras).
- epochs: Number of training epochs (default: 10).
- batch_size: Batch size for training (default: 64).
- sample_size: Optional. Percentage of training data to use (default: all data).

## Evaluating the Model

- After training, you can evaluate the model's performance on a separate test set or individual sentences.

Evaluate on a new sentence:

```
python sentiment_analysis.py \
--mode evaluate \
--model_path sentiment_model.keras \
--sentence "This movie was a masterpiece!"
```
- sentence: The sentence to classify.
- The output will be a JSON string indicating the predicted sentiment and score (e.g., {"sentiment": "positive", "sentiment_score": "0.98"}).

# REST-API: Google Cloud Function for Sentiment Analysis

This repository contains a Google Cloud Function that analyzes the sentiment of a given sentence. It utilizes a pre-trained sentence transformer model (all-MiniLM-L6-v2) to encode the sentence into a numerical representation and a custom Keras sentiment model to predict the sentiment (positive or negative) based on the embedding.

### Features

- Leverages powerful pre-trained sentence transformer model for accurate sentence encoding.
- Utilizes a custom Keras sentiment model for sentiment prediction.
- Includes thorough logging for monitoring and debugging.
- Handles potential errors gracefully by returning informative error messages.

### Requirements

- Google Cloud project with Cloud Functions and Cloud Storage enabled.
- A pre-trained Keras sentiment model stored in a Cloud Storage bucket named sentiment-model-tf 

## Deployment

### Create a Cloud Storage bucket:
- If you haven't already, create a Cloud Storage bucket named sentiment-model-tf to store your pre-trained Keras sentiment model.

### Upload the sentiment model:
- Upload the v2_sentiment_model.keras file to the sentiment-model-tf bucket.

### Deploy the Cloud Function:
- You can deploy the Cloud Function in several ways, such as using the gcloud command-line tool or the Cloud Console. Here's an example using gcloud:

```
gcloud functions deploy evaluate \
  --runtime python39 \
  --trigger-http \
  --allow-unauthenticated
```
## Usage

### HTTP request:

This Cloud Function expects an HTTP POST request with the following JSON payload in the request body:

```
JSON
{
  "sentence": "Your sentence to analyze here"
}
```

### Example cURL command:

```
curl -X POST -H "Content-Type: application/json" -d '{"sentence": "This is a good book to read during travel."}' https://us-central1-jaldi-bol-ai.cloudfunctions.net/evaluate
```

### Response:

- The function will return a JSON response containing the predicted sentiment score (between 0.0 and 1.0) and the overall sentiment (positive or negative).

```
JSON
{
  "sentiment_score": 0.87,
  "sentiment": "positive"
}
```

## Logging

- Cloud logging is enabled for this Cloud Function, providing detailed information about the execution process, including model loading times and sentiment predictions. You can access these logs in the Google Cloud Console to monitor and debug the function.

## Error Handling

- The function includes error handling to catch exceptions and return informative error messages in the response, allowing you to troubleshoot issues more effectively.

## Security Considerations

- This example uses --allow-unauthenticated for demonstration purposes. In production environments, consider implementing authentication and authorization mechanisms to control access to your Cloud Function.

## Contributing

- We encourage contributions to this project. Feel free to submit pull requests to improve the code or documentation.