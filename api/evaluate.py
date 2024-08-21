from flask import jsonify, make_response
from google.cloud import storage
from sentence_transformers import SentenceTransformer
import functions_framework
import tensorflow as tf
import requests
import keras
import time
import flask
import os
from google.cloud import logging as cloudlogging
import logging
import re

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

lg_client = cloudlogging.Client()
lg_handler = lg_client.get_default_handler()
cloud_logger = logging.getLogger("cloudLogger")
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(lg_handler)


start_time = time.time()
# Load the sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
cloud_logger.info(f"Loaded sentence model in {time.time() - start_time:.2f} seconds")

def remove_handles(sentence):
    return re.sub(r'@\w+', '', sentence)

def remove_url(sentence):
    url_pattern = r"(http|https):\/\/\S+"
    return re.sub(url_pattern, "", sentence)

def load_sentiment_model():
    bucket_name = "sentiment-model-tf"
    client = storage.Client()

    model_name = 'v2_sentiment_model.keras'
    blob = client.bucket(bucket_name).blob(model_name)
    blob.download_to_filename(model_name)

    return keras.models.load_model(model_name)


@functions_framework.http
def evaluate(request):

    cloud_logger.info("generate sentiment score")
    
    try:
        data = request.get_json()
        sentence = data.get('sentence')
        cloud_logger.info(f"original sentence: {sentence}")

        if not sentence:
            return jsonify({'error': 'Sentence is required'}), 400
        
        # process sentence
        sentence = remove_handles(sentence)
        sentence = remove_url(sentence)
        cloud_logger.info(f"processed sentence: {sentence}")

        start_time = time.time()
        cloud_logger.info("generate sentence embedding")
        device = "cpu"
        # Use the sentence transformer to encode the sentence into a numerical representation
        sentence_embedding = sentence_model.encode([sentence], device=device, normalize_embeddings=True)
        cloud_logger.info(f"Generated sentence_embedding in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        sentiment_model = load_sentiment_model()
        cloud_logger.info(f"load_sentiment_model done in {time.time() - start_time:.2f} seconds")

        start_time = time.time()
        prediction = sentiment_model.predict(sentence_embedding)
        cloud_logger.info(f"Made predictions in {time.time() - start_time:.2f} seconds")
        sentiment_score = prediction[0][0]
        
        sentiment = 'positive' if sentiment_score >= 0.5 else 'negative'
        cloud_logger.info(f"sentiment_score: {sentiment_score}, sentiment:{sentiment}")

        return jsonify({'sentiment_score': str(sentiment_score), 'sentiment':sentiment})

    except Exception as e:
        # Handle errors gracefully
        # print(f"Error processing request: {e}")
        
        cloud_logger.error(f"An error occurred: {e}")
        
        response = make_response(jsonify({'error': str(e)}), 400)  # Send error response with status code 400
        return response