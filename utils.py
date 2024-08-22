import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
import torch
import random


# Function to remove Twitter handles
def remove_handles(tweet):
    return re.sub(r'@\w+', '', tweet)


# Function to remove url
def remove_url(tweet):
  url_pattern = r"(http|https):\/\/\S+"
  return re.sub(url_pattern, "", tweet)


def encode_tweets_in_batches(tweets, model, device, batch_size=32):
  """Encodes tweets in batches using the given Sentence Transformer model.

  Args:
    tweets: A list of tweets.
    model: The Sentence Transformer model.
    batch_size: The size of each batch.

  Returns:
    A numpy array of embeddings.
  """

  embeddings = []
  for start in range(0, len(tweets), batch_size):
    # batch = pd.DataFrame(tweets[start:start + batch_size]).reset_index(drop=True).values
    batch = np.array(tweets[start:start + batch_size])
    batch_embeddings = model.encode(batch, device=device, normalize_embeddings=True)
    embeddings.extend(batch_embeddings)
  
  return np.array(embeddings)


def preprocess_data(filename="training-sample.csv", sample_size=None, batch_size=32):

  if sample_size:
    data = pd.read_csv(filename
                      # , header=None
                      , encoding_errors='ignore'
                      , skiprows=lambda i: i>0 and random.random() > sample_size
                      )
  else:
    data = pd.read_csv(filename
                      # , header=None
                      , encoding_errors='ignore'
                      )
  
  print(f"Data Info : {data.info()}")
    
  data.columns = ['label', 'date_int', 'date', 'query', 'author', 'tweet']

  # clean tweets
  print("process tweets")
  data['tweet_processed'] = data['tweet'].apply(remove_handles)
  data['tweet_processed'] = data['tweet_processed'].apply(remove_url)
  print(data['tweet_processed'].head())
  
  # print(data[data.label==2]['tweet'].sample(frac=1).values)

  print("get sentence embeddings in batch")
  sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
  tweets = data['tweet_processed']
  device = "cuda" if torch.cuda.is_available() else "cpu"
  embeddings = encode_tweets_in_batches(tweets, model=sentence_transformer, device=device, batch_size=batch_size)

  # Encode Labels
  label_encoder = LabelEncoder()
  data['encoded_label'] = label_encoder.fit_transform(data['label'])

  X = embeddings
  y = data['encoded_label'].values

  return X, y