import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.calibration import LabelEncoder
import torch
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import keras


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
                      , header=None
                      , encoding_errors='replace'
                      , skiprows=lambda i: i>0 and random.random() > sample_size
                      )
  else:
    data = pd.read_csv(filename
                      , header=None
                      , encoding_errors='replace'
                      )
  
    
  data.columns = ['label', 'date_int', 'date', 'query', 'author', 'tweet']
  print(f"Data Info : {data.info()}")

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

def testdata_remove_neutral(filename='testdata.manual.2009.06.14.csv'):

      testdata = pd.read_csv(filename, header=None, encoding_errors='replace')
      
      # remove neutral labels
      testdata_no_neutral = testdata[testdata[0] != 2]

      return testdata_no_neutral

def evaluate_testdata(test_file, model_path, threshold=0.5):
    
    model = keras.models.load_model(model_path)

    X, y = preprocess_data(filename=test_file)
    print(X.shape, y.shape)
    
    # Predict the sentiment
    y_pred_prob = model.predict(X)
    print(y_pred_prob.shape)

    y_pred = np.where(y_pred_prob > threshold, 1, 0)

    return y_pred_prob, y_pred

def plot_classifier_metrics(testdata_no_neutral, y_pred, y_pred_prob):

  label_encoder = LabelEncoder()
  testdata_no_neutral['encoded_label'] = label_encoder.fit_transform(testdata_no_neutral[0])

  y_true = testdata_no_neutral['encoded_label'].values

  ###### confusion matrix ######
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

  # Set labels and title
  plt.xlabel('Predicted Labels')
  plt.ylabel('True Labels')
  plt.title('Confusion Matrix - Test data (no neutral)')
  plt.savefig('Test data - Confusion Matrix.png')
  plt.show()
  plt.close()

  ###### PR curve ######
  precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
  plt.figure(figsize=(8, 6))
  plt.plot(recall, precision, color='blue', linewidth=2, label='PR Curve')

  # Set labels and title
  plt.xlabel('Recall')
  plt.ylabel('Precision')
  plt.title('Test data - Precision-Recall Curve')

  # Add grid and legend
  plt.grid(True)
  plt.legend()
  plt.savefig('Test data - Precision-Recall Curve.png')
  plt.show()
  plt.close()

  ##### ROC-AUC Curve ######

  fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
  roc_auc = auc(fpr, tpr)
  plt.figure(figsize=(8, 6))
  plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)

  # Plot random guess line
  plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

  # Set labels and title
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Test data - ROC-AUC Curve')

  # Add legend
  plt.legend(loc='lower right')
  plt.savefig('Test data - ROC-AUC Curve.png')
  plt.show()
  plt.close()
