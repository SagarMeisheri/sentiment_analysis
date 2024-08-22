import datetime
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import preprocess_data
import argparse
import json

class SentimentClassifier:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def train(self, train_file, epochs=3, batch_size=32, sample_size=0.2):
        X, y = preprocess_data(filename=train_file
                  , sample_size=sample_size
                  , batch_size=batch_size)
        print(X.shape, y.shape)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=36)
        print(X_train.shape, y_train.shape)
        print(X_val.shape, y_val.shape)

        # Build the model
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size
                       , validation_data=(X_val, y_val)
                       , callbacks=[tensorboard_callback]
                       )

        # Save the model
        self.model.save(self.model_path)

    def evaluate(self, sentence):
        if self.model is None:
            # Load the saved model
            self.model = tf.keras.models.load_model(self.model_path)

        # Encode and preprocess the sentence
        model_embed = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model_embed.encode([sentence], normalize_embeddings=True)

        # Predict the sentiment
        sentiment_score = self.model.predict(embeddings)[0][0]
        sentiment = 'positive' if sentiment_score >= 0.5 else 'negative'

        return json.dumps({'sentiment':sentiment, 'sentiment_score': str(sentiment_score)})
    
def main():
    parser = argparse.ArgumentParser(description="Sentiment Classifier")
    parser.add_argument("--train_file", help="Path to the train data file")
    parser.add_argument("--model_path", default="sentiment_model.keras", help="Path to save/load the trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--sample_size", type=float, default=None, help="Percent of training samples to use")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train", help="Mode: train or evaluate")
    parser.add_argument("--sentence", default=None, help="Sentence to evaluate (optional)")

    args = parser.parse_args()

    # Create an instance of SentimentClassifier
    classifier = SentimentClassifier(args.model_path)

    if args.mode == "train":
        # Train the classifier
        classifier.train(args.train_file, epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample_size)
    elif args.mode == "evaluate":
        # Evaluate the classifier
        if args.sentence:
            sentiment = classifier.evaluate(args.sentence)
            print("Sentiment:", sentiment)
        else:
            print("No sentence provided for evaluation.")



if __name__ == "__main__":
    main()
    
'''
train example:

python sentiment_analysis.py \
--mode train \
--train_file training-sample.csv \
--model_path sentiment_model.keras \
--epochs 10 \
--batch_size 64 \
--sample_size 0.1

evaluate example:

python sentiment_analysis.py \
--mode evaluate \
--model_path sentiment_model.keras \
--sentence "today is a sunny day" 

'''