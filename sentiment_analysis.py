import datetime
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils import preprocess_data
import argparse


class SentimentClassifier:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path

    def train(self, filename, epochs=3, batch_size=32, sample_size=None):
        X, y = preprocess_data(filename, sample_size)
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

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size
                       , validation_data=(X_val, y_val)
                       , callbacks=[tensorboard_callback])

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
        sentiment = self.model.predict(embeddings)[0][0]
        return sentiment > 0.5  # Assuming 0.5 as threshold
    
def main():
    parser = argparse.ArgumentParser(description="Sentiment Classifier")
    parser.add_argument("filename", help="Path to the input data file")
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
        classifier.train(args.filename, epochs=args.epochs, batch_size=args.batch_size, sample_size=args.sample_size)
    elif args.mode == "evaluate":
        # Evaluate the classifier
        if args.sentence:
            sentiment = classifier.evaluate(args.sentence)
            print("Sentiment:", sentiment)
        else:
            print("No sentence provided for evaluation.")



if __name__ == "__main__":
    main()