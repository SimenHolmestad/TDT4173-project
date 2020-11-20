"""TDT4173 Machine learning model training

This script takes an input file containing processed review data,
trains and saves an LSTM-model, the corresponding tokenizer and stores
plots related to the accuracy and loss of the model.

Run script with:

python3 lstm_model_training.py <file-path-to-data-file> <directory-for-output-files>
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# The maximum number of words to be used, only most frequent
VOCABULARY_SIZE = 50000
# Max number of words in each review
MAX_REVIEW_SIZE = 100


def create_and_save_tokenizer(tokenizer_file_path, corpus):
    # Tokenize the corpus with the `vocabulary_size` most frequent words
    tokenizer = Tokenizer(num_words=VOCABULARY_SIZE)
    tokenizer.fit_on_texts(corpus)

    # Save tokenizer to file
    tokenizer_json = tokenizer.to_json()
    text_file = open(tokenizer_file_path, "w")
    text_file.write(tokenizer_json)
    text_file.close()

    return tokenizer


def create_balanced_dataset(filename):
    """Loading of the dataframe and balancing the data

    We want equal amounts of reviews for all ratings, so we set the
    `cap_number` as the minimum number of reviews for one rating.
    (this is the number of revies rated 2 stars)

    """
    df = pd.read_csv(filename, error_bad_lines=False, engine="python")

    unique, counts = np.unique(df["label"], return_counts=True)
    cap_number = min(counts)

    # Create one dataframe for reviews with each rating and sample `cap_number` rows for each.
    dfs = []
    for x in range(5):
        # x_df = df[df["label"]==x]
        number_of_rows = len(df[df["label"] == x].index)
        n = min(cap_number, number_of_rows)
        dfs.append(df[df["label"] == x].sample(n=n))  # Sample chooses random rows

    # Return the concatinated dataframes in randomised order
    return pd.concat(dfs).sample(frac=1.0)


def main():
    if len(sys.argv) < 3:
        print("You have to specify a filename for the original file and a filename for the output file")
        print("You should run script as:")
        print("python3 lstm_model_training.py <file-path-to-data-file> <directory-for-output-files>")
        sys.exit()

    data_filepath = sys.argv[1]
    output_directory = sys.argv[1]
    model_base_name = 'balanced_model_2epochs'
    model_file_path = model_base_name + '.h5'
    tokenizer_file_path = os.path.join(output_directory, model_base_name + '_tokenizer.json')

    print("Creating balanced dataset")
    df = create_balanced_dataset(data_filepath)

    print(df.head())
    print("Length of corpus:", df.shape[0])

    # --- Get corpus
    corpus = df['training'].tolist()

    # Tokenize the corpus with the `vocabulary_size` most frequent words
    print("Creating and saving tokenizer to " + tokenizer_file_path)
    tokenizer = create_and_save_tokenizer(tokenizer_file_path, corpus)
    total_tokens = len(tokenizer.word_index)
    print('#unique tokens:', total_tokens)

    # ---  Create training vectors with padding, where applicable, is at the end
    X = tokenizer.texts_to_sequences(corpus)
    X = pad_sequences(X, maxlen=MAX_REVIEW_SIZE)
    print('Shape of training data:', X.shape)

    # Get labels
    Y = df['label'].to_numpy()
    print('Shape of label tensor:', len(Y))

    # --- Split features and labels into training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.10, random_state=42)
    print("Shape of training features: ", X_train.shape)
    print("Shape of training labels: ", Y_train.shape)
    print("Shape of test features: ", X_test.shape)
    print("Shape of test labels: ", Y_test.shape)

    # --- Define model
    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, 64, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(32))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Define constants for training
    epochs = 2
    batch_size = 64

    history = model.fit(X_train, Y_train, epochs=epochs,
                        batch_size=batch_size, validation_split=0.1)
    model.save(os.path.join(output_directory, model_file_path))

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_directory, "loss_lstm.png"))
    plt.clf()

    plt.title('Accuracy for LSTM model')
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(output_directory, "accuracy_lstm.png"))

    accr = model.evaluate(X_test, Y_test, verbose=0)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
        accr[0], accr[1]))


if __name__ == "__main__":
    main()
