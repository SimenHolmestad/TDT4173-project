from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense,  SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

"""
--- Marks the beginning of where a new cell should start in a notebook
"""

# --- Define filepaths
data_file_path = 'Data/first_100000_processed_reviews2.csv'
model_file_path = 'Models/base_model_epochs5.h5'

# The maximum number of words to be used, only most frequent
vocabulary_size = 50000
# Max number of words in each review
max_review_size = 100

# One of the lines (5945667) apparently contains an EOF-character
df = pd.read_csv(data_file_path, error_bad_lines=False, engine="python")

print(df.head())
print("Length of corpus:", df.shape[0])

# --- Get corpus
corpus = df['training'].tolist()

# Tokenize the corpus with the `vocabulary_size` most frequent words
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(corpus)
total_tokens = len(tokenizer.word_index)
print('#unique tokens:', total_tokens)

# ---  Create training vectors with padding, where applicable, is at the end
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=max_review_size)
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
model.add(Embedding(vocabulary_size, 64, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])


# Define constants for training
epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size, validation_split=0.1)
model.save(model_file_path)

# plt.title('Loss')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

plt.title('Accuracy for LSTM model')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Plots/accuracy_lstm")

accr = model.evaluate(X_test, Y_test, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(
    accr[0], accr[1]))
