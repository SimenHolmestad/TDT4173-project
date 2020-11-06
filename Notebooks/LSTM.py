from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as ku 
import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Define filepaths
data_file_path = 'Data/first_100000_processed_reviews2.csv'
model_file_path = 'Models/base_model2_5epochs.h5'

# The maximum number of words to be used, only most frequent
vocabulary_size = 50000
# Max number of words in each review
max_review_size = 100

df = pd.read_csv(data_file_path, error_bad_lines=False, engine="python") # One of the lines (5945667) apparently contains an EOF-character

print(df.head())
print("Length of corpus:", df.shape[0])


# Get corpus
corpus = df['training'].tolist()

# Tokenize the corpus with the `vocabulary_size` most frequent words
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print('Total words:', total_words)
print('#unique tokens:', total_words-1)

# Create training vectors with padding, where applicable, is at the end 
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=max_review_size)
print('Shape of training data:', X.shape)

# Get labels
Y = df['label'].to_numpy()
print('Shape of label tensor:', len(Y))

# Split features and labels into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print("Shape of training features: ", X_train.shape)
print("Shape of training labels: ", Y_train.shape)
print("Shape of test features: ", X_test.shape)
print("Shape of test labels: ", Y_test.shape)

# Define model
model = Sequential()
model.add(Embedding(vocabulary_size, 64, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(32, dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

try:
    model = tf.keras.models.load_model(model_file_path)
except:
  # Define constants for training 
  epochs = 5
  batch_size = 128
  
  history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)
  model.save(model_file_path)

  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show()

  plt.title('Accuracy')
  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.legend()
  plt.show()

accr = model.evaluate(X_test,Y_test, verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))