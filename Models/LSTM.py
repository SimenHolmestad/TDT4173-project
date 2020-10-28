from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
import tensorflow.keras.utils as ku 
import numpy as np 
import pandas as pd

df = pd.read_csv('Data/first_100000_processed_reviews.csv')

print(df.head())
print("Length of corpus:", df.shape[0])

df = df.dropna()
print(df.shape)

corpus = df['training'].tolist()

print(len(corpus))

Y = df[' label'].to_numpy()-1
print('Shape of label tensor:', len(Y))

# The maximum number of words to be used. (most frequent)
vocabulary_size = 50000
# Max number of words in each complaint.
max_review_size = 100
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print('Total words:', total_words)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=max_review_size)
print('Shape of data tensor:', X.shape)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(vocabulary_size, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(5, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 128

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()