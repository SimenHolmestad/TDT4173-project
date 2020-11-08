import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('Data/output.csv')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='utf-8')

for i, text in enumerate(df.training):
    if type(text) != str:
        print(i)
        df.drop(index=i, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.training, df.label, random_state=0)
print("Vectorizing...")
training_features = tfidf.fit_transform(X_train)
test_features = tfidf.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=20)

print("Fitting...")
classifier.fit(training_features, y_train)
predicted = classifier.predict(test_features)
accuracy = np.mean(predicted == y_test)
print("Accuracy was: " + str(accuracy))
