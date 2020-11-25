from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
try:
    import cPickle as pickle
except:
    import pickle


def create_balanced_dataset(filename):
    df = pd.read_csv(filename, error_bad_lines=False, engine="python")

    unique, counts = np.unique(df["label"], return_counts=True)
    cap_number = min(counts)

    # Create one dataframe for reviews with each rating and sample `cap_number` rows for each.
    dfs = []
    for x in range(5):
        # x_df = df[df["label"]==x]
        number_of_rows = len(df[df["label"] == x].index)
        n = min(cap_number, number_of_rows)
        # Sample chooses random rows
        dfs.append(df[df["label"] == x].sample(n=n))

    # Return the concatinated dataframes in randomised order
    return pd.concat(dfs).sample(frac=1)


path = Path(__file__).parent / "../Data/output_final2.csv"

print("Loading dataset...")
df = create_balanced_dataset(path.absolute())
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, encoding='utf-8')

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df.training, df.label, random_state=0, test_size=0.05)
print("Vectorizing...")
training_features = tfidf.fit_transform(X_train)
with open("Source/gcloud_function_2/TFIDFvectorizer.bin", "wb") as file:
    pickle.dump(tfidf, file)
test_features = tfidf.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=20)

print("Fitting...")
classifier.fit(training_features, y_train)

with open("Source/gcloud_function_2/kNNClassifier.bin", "wb") as file:
    pickle.dump(classifier, file)

print("Predicting...")
accuracy = classifier.score(test_features, y_test)
print("Accuracy was: " + str(accuracy))
