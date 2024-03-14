import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load the data
data = pd.read_csv("data/balanced_corpus.csv")

# Preprocess the data
X = data["sentence"]
y = data["label"]  # Assuming label column contains integers 1 and 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Support Vector Machine classifier
clf = SVC(kernel='linear')
clf.fit(X_train_vectorized, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_vectorized)

# Evaluate the model
print(classification_report(y_test, y_pred))


# Save the results to a new CSV file
test_results = pd.DataFrame({'sentence': X_test, 'label': y_test, 'Model Label': y_pred})
test_results.to_csv('data/testing_results.csv', index=False)
