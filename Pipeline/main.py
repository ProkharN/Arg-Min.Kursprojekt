import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("data/balanced_corpus.csv")

# Preprocess the data
X = data["sentence"]
y = data["label"]

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

# Function to generate text from 20 random sentences
def generate_text(corpus, n=20):
    random_sentences = random.sample(list(corpus['sentence']), n)
    ids = ','.join(str(corpus[corpus['sentence'] == sent].index[0]) for sent in random_sentences)
    text = '. '.join(random_sentences)
    return text, ids

# Generate text and count argumentative sentences for 1000 rows
data_rows = []
for _ in range(1000):
    text, ids = generate_text(data)
    real_argumentative = sum(data.loc[int(id), 'label'] == 1 for id in ids.split(','))
    model_argumentative = sum(clf.predict(vectorizer.transform([sentence]))[0] == 1 for sentence in re.split(r'(?<=[.!?])\s+', text.lower().strip()))
    data_rows.append({
        'ids': ids,
        'texts': text,
        'real number of argumentative sentences': real_argumentative,
        'number of argumentative sentences assigned by the model': model_argumentative
    })

# Create DataFrame from generated data
new_corpus = pd.DataFrame(data_rows)

# Calculate statistics
same_count = new_corpus[new_corpus['real number of argumentative sentences'] == new_corpus['number of argumentative sentences assigned by the model']].shape[0] / new_corpus.shape[0] * 100
model_less_count = new_corpus[new_corpus['real number of argumentative sentences'] > new_corpus['number of argumentative sentences assigned by the model']].shape[0] / new_corpus.shape[0] * 100
model_more_count = new_corpus[new_corpus['real number of argumentative sentences'] < new_corpus['number of argumentative sentences assigned by the model']].shape[0] / new_corpus.shape[0] * 100

# Print F1 score and other statistics
print("F1 Score:", classification_report(new_corpus['real number of argumentative sentences'], new_corpus['number of argumentative sentences assigned by the model']))

# Plot the bar chart
categories = ['Same', 'Model Less', 'Model More']
percentages = [same_count, model_less_count, model_more_count]
colors = ['#50C878', '#E6E6FA', '#FFDAB9']

plt.bar(categories, percentages, color=colors)
plt.xlabel('Category')
plt.ylabel('Percentage')
plt.title('Comparison of Real vs. Model Assigned Argumentative Sentences')
plt.show()
