import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report  # Add this import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd

# Load your trained model here
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

# Assuming model is already defined and trained
model = clf


# Load the original corpus
corpus_file = 'data/balanced_corpus.csv'
corpus = pd.read_csv(corpus_file)

# Function to generate text from 10 random sentences


# def generate_text(corpus, n=20):
#     random_sentences = random.sample(list(corpus['sentence']), n)
#     ids = ','.join(str(corpus[corpus['sentence'] == sent].index[0]) for sent in random_sentences)
#     text = '. '.join(random_sentences)
#     return text, ids




def generate_text(corpus, n=20):
    # Filter the corpus to include only sentences with a label of "0"
    non_argumentative_corpus = corpus[corpus['label'] == 1]

    random_sentences = random.sample(list(non_argumentative_corpus['sentence']), n)
    ids = ','.join(str(non_argumentative_corpus[non_argumentative_corpus['sentence'] == sent].index[0]) for sent in
                   random_sentences)
    text = '. '.join(random_sentences)

    return text, ids


# Generate text and count argumentative sentences
def count_real_argumentative_sentences(ids, corpus):
    real_argumentative = 0
    for id in ids.split(','):
        id = int(id)
        label = corpus.loc[id, 'label']
        if label == 1:
            real_argumentative += 1
    return real_argumentative

def count_model_argumentative_sentences(text, clf):
    sentences = list(re.split(r'(?<=[.!?])\s+', text))# Ensure sentences is a list
    model_argumentative = 0
    for sentence in sentences:
        # Preprocess the sentence (e.g., convert to lowercase, remove punctuation)
        sentence = sentence.lower().strip()
        sentence_vectorized = vectorizer.transform([sentence])  # Assuming vectorizer is defined earlier
        prediction = clf.predict(sentence_vectorized)[0]
        # If the sentence is predicted as argumentative, increment the count
        if prediction == 1:
            model_argumentative += 1
    return model_argumentative



# Generate text and count argumentative sentences for 10 rows
data_rows = []
for _ in range(10):
    text, ids = generate_text(corpus)
    real_argumentative = count_real_argumentative_sentences(ids, corpus)  # Corrected variable name
    # Assuming model is already defined and trained
    model_argumentative = count_model_argumentative_sentences(text, model)  # Corrected variable name
    data_rows.append({
        'ids': ids,
        'texts': text,
        'real number ': real_argumentative,
        'assigned by the model': model_argumentative
    })
# Create a DataFrame from the generated data
new_corpus = pd.DataFrame(data_rows)

# Save the DataFrame to a new CSV file
new_corpus.to_csv('data/generated_corpus.csv', index=False)





