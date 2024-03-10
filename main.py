import pandas as pd
import random

# Load the corpus
corpus_file = "data/IBM_Debater_(R)_ArgsInASR_Findings-2020.v1/argumentative_sentences_in_spoken_language_with split.csv"
corpus = pd.read_csv(corpus_file)

# Separate entries with label '0' and '1'
label_0 = corpus[corpus['label'] == 0]
label_1 = corpus[corpus['label'] == 1]

# Sample 260 entries with label '0'
random.seed(42)  # For reproducibility
label_0_sampled = label_0.sample(n=260, random_state=42)

# Concatenate sampled entries with entries with label '1'
balanced_corpus = pd.concat([label_0_sampled, label_1])

# Shuffle the rows to mix the labels
balanced_corpus = balanced_corpus.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced corpus to a new CSV file
balanced_corpus.to_csv('data/balanced_corpus.csv', index=False)
