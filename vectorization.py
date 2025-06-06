import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus (collection of documents)
corpus = [
    "Machine learning is fascinating and powerful.",
    "Deep learning is a subset of machine learning.",

]

# Step 1: Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Step 2: Fit and transform the corpus
X = vectorizer.fit_transform(corpus)

# Step 3: Extract feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Step 4: Convert sparse matrix to dense representation
X_dense = X.toarray()

# Display TF-IDF scores as a DataFrame
df = pd.DataFrame(X_dense, columns=feature_names)
print("TF-IDF Matrix:")
print(df)

# Visualizing TF-IDF Scores
plt.figure(figsize=(12, 6))
plt.imshow(X_dense, cmap='coolwarm', aspect='auto')
plt.colorbar(label='TF-IDF Score')
plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
plt.yticks(ticks=np.arange(len(corpus)), labels=[f'Doc {i+1}' for i in range(len(corpus))])
plt.title("TF-IDF Heatmap of Terms in the Corpus")
plt.xlabel("Terms")
plt.ylabel("Documents")
plt.show()

# Explanation:
print("\nExplanation:")
print("TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate how important a word is to a document in a collection (corpus).")
print("- Term Frequency (TF): Measures how often a word appears in a document.")
print("- Inverse Document Frequency (IDF): Measures how important or unique a word is across the entire corpus.")
print("- TF-IDF Score: Computed as TF * IDF, giving higher scores to words that are more informative.")