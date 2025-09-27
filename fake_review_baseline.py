""" 
Fake Product Review Monitoring Prototype (Baseline)
--------------------------------------------------
This script demonstrates a baseline fake product review detection pipeline:

- Generates a synthetic dataset with benign and fake reviews (with near-duplicate groups).
- Trains a TF-IDF + Logistic Regression classifier.
- Evaluates metrics (precision, recall, f1, confusion matrix, precision@k).
- Detects near-duplicate review groups among high-scoring suspicious reviews.
- Saves trained model artifacts for reuse.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import random
import os
import joblib
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

# 1) Generate synthetic dataset
def generate_benign_samples(n):
    templates = [
        "I bought this {product} last month and it's been great. The quality is excellent and it works as expected.",
        "Solid product — arrived on time, packaging was good. It does exactly what it says.",
        "Good value for money. I've used it for weeks and I'm satisfied with the build and performance.",
        "Decent product for the price. Customer service answered my questions quickly.",
        "Works well. I would recommend this to friends and family."
    ]
    products = ["phone case", "wireless mouse", "coffee maker", "laptop stand", "vacuum cleaner", "smartwatch"]
    texts = []
    for _ in range(n):
        t = random.choice(templates)
        p = random.choice(products)
        suffix = ""
        if random.random() < 0.2:
            suffix = " " + random.choice(["Five stars.", "Would buy again.", "No complaints."])
        texts.append(t.format(product=p) + suffix)
    return texts

def generate_fake_samples(n, templated_groups=5):
    group_templates = [
        "Amazing {product}! Best purchase ever. Highly recommend to everyone!",
        "This {product} is a lifesaver. Quality is superb and it arrived fast!",
        "I love this {product} — fantastic performance and excellent customer support.",
        "If you need a {product}, this is the one. Cannot believe how good it is!",
        "Top quality {product}. Five stars, will buy more."
    ]
    products = ["phone case", "wireless mouse", "coffee maker", "laptop stand", "vacuum cleaner", "smartwatch"]
    texts = []
    for _ in range(templated_groups):
        t = random.choice(group_templates)
        p = random.choice(products)
        base = t.format(product=p)
        group_size = max(3, int(n * 0.6 / templated_groups))
        for i in range(group_size):
            edit = base
            if random.random() < 0.5:
                edit += " " + random.choice(["Fast shipping!", "Highly recommended!", "Great seller!"])
            if random.random() < 0.3:
                edit = edit.replace("Best purchase ever", "Best buy ever")
            texts.append(edit)
    while len(texts) < n:
        p = random.choice(products)
        texts.append("Excellent " + p + ". 100% satisfied. Buy now!")
    return texts[:n]

# Dataset
N = 3500
n_fake = int(0.08 * N)
n_benign = N - n_fake

benign_texts = generate_benign_samples(n_benign)
fake_texts = generate_fake_samples(n_fake, templated_groups=12)

texts = benign_texts + fake_texts
labels = [0] * len(benign_texts) + [1] * len(fake_texts)

df = pd.DataFrame({"review_text": texts, "label": labels}).sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset size:", len(df), "Fake count:", df['label'].sum(), "Benign count:", len(df)-df['label'].sum())

# 2) Split and vectorize
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2)
X_train = vectorizer.fit_transform(train_df['review_text'])
X_test = vectorizer.transform(test_df['review_text'])
y_train = train_df['label'].values
y_test = test_df['label'].values

# 3) Train classifier
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# 4) Evaluation
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

print("\nClassification report (test set):")
print(classification_report(y_test, y_pred, digits=4))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Precision@k utility
def precision_at_k(y_true, scores, k):
    idx = np.argsort(scores)[::-1][:k]
    return y_true[idx].sum() / k

for k in [10, 25, 50, 100]:
    print(f"Precision@{k}: {precision_at_k(y_test, y_proba, k):.3f}")

# 5) Top tokens for fake class
feature_names = np.array(vectorizer.get_feature_names_out())
coef = clf.coef_[0]
top_pos_idx = np.argsort(coef)[-20:][::-1]
print("\nTop indicators for fake reviews:")
print(feature_names[top_pos_idx])

# 6) Near-duplicate detection among top-scoring suspicious reviews
top_indices = np.argsort(y_proba)[::-1][:30]
X_top = X_test[top_indices]
texts_top = test_df.reset_index(drop=True).loc[top_indices, 'review_text'].values
probas_top = y_proba[top_indices]

nn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(X_top.toarray())
distances, neighbors = nn.kneighbors(X_top.toarray())

near_dup_groups = []
visited = set()
for i in range(len(texts_top)):
    if i in visited: continue
    group = [i]
    for j, d in zip(neighbors[i], distances[i]):
        if d < 0.18:
            group.append(j)
    for g in group: visited.add(g)
    near_dup_groups.append(sorted(set(group)))

print("\nNear-duplicate groups among suspicious reviews:")
for g in near_dup_groups:
    if len(g) > 1:
        print([texts_top[i] for i in g])

# 7) Save model artifacts
os.makedirs("fake_review_model", exist_ok=True)
joblib.dump(clf, "fake_review_model/logreg_model.joblib")
joblib.dump(vectorizer, "fake_review_model/tfidf_vectorizer.joblib")
test_out = test_df.reset_index(drop=True).copy()
test_out['pred_proba'] = y_proba
test_out['pred_label'] = y_pred
test_out.to_csv("fake_review_model/test_predictions.csv", index=False)
print("Artifacts saved in ./fake_review_model/")
