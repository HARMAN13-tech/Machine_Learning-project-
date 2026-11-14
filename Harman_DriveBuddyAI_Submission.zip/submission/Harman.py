import json
import os
import re
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ==========================================
# 1. LOAD DATA
# ==========================================

JSON_FILE = "dataset/pavbhaji.json"
IMAGE_FOLDER = "dataset/images"

with open(JSON_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

print("Total JSON posts:", len(data))


# ==========================================
# 2. CREATE IMAGE LABEL MAP
# ==========================================

label_map = {}

for label in ["0", "1"]:
    folder = os.path.join(IMAGE_FOLDER, label)

    for filename in os.listdir(folder):
        label_map[filename] = int(label)


# ==========================================
# 3. EXTRACT TEXT AND LABELS
# ==========================================

texts = []
labels = []

for post in data:

    url = post.get("display_url", "")
    filename = os.path.basename(urlparse(url).path)

    if filename not in label_map:
        continue

    caption = ""

    caption_data = post.get("edge_media_to_caption", {})

    if isinstance(caption_data, dict):
        edges = caption_data.get("edges", [])

        if edges:
            caption = edges[0].get("node", {}).get("text", "")

    tags = post.get("tags", [])

    if isinstance(tags, list):
        tags_text = " ".join(tags)
    else:
        tags_text = ""

    text = caption + " " + tags_text

    text = text.lower()

    # Remove target name to avoid direct leakage
    text = text.replace("pavbhaji", "")
    text = text.replace("pav bhaji", "")

    # Remove excessive punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    texts.append(text)
    labels.append(label_map[filename])


texts = np.array(texts)
labels = np.array(labels)

print("Matched samples:", len(texts))
print("Not Pav Bhaji:", sum(labels == 0))
print("Pav Bhaji:", sum(labels == 1))


# ==========================================
# 4. BUILD FINAL MODEL
# ==========================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=10000
        )
    ),
    (
        "classifier",
        LinearSVC()
    )
])


# ==========================================
# 5. 5-FOLD CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    texts,
    labels,
    cv=cv,
    scoring="accuracy"
)

print("\n========================================")
print("5-FOLD CROSS VALIDATION")
print("========================================")

for i, score in enumerate(scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"Mean Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")


# ==========================================
# 6. FINAL TRAIN/TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ==========================================
# 7. TEST RESULTS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Not Pav Bhaji", "Pav Bhaji"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# ==========================================
# 8. TRAIN FINAL MODEL ON ALL DATA
# ==========================================

model.fit(texts, labels)


# ==========================================
# 9. PREDICTION FUNCTION
# ==========================================

def predict_text(text):

    text = text.lower()

    text = text.replace("pavbhaji", "")
    text = text.replace("pav bhaji", "")

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    prediction = model.predict([text])[0]

    score = model.decision_function([text])[0]

    if prediction == 1:
        result = "Pav Bhaji"
    else:
        result = "Not Pav Bhaji"

    return result, score


# ==========================================
# 10. EXAMPLE PREDICTIONS
# ==========================================

examples = [
    "buttery toasted bread with spicy vegetable masala and lots of butter",
    "delicious cheese pizza with fresh vegetables",
    "crispy samosa with mint chutney"
]

print("\n========================================")
print("EXAMPLE PREDICTIONS")
print("========================================")

for example in examples:

    result, score = predict_text(example)

    print("\nText:", example)
    print("Prediction:", result)
    print(f"Decision Score: {score:.4f}")


print("\n========================================")
print("MODEL EXECUTION COMPLETED")
print("========================================")