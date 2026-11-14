import json
import os
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# ==========================================
# 1. LOAD JSON
# ==========================================

with open("dataset/pavbhaji.json", "r", encoding="utf-8") as file:
    data = json.load(file)


# ==========================================
# 2. IMAGE -> LABEL MAPPING
# ==========================================

label_map = {}

for label in ["0", "1"]:

    folder = os.path.join("dataset/images", label)

    for filename in os.listdir(folder):
        label_map[filename] = int(label)


# ==========================================
# 3. CREATE DATASET
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

    # Remove direct target name
    text = text.replace("pavbhaji", "")
    text = text.replace("pav bhaji", "")

    texts.append(text)
    labels.append(label_map[filename])


# ==========================================
# 4. TRAIN TF-IDF
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="char",
    ngram_range=(3, 5),
    max_features=10000
)

X = vectorizer.fit_transform(texts)


# ==========================================
# 5. TRAIN FINAL MODEL
# ==========================================

model = LinearSVC()

model.fit(X, labels)


# ==========================================
# 6. PREDICTION FUNCTION
# ==========================================

def predict_pav_bhaji(text):

    text = text.lower()

    # Remove target name
    text = text.replace("pavbhaji", "")
    text = text.replace("pav bhaji", "")

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    confidence_score = model.decision_function(text_vector)[0]

    if prediction == 1:

        result = "Pav Bhaji"

    else:

        result = "Not Pav Bhaji"

    return result, confidence_score


# ==========================================
# 7. TEST PREDICTIONS
# ==========================================

print("\n========================================")
print("PAV BHAJI TEXT CLASSIFIER")
print("========================================")

while True:

    user_text = input(
        "\nEnter caption/hashtags "
        "(type 'exit' to stop):\n"
    )

    if user_text.lower() == "exit":
        break

    result, score = predict_pav_bhaji(user_text)

    print("\nPrediction:", result)
    print("Decision Score:", round(score, 4))