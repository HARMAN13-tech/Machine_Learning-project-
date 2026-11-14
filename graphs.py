import os
import matplotlib.pyplot as plt


# ==========================================
# CREATE RESULTS FOLDER
# ==========================================

os.makedirs("results", exist_ok=True)


# ==========================================
# 1. CLASS DISTRIBUTION
# ==========================================

classes = ["Not Pav Bhaji", "Pav Bhaji"]
counts = [269, 183]

plt.figure(figsize=(7, 5))

plt.bar(classes, counts)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")

for i, count in enumerate(counts):
    plt.text(i, count + 5, str(count), ha="center")

plt.tight_layout()

plt.savefig(
    "results/class_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 2. MODEL COMPARISON
# ==========================================

models = [
    "Logistic Regression\nWord TF-IDF",
    "Naive Bayes\nWord TF-IDF",
    "Linear SVM\nWord TF-IDF",
    "Linear SVM\nCharacter TF-IDF"
]

accuracies = [
    62.59,
    64.59,
    65.69,
    66.35
]

plt.figure(figsize=(9, 5))

plt.bar(models, accuracies)

plt.title("Model Comparison")
plt.xlabel("Model")
plt.ylabel("Mean 5-Fold Accuracy (%)")

plt.ylim(0, 100)

for i, accuracy in enumerate(accuracies):
    plt.text(
        i,
        accuracy + 1,
        f"{accuracy:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=300
)

plt.close()


# ==========================================
# 3. CONFUSION MATRIX
# ==========================================

matrix = [
    [35, 19],
    [17, 20]
]

plt.figure(figsize=(6, 5))

plt.imshow(matrix)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Not Pav Bhaji", "Pav Bhaji"]
)

plt.yticks(
    [0, 1],
    ["Not Pav Bhaji", "Pav Bhaji"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            matrix[i][j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# DONE
# ==========================================

print("========================================")
print("GRAPHS CREATED SUCCESSFULLY")
print("========================================")

print("\nFiles created:")

print("results/class_distribution.png")
print("results/model_comparison.png")
print("results/confusion_matrix.png")