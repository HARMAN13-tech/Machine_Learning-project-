# 🍛 Pav Bhaji Classifier

A Machine Learning project that classifies text-based social media content to identify whether a post is related to **Pav Bhaji or Not Pav Bhaji**.

## 📌 Project Overview

The Pav Bhaji Classifier uses Natural Language Processing (NLP) and Machine Learning techniques to analyze captions and hashtags and predict whether the given content is related to Pav Bhaji.

The project was developed to explore text classification, data preprocessing, model training, and performance evaluation.

## 🚀 Features

- Text classification (Pav Bhaji / Not Pav Bhaji).
- Caption and hashtag analysis.
- Dataset preprocessing.
- Machine Learning model training.
- 5-Fold Cross Validation.
- Accuracy evaluation.

## 📊 Dataset Information

- Total JSON posts: 1,500
- Matched samples: 452
- Pav Bhaji samples: 183
- Not Pav Bhaji samples: 269

## 📈 Model Performance

5-Fold Cross Validation Results:

| Fold | Accuracy |
|---|---:|
| Fold 1 | 67.03% |
| Fold 2 | 68.13% |
| Fold 3 | 56.67% |
| Fold 4 | 72.22% |
| Fold 5 | 63.33% |
| **Mean Accuracy** | **65.48%** |

## 🛠️ Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- JSON Dataset
- Scikit-learn

## ⚙️ Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/HARMAN13-tech/Machine_Learning-project-.git
```

**2. Navigate to the project folder**

```bash
cd Machine_Learning-project-
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the environment (Windows)**

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Run the project**

```bash
python main.py
```

Enter a caption or hashtags when prompted to get the classification result.

## 📂 Project Structure

```text
Pavbhaji/
│
├── dataset/
│   └── pavbhaji.json
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 🎯 Learning Outcomes

- Understanding text classification.
- Working with JSON datasets.
- Applying Machine Learning concepts.
- Evaluating models using cross-validation.
- Building an interactive Python application.

## 👨‍💻 Author

**Harman**

GitHub: [HARMAN13-tech](https://github.com/HARMAN13-tech)

---

⭐ If you find this project interesting, consider giving it a star!
