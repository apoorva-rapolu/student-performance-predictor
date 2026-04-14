# 🎓 Student Performance Predictor

A machine learning web app that predicts a student's grade (A/B/C/D/F) based on demographic background and exam scores — with a live model comparison dashboard.

---

## Live Demo
[check live website](https://student-performance-predictor-qfr5shjjbeklzmaun7a97q.streamlit.app/)
---

## Features

- Predicts grade from gender, parental education, lunch type, test prep, and math/reading/writing scores
- Compares **4 models** side-by-side: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- Shows subject-level insights (strongest/weakest subject, test prep impact)
- User-selectable model at prediction time

---

## Model Performance

| Model | Test Accuracy |
|---|---|
| Gradient Boosting | ~93% |
| Random Forest | ~92% |
| Decision Tree | ~88% |
| Logistic Regression | ~85% |

---

## Run Locally

```bash
git clone https://github.com/apoorva-rapolu/student-performance-predictor
cd student-performance-predictor
pip install -r requirements.txt
streamlit run student_app.py
```

---

## Dataset

[Students Performance in Exams — Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)  
1000 samples · 8 features · 5-class grade prediction (A/B/C/D/F)

---

## Tech Stack

`Python` `Streamlit` `Scikit-Learn` `Pandas` `NumPy`
