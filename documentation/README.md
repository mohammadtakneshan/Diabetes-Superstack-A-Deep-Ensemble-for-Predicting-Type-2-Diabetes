# Diabetes Superstack: A Deep Ensemble for Predicting Type 2 Diabetes

## 📌 Project Overview
This project presents a carefully engineered machine learning system that predicts whether a person is likely to develop type 2 diabetes based on key clinical variables. The system combines classical machine learning, deep learning, domain-specific feature engineering, and AutoML techniques into one cohesive pipeline. The goal is to go beyond standard approaches and show how integrating multiple perspectives in model building can lead to significantly higher performance.

---

## 🎯 Project Goal
Our goal is to build a model that can:
- Achieve **high predictive accuracy and recall**, especially for diabetic cases (positive class).
- **Surpass the traditional 77% accuracy ceiling** on this dataset.
- **Explain the why** behind every modeling choice so it becomes an educational and reusable template.

---

## 💡 Why This Approach? (Explain Like I'm 5)
This project answers a simple but powerful question:

> "Can we teach a computer to spot someone at risk of diabetes more accurately than existing methods, using everything we know?"

To do this, we use not one model but a **team of models**, each contributing their own point of view, just like how multiple doctors might weigh in on a diagnosis.

We don't just throw models at the data. We:
- Clean the data like a real nurse would before entering into the system.
- Add new features that mimic the way doctors think (e.g. flagging if BMI is too high).
- Use AI to **search for the best neural network structure**.
- Use **a robot scientist (TPOT)** to explore the best combo of steps.
- Combine all models together and **ask another model to vote smartly** based on all of them.

This is not just ML — it's an orchestrated, explainable ML system.

---

## 🔍 Step-by-Step Logical Flow

### 1. 📦 Data Cleaning and Preprocessing
**Problem**: Some clinical fields have `0` values — medically invalid (e.g. Glucose = 0).

**Solution**:
- Replace 0s with the **median** of each feature.
- Normalize with `PowerTransformer` (makes skewed distributions like Insulin more Gaussian).
- Standardize with `StandardScaler` to make sure all features contribute equally.

👉 This ensures clean, normalized input for all models.

### 2. 🏥 Domain-Inspired Feature Engineering
Doctors know that some variable interactions are more telling than the raw values.
We recreate this logic:

- `Glucose * BMI`: Indicates risk via sugar level and obesity combined.
- `High_BMI`: A binary flag for BMI > 30 — common clinical threshold.
- `Glucose / Insulin`: Helps detect insulin resistance.
- `Age * BMI`: Older, overweight individuals have higher risk.

👉 These crafted features help the model **think more like a doctor.**

### 3. 🧠 Neural Networks (with Optuna Tuning)
**Why a Neural Network?**
- Traditional models flatten out at ~75% accuracy.
- NNs can capture **non-linear interactions** and patterns traditional models miss.

**Why Optuna?**
- Manually tuning a neural net is hard.
- Optuna uses smart trial and error to **search for the best architecture automatically**.

👉 This gives us a custom-fit neural network for our data.

### 4. 🧩 CNN on Tabular Data
**Why CNNs?**
- CNNs are good at detecting **local patterns**.
- We reshape tabular features into a 4x2 grid.

**Why?**
- This simulates how combinations of features (e.g., Glucose next to Insulin) can influence each other.
- A CNN learns these **spatial-like relationships**, even in non-image data.

👉 We're giving the model a different way to "look" at the data.

### 5. 🧬 TPOT for Genetic AutoML
**Why use TPOT?**
- It's like having an AI assistant that experiments with dozens of pipelines for you.
- TPOT tries different algorithms, scalers, and feature selectors using **genetic evolution**.

**What does it find?**
- Sometimes combinations like `SelectPercentile + ExtraTrees` work better than you’d guess.

👉 TPOT helps find pipelines that we might overlook.

### 6. 🔁 Model Stacking and Voting
**The Big Idea:** Use all good models, and let one smart model decide how to combine their outputs.

**We combine:**
- Optuna-tuned Neural Net
- CNN
- Random Forest
- Gradient Boosting
- LightGBM

**Then:**
- Their predicted probabilities become input features for a logistic regression meta-model.
- We **tune the final decision threshold** (not just 0.5!) to maximize **F1-score**.

👉 This allows the final decision to consider strengths of each base model.

---

## 📊 Final Results
| Metric               | Value    |
|----------------------|----------|
| Accuracy             | 80%      |
| F1 Score (Class 1)   | 0.7467   |
| Recall (Class 1)     | 0.85     |
| ROC AUC              | 0.8860   |

Confusion Matrix:
```
[[387 113]   # Non-diabetics: Correct and incorrect
 [ 41 227]]  # Diabetics: Missed and correctly identified
```

✅ This means:
- We're catching **85% of actual diabetics**
- With a very balanced **precision and recall**
- Using all tools from deep learning to AutoML

---

## 🧠 Conclusion
This is not just a project to boost accuracy.
It’s a **blueprint** showing:
- How to combine feature engineering, neural tuning, AutoML, CNNs, and model stacking
- How to think **beyond Kaggle scripts** and build **multi-perspective ML pipelines**

The takeaway: **Good ML comes from diverse tools, not a single model.**

---

## 🗂️ Project Structure
```bash
diabetes_superstack/
├── data/                          # Raw data CSV
├── features/                     # Domain feature generation
│   └── domain_features.py
├── genetics/                     # Genetic AutoML logic
│   └── feature_selector_tpot.py
├── models/                       # Neural net, CNN, and ensemble logic
│   ├── tuned_nn.py
│   ├── cnn_model.py
│   └── stacking_ensemble.py
├── full_superstack.py            # Run all steps end-to-end
├── requirements.txt              # Dependencies
└── README.md                     # You are here
```

---

## 🚀 Running the Pipeline
```bash
pip install -r requirements.txt
python full_superstack.py
```

---

## 📄 Dataset
- Source: [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 768 rows, 8 features, binary classification

---

## 🙌 Author
Created by **Mohammad Takneshan**  
For applied ML practitioners, health AI researchers, and students.

---

## 📜 License
MIT License — free to use and modify with credit.

---

## ⭐ Like it?
Star it, fork it, or use it in your portfolio!
