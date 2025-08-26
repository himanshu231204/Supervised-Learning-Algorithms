# 📘 Supervised Learning Algorithms  

---
A comprehensive repository that covers all major Supervised Machine Learning algorithms 📘 — explained in detail with theory, Python code, ASCII flowcharts, and practical implementation.

To make it even more valuable, I implemented every model on the Titanic Dataset 🛳️ (Survival Prediction) — one of the most popular datasets in Data Science.


## 📑 Table of Contents  
1. [Overview of Supervised Learning](#-overview-of-supervised-learning)  
2. [Algorithms](#-algorithms)  
   - [Linear Regression](#1-linear-regression)  
   - [Logistic Regression](#2-logistic-regression)  
   - [K-Nearest Neighbors (KNN)](#3-k-nearest-neighbors-knn)  
   - [Decision Tree](#4-decision-tree)  
   - [Random Forest](#5-random-forest)  
   - [Support Vector Machine (SVM)](#6-support-vector-machine-svm)  
   - [Naïve Bayes](#7-naïve-bayes)  
   - [Gradient Boosting](#8-gradient-boosting-xgboost-lightgbm-catboost)  
   - [Neural Networks (Intro)](#9-neural-networks-intro)  
3. [Comparison Table](#-comparison-table)  
4. [Choosing the Right Algorithm (Flowchart)](#-choosing-the-right-algorithm-ascii-flowchart)  
5. [Installation](#-installation)  
6. [Cheatsheet (Quick Revision)](#-cheatsheet-quick-revision)  
7. [References](#-references)  
8. [Credits](#-credits)  

---

## 🔹 Overview of Supervised Learning  
- **Definition**: Algorithms that learn from **labeled data** (input `X` → output `Y`).  
- **Goal**: Map input features → output target.  
- **Types**:  
  - **Regression** → Predict continuous values (e.g., house price).  
  - **Classification** → Predict categorical values (e.g., spam or not spam).  
- **Workflow**:  
```

Data → Preprocess → Train → Test → Evaluate → Deploy

````

---

## 🔹 Algorithms  

### 1. Linear Regression  
- **Type**: Regression  
- **Mathematical Idea**:  
- Predicts continuous target using a **linear equation**.  
- Equation: `y = b0 + b1x1 + b2x2 + ... + bn*xn`  
- **When to Use**:  
- Relationship between features & target is linear.  
- **Pros**: Simple, interpretable.  
- **Cons**: Sensitive to outliers.  
- **Code**:  
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
````

* **Flowchart**:

  ```
  Input Data
     ↓
  Fit Line y = mx + c
     ↓
  Minimize Squared Error
     ↓
  Predict Continuous Value
  ```

---

### 2. Logistic Regression

* **Type**: Classification
* **Mathematical Idea**:

  * Uses **sigmoid function** for probability.
* **When to Use**: Binary classification.
* **Pros**: Simple, probabilistic.
* **Cons**: Poor for non-linear data.
* **Code**:

  ```python
  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression().fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Features
      ↓
  Linear Combination (z = b0 + b1x1 + ...)
      ↓
  Apply Sigmoid σ(z)
      ↓
  Probability (0-1)
      ↓
  Threshold → Class Label
  ```

---

### 3. K-Nearest Neighbors (KNN)

* **Type**: Both
* **Idea**: Classifies based on nearest `k` neighbors.
* **When to Use**: Small datasets, pattern recognition.
* **Pros**: No training.
* **Cons**: Slow on large datasets.
* **Code**:

  ```python
  from sklearn.neighbors import KNeighborsClassifier
  model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Data Point
        ↓
  Compute Distance to All Points
        ↓
  Select k Nearest Neighbors
        ↓
  Majority Vote (Classification)
  or Average (Regression)
  ```

---

### 4. Decision Tree

* **Type**: Both
* **Idea**: Splits data using rules → tree.
* **Pros**: Easy to interpret.
* **Cons**: Overfits.
* **Code**:

  ```python
  from sklearn.tree import DecisionTreeClassifier
  model = DecisionTreeClassifier().fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Start
    ↓
  Choose Best Feature (Gini/Entropy)
    ↓
  Split Data into Subsets
    ↓
  Repeat Until Leaf Node
    ↓
  Leaf → Prediction
  ```

---

### 5. Random Forest

* **Type**: Both
* **Idea**: Multiple trees → voting/averaging.
* **Pros**: Robust, less overfit.
* **Cons**: Black-box.
* **Code**:

  ```python
  from sklearn.ensemble import RandomForestClassifier
  model = RandomForestClassifier().fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Data
     ↓
  Bootstrap Samples
     ↓
  Build Many Decision Trees
     ↓
  Aggregate Predictions
     ↓
  Final Output
  ```

---

### 6. Support Vector Machine (SVM)

* **Type**: Both
* **Idea**: Finds best hyperplane with max margin.
* **Pros**: Works well in high dimensions.
* **Cons**: Slow for large data.
* **Code**:

  ```python
  from sklearn.svm import SVC
  model = SVC(kernel='rbf').fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Data
     ↓
  Map Data (Kernel Trick)
     ↓
  Find Optimal Hyperplane
     ↓
  Maximize Margin Between Classes
     ↓
  Predict Class
  ```

---

### 7. Naïve Bayes

* **Type**: Classification
* **Idea**: Uses **Bayes’ theorem** + independence assumption.
* **Pros**: Fast, works with text.
* **Cons**: Assumes independence.
* **Code**:

  ```python
  from sklearn.naive_bayes import GaussianNB
  model = GaussianNB().fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Features
     ↓
  Calculate Prior Probability
     ↓
  Calculate Likelihood for Each Class
     ↓
  Apply Bayes’ Rule
     ↓
  Choose Class with Max Probability
  ```

---

### 8. Gradient Boosting (XGBoost, LightGBM, CatBoost)

* **Type**: Both
* **Idea**: Sequentially builds trees → corrects errors.
* **Pros**: High accuracy.
* **Cons**: Complex, tuning required.
* **Code**:

  ```python
  from xgboost import XGBClassifier
  model = XGBClassifier().fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Start with Weak Model
        ↓
  Compute Residual Errors
        ↓
  Train New Tree on Errors
        ↓
  Add Tree to Ensemble
        ↓
  Repeat → Final Strong Model
  ```

---

### 9. Neural Networks (Intro)

* **Type**: Both
* **Idea**: Layers of neurons → non-linear transformations.
* **Pros**: Very powerful.
* **Cons**: Needs huge data & compute.
* **Code**:

  ```python
  from sklearn.neural_network import MLPClassifier
  model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500).fit(X_train, y_train)
  pred = model.predict(X_test)
  ```
* **Flowchart**:

  ```
  Input Layer (Features)
        ↓
  Hidden Layers (Weights + Activation)
        ↓
  Backpropagation (Error Correction)
        ↓
  Output Layer (Prediction)
  ```

---

## 📊 Comparison Table

| Algorithm           | Type           | Pros                    | Cons                       |
| ------------------- | -------------- | ----------------------- | -------------------------- |
| Linear Regression   | Regression     | Simple, interpretable   | Only linear data           |
| Logistic Regression | Classification | Fast, interpretable     | Poor for non-linear data   |
| KNN                 | Both           | Simple, no training     | Slow, scaling needed       |
| Decision Tree       | Both           | Easy visualization      | Overfits                   |
| Random Forest       | Both           | Accurate, stable        | Slower, less interpretable |
| SVM                 | Both           | Works in high-dim space | Slow for big datasets      |
| Naïve Bayes         | Classification | Fast, good for text     | Assumes independence       |
| Gradient Boosting   | Both           | Very accurate           | Complex, tuning needed     |
| Neural Networks     | Both           | Powerful, flexible      | Data & compute hungry      |

---

## 🎯 Choosing the Right Algorithm (ASCII Flowchart)

```
                    ┌─────────────────────────┐
                    │ How big is your dataset?│
                    └───────────┬────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
        Small Dataset                    Large Dataset
                │                               │
   ┌────────────┴─────────────┐       ┌────────┴─────────┐
   │                          │       │                  │
 Linear / Logistic       Non-linear?   Need high accuracy?
 Regression                   │             │
                              │             │
                        ┌─────┴─────┐  ┌───┴─────────────┐
                        │   Yes     │  │       Yes        │
                        └─────┬─────┘  └───────┬─────────┘
                              │                │
                         KNN / Decision   Random Forest /
                             Tree         Gradient Boosting
                                              │
                                    Deep patterns? (images/text)
                                              │
                                         Neural Networks
```

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost
```

---

## 📄 Cheatsheet (Quick Revision)

* **Linear Regression** → `y = b0 + b1x` → Continuous output
* **Logistic Regression** → `σ(z) = 1 / (1 + e^-z)` → Binary classification
* **KNN** → Look at `k` closest neighbors → vote/average
* **Decision Tree** → Splits dataset by features → if/else rules
* **Random Forest** → Multiple trees → majority vote → stable
* **SVM** → Max-margin hyperplane → kernel trick
* **Naïve Bayes** → Bayes theorem + independence assumption
* **Gradient Boosting** → Sequential trees fixing errors → very accurate
* **Neural Networks** → Layers + activations + backpropagation → powerful

---

## 📚 References

* [Scikit-Learn Documentation](https://scikit-learn.org/stable/)
* [XGBoost Documentation](https://xgboost.readthedocs.io/)
* [Deep Learning Book (Ian Goodfellow)](https://www.deeplearningbook.org/)

---

## 🙌 Credits

* Created by **Himanshu Kumar**

