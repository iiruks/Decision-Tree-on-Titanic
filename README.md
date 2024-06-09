# Titanic Survival Prediction

This project demonstrates a machine learning approach to predict survival on the Titanic dataset using decision trees and random forests. The workflow involves data preprocessing, model training, hyperparameter tuning, and evaluation.

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Model Training and Evaluation](#model-training-and-evaluation)
    1. [Decision Tree Classifier](#decision-tree-classifier)
    2. [Cross-Validation and Hyperparameter Tuning](#cross-validation-and-hyperparameter-tuning)
    3. [Pruned Decision Tree](#pruned-decision-tree)
    4. [Random Forest Classifier](#random-forest-classifier)
4. [Results](#results)
5. [Conclusion](#conclusion)

## Installation

1. Clone the repository.
2. Ensure you have Python 3.x installed.
3. Install the required packages:

```bash
pip install pandas scikit-learn numpy matplotlib
```

## Data Preparation

1. Read the Titanic dataset (`Titanic.csv`).
2. Select relevant features: `pclass`, `sex`, `age`, `sibsp`, and `survived`.
3. Convert categorical variables to numerical: `pclass` and `sex`.
4. Handle missing values in the `age` column by replacing them with the mean age.
5. Split the data into training (70%) and testing (30%) sets.

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Titanic.csv')
df = df[['pclass', 'sex', 'age', 'sibsp', 'survived']]
df['pclass'] = df['pclass'].map({'1st': 1, '2nd': 2, '3rd': 3})
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['age'].fillna(df['age'].mean(), inplace=True)
df = df.dropna()

X = df.drop('survived', axis=1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

## Model Training and Evaluation

### Decision Tree Classifier

Train a decision tree classifier on the training set and evaluate its performance on the test set.

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

print("Classification: \n\n", classification_report(y_test, y_pred))
print("Accuracy: ", accuracy_score(y_test, y_pred))
```

### Cross-Validation and Hyperparameter Tuning

Use grid search with cross-validation to find the best parameters for pruning the decision tree.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': ['entropy'],
    'splitter': ['best'],
    'max_depth': [None, 1, 2, 3, 4],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 3, 5],
    'max_leaf_nodes': [None, 3, 5],
    'min_impurity_decrease': [0, 0.2, 0.3, 0.5]
}

gsearch = GridSearchCV(DecisionTreeClassifier(), param_grid, verbose=1)
gsearch.fit(X_train, y_train)
best_params = gsearch.best_params_
```

### Pruned Decision Tree

Train a pruned decision tree using the best parameters found and evaluate its performance.

```python
clf_best = DecisionTreeClassifier(**best_params)
clf_best.fit(X_train, y_train)
y_predict = clf_best.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_predict))
print("Classification Report:\n", classification_report(y_test, y_predict))
```

### Random Forest Classifier

Train a random forest classifier using the optimal tree size and evaluate its performance.

```python
from sklearn.ensemble import RandomForestClassifier

R = RandomForestClassifier(max_leaf_nodes=None, n_estimators=50)
R.fit(X_train, y_train)
R_predict = R.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, R_predict))
print("Classification Report:\n", classification_report(y_test, R_predict))
```

## Results

### Decision Tree Classifier
- Accuracy: 0.74
- Precision, recall, and F1-score details for each class are provided in the classification report.

### Pruned Decision Tree
- Improved accuracy: 0.77
- Better precision and recall for the 'survived' class compared to the initial decision tree.

### Random Forest Classifier
- Accuracy: 0.76
- Comparable precision and recall for both classes, indicating a robust performance.

## Conclusion

This project demonstrates the use of decision trees and random forests to predict Titanic survival. The pruned decision tree and random forest classifiers showed improved performance compared to the initial decision tree model. Hyperparameter tuning and cross-validation were crucial in optimizing the models. Further improvements can be explored by incorporating more features and advanced techniques.

Feel free to explore and modify the code for better insights and results.
