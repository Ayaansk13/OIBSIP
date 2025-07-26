# Iris Flower Classification using Machine Learning

# Step 1: Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris

# Step 2: Load dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].apply(lambda x: iris.target_names[x])

# Step 3: Exploratory Data Analysis
print("First 5 rows of the dataset:")
print(df.head())

print("\nClass Distribution:")
print(df['species'].value_counts())

# Visualizations
sns.pairplot(df, hue='species')
plt.suptitle("Pairplot of Iris Features", y=1.02)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Feature and target split
X = df.iloc[:, :-1]
y = df['species']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Predict on new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # Sample input
prediction = model.predict(sample)
print("\nPrediction for sample [5.1, 3.5, 1.4, 0.2]:", prediction[0])
