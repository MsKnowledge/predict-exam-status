"""
Predict Student Pass/Fail (Logistic Regression)
A simple Logistic Regression model that predicts whether a student
will pass or fail based on hours studied.
Built while learning AI basics.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create the dataset
# 0 = fail, 1 = pass
data = {
    "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "passed":        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
print("Dataset:")
print(df)

# Step 2: Separate input (X) and output (y)
X = df[["hours_studied"]]
y = df["passed"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Test the model
predictions = model.predict(X_test)
print("\nPredicted:", predictions)
print("Actual:", y_test.values)

# Step 6: Predict a new value
hours = 4.5
result = model.predict([[hours]])[0]
print(f"\nIf you study {hours} hours, will you pass? {'Yes' if result == 1 else 'No'}")
