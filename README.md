# Predict Student Logistic

A simple Logistic Regression model that predicts whether a student will **pass or fail** based on hours studied. Built while learning AI basics.

## How it works

- Uses **pandas** to organize the dataset into a table
- Uses **scikit-learn**'s `LogisticRegression` to train a classification model (predicting a category — Yes/No — instead of a number)
- Splits the data into training and testing sets to fairly check the model's accuracy
- Predicts Pass/Fail for any given number of study hours

## Example output

```
Dataset:
   hours_studied  passed
0              1       0
1              2       0
2              3       0
3              4       0
4              5       1
5              6       1
6              7       1
7              8       1
8              9       1
9             10       1

Predicted: [1 0]
Actual: [1 0]

If you study 4.5 hours, will you pass? Yes
```

## Run it

```bash
pip install pandas scikit-learn
python predictstudentlogistic.py
```

## What I learned building this

- The difference between regression (predicting a number) and classification (predicting a category)
- How Logistic Regression calculates a probability, then applies a cutoff (default 50%) to decide Yes or No
- How to test and use a trained classification model to predict new values

---
*Part of my "AI Simply Explained" weekly beginner project series.*
