# Diabetes prediction

The dataset is from kaggle (diabetes prediction dataset).

It contains medical information about patients and each row already has a label indicating whether the patient has diabetes or not. The model learns patterns in the health data to predict if a new patient is likely to have diabetes.

The dataset includes the following features:

* Gender
* Age
* Hypertension
* Heart disease
* Smoking history
* BMI
* HbA1c level
* Blood glucose level
* Diabetes (target)

I encoded the categorical features, split the data into training and testing sets, and trained both a Logistic Regression and a Random Forest model to compare their performance. I evaluated them using accuracy, a classification report, and a confusion matrix, visualized the most important features and added an interactive interface where users can enter patient information to receive a prediction.

Update 1: reduced a lot of manual code while adding more sophisticated methodology

This version demonstrates: 5-fold cross-validation → F1 optimization → ROC-AUC evaluation → final model comparison
