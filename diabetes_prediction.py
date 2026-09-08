import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, roc_auc_score
)

# we load our data
df = pd.read_csv("diabetes_prediction_dataset.csv").drop_duplicates()

print(f"Shape: {df.shape}")
print("\nMissing values:\n", df.isna().sum())
print("\nClass distribution:\n", df["diabetes"].value_counts(normalize=True))

X, y = df.drop(columns="diabetes"), df["diabetes"]

cat = ["gender", "smoking_history"]
num = [c for c in X.columns if c not in cat]

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, stratify=y, random_state=42
)

# hyperparameter tuning 
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"model__C": [.1, 1, 10]}
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }
    )
}

results = {}

for name, (model, params) in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])

    search = GridSearchCV(
        pipe, params, cv=5, scoring="f1", n_jobs=-1
    ).fit(X_train, y_train)

    pred = search.predict(X_test)
    prob = search.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)

    results[name] = (search.best_estimator_, auc)

    print(f"\n{name}")
    print("Best params:", search.best_params_)
    print(f"CV F1: {search.best_score_:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, pred))

# select best model
best_name, (best_model, best_auc) = max(
    results.items(), key=lambda x: x[1][1]
)

print(f"\nBest model: {best_name} | ROC-AUC: {best_auc:.4f}")

# evaluation plots
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title(f"{best_name} — Confusion Matrix")
plt.show()

for name, (model, _) in results.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)

plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curve Comparison")
plt.show()

#feature importance
prep = best_model.named_steps["prep"]
model = best_model.named_steps["model"]
features = prep.get_feature_names_out()

importance = (
    abs(model.coef_[0])
    if isinstance(model, LogisticRegression)
    else model.feature_importances_
)

importance = (
    pd.Series(importance, index=features)
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop features:\n", importance)

importance.sort_values().plot(kind="barh")
plt.title(f"Top Features — {best_name}")
plt.tight_layout()
plt.show()

# save model
joblib.dump(best_model, "diabetes_model.pkl")

# interactive prediction
def ask(prompt, cast=float):
    while True:
        try:
            return cast(input(prompt))
        except ValueError:
            print("Invalid input. Try again.")

patient = pd.DataFrame([{
    "gender": input("Gender: "),
    "age": ask("Age: "),
    "hypertension": ask("Hypertension (0/1): ", int),
    "heart_disease": ask("Heart disease (0/1): ", int),
    "smoking_history": input("Smoking history: "),
    "bmi": ask("BMI: "),
    "HbA1c_level": ask("HbA1c level: "),
    "blood_glucose_level": ask("Blood glucose level: ")
}])

prediction = best_model.predict(patient)[0]
probability = best_model.predict_proba(patient)[0, 1]

print("\nPrediction:", "Diabetes" if prediction else "No Diabetes")
print(f"Estimated probability: {probability:.2%}")
