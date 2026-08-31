# train.py  -  run this ONCE:  python train.py
import pathlib
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# 2. SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 3. TRAIN - scaler a
# nd model in ONE pipeline object
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, random_state=42))
pipeline.fit(X_train, y_train)

# 4. EVALUATE - you must know what you are shipping
y_pred = pipeline.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. SAVE - the whole pipeline, plus everything the app needs
pathlib.Path("models").mkdir(exist_ok=True)
joblib.dump({
    "pipeline": pipeline,
    "features": list(X.columns),
    "classes": list(iris.target_names),
    "accuracy": accuracy_score(y_test, y_pred),
}, "models/iris_model.joblib")

print("saved -> models/iris_model.joblib")