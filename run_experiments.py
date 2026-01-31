from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

OUTPUT_DIR = Path("outputs")
CM_DIR = OUTPUT_DIR / "confusion_matrices"
OUTPUT_DIR.mkdir(exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)


def load_heart_failure(csv_path="data_cache/heart_failure_clinical_records_dataset.csv"):
    df = pd.read_csv(csv_path)
    target = "DEATH_EVENT"
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def build_preprocessors(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = [c for c in X.columns if c not in categorical_cols]

    label_encoder = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_cols)],
        remainder="passthrough"
    )

    one_hot = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough"
    )

    return {
        "labelEncoder": label_encoder,
        "labelEncoder+standardScaler": Pipeline([
            ("label", label_encoder),
            ("scaler", StandardScaler(with_mean=False))
        ]),
        "labelEncoder+oneHotEncoder": one_hot,
        "labelEncoder+oneHotEncoder+standardScaler": Pipeline([
            ("onehot", one_hot),
            ("scaler", StandardScaler(with_mean=False))
        ]),
    }

MODELS = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200),
    # < 50 trees → unstable results | ~100 trees → reasonably stable | 150–300 trees → “safe” zone
    "kNN": KNeighborsClassifier(n_neighbors=5),
    # default value at scikit-learn is 5
}

def evaluate_model(
    X, y,
    dataset_name,
    prep_name, preprocessor,
    model_name, model,
    test_size,
    random_state=42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", model)
    ])

    y_pred = pipeline.fit(X_train, y_train).predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    cm_filename = (
        f"{dataset_name}__{prep_name}__{model_name}__test{int(test_size*100)}.png"
        .replace(" ", "")
    )

    fig = ConfusionMatrixDisplay(cm).plot().figure_
    fig.savefig(CM_DIR / cm_filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "test_size": test_size,
        "preprocess": prep_name,
        "model": model_name,
        "accuracy": accuracy,
        "precision_weighted": precision,
        "cm_image": cm_filename,
        "cm_flat": cm.ravel().tolist(),
    }

def main():
    datasets = {}

    breast = load_breast_cancer(as_frame=True)
    datasets["BreastCancer"] = (breast.data, breast.target)

    datasets["HeartFailure"] = load_heart_failure()

    test_sizes = [0.15, 0.30, 0.50]
    results = []

    for dataset_name, (X, y) in datasets.items():
        preprocessors = build_preprocessors(X)
        for test_size in test_sizes:
            for prep_name, preprocessor in preprocessors.items():
                for model_name, model in MODELS.items():
                    results.append(
                        evaluate_model(
                            X, y,
                            dataset_name,
                            prep_name, preprocessor,
                            model_name, model,
                            test_size
                        )
                    )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    print("Results saved to:", OUTPUT_DIR / "results.csv")


if __name__ == "__main__":
    main()
