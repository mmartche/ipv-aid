import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


UCI_HEART_FAILURE_ZIP = (
    "https://archive.ics.uci.edu/static/public/519/heart%2Bfailure%2Bclinical%2Brecords.zip"
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: pd.DataFrame
    y: pd.Series
    categorical_cols: List[str]


def load_heart_failure_from_uci(cache_dir: Path) -> DatasetSpec:
    """Load Heart Failure Clinical Records.

    Priority order:
    1) If a cached CSV exists at `<cache_dir>/heart_failure_clinical_records_dataset.csv`, load it.
    2) Otherwise, download the official UCI ZIP and read the CSV from inside it.

    This makes the project usable in restricted environments (e.g., where Python cannot
    access the network) as long as the CSV is provided.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_path = cache_dir / "heart_failure_clinical_records_dataset.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        zip_path = cache_dir / "heart_failure_clinical_records.zip"

        if not zip_path.exists():
            print(f"Downloading: {UCI_HEART_FAILURE_ZIP}")
            import urllib.request

            urllib.request.urlretrieve(UCI_HEART_FAILURE_ZIP, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = None
            for n in zf.namelist():
                if n.lower().endswith(".csv"):
                    csv_name = n
                    break
            if csv_name is None:
                raise FileNotFoundError("No CSV found inside the downloaded ZIP.")

            with zf.open(csv_name) as f:
                df = pd.read_csv(f)

    # Target column in this dataset is commonly "DEATH_EVENT".
    if "DEATH_EVENT" not in df.columns:
        raise ValueError(f"Expected target column 'DEATH_EVENT' not found. Columns: {list(df.columns)}")

    y = df["DEATH_EVENT"].astype(int)
    X = df.drop(columns=["DEATH_EVENT"])

    # Detect categorical columns by dtype (object/category). This dataset is numeric,
    # but keeping detection makes the pipeline generic.
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]

    return DatasetSpec(name="Heart Failure", X=X, y=y, categorical_cols=cat_cols)


def load_breast_cancer_dataset() -> DatasetSpec:
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = pd.Series(data.target, name="target")

    cat_cols: List[str] = []
    return DatasetSpec(name="Breast Cancer", X=X, y=y, categorical_cols=cat_cols)


def build_preprocessor(
    X: pd.DataFrame,
    categorical_cols: List[str],
    variant: str,
) -> Tuple[object, str]:
    """Return a transformer (or 'passthrough') according to variant."""

    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # OneHotEncoder: handle unknowns, sparse output; then later StandardScaler can be
    # applied with with_mean=False if output is sparse.
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    if variant == "labelEncoder":
        return "passthrough", variant

    if variant == "labelEncoder+standardScaler":
        # scale numeric columns, passthrough categorical (if any)
        if categorical_cols:
            pre = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", "passthrough", categorical_cols),
                ]
            )
        else:
            pre = StandardScaler()
        return pre, variant

    if variant == "labelEncoder+oneHotEncoder":
        if categorical_cols:
            pre = ColumnTransformer(
                transformers=[
                    ("num", "passthrough", numeric_cols),
                    ("cat", onehot, categorical_cols),
                ],
                remainder="drop",
            )
        else:
            pre = "passthrough"
        return pre, variant

    if variant == "labelEncoder+oneHotEncoder+standardScaler":
        if categorical_cols:
            pre = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", onehot, categorical_cols),
                ],
                remainder="drop",
            )
        else:
            pre = StandardScaler()
        return pre, variant

    raise ValueError(f"Unknown preprocessing variant: {variant}")


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # annotate cells
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "kNN": KNeighborsClassifier(n_neighbors=7),
    }


def evaluate(
    dataset: DatasetSpec,
    test_size: float,
    preprocessing_variant: str,
    model_name: str,
    model,
    out_dir: Path,
    random_state: int = 42,
) -> Dict[str, object]:
    X = dataset.X
    y = dataset.y

    # label encode target (even if already numeric) to match assignment wording
    le = LabelEncoder()
    y_enc = pd.Series(le.fit_transform(y), name="target")
    class_labels = [str(c) for c in le.classes_]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    pre, variant_name = build_preprocessor(X_train, dataset.categorical_cols, preprocessing_variant)

    pipe_steps = []
    if pre != "passthrough":
        pipe_steps.append(("preprocess", pre))
    pipe_steps.append(("model", model))

    pipe = Pipeline(pipe_steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec_w = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred)

    # save confusion matrix plot
    test_pct = int(round(test_size * 100))
    safe_ds = dataset.name.replace(" ", "_").lower()
    safe_prep = preprocessing_variant.replace("+", "_").replace(" ", "_").lower()
    safe_model = model_name.replace(" ", "_").lower()

    cm_path = out_dir / "confusion_matrices" / f"{safe_ds}__test{test_pct}__{safe_prep}__{safe_model}.png"
    title = f"{dataset.name} | {model_name} | test={test_pct}%\n{preprocessing_variant}"
    plot_confusion_matrix(cm, labels=class_labels, title=title, out_path=cm_path)

    # Flatten cm for CSV
    flat_cm = {f"cm_{i}{j}": int(cm[i, j]) for i in range(cm.shape[0]) for j in range(cm.shape[1])}

    row = {
        "dataset": dataset.name,
        "test_size": test_size,
        "test_pct": test_pct,
        "preprocessing": preprocessing_variant,
        "model": model_name,
        "accuracy": acc,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "f1_weighted": f1_w,
        "cm_image": str(cm_path),
    }
    row.update(flat_cm)
    return row


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    out_dir = project_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        load_breast_cancer_dataset(),
        load_heart_failure_from_uci(project_dir / "data_cache"),
    ]

    preprocess_variants = [
        "labelEncoder",
        "labelEncoder+standardScaler",
        "labelEncoder+oneHotEncoder",
        "labelEncoder+oneHotEncoder+standardScaler",
    ]

    test_sizes = [0.15, 0.30, 0.50]
    models = make_models(random_state=42)

    rows: List[Dict[str, object]] = []

    for ds in datasets:
        for ts in test_sizes:
            for prep in preprocess_variants:
                for model_name, model in models.items():
                    print(f"Running: {ds.name} | test={ts} | {prep} | {model_name}")
                    rows.append(
                        evaluate(
                            dataset=ds,
                            test_size=ts,
                            preprocessing_variant=prep,
                            model_name=model_name,
                            model=model,
                            out_dir=out_dir,
                            random_state=42,
                        )
                    )

    df_long = pd.DataFrame(rows)
    df_long.to_csv(out_dir / "results_long.csv", index=False)

    # Summary: show accuracy and precision_weighted
    df_summary = (
        df_long
        .assign(test_pct=df_long["test_pct"].astype(int))
        .pivot_table(
            index=["dataset", "test_pct", "preprocessing"],
            columns="model",
            values=["precision_weighted", "accuracy"],
            aggfunc="first",
        )
    )

    # Flatten multi-index columns
    df_summary.columns = [f"{metric}__{model}" for metric, model in df_summary.columns]
    df_summary = df_summary.reset_index().sort_values(["dataset", "test_pct", "preprocessing"])
    df_summary.to_csv(out_dir / "results_summary.csv", index=False)

    print("\nDone.")
    print(f"Wrote: {out_dir / 'results_long.csv'}")
    print(f"Wrote: {out_dir / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
