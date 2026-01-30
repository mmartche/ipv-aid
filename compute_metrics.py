from __future__ import annotations

from pathlib import Path
from typing import List, Dict

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
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix


def build_preprocessor(X: pd.DataFrame, categorical_cols: List[str], variant: str):
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    if variant == 'labelEncoder':
        return 'passthrough'
    if variant == 'labelEncoder+standardScaler':
        return ColumnTransformer([('num', StandardScaler(), numeric_cols)], remainder='drop')
    if variant == 'labelEncoder+oneHotEncoder':
        if not categorical_cols:
            return 'passthrough'
        return ColumnTransformer([('cat', onehot, categorical_cols), ('num', 'passthrough', numeric_cols)], remainder='drop')
    if variant == 'labelEncoder+oneHotEncoder+standardScaler':
        transformers = []
        if categorical_cols:
            transformers.append(('cat', onehot, categorical_cols))
        transformers.append(('num', StandardScaler(), numeric_cols))
        return ColumnTransformer(transformers, remainder='drop')
    raise ValueError(variant)


def load_breast_cancer_df():
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = pd.Series(data.target, name='target')
    return 'Breast Cancer', X, y, []


def load_heart_failure_df():
    p = Path(__file__).resolve().parent / 'data_cache' / 'heart_failure_clinical_records_dataset.csv'
    df = pd.read_csv(p)
    X = df.drop(columns=['DEATH_EVENT'])
    y = df['DEATH_EVENT']
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ('object','category')]
    return 'Heart Failure', X, y, cat_cols


def main():
    out_dir = Path(__file__).resolve().parent / 'outputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [load_breast_cancer_df(), load_heart_failure_df()]
    test_sizes = [0.15, 0.30, 0.50]
    preprocess_variants = [
        'labelEncoder',
        'labelEncoder+standardScaler',
        'labelEncoder+oneHotEncoder',
        'labelEncoder+oneHotEncoder+standardScaler',
    ]
    models = {
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
    }

    rows: List[Dict[str, object]] = []

    for ds_name, X, y, cat_cols in datasets:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        for ts in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_enc, test_size=ts, random_state=42, stratify=y_enc
            )
            for prep in preprocess_variants:
                pre = build_preprocessor(X_train, cat_cols, prep)
                for model_name, model in models.items():
                    steps = []
                    if pre != 'passthrough':
                        steps.append(('preprocess', pre))
                    steps.append(('model', model))
                    pipe = Pipeline(steps)
                    pipe.fit(X_train, y_train)
                    y_pred = pipe.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred)
                    rows.append({
                        'dataset': ds_name,
                        'test_pct': int(round(ts*100)),
                        'preprocessing': prep,
                        'model': model_name,
                        'accuracy': float(accuracy_score(y_test, y_pred)),
                        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                        'cm_00': int(cm[0,0]),
                        'cm_01': int(cm[0,1]),
                        'cm_10': int(cm[1,0]),
                        'cm_11': int(cm[1,1]),
                    })

    df_long = pd.DataFrame(rows)
    df_long.to_csv(out_dir / 'results_long.csv', index=False)

    df_summary = (
        df_long.pivot_table(
            index=['dataset','test_pct','preprocessing'],
            columns='model',
            values='precision_weighted',
            aggfunc='first'
        ).reset_index()
    )
    df_summary.to_csv(out_dir / 'results_precision_summary.csv', index=False)


if __name__ == '__main__':
    main()
