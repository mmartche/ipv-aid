"""Driver script to run all experiments and save CSVs.

This script exists as a robust entrypoint (in some environments, running
run_experiments.py directly may be constrained).
"""

from pathlib import Path
from typing import Dict, List

import pandas as pd

import run_experiments as rexp


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    out_dir = project_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    datasets = [
        rexp.load_breast_cancer_dataset(),
        rexp.load_heart_failure_from_uci(project_dir / "data_cache"),
    ]

    preprocess_variants = [
        "labelEncoder",
        "labelEncoder+standardScaler",
        "labelEncoder+oneHotEncoder",
        "labelEncoder+oneHotEncoder+standardScaler",
    ]

    test_sizes = [0.15, 0.30, 0.50]
    models = rexp.make_models(random_state=42)

    rows: List[Dict[str, object]] = []

    for ds in datasets:
        for ts in test_sizes:
            for prep in preprocess_variants:
                for model_name, model in models.items():
                    rows.append(
                        rexp.evaluate(
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
    df_summary.columns = [f"{metric}__{model}" for metric, model in df_summary.columns]
    df_summary = df_summary.reset_index().sort_values(["dataset", "test_pct", "preprocessing"])
    df_summary.to_csv(out_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
