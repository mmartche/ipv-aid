from pathlib import Path
import ast
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet


OUT_DIR = Path("outputs")
CSV_PATH = OUT_DIR / "results.csv"
CM_DIR = OUT_DIR / "confusion_matrices"
PDF_PATH = OUT_DIR / "Report.pdf"


def to_cm_matrix(cm_flat):
    """Convert flattened confusion matrix (list or stringified list) to square matrix."""
    if isinstance(cm_flat, str):
        cm_flat = ast.literal_eval(cm_flat)
    n = int(len(cm_flat) ** 0.5)
    return [cm_flat[i * n:(i + 1) * n] for i in range(n)]


def nice_table(df, cols, title, styles):
    """Create a simple ReportLab table with a title paragraph."""
    data = [cols] + df[cols].values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    return [Paragraph(title, styles["Heading2"]), Spacer(1, 8), t, Spacer(1, 12)]


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing: {CSV_PATH}. Run run_experiments_min.py first.")

    df = pd.read_csv(CSV_PATH)

    # Basic formatting
    df["test_size"] = df["test_size"].astype(float)
    df["accuracy"] = df["accuracy"].astype(float)
    df["precision_weighted"] = df["precision_weighted"].astype(float)

    styles = getSampleStyleSheet()
    story = []
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, title="ML Classification Report")

    story.append(Paragraph("Machine Learning Classification Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Models: Naive Bayes, Decision Tree, Random Forest, kNN. "
        "Datasets: BreastCancer, HeartFailure. "
        "Metrics: Accuracy, Weighted Precision, Confusion Matrix.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 16))

    # Summary table (keep it compact)
    summary = df[["dataset", "test_size", "preprocess", "model", "accuracy", "precision_weighted"]].copy()
    summary["accuracy"] = summary["accuracy"].round(4)
    summary["precision_weighted"] = summary["precision_weighted"].round(4)
    summary = summary.sort_values(["dataset", "test_size", "preprocess", "accuracy"],
                                  ascending=[True, True, True, False])

    for dataset_name in sorted(summary["dataset"].unique()):
        story.append(Paragraph(f"Dataset: {dataset_name}", styles["Heading1"]))
        story.append(Spacer(1, 6))

        for test_size in sorted(summary.loc[summary["dataset"] == dataset_name, "test_size"].unique()):
            block = summary[(summary["dataset"] == dataset_name) & (summary["test_size"] == test_size)].copy()

            story.append(Paragraph(f"Test size: {int(test_size * 100)}%", styles["Heading2"]))
            story.append(Spacer(1, 6))

            cols = ["preprocess", "model", "accuracy", "precision_weighted"]
            story += nice_table(block, cols, "Results (sorted by accuracy):", styles)

            # For each preprocessing, include the best model confusion matrix (image if available)
            for prep_name in block["preprocess"].unique():
                best = block[block["preprocess"] == prep_name].iloc[0]
                story.append(Paragraph(f"Best for preprocess: {prep_name} → {best['model']}", styles["Heading3"]))

                # IMPORTANT: fetch cm_image from the ORIGINAL df (summary does not include it)
                row = df[
                    (df["dataset"] == dataset_name) &
                    (df["test_size"] == test_size) &
                    (df["preprocess"] == prep_name) &
                    (df["model"] == best["model"])
                ].head(1)

                cm_image = None
                if len(row) == 1 and "cm_image" in row.columns and pd.notna(row.iloc[0]["cm_image"]):
                    cm_image = str(row.iloc[0]["cm_image"]).strip()

                # Use image only if it's a real file
                if cm_image:
                    cm_path = CM_DIR / cm_image
                    if cm_path.is_file():
                        story.append(Image(str(cm_path), width=400, height=300))
                        story.append(Spacer(1, 10))
                        continue  # done for this block

                # Fallback: build confusion matrix table from cm_flat
                if len(row) == 1 and "cm_flat" in row.columns and pd.notna(row.iloc[0]["cm_flat"]):
                    cm = to_cm_matrix(row.iloc[0]["cm_flat"])
                    cm_tbl = Table(cm)
                    cm_tbl.setStyle(TableStyle([
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ]))
                    story.append(Paragraph("Confusion matrix (from CSV):", styles["BodyText"]))
                    story.append(Spacer(1, 6))
                    story.append(cm_tbl)
                    story.append(Spacer(1, 12))
                else:
                    story.append(Paragraph("Confusion matrix not available.", styles["BodyText"]))
                    story.append(Spacer(1, 12))

            story.append(PageBreak())

    doc.build(story)
    print("PDF saved to:", PDF_PATH)


if __name__ == "__main__":
    main()
