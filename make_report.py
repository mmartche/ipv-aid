from pathlib import Path
import ast
import math
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepInFrame
)
from reportlab.lib.styles import getSampleStyleSheet

OUT_DIR = Path("outputs")
CSV_PATH = OUT_DIR / "results.csv"
CM_DIR = OUT_DIR / "confusion_matrices"
PDF_PATH = OUT_DIR / "Report.pdf"

MODEL_ORDER = ["Decision Tree", "Naive Bayes", "Random Forest", "kNN"]


def cm_from_flat(cm_flat):
    if isinstance(cm_flat, str):
        cm_flat = ast.literal_eval(cm_flat)
    n = int(len(cm_flat) ** 0.5)
    return [cm_flat[i * n:(i + 1) * n] for i in range(n)]


def pivot_data(df, value_col):
    p = (
        df.pivot_table(index="preprocess", columns="model", values=value_col, aggfunc="mean")
        .reindex(columns=MODEL_ORDER)
        .round(4)
    )
    data = [["Preprocessing"] + MODEL_ORDER]
    for prep, row in p.iterrows():
        data.append([prep] + [("" if pd.isna(row[m]) else f"{row[m]:.4f}") for m in MODEL_ORDER])
    return data


def styled_table(data):
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def add_best_run_summary_line(story, styles, df_block):
    best = df_block.sort_values(["accuracy", "precision_weighted"], ascending=[False, False]).iloc[0]
    line = (
        f"Melhor execucao (por Exatidao): {best['model']} com {best['preprocess']}. "
        f"Accuracy={float(best['accuracy']):.4f}, Precision={float(best['precision_weighted']):.4f}."
    )
    story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 8))


def cm_text_lines(cm):
    n = len(cm)
    if n == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        line1 = f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}"
        if fn == 0 and fp == 0:
            line2 = "Sem falsos positivos nem falsos negativos."
        else:
            line2 = f"Erros: {fn} Falsos Negativos e {fp} Falsos Positivos."
        return line1, line2
    return f"Matriz {n}x{n}", ""


def best_row_for_preprocess(df_block, preprocess_name):
    sub = df_block[df_block["preprocess"] == preprocess_name].copy()
    if sub.empty:
        return None
    return sub.sort_values(["accuracy", "precision_weighted"], ascending=[False, False]).iloc[0]


def cm_cell(styles, preprocess_name, best_row, img_w=200, img_h=200):
    elems = []
    elems.append(Paragraph(f"<b>{preprocess_name}</b>", styles["BodyText"]))
    elems.append(Spacer(1, 4))

    cm = None
    if best_row is not None and pd.notna(best_row.get("cm_flat", "")):
        cm = cm_from_flat(best_row["cm_flat"])

    if best_row is not None and pd.notna(best_row.get("cm_image", "")):
        img_path = CM_DIR / str(best_row["cm_image"]).strip()
        if img_path.is_file():
            elems.append(Image(str(img_path), width=img_w, height=img_h))
        else:
            elems.append(Paragraph("Imagem não encontrada.", styles["BodyText"]))
    else:
        elems.append(Paragraph("Imagem indisponível.", styles["BodyText"]))

    elems.append(Spacer(1, 4))

    if cm is not None:
        l1, l2 = cm_text_lines(cm)
        elems.append(Paragraph(l1, styles["BodyText"]))
        if l2:
            elems.append(Paragraph(l2, styles["BodyText"]))
    else:
        elems.append(Paragraph("Resumo TN/FP/FN/TP indisponível.", styles["BodyText"]))

    return KeepInFrame(240, 280, elems, mode="shrink")


def add_confusion_grid(story, styles, df_block, preprocess_list):

    items = []
    for prep in preprocess_list:
        best = best_row_for_preprocess(df_block, prep)
        if best is None:
            continue
        items.append((prep, best))

    if not items:
        story.append(Paragraph("Sem matrizes de confusão disponíveis.", styles["BodyText"]))
        story.append(Spacer(1, 12))
        return

    chunk_size = 4
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

    for ci, chunk in enumerate(chunks):
        story.append(Paragraph("Matriz de confusão (melhor execução por pré-processamento):", styles["BodyText"]))
        story.append(Spacer(1, 6))

        cells = [cm_cell(styles, prep, row) for prep, row in chunk]
        while len(cells) < 4:
            cells.append(KeepInFrame(240, 280, [Paragraph("", styles["BodyText"])], mode="shrink"))

        grid_data = [
            [cells[0], cells[1]],
            [cells[2], cells[3]],
        ]

        grid = Table(grid_data, colWidths=[260, 260], rowHeights=[300, 300])
        grid.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))

        story.append(grid)
        story.append(Spacer(1, 10))

        if ci < len(chunks) - 1:
            story.append(PageBreak())


def main():
    df = pd.read_csv(CSV_PATH)

    df["test_size"] = pd.to_numeric(df["test_size"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["precision_weighted"] = pd.to_numeric(df["precision_weighted"], errors="coerce")

    styles = getSampleStyleSheet()
    story = []
    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4)

    story.append(Paragraph("Relatorio de classificacao Algoritmos de Aprendizagem", styles["Title"]))
    story.append(Spacer(1, 12))

    for dataset in sorted(df["dataset"].dropna().unique()):
        story.append(Paragraph(f"Dataset: {dataset}", styles["Heading1"]))
        story.append(Spacer(1, 8))

        for test_size in sorted(df[df["dataset"] == dataset]["test_size"].dropna().unique()):
            story.append(Paragraph(f"Set de teste: {int(float(test_size) * 100)}%", styles["Heading2"]))
            story.append(Spacer(1, 6)) 

            block = df[(df["dataset"] == dataset) & (df["test_size"] == test_size)].copy()

            story.append(Paragraph("Tabela: Exatidao (media por modelo)", styles["BodyText"]))
            story.append(styled_table(pivot_data(block, "accuracy")))
            story.append(Spacer(1, 10))

            story.append(Paragraph("Tabela: Precisão (media ponderada)", styles["BodyText"]))
            story.append(styled_table(pivot_data(block, "precision_weighted")))
            story.append(Spacer(1, 10))
            add_best_run_summary_line(story, styles, block)
            story.append(PageBreak())

            preprocess_list = sorted(block["preprocess"].dropna().unique())
            add_confusion_grid(story, styles, block, preprocess_list)

            story.append(PageBreak())
        
        story.append(Paragraph("Analise critica (sintese)", styles["Heading2"]))
        if dataset.lower().startswith("breast"):
            story.append(Paragraph("No dataset Breast Cancer, os resultados obtidos demonstram um elevado desempenho global em todos os cenários de teste considerados (15%, 30% e 50%). Mesmo com o aumento da percentagem do conjunto de teste, os valores de accuracy e precision mantiveram-se elevados, indicando que os padrões presentes nos dados são bem definidos e facilmente aprendidos pelos algoritmos de classificação.",styles["BodyText"]))
            story.append(Paragraph("A análise das matrizes de confusão revelou um número reduzido de erros, com especial destaque para a baixa incidência de falsos negativos, que correspondem a tumores malignos classificados como benignos. Este aspeto é particularmente relevante em contexto clínico, uma vez que este tipo de erro pode ter consequências graves. Em alguns cenários, nomeadamente com 30% de dados de teste, verificou-se mesmo a ausência de falsos negativos nos melhores modelos.",styles["BodyText"]))
            story.append(Paragraph("Os algoritmos Random Forest e kNN com normalização destacaram-se pela sua robustez, apresentando matrizes de confusão mais equilibradas. Estes resultados indicam que, para este dataset, os modelos conseguem generalizar adequadamente mesmo com uma redução significativa do conjunto de treino.",styles["BodyText"]))
            story.append(Paragraph("Breast Cancer Dataset", styles["Heading2"]))
            story.append(
                Paragraph(
                    "- Breast Cancer (Wisconsin – Diagnostic)<br/><br/>"
                    "- Fonte: UCI Machine Learning Repository<br/>"
                    "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)<br/><br/>"
                    "- Autores: University of Wisconsin–Madison<br/>"
                    "- Disponivel em: scikit-learn datasets module<br/>"
                    "https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset",
                    styles["BodyText"],
                )
            )
        else:
            story.append(Paragraph("Em contraste, o dataset Heart Failure revelou-se substancialmente mais desafiante. Apesar de valores aceitáveis de accuracy, a análise das matrizes de confusão evidenciou um número considerável de falsos negativos, particularmente nos cenários com 30% e 50% de dados de teste. Este comportamento indica que os modelos apresentam dificuldade em identificar corretamente todos os casos de óbito.",styles["BodyText"]))
            story.append(Paragraph("Este fenómeno pode ser explicado pela complexidade clínica dos dados, pela possível existência de relações não lineares entre variáveis e pelo desequilíbrio entre classes. Mesmo o Random Forest, que apresentou os melhores resultados globais, mostrou limitações evidentes na identificação da classe minoritária.",styles["BodyText"]))
            story.append(Paragraph("Assim, para este dataset, a matriz de confusão revelou-se essencial para uma interpretação adequada dos resultados, demonstrando que a accuracy, isoladamente, pode mascarar problemas relevantes do ponto de vista clínico.",styles["BodyText"]))
            story.append(Paragraph("Heart Failure Dataset", styles["Heading2"]))
            story.append(
                Paragraph(
                    "- Heart Failure (Clinical Records)<br/><br/>"
                    "- Fonte: UCI Machine Learning Repository<br/>"
                    "https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records<br/><br/>"
                    "- Disponivel em: Kaggle<br/>"
                    "https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data",
                    styles["BodyText"],
                )
            )            
        story.append(PageBreak())

    doc.build(story)
    print("PDF gerado em:", PDF_PATH)


if __name__ == "__main__":
    main()
