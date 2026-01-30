"""Generate a PDF report from results_long.csv.

The report is aimed at the assignment deliverable: explain methodology, show tables
similar to the requested template, and provide a short critical analysis.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.lib.units import cm


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def _fmt(x: float, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "-"
    return f"{x:.{nd}f}"


def _make_pivot(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # metric in {"accuracy", "precision_weighted"}
    pivot = (
        df.pivot_table(
            index="preprocessing",
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        .reindex(
            [
                "labelEncoder",
                "labelEncoder+standardScaler",
                "labelEncoder+oneHotEncoder",
                "labelEncoder+oneHotEncoder+standardScaler",
            ]
        )
        .sort_index(axis=1)
    )
    return pivot


def _table_from_df(df: pd.DataFrame, title: str) -> list:
    styles = getSampleStyleSheet()
    elems = [Paragraph(title, styles["Heading3"]), Spacer(1, 6)]

    header = ["Preprocessing"] + list(df.columns)
    body = []
    for idx, row in df.iterrows():
        body.append([idx] + [_fmt(v) for v in row.tolist()])

    t = Table([header] + body, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightyellow]),
            ]
        )
    )
    elems.extend([t, Spacer(1, 10)])
    return elems


def build_report(results_csv: Path, pdf_out: Path) -> None:
    df = pd.read_csv(results_csv)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
        )
    )

    doc = SimpleDocTemplate(str(pdf_out), pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm)
    elems = []

    elems.append(Paragraph("Relatorio - Avaliacao de Modelos de Machine Learning", styles["Title"]))
    elems.append(Paragraph("Datasets: Breast Cancer (Wisconsin Diagnostic) e Heart Failure Clinical Records", styles["Heading2"]))
    elems.append(Spacer(1, 10))

    elems.append(
        Paragraph(
            "<b>Objetivo.</b> Implementar um conjunto de programas em Python que construa modelos de classificacao com "
            "Naive Bayes, Decision Tree, Random Forest e kNN, e calcule <i>accuracy</i> e <i>confusion matrix</i> para "
            "diferentes configuracoes de preparacao dos dados e diferentes percentagens de teste (15%, 30% e 50%).",
            styles["BodyText"],
        )
    )
    elems.append(Spacer(1, 10))

    elems.append(Paragraph("Metodologia", styles["Heading2"]))
    elems.append(
        Paragraph(
            "<b>Particao treino/teste.</b> Para cada dataset foram efetuadas particoes estratificadas (preservando a distribuicao "
            "da classe) com <b>random_state=42</b> para garantir reprodutibilidade.",
            styles["BodyText"],
        )
    )
    elems.append(
        Paragraph(
            "<b>Variantes de pre-processamento.</b> Foram avaliadas quatro configuracoes: "
            "(1) apenas codificacao do alvo (LabelEncoder para y); "
            "(2) + StandardScaler nas features numericas; "
            "(3) + OneHotEncoder para colunas categoricas (quando existirem); "
            "(4) OneHotEncoder + StandardScaler (escala apenas em features numericas).",
            styles["BodyText"],
        )
    )
    elems.append(
        Paragraph(
            "<b>Modelos.</b> GaussianNB (Naive Bayes), DecisionTreeClassifier, RandomForestClassifier e KNeighborsClassifier. "
            "Para Random Forest foi usado n_estimators=300 e random_state=42. Para kNN foi usado k=5.",
            styles["BodyText"],
        )
    )
    elems.append(
        Paragraph(
            "<b>Metricas.</b> Alem de accuracy, foram calculadas precision, recall e f1 (todas com media <i>weighted</i>). "
            "A matriz de confusao foi gerada e guardada como imagem para cada execucao.",
            styles["BodyText"],
        )
    )
    elems.append(Spacer(1, 12))

    for dataset in sorted(df["dataset"].unique()):
        elems.append(PageBreak())
        elems.append(Paragraph(f"Resultados - {dataset}", styles["Heading1"]))
        elems.append(Spacer(1, 6))

        for test_size in [0.15, 0.30, 0.50]:
            sub = df[(df["dataset"] == dataset) & (df["test_size"] == test_size)].copy()
            elems.append(Paragraph(f"Teste = {int(test_size*100)}%", styles["Heading2"]))

            piv_acc = _make_pivot(sub, "accuracy")
            elems.extend(_table_from_df(piv_acc, "Tabela: Accuracy (media por modelo)"))

            piv_prec = _make_pivot(sub, "precision_weighted")
            elems.extend(_table_from_df(piv_prec, "Tabela: Precision (weighted)"))

            # pick best run by accuracy and include its confusion matrix
            best = sub.sort_values(["accuracy", "precision_weighted"], ascending=False).iloc[0]
            cm_path = OUT_DIR / "confusion_matrices" / best["cm_image"]
            if cm_path.exists():
                elems.append(
                    Paragraph(
                        f"Melhor execucao (por accuracy): <b>{best['model']}</b> com <b>{best['preprocessing']}</b>. "
                        f"Accuracy={_fmt(best['accuracy'])}, Precision={_fmt(best['precision_weighted'])}.",
                        styles["BodyText"],
                    )
                )
                elems.append(Spacer(1, 6))
                elems.append(Image(str(cm_path), width=14 * cm, height=10 * cm))
                elems.append(Spacer(1, 12))

        # Critical analysis per dataset
        elems.append(Paragraph("Analise critica (sintese)", styles["Heading2"]))
        if dataset.lower().startswith("breast"):
            txt = (
                "O dataset Breast Cancer e relativamente equilibrado e contem apenas features numericas (30 variaveis). "
                "Nestas condicoes, algoritmos baseados em distancia (kNN) e modelos de conjunto (Random Forest) tendem a beneficiar "
                "da normalizacao (StandardScaler). Naive Bayes costuma ser um baseline forte, mas pode perder desempenho se as "
                "distribuicoes das features se afastarem da hipotese Gaussiana. Decision Tree pode ter variancia elevada, e por isso "
                "a Random Forest costuma melhorar a generalizacao. Quando o tamanho do teste aumenta (ex.: 50%), o treino fica menor "
                "e as diferencas entre modelos podem aumentar; modelos mais robustos (RF) tendem a manter melhor desempenho." 
            )
        else:
            txt = (
                "O dataset Heart Failure tem apenas 299 instancias e inclui variaveis binarias e continuas, com potencial desbalanceamento da classe "
                "(morte/nao-morte). Com poucos dados, o resultado fica mais sensivel a como o treino/teste e separado: ao aumentar o teste para 50%, "
                "o modelo treina com muito menos exemplos e a variancia aumenta. Decision Tree pode sobre-ajustar; Random Forest tende a ser mais estavel. "
                "kNN depende fortemente de escala, por isso a normalizacao costuma ser determinante. Como as features ja sao numericas/boleanas, "
                "o OneHotEncoder normalmente nao altera o desempenho (na pratica, nao ha colunas categoricas 'texto' a converter)."
            )
        elems.append(Paragraph(txt, styles["BodyText"]))

    elems.append(PageBreak())
    elems.append(Paragraph("Reprodutibilidade", styles["Heading2"]))
    elems.append(
        Paragraph(
            "Para reproduzir localmente: (1) instalar dependencias: numpy, pandas, scikit-learn, matplotlib, reportlab; "
            "(2) executar <b>python run_experiments.py</b>; (3) executar <b>python make_report.py</b>. "
            "Os resultados e matrizes de confusao serao guardados na pasta <b>outputs/</b>.",
            styles["BodyText"],
        )
    )

    doc.build(elems)


def main() -> None:
    pdf_out = ROOT / "Relatorio_ML.pdf"
    build_report(OUT_DIR / "results_long.csv", pdf_out)
    print(f"OK: {pdf_out}")


if __name__ == "__main__":
    main()
