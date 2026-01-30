# Avaliacao de Modelos de ML (Accuracy e Confusion Matrix)

Este projeto executa e compara 4 algoritmos de classificacao (Naive Bayes, Decision Tree,
Random Forest e kNN) em 2 datasets (Breast Cancer e Heart Failure), avaliando **accuracy**
(e exatidao) e **matriz de confusao** (confusion matrix).

## Datasets
- **Breast Cancer Wisconsin (Diagnostic)**: carregado via `sklearn.datasets.load_breast_cancer`.
- **Heart Failure Clinical Records**: obtido do repositorio UCI (CSV dentro de um ZIP).

## Como executar
Dentro da pasta do projeto:

```bash
python run_experiments.py
python make_report.py
```

## Saidas
- `outputs/results_long.csv`: resultados detalhados (1 linha por modelo).
- `outputs/results_summary.csv`: resumo (accuracy e precision ponderada) em formato de tabela.
- `outputs/confusion_matrices/*.png`: imagens das matrizes de confusao.
- `Relatorio_ML_Classification.pdf`: relatorio em PDF com metodologia e analise.

## Observacoes
- O script baixa automaticamente o dataset de Heart Failure do UCI.
- Os resultados sao sensiveis ao `random_state` e a hiperparametros; para garantir
  comparabilidade, usamos valores padrao e `random_state=42` sempre que aplicavel.
