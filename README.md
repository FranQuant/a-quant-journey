# Forecasting Asset Returns Using Machine Learning and Artificial Intelligence

![Python](https://img.shields.io/badge/Python-3.11.15-465766.svg) ![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-465766.svg) ![pandas](https://img.shields.io/badge/pandas-2.2.3-465766.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-465766.svg) [![MIT License](https://img.shields.io/badge/License-MIT-465766.svg)](LICENSE)

![AI in Finance conference artwork](Images/Presentation.png)

### Conference Presentation

- **Title:** A Quant Journey
- **Presenter:** J. Francisco Salazar
- **Event:** AI in Finance: Navigating the Future — Inaugural Artificial Intelligence in Finance Conference
- **Date:** April 26–27, 2024

- 📄 [Original Conference Slides (PDF)](presentations/AI%20in%20Finance-%20Navigating%20the%20Future.pdf)
- ▶️ [Original Conference Recording (YouTube)](https://www.youtube.com/watch?v=viCfp-AWUvQ)

## Project Overview

*A Quant Journey* explores machine learning for stock market forecasting using a decade of daily financial data. Framing directional price movements as a binary classification problem, the original research compares linear, tree-based, and other machine-learning models through historical backtesting against an SPY benchmark.

The project examines the predictive potential of random forests alongside the challenges of overfitting, transaction costs, and execution assumptions, while identifying deep learning and causal inference as avenues for further research.

![Historical strategy backtests from the 2024 conference presentation](Images/Algo_Trade_Backt.png)

## Research Notebooks

The research workflow is organized into four sequential teaching notebooks.

| Notebook | Research Focus |
| --- | --- |
| [01 · Understand the Data](notebooks/teaching/01_understand_the_data.ipynb) | Market-data exploration, returns, volatility and correlations. |
| [02 · Construct Candidate Features](notebooks/teaching/02_construct_candidate_features.ipynb) | Technical indicators, PCA, clustering and feature construction. |
| [03 · Test Predictive Value](notebooks/teaching/03_test_predictive_value.ipynb) | Seven classifiers, chronological validation and majority-class benchmark. |
| [04 · From Predictions to Portfolio Accounting](notebooks/teaching/04_predictions_to_portfolio_accounting.ipynb) | SPY portfolio accounting, transaction costs and benchmark comparisons. |

## Getting Started

Clone the repository and reproduce the teaching workflow using Python 3.11.

```bash
git clone https://github.com/FranQuant/a-quant-journey.git
cd a-quant-journey

python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name a-quant-journey
.venv/bin/python scripts/execute_teaching_notebooks.py
```

The four notebooks run sequentially and save their outputs locally. See the [Teaching Guide](notebooks/teaching/README.md) for methodological details and additional instructions.

## License

This project's source code is available under the [MIT License](LICENSE). Third-party data and conference materials may be subject to separate terms.

## Disclaimer

This repository is provided for educational and research purposes only and does not constitute investment advice. Historical results and hypothetical backtests do not guarantee future performance.
