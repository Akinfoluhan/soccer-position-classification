# Soccer Position Classification

A machine learning project that predicts a soccer player's primary field position (**Defender, Midfielder, or Forward**) based on their performance statistics.

## Overview

This project explores how player statistics can be used to classify playing positions. It compares several machine learning algorithms and evaluates their performance using cross-validation, hyperparameter tuning, and learning curves.

The project also explores cost-sensitive classification to investigate how different types of prediction errors affect model decisions.

## Models

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Dummy Classifier** (baseline)

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- Matplotlib

## Methodology

1. Clean and preprocess player statistics.
2. Exclude goalkeepers and extract each player's primary position.
3. Split data into training and test sets while preventing player overlap.
4. Train and tune models using 5-fold group-aware cross-validation.
5. Compare validation accuracy and analyze overfitting using learning curves.
6. Explore cost-sensitive classification and evaluate model performance.

## Current Results

| Model                   | Cross-Validation Accuracy |
| ----------------------- | ------------------------: |
| Logistic Regression     |                    80.75% |
| K-Nearest Neighbors     |                    72.02% |
| Decision Tree           |                    70.27% |
| Majority-Class Baseline |                    38.83% |

Logistic Regression currently achieves the highest cross-validation accuracy.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Akinfoluhan/soccer-position-classification.git
cd soccer-position-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python3 src/main.py
```

The script uses the dataset located at `data/soccer_players.csv`. Generated plots are saved in the `results/` directory.

## Project Status

Work in progress. Model comparison and learning-curve analysis are complete; cost-sensitive evaluation and final testing are in progress.
