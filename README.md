# Soccer Position Classification Using Machine Learning

A machine learning project that predicts soccer players' primary field positions—**Defender (DF), Midfielder (MF), or Forward (FW)**—using performance statistics from the 2024–25 season.

The project compares multiple classification algorithms, investigates their generalization performance, and explores cost-sensitive decisions and confidence-based human review.

## Project Overview

Can a player's statistical performance accurately predict their primary playing position?

This project investigates that question using supervised machine learning. Beyond classification accuracy, it examines which positions are hardest to distinguish, how model complexity affects performance, and how different error costs influence predictions.

A potential application is supporting scouting and player-development analysis by highlighting players whose statistical profiles warrant further investigation. The model predicts recorded positions, **not necessarily a player's optimal position**.

## Dataset and Preprocessing

The dataset contains performance statistics for players in major European soccer leagues during the 2024–25 season.

- **Original dataset:** 2,854 records
- **After excluding goalkeepers:** 2,642 records
- **Target classes:** DF, MF, FW
- **Selected features:** 19 numerical attributes covering playing time, goals, assists, expected goals, passing, progression, defensive actions, and aerial duels
- **Training set:** 2,109 records
- **Held-out test set:** 533 records

Preprocessing includes median imputation for missing numerical values and feature standardization for Logistic Regression and KNN.

To reduce data leakage, the project uses player-group-aware splitting and cross-validation, preventing records sharing a player name from appearing in both training and validation sets.

## Models and Results

Four classifiers were evaluated using five-fold stratified group-aware cross-validation.

| Model               | Best Hyperparameter | Mean CV Accuracy |
| ------------------- | ------------------- | ---------------: |
| Logistic Regression | C = 10,000          |       **80.75%** |
| K-Nearest Neighbors | k = 51              |           72.02% |
| Decision Tree       | max_depth = 7       |           70.27% |
| Dummy Classifier    | Most frequent class |           38.83% |

**Logistic Regression was selected** because it achieved the highest mean cross-validation accuracy.

### Final Test Performance

The selected model was evaluated once on the held-out test set.

| Metric                    |     Result |
| ------------------------- | ---------: |
| Test accuracy             | **76.17%** |
| Macro precision           |       0.75 |
| Macro recall              |       0.76 |
| Macro F1-score            |       0.76 |
| Macro one-vs-rest ROC-AUC | **0.9208** |

Defenders were classified most reliably, while midfielders had the lowest recall (66%), suggesting greater statistical overlap with other playing roles.

## Model Diagnostics

Hyperparameter tuning and learning curves were used to examine bias, variance, and generalization.

The Decision Tree showed clear overfitting at larger depths, while Logistic Regression exhibited a relatively small training-validation gap. A group-aware learning curve showed validation accuracy approaching approximately 81% as additional training records were included.

Figures are generated automatically in the `results/` directory.

## Cost-Sensitive Classification

A hypothetical classification cost matrix was introduced to penalize defender–forward confusion more heavily than errors involving midfielders.

Under the selected cost assumptions:

| Decision Rule                    | Test Accuracy | Test Classification Cost |
| -------------------------------- | ------------: | -----------------------: |
| Standard prediction              |        76.17% |                      311 |
| Minimum expected-cost prediction |        76.17% |                      272 |

The cost-sensitive decision rule reduced the hypothetical test classification penalty by **12.54%** without changing test accuracy.

A separate confidence-based review policy flags uncertain predictions for human verification.

- **Selected confidence threshold:** 0.71
- **Test records flagged:** 207 of 533
- **Incorrect predictions caught:** 94 of 127
- **Simulated test review cost:** 372 units
- **Never-review baseline:** 635 units

Thresholds were selected using out-of-fold training predictions, not the test set. Sensitivity analysis also examined missed-error-to-review-cost ratios of 5:1, 10:1, and 20:1.

All costs are hypothetical and intended to illustrate decision-making trade-offs rather than established financial or scouting outcomes.

## Project Structure

```text
soccer-position-classification/
├── data/
│   └── soccer_players.csv
├── src/
│   └── main.py
├── results/
│   └── *.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Usage

**Requirements:** Python 3.14 and the dependencies listed in `requirements.txt`.

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

The script prints model comparisons, cross-validation results, classification metrics, and cost analyses. It also saves diagnostic and evaluation figures to the `results/` directory.

Set `DEBUG = True` in `src/main.py` to display additional dataset and hyperparameter information.

## Limitations and Future Work

- **Recorded position vs. optimal position:** A classification mismatch does not prove that a player should play another position.
- **Player identity:** Grouping by player name reduces leakage but may incorrectly combine different players with identical names.
- **Data scope:** Results may not generalize to other seasons, leagues, or tactical systems.
- **Feature representation:** Season-level statistics may not capture formations, tactical instructions, or contextual playing roles.
- **Review costs:** The human-review simulation assumes reviewed errors can be resolved and uses illustrative costs rather than observed outcomes.

Future work could incorporate per-90 statistics, additional seasons, richer tactical features, probability calibration, and independent expert labels for player-review decisions.

## Technologies

Python, NumPy, pandas, scikit-learn, and Matplotlib.
