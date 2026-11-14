# Soccer Position Classification Using Machine Learning

A machine learning project that predicts soccer players' recorded primary field positions—**Defender (DF), Midfielder (MF), or Forward (FW)**—using performance statistics from the 2024–25 season.

The project compares classification algorithms, evaluates generalization, and explores cost-sensitive predictions and confidence-based human review.

## Project Overview

**Research question:** Can a player's statistical performance predict their primary playing position?

A potential use is supporting scouting and player-development analysis by identifying unusual statistical profiles for further investigation. The classifier predicts a player's _recorded position_, not necessarily their optimal position.

## Dataset and Preprocessing

**Source:** [Football Players Stats (2024–2025)](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025), published on Kaggle by **Hubert Sidorowicz**. The dataset is described as derived from [FBref](https://fbref.com/) and covers the five major European leagues during the 2024–25 season. The project uses the CSV included in `data/`.

- Original dataset: **2,854 records and 165 columns**
- After excluding goalkeepers: **2,642 records** (1,022 DF, 900 MF, 720 FW)
- Target: first listed position for players with multiple listed roles
- Features: **19 numerical attributes** describing age, playing time, goals, assists, expected goals, shooting, progression, passing, defending, recoveries, and aerial duels
- Training: **2,109 records**; held-out test: **533 records**

Median imputation handles missing values within each training fold; Logistic Regression and KNN also use standardization inside their pipelines. Player-name grouping is used for the train/test split and five-fold stratified group-aware cross-validation to reduce leakage across multiple club records for the same player.

## Model Comparison

| Model                   | Selected setting    | Mean 5-fold CV accuracy |
| ----------------------- | ------------------- | ----------------------: |
| **Logistic Regression** | `C = 10000`         |              **80.75%** |
| K-Nearest Neighbors     | `k = 51`            |                  72.02% |
| Decision Tree           | `max_depth = 7`     |                  70.27% |
| Dummy baseline          | Most frequent class |                  38.83% |

Logistic Regression achieved the highest mean cross-validation accuracy. Hyperparameter tuning curves and a group-aware learning curve are saved in `results/`.

## Final Held-Out Test Results

| Metric                    |                Value |
| ------------------------- | -------------------: |
| Accuracy                  | **76.17%** (406/533) |
| Macro precision           |                 0.75 |
| Macro recall              |                 0.76 |
| Macro F1                  |                 0.76 |
| Macro one-vs-rest ROC-AUC |           **0.9208** |

Defenders were the easiest group to identify by recall (**86%**); midfielders were the hardest (**66%** recall).

## Cost-Sensitive Classification and Human Review

The project distinguishes **multiclass position classification** from a separate **binary human-review decision**.

A hypothetical position-error cost matrix assigns 2 units to errors involving midfielders and 5 units to DF–FW confusion. On the test set, minimum-expected-cost classification reduced the total penalty from **311 to 272 units (12.54%)**, with unchanged accuracy of **76.17%**.

For the review policy, predictions with maximum probability below a confidence threshold are flagged for verification. A **0.71** threshold was selected using out-of-fold training predictions under hypothetical costs of 1 unit per review and 5 units per missed error. On the held-out test set, this policy flagged **207/533** predictions, captured **94/127** classification errors, and incurred a simulated cost of **372** units versus **635** when never reviewing and **533** when always reviewing.

Training-set sensitivity analysis also tested missed-error-to-review-cost ratios of **5:1, 10:1, and 20:1**, selecting confidence thresholds of **0.71, 0.91, and 0.98**, respectively.

**These are illustrative costs, not measured financial values.** The simulation assumes that human review resolves flagged classification errors; it does not establish which players ought to change position.

## Repository Structure

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

## Setup and Usage

Use **Python 3.14** with the pinned dependencies in `requirements.txt`.

```bash
git clone https://github.com/Akinfoluhan/soccer-position-classification.git
cd soccer-position-classification
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```

The script prints model-comparison and evaluation outputs and saves plots to `results/`. Set `DEBUG = True` in `src/main.py` for additional intermediate diagnostics.

## Limitations and Future Work

- A recorded playing position is not an independently verified label for a player's _best_ position.
- Grouping by player names is an imperfect substitute for unique player identifiers.
- Results may not transfer to other seasons, leagues, or tactical systems.
- Aggregate season statistics omit tactical assignments and some match context.
- Probability calibration and real coach-reviewed outcomes would be needed before operational use.

Potential extensions include per-90 features, multiple seasons, richer tactical context, probability calibration, and independent expert labels for human-review value.

## Technologies

Python, pandas, NumPy, scikit-learn, and Matplotlib.

## Data Attribution

Sidorowicz, H. (n.d.). _Football Players Stats (2024–2025)_ [Dataset]. Kaggle. https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025

Statistical data are described as FBref-derived. See [FBref](https://fbref.com/) for the underlying football-statistics source.
