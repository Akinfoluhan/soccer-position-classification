from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DEBUG = False

# Load the soccer players dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soccer_players.csv"
df = pd.read_csv(DATA_DIR).copy()

# Explore the original dataset
if DEBUG:
    print("\n--- Dataset Exploration ---")
    print(df.head())
    print("Dataset shape:", df.shape)
    print(df.dtypes)

    missing_values = df.isnull().sum()
    print("Missing values:")
    print(missing_values[missing_values > 0])

    print("Original position distribution:")
    print(df["Pos"].value_counts())


# Extract primary position and remove goalkeepers
df["Primary_Pos"] = df["Pos"].str.split(",").str[0]
df = df[df["Primary_Pos"] != "GK"].copy()

print("\n--- Cleaned Dataset ---")
print("Total records:", len(df))
print("Position distribution:")
print(df["Primary_Pos"].value_counts())

# Inspect missing values and retained columns when debugging
if DEBUG:
    print("\nMissing values after removing goalkeepers:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

# Remove columns containing only missing values
df = df.dropna(axis=1, how="all")

if DEBUG:
    print("Remaining columns:", df.shape[1])

# Select features for our model
features = [
    "Age",
    "90s",
    "Gls",
    "Ast",
    "xG",
    "xAG",
    "Sh",
    "SoT",
    "PrgC",
    "PrgP",
    "PrgR",
    "Cmp%",
    "Tkl",
    "Int",
    "Blocks_stats_defense",
    "Clr",
    "Recov",
    "Won",
    "Lost"
]

# Check whether all selected features exist
missing_features = [col for col in features if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

# Identify repeated player names
player_counts = df["Player"].value_counts()
repeated_players = player_counts[player_counts > 1]

print("\nRepeated player names:", len(repeated_players))

if DEBUG:
    print("\nFirst 20 repeated names:")
    print(repeated_players.head(20))

    print("\nExample of a player appearing for multiple clubs:")
    print(
        df[df["Player"] == "Max Aarons"][
            ["Player", "Squad", "Comp", "Pos", "MP", "Gls", "Ast"]
        ]
    )

# Define input features and target variable
X = df[features]
y = df["Primary_Pos"]

# Identify groups using player names
groups = df["Player"]

# Create an 80/20 group-aware split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

groups_train = groups.iloc[train_idx]

print("Training records:", len(X_train))
print("Test records:", len(X_test))

# Examine class distributions
print("Training class distribution:")
print(y_train.value_counts(normalize=True) * 100)

print("Test class distribution:")
print(y_test.value_counts(normalize=True) * 100)

# Verify that no player appears in both sets
train_players = set(groups.iloc[train_idx])
test_players = set(groups.iloc[test_idx])

overlap = train_players.intersection(test_players)

# Stop execution if any players appear in both sets
assert len(overlap) == 0, "Player overlap detected!"

print("Players in both sets:", len(overlap))

# Create a dummy classifier that predicts the most frequent class
baseline = DummyClassifier(strategy="most_frequent")

# Use stratified k-fold cross-validation to evaluate the baseline model
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

baseline_scores = cross_val_score(
    baseline,
    X_train,
    y_train,
    groups=groups_train,
    cv=cv,
    scoring="accuracy"
)

print("\n--- Baseline Model ---")
print("Mean CV accuracy:", baseline_scores.mean())
print("CV standard deviation:", baseline_scores.std())

if DEBUG:
    print("Individual fold scores:", baseline_scores)


log_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

# Values of C we want to test
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

# Search for the best regularization value
log_reg_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

# Train and evaluate using group-aware cross-validation
log_reg_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\n--- Logistic Regression ---")
print("Best C:", log_reg_search.best_params_)
print("Best CV accuracy:", log_reg_search.best_score_)

if DEBUG:
    cv_results = pd.DataFrame(log_reg_search.cv_results_)
    print(
        cv_results[
            [
                "param_classifier__C",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ]
    )


# Build the KNN pipeline
knn = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])

# Hyperparameter values to test
knn_param_grid = {
    "classifier__n_neighbors": [3, 5, 7, 11, 15, 21, 31, 51, 71, 101, 151, 201]
}

# Tune KNN using group-aware cross-validation
knn_search = GridSearchCV(
    estimator=knn,
    param_grid=knn_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

knn_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\n--- KNN ---")
print("Best K:", knn_search.best_params_)
print("Best CV accuracy:", knn_search.best_score_)

if DEBUG:
    knn_results = pd.DataFrame(knn_search.cv_results_)
    print(
        knn_results[
            [
                "param_classifier__n_neighbors",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ]
    )

# Create the Decision Tree pipeline
tree = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

# Hyperparameters to tune
tree_param_grid = {
    "classifier__max_depth": [2, 3, 5, 7, 10, 15, None]
}

# Perform group-aware cross-validation
tree_search = GridSearchCV(
    estimator=tree,
    param_grid=tree_param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

tree_search.fit(
    X_train,
    y_train,
    groups=groups_train
)

print("\n--- Decision Tree ---")
print("Best depth:", tree_search.best_params_)
print("Best CV accuracy:", tree_search.best_score_)

if DEBUG:
    tree_results = pd.DataFrame(tree_search.cv_results_)
    print(
        tree_results[
            [
                "param_classifier__max_depth",
                "mean_test_score",
                "std_test_score",
                "rank_test_score"
            ]
        ]
    )

# Summarize cross-validation results
model_results = pd.DataFrame({
    "Model": [
        "Baseline",
        "Logistic Regression",
        "KNN",
        "Decision Tree"
    ],
    "CV Accuracy": [
        baseline_scores.mean(),
        log_reg_search.best_score_,
        knn_search.best_score_,
        tree_search.best_score_
    ],
    "Best Parameter": [
        "Most frequent",
        str(log_reg_search.best_params_),
        str(knn_search.best_params_),
        str(tree_search.best_params_)
    ]
})

# Sort by highest validation accuracy
model_results = model_results.sort_values(
    by="CV Accuracy",
    ascending=False
)

print("\nFinal Model Comparison:")
print(model_results.to_string(index=False))
