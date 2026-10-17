import pandas as pd
from pathlib import Path

# Load the soccer players dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soccer_players.csv"
df = pd.read_csv(DATA_DIR)

# Display the first few rows of the dataset
print(df.head())

# Display the number of rows and columns in the dataset
print(df.shape)

# Display the data types of each column
print(df.dtypes)

# Count missing values in each column
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

# Examine all unique positions in the dataset
unique_positions = df['Pos'].value_counts()
print("Unique positions and their counts:")
print(unique_positions)


# Extract the primary position
df["Primary_Pos"] = df["Pos"].str.split(",").str[0]

# Remove goalkeepers
df = df[df["Primary_Pos"] != "GK"].copy()

# Check our cleaned positions
print(df["Primary_Pos"].value_counts())

# Check for missing values in the cleaned dataset
print(df.isnull().sum()[df.isnull().sum() > 0])

# Remove columns where every value is missing
df = df.dropna(axis=1, how="all")

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

print("Missing feature columns:", missing_features)

print(df.columns.tolist())

# Count how many times each player appears
player_counts = df["Player"].value_counts()

# Find players who appear more than once
repeated_players = player_counts[player_counts > 1]

print("Number of repeated player names:", len(repeated_players))
print(repeated_players.head(20))

# Inspect records belonging to one repeated player
print(
    df[df["Player"] == "Max Aarons"][
        ["Player", "Squad", "Comp", "Pos", "MP", "Gls", "Ast"]
    ]
)

from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

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

# Verify that no player appears in both sets
train_players = set(groups.iloc[train_idx])
test_players = set(groups.iloc[test_idx])

overlap = train_players.intersection(test_players)

print("Players in both sets:", len(overlap))

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

print("Baseline fold scores:", baseline_scores)
print("Baseline mean accuracy:", baseline_scores.mean())
print("Baseline standard deviation:", baseline_scores.std())


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

log_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000))
])

from sklearn.model_selection import GridSearchCV

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

print("Best C:", log_reg_search.best_params_)
print("Best CV accuracy:", log_reg_search.best_score_)

# Convert cross-validation results to a DataFrame
cv_results = pd.DataFrame(log_reg_search.cv_results_)

# Display the regularization values and their scores
print(
    cv_results[
        ["param_classifier__C",
         "mean_test_score",
         "std_test_score",
         "rank_test_score"]
    ]
)
