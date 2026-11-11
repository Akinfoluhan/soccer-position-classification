from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_score,
    learning_curve,
    cross_val_predict
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    auc,
    RocCurveDisplay
)

DEBUG = True

def plot_tuning_curve(search, param_name, title, filename):
    results = pd.DataFrame(search.cv_results_)

    params = results[f"param_{param_name}"].tolist()

    # Handle None as a category for unrestricted tree depth
    labels = [str(p) for p in params]
    positions = range(len(params))

    plt.figure(figsize=(8, 5))

    plt.plot(
        positions,
        results["mean_train_score"],
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        positions,
        results["mean_test_score"],
        marker="o",
        label="Validation Accuracy"
    )

    best_param = search.best_params_[param_name]
    best_idx = params.index(best_param)

    plt.axvline(
        x=best_idx,
        linestyle="--",
        label=f"Selected: {best_param}"
    )

    plt.xticks(positions, labels, rotation=45)
    plt.xlabel(param_name.split("__")[-1])
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()

# Load the soccer players dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "soccer_players.csv"
df = pd.read_csv(DATA_DIR).copy()

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    "classifier__C": [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]
}

# Search for the best regularization value
log_reg_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=True
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
    n_jobs=-1,
    return_train_score=True
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
    n_jobs=-1,
    return_train_score=True
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


# Generate a group-aware learning curve
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

learning_sizes = []
learning_train_scores = []
learning_val_scores = []

for fraction in fractions:
    fold_sizes = []
    fold_train_scores = []
    fold_val_scores = []

    for train_idx, val_idx in cv.split(
        X_train, y_train, groups_train
    ):
        # Separate training and validation folds
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_groups = groups_train.iloc[train_idx]

        # Select whole player groups
        unique_players = fold_groups.unique()

        rng = np.random.default_rng(42)
        shuffled_players = rng.permutation(unique_players)

        n_players = max(1, int(len(shuffled_players) * fraction))
        selected_players = shuffled_players[:n_players]

        mask = fold_groups.isin(selected_players)

        X_subset = X_fold_train.loc[mask]
        y_subset = y_fold_train.loc[mask]

        # Train a fresh copy of the selected model
        from sklearn.base import clone

        model = clone(log_reg_search.best_estimator_)
        model.fit(X_subset, y_subset)

        fold_sizes.append(len(X_subset))
        fold_train_scores.append(model.score(X_subset, y_subset))
        fold_val_scores.append(model.score(X_fold_val, y_fold_val))

    learning_sizes.append(np.mean(fold_sizes))
    learning_train_scores.append(np.mean(fold_train_scores))
    learning_val_scores.append(np.mean(fold_val_scores))

print("\n--- Group-Aware Learning Curve ---")
print("Training sizes:", np.round(learning_sizes, 1))
print("Training accuracy:", np.round(learning_train_scores, 4))
print("Validation accuracy:", np.round(learning_val_scores, 4))

# Plot the learning curve
plt.figure(figsize=(8, 5))

plt.plot(
    learning_sizes,
    learning_train_scores,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    learning_sizes,
    learning_val_scores,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Number of Training Records")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Learning Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "logistic_regression_learning_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plot_tuning_curve(
    log_reg_search,
    "classifier__C",
    "Logistic Regression Hyperparameter Tuning",
    "logistic_regression_tuning.png"
)

plot_tuning_curve(
    knn_search,
    "classifier__n_neighbors",
    "KNN Hyperparameter Tuning",
    "knn_tuning.png"
)

plot_tuning_curve(
    tree_search,
    "classifier__max_depth",
    "Decision Tree Hyperparameter Tuning",
    "decision_tree_tuning.png"
)

# Position labels in a fixed order
classes = ["DF", "MF", "FW"]

# Rows = actual positions, columns = predicted positions
cost_matrix = np.array([
    [0, 2, 5],
    [2, 0, 2],
    [5, 2, 0]
])

print("\n--- Classification Cost Matrix ---")
print(pd.DataFrame(cost_matrix, index=classes, columns=classes))

def calculate_cost(y_true, y_pred, cost_matrix, classes):
    label_to_idx = {
        label: i for i, label in enumerate(classes)
    }

    total_cost = 0

    for actual, predicted in zip(y_true, y_pred):
        actual_idx = label_to_idx[actual]
        predicted_idx = label_to_idx[predicted]

        total_cost += cost_matrix[actual_idx, predicted_idx]

    return total_cost

def predict_min_cost(probabilities, cost_matrix, classes):
    # Calculate the expected cost of predicting each class
    expected_costs = probabilities @ cost_matrix

    # Select the prediction with the lowest expected cost
    predicted_indices = np.argmin(expected_costs, axis=1)

    return np.array(classes)[predicted_indices]

example_probs = np.array([
    [0.40, 0.35, 0.25]
])

normal_prediction = np.array(classes)[
    np.argmax(example_probs, axis=1)
]

cost_prediction = predict_min_cost(
    example_probs,
    cost_matrix,
    classes
)

print("Normal prediction:", normal_prediction)
print("Minimum-cost prediction:", cost_prediction)


# Generate group-aware out-of-fold probabilities
oof_probs = cross_val_predict(
    estimator=log_reg_search.best_estimator_,
    X=X_train,
    y=y_train,
    groups=groups_train,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)

# Align probability columns with our cost matrix
model_classes = log_reg_search.best_estimator_.classes_

column_order = [
    list(model_classes).index(label)
    for label in classes
]

oof_probs = oof_probs[:, column_order]

# Standard predictions: choose highest probability
standard_predictions = np.array(classes)[
    np.argmax(oof_probs, axis=1)
]

# Cost-sensitive predictions: choose lowest expected cost
cost_predictions = predict_min_cost(
    oof_probs,
    cost_matrix,
    classes
)

# Calculate realized classification costs
standard_cost = calculate_cost(
    y_train,
    standard_predictions,
    cost_matrix,
    classes
)

sensitive_cost = calculate_cost(
    y_train,
    cost_predictions,
    cost_matrix,
    classes
)

print("\n--- Out-of-Fold Cost Comparison ---")
print("Standard total cost:", standard_cost)
print("Cost-sensitive total cost:", sensitive_cost)
print("Standard average cost:", standard_cost / len(y_train))
print("Cost-sensitive average cost:", sensitive_cost / len(y_train))

# Compare accuracy
standard_accuracy = accuracy_score(
    y_train, standard_predictions
)

sensitive_accuracy = accuracy_score(
    y_train, cost_predictions
)

# Count predictions changed by cost-sensitive classification
changed_predictions = np.sum(
    standard_predictions != cost_predictions
)

print("\n--- Cost-Sensitive Decision Analysis ---")
print(f"Standard accuracy: {standard_accuracy:.4f}")
print(f"Cost-sensitive accuracy: {sensitive_accuracy:.4f}")
print("Predictions changed:", changed_predictions)

# Compare confusion matrices
print("\nStandard confusion matrix:")
print(
    pd.DataFrame(
        confusion_matrix(y_train, standard_predictions, labels=classes),
        index=classes,
        columns=classes
    )
)

print("\nCost-sensitive confusion matrix:")
print(
    pd.DataFrame(
        confusion_matrix(y_train, cost_predictions, labels=classes),
        index=classes,
        columns=classes
    )
)


# Compare standard and cost-sensitive decisions under each cost scenario
penalties = [2, 5, 10, 20]

sensitivity_results = []

for penalty in penalties:
    adjusted_cost_matrix = np.array([
        [0, 2, penalty],
        [2, 0, 2],
        [penalty, 2, 0]
    ])

    adjusted_predictions = predict_min_cost(
        oof_probs,
        adjusted_cost_matrix,
        classes
    )

    # Evaluate both rules under the same cost matrix
    standard_total = calculate_cost(
        y_train,
        standard_predictions,
        adjusted_cost_matrix,
        classes
    )

    sensitive_total = calculate_cost(
        y_train,
        adjusted_predictions,
        adjusted_cost_matrix,
        classes
    )

    # Count expensive mistakes and midfielder predictions
    adjusted_cm = confusion_matrix(
        y_train,
        adjusted_predictions,
        labels=classes
    )

    df_fw_errors = adjusted_cm[0, 2] + adjusted_cm[2, 0]
    mf_predictions = np.sum(adjusted_predictions == "MF")

    sensitivity_results.append({
        "DF-FW Penalty": penalty,
        "Accuracy": accuracy_score(y_train, adjusted_predictions),
        "Standard Cost": standard_total,
        "Sensitive Cost": sensitive_total,
        "Cost Savings": standard_total - sensitive_total,
        "DF-FW Errors": df_fw_errors,
        "MF Predictions": mf_predictions
    })

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\n--- Cost Sensitivity Comparison ---")
print(sensitivity_df.to_string(index=False))

# Threshold analysis: flag uncertain predictions for human review

# Use the standard Logistic Regression predictions
confidence = np.max(oof_probs, axis=1)

# Identify whether each prediction was incorrect
actual_errors = (
    standard_predictions != y_train.to_numpy()
)

# Hypothetical deployment costs
review_cost = 1
missed_error_cost = 5

thresholds = np.linspace(0, 1, 101)
threshold_results = []

for threshold in thresholds:
    # Flag predictions with confidence below threshold
    flagged = confidence < threshold

    # Every flagged prediction requires human review
    review_expense = np.sum(flagged) * review_cost

    # Incorrect predictions that weren't flagged incur a penalty
    missed_errors = np.sum(~flagged & actual_errors)
    missed_expense = missed_errors * missed_error_cost

    total_cost = review_expense + missed_expense

    threshold_results.append({
        "Threshold": threshold,
        "Flagged": np.sum(flagged),
        "Missed Errors": missed_errors,
        "Total Cost": total_cost
    })

threshold_df = pd.DataFrame(threshold_results)

# Select the threshold with the lowest development cost
best_row = threshold_df.loc[
    threshold_df["Total Cost"].idxmin()
]

best_threshold = float(best_row["Threshold"])

print("\n--- Threshold Optimization ---")
print("Best threshold:", best_threshold)
print("Predictions flagged:", int(best_row["Flagged"]))
print("Errors not flagged:", int(best_row["Missed Errors"]))
print("Minimum total cost:", int(best_row["Total Cost"]))

# Compare with a conventional 50% threshold
print("\nThreshold comparison:")
print(
    threshold_df[
        threshold_df["Threshold"].round(2).isin(
            [0.0, 0.5, best_threshold, 1.0]
        )
    ].to_string(index=False)
)

# Plot review cost against confidence threshold
plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Total Cost"],
    label="Total Simulated Cost"
)

plt.axvline(
    best_threshold,
    color="green",
    linestyle="--",
    label=f"Selected Threshold = {best_threshold:.2f}"
)

plt.axvline(
    0.5,
    color="orange",
    linestyle=":",
    label="Reference Threshold = 0.50"
)

plt.xlabel("Confidence Threshold")
plt.ylabel("Total Simulated Cost")
plt.title("Cost vs. Human Review Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "review_threshold_cost.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Evaluate the selected review policy
flagged = confidence < best_threshold

# Treat an incorrect classification as a review-worthy error
true_positives = np.sum(flagged & actual_errors)
false_positives = np.sum(flagged & ~actual_errors)
false_negatives = np.sum(~flagged & actual_errors)
true_negatives = np.sum(~flagged & ~actual_errors)

print("\n--- Review Policy Outcomes ---")
print("TP: Incorrect predictions flagged:", true_positives)
print("FP: Correct predictions flagged:", false_positives)
print("FN: Incorrect predictions missed:", false_negatives)
print("TN: Correct predictions accepted:", true_negatives)

# Final evaluation on the held-out test set
final_model = log_reg_search.best_estimator_

# Standard class predictions
test_predictions = final_model.predict(X_test)

# Predicted probabilities
test_probs = final_model.predict_proba(X_test)

# Align probability columns with the cost matrix
test_column_order = [
    list(final_model.classes_).index(label)
    for label in classes
]

test_probs = test_probs[:, test_column_order]

print("\n--- Final Test Evaluation ---")
print("Test accuracy:", accuracy_score(y_test, test_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=classes,
        zero_division=0
    )
)

print("\nTest Confusion Matrix:")
test_cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=classes
)

print(
    pd.DataFrame(
        test_cm,
        index=classes,
        columns=classes
    )
)

# Visualize the test confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm,
    display_labels=classes
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Logistic Regression: Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# Calculate multiclass ROC-AUC using sorted class labels
auc_classes = sorted(classes)

auc_column_order = [
    classes.index(label)
    for label in auc_classes
]

auc_probs = test_probs[:, auc_column_order]

test_auc = roc_auc_score(
    y_test,
    auc_probs,
    labels=auc_classes,
    multi_class="ovr",
    average="macro"
)

print(f"\nTest ROC-AUC (macro OvR): {test_auc:.4f}")

# Apply cost-sensitive classification to test probabilities
test_cost_predictions = predict_min_cost(
    test_probs,
    cost_matrix,
    classes
)

standard_test_cost = calculate_cost(
    y_test,
    test_predictions,
    cost_matrix,
    classes
)

sensitive_test_cost = calculate_cost(
    y_test,
    test_cost_predictions,
    cost_matrix,
    classes
)

print("\n--- Final Test Cost Comparison ---")
print("Standard cost:", standard_test_cost)
print("Cost-sensitive cost:", sensitive_test_cost)

print(
    "Cost-sensitive accuracy:",
    accuracy_score(y_test, test_cost_predictions)
)

# Apply the previously selected review threshold
test_confidence = np.max(test_probs, axis=1)
test_flagged = test_confidence < best_threshold

test_errors = (
    test_predictions != y_test.to_numpy()
)

test_tp = np.sum(test_flagged & test_errors)
test_fp = np.sum(test_flagged & ~test_errors)
test_fn = np.sum(~test_flagged & test_errors)
test_tn = np.sum(~test_flagged & ~test_errors)

test_review_cost = (
    np.sum(test_flagged) * review_cost
    + test_fn * missed_error_cost
)

print("\n--- Final Test Review Policy ---")
print("Threshold:", best_threshold)
print("TP: Incorrect predictions flagged:", test_tp)
print("FP: Correct predictions flagged:", test_fp)
print("FN: Incorrect predictions missed:", test_fn)
print("TN: Correct predictions accepted:", test_tn)
print("Total simulated review cost:", test_review_cost)
print("Never-review cost:", np.sum(test_errors) * missed_error_cost)
print("Always-review cost:", len(y_test) * review_cost)

# Plot one-vs-rest ROC curves for each position
plt.figure(figsize=(8, 5))

for i, position in enumerate(classes):
    # Convert the multiclass labels into a binary target
    binary_labels = (y_test.to_numpy() == position).astype(int)

    # Probability of belonging to this position
    position_probs = test_probs[:, i]

    # Calculate ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(binary_labels, position_probs)
    position_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{position} (AUC = {position_auc:.3f})"
    )

# Reference line: random classification
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression: Test ROC Curves (One-vs-Rest)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "test_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()
