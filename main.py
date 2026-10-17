import pandas as pd

# Load the soccer players dataset
df = pd.read_csv('soccer_players.csv')

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