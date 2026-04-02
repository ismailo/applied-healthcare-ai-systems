from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split

from src.data_cleaning import clean_healthcare_data


def build_healthcare_features():
    """
    Build model-ready features for healthcare readmission prediction.
    """

    # ---------------------------------
    # Step 1: Load cleaned dataset
    # ---------------------------------
    df = clean_healthcare_data()

    print("Cleaned dataset shape:", df.shape)

    # ---------------------------------
    # Step 2: Separate target
    # ---------------------------------
    target_col = "readmitted_binary"

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    print("Feature matrix shape before encoding:", X.shape)
    print("Target shape:", y.shape)

    # ---------------------------------
    # Step 3: Identify categorical columns
    # ---------------------------------
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    print("Categorical columns:", len(categorical_cols))

    # ---------------------------------
    # Step 4: One-hot encode categorical features
    # ---------------------------------
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print("Feature matrix shape after encoding:", X.shape)

    # ---------------------------------
    # Step 5: Train/test split
    # ---------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train/test split complete")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    # ---------------------------------
    # Step 6: Save processed datasets
    # ---------------------------------
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    print("Processed training and test files saved to data/processed/")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = build_healthcare_features()

    print("\nFeature engineering step complete")