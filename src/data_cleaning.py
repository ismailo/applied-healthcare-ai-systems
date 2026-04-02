from pathlib import Path
import pandas as pd

from src.data_loader import load_raw_data


def clean_healthcare_data():
    """
    Load and clean the diabetes hospital dataset.
    """

    # -----------------------------
    # Step 1: Load raw data
    # -----------------------------
    df = load_raw_data()

    print("Initial shape:", df.shape)

    # -----------------------------
    # Step 2: Replace '?' with missing values
    # -----------------------------
    df = df.replace("?", pd.NA)

    # -----------------------------
    # Step 3: Drop ID columns
    # -----------------------------
    id_columns = ["encounter_id", "patient_nbr"]

    existing_id_columns = [col for col in id_columns if col in df.columns]
    df = df.drop(columns=existing_id_columns)

    print("Dropped ID columns:", existing_id_columns)

    # -----------------------------
    # Step 4: Drop columns with too many missing values
    # -----------------------------
    high_missing_cols = ["weight", "payer_code", "medical_specialty"]

    existing_missing_cols = [col for col in high_missing_cols if col in df.columns]
    df = df.drop(columns=existing_missing_cols)

    print("Dropped high-missing columns:", existing_missing_cols)

    # -----------------------------
    # Step 5: Remove rows with invalid gender
    # -----------------------------
    if "gender" in df.columns:
        df = df[df["gender"] != "Unknown/Invalid"]

    # -----------------------------
    # Step 6: Convert target column
    # readmitted: NO, >30, <30
    # target = 1 if <30 else 0
    # -----------------------------
    if "readmitted" not in df.columns:
        raise ValueError("Target column 'readmitted' not found.")

    df["readmitted_binary"] = df["readmitted"].apply(
        lambda x: 1 if x == "<30" else 0
    )

    # Drop original target column
    df = df.drop(columns=["readmitted"])

    # -----------------------------
    # Step 7: Remove duplicate rows
    # -----------------------------
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]

    print(f"Removed {before - after} duplicate rows")

    # -----------------------------
    # Step 8: Save cleaned dataset
    # -----------------------------
    output_path = Path("data/processed")
    output_path.mkdir(parents=True, exist_ok=True)

    cleaned_file = output_path / "clean_diabetic_data.csv"
    df.to_csv(cleaned_file, index=False)

    print("Cleaned dataset saved to:", cleaned_file)
    print("Final shape:", df.shape)

    return df


if __name__ == "__main__":
    df = clean_healthcare_data()

    print("\nPreview cleaned data:")
    print(df.head())