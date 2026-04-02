from pathlib import Path
import pandas as pd


def load_raw_data():

    data_path = Path("data/raw/diabetic_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            "Dataset not found. Place diabetic_data.csv in data/raw/"
        )

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully")
    print("Shape:", df.shape)

    return df


def basic_validation(df):

    print("\nBasic dataset info")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nPreview:")
    print(df.head())

    return df


if __name__ == "__main__":

    df = load_raw_data()

    df = basic_validation(df)