from utils import (
    load_data,
    preprocess_data,
    create_eda
)

from model import train_models


def main():

    print("=" * 60)
    print("NUMBER OF ORDERS PREDICTION")
    print("=" * 60)

    df = load_data()

    processed = preprocess_data(df)

    create_eda(processed)

    results = train_models(processed)

    print("\nModel Performance\n")

    for name, metrics in results.items():

        print("-" * 30)

        print(name)

        print(f"MAE  : {metrics['MAE']:.3f}")

        print(f"RMSE : {metrics['RMSE']:.3f}")

        print(f"R²   : {metrics['R2']:.3f}")

    print("\nProject Completed Successfully!")
    print("Outputs saved inside the outputs folder.")


if __name__ == "__main__":
    main()