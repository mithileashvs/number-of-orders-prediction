import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_data():
    df = pd.read_csv(
        "dataset/Sample - Superstore.csv",
        encoding="latin1"
    )
    return df


def preprocess_data(df):

    df["Order Date"] = pd.to_datetime(df["Order Date"])

    daily = (
        df.groupby("Order Date")
        .agg(
            Orders=("Order ID", "nunique"),
            Sales=("Sales", "sum"),
            Quantity=("Quantity", "sum"),
            Discount=("Discount", "mean"),
            Profit=("Profit", "sum"),
        )
        .reset_index()
    )

    daily["Day"] = daily["Order Date"].dt.day
    daily["Month"] = daily["Order Date"].dt.month
    daily["Year"] = daily["Order Date"].dt.year
    daily["Weekday"] = daily["Order Date"].dt.weekday

    return daily


def create_eda(df):

    os.makedirs("outputs", exist_ok=True)

    # Missing values
    print("\nMissing Values:\n")
    print(df.isnull().sum())

    # Correlation Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(df.drop(columns=["Order Date"]).corr(),
                annot=True,
                cmap="Blues")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    # Orders over time
    plt.figure(figsize=(12,5))
    plt.plot(df["Order Date"], df["Orders"])
    plt.title("Orders Over Time")
    plt.tight_layout()
    plt.savefig("outputs/orders_over_time.png")
    plt.close()

    # Monthly Orders
    monthly = df.groupby("Month")["Orders"].mean()

    plt.figure(figsize=(8,5))
    monthly.plot(kind="bar")
    plt.title("Average Monthly Orders")
    plt.tight_layout()
    plt.savefig(
    "outputs/monthly_orders.png",
    dpi=300,
    bbox_inches="tight"
)
    plt.close()