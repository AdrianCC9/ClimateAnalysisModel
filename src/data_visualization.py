# data_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from data_preprocessing import preprocess_data

def visualize_data(df):
    """
    Takes a cleaned DataFrame and creates some basic plots.
    """

    if df.empty:
        print("DataFrame is empty. No data to visualize.")
        return

    # 1. Histogram of the target variable (tas)
    plt.figure(figsize=(8, 5))
    sns.histplot(df['tas'].dropna(), kde=True)
    plt.title("Distribution of Mean Daily Temperatures (tas)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Count")
    plt.show()

    # 2. Relationship between Day of Year and Temperature
    #    (If your dataset is large, consider sampling to reduce clutter.)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='day_of_year', y='tas', alpha=0.3)
    plt.title("Daily Mean Temperature vs. Day of Year")
    plt.xlabel("Day of Year")
    plt.ylabel("Mean Temperature (°C)")
    plt.show()

    # 3. Relationship between Year and Average Temperature
    #    Group by year and plot the mean tas per year.
    yearly_avg = df.groupby('year')['tas'].mean().reset_index()
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=yearly_avg, x='year', y='tas')
    plt.title("Average Temperature by Year")
    plt.xlabel("Year")
    plt.ylabel("Mean Temperature (°C)")
    plt.show()

if __name__ == "__main__":
    # 1. Load the cleaned data
    csv_path = r"Q:\adria\Documents\climate_data.csv"
    cleaned_df = preprocess_data(filepath=csv_path)

    # 2. Save the cleaned dataframe
    cleaned_df.to_csv("cleaned_climate_data.csv", index=False)
    print("Saved cleaned DataFrame to cleaned_climate_data.csv in your project folder.")

    # 3. Visualize the data
    visualize_data(cleaned_df)


