import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv("dataset.csv")
    df_values = df.values
    temp = sns.lineplot(x="date", y="meantemp", data=df)

    # Split the data into training and testing sets
    train_size = int(len(df) * 0.8)
    train, test = df_values[0:train_size, :], df_values[train_size:, :]

    train_df = pd.DataFrame(train, columns=df.columns)
    test_df = pd.DataFrame(test, columns=df.columns)

    temp_X = sns.lineplot(x="date", y="meantemp", data=train_df, color="red")
    temp_y = sns.lineplot(x="date", y="meantemp", data=test_df, color="blue")

    plt.show()
