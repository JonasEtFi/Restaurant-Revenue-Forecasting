import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def prepare_data(wheater_data: str, revenue_data: str) -> pd.DataFrame:
    """
    Reads, cleans, and merges weather data with revenue and holiday data.

    Parameters:
    -------------
    wheater_data (str): Path to the CSV file containing weather data.
    revenue_data (str): Path to the CSV file containing revenue and holiday data.


    Returns:
    ------------
    df: A merged DataFrame containing both weather and revenue data.
        The weather data has missing columns removed,
        while the revenue data replaces `1` with `True`
        for holidays and fills missing values with `False`.
    """
    pd.set_option("future.no_silent_downcasting", True)

    # read and prepare wheater data
    wheater_df = pd.read_csv(wheater_data, parse_dates=True, index_col=0)
    # delete the rows where we have no information. all other columns have no NaN value
    wheater_df = wheater_df.dropna(axis=1)

    # read and prepare revenua and holiday data
    revenue_df = pd.read_csv(revenue_data, sep=";", parse_dates=True, index_col=0)
    revenue_df = revenue_df.fillna(0)

    revenue_df["national_holiday"] = revenue_df["national_holiday"].astype(int)

    revenue_df["holiday_not_bw"] = revenue_df["holiday_not_bw"].astype(int)
    revenue_df["holiday"] = revenue_df["holiday"].astype(int)

    revenue_df["holiday_all_germany"] = (
        revenue_df["holiday_not_bw"] * revenue_df["holiday"]
    )
    revenue_df["national_and_holiday"] = (
        revenue_df["national_holiday"] * revenue_df["holiday"]
    )

    df = pd.merge(wheater_df, revenue_df, left_index=True, right_index=True)
    df = weekday_mapping(df)
    df = add_lagged_features(df)
    df = add_bridge_day(df)

    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year

    return df


def add_bridge_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'bridge_day' column to the DataFrame. The 'bridge_day' column is set to 1 if the 'weekday' is Monday
    and the next day is a national holiday; otherwise, it is set to 0.

    Args:
        df (pd.DataFrame): A DataFrame containing at least the following two columns:
            - 'weekday' (str): The day of the week (e.g., 'Monday', 'Tuesday', etc.).
            - 'national_holiday' (int): 1 if the day is a national holiday, 0 otherwise.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'bridge_day' column.
    """
    # Condition for Friday with the previous day being a national holiday
    friday_condition = (df["weekday"].str.lower() == "friday") & (
        df["national_holiday"].shift(1) == 1
    )

    # Condition for Monday with the next day being a national holiday
    monday_condition = (df["weekday"].str.lower() == "monday") & (
        df["national_holiday"].shift(-1) == 1
    )

    # Combining both conditions with OR (|) operator and handling NaN values before converting to int
    df["bridge_day"] = (friday_condition | monday_condition).fillna(0).astype(int)
    return df


def create_test_and_train_set(df: pd.DataFrame):
    """
    Splits the dataframe in a test and train frame 90/10

    Parameters
    -------------
    df: pandas Dataframe containing the prepared data

    Returns
    --------------
    train_df: pd.Dataframe containing the train
    test_df: pd.Dataframe containing the test
    """
    eighty_pct = int(0.8 * df.shape[0])

    train_df = df.iloc[: eighty_pct - 1, :]
    test_df = df.iloc[eighty_pct:, :]
    train_df = train_df.drop("weekday", axis=1)
    test_df = test_df.drop("weekday", axis=1)
    train_df["DOW"] = train_df["DOW"].astype(int)
    test_df["DOW"] = test_df["DOW"].astype(int)

    return train_df, test_df


def weekday_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """ "
    Map the weekday string to integer for easyier datamanipulation

    Args
    ------------
    df: current dataframe with a weekday column
    """
    mapping = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    df["DOW"] = df["weekday"].replace(mapping)
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lagged features to the DataFrame by incorporating revenue data from previous days.

    This function calls helper functions to:
    - Add the previous day's revenue as a feature.
    - Add a three-day and seven-day rolling average of past revenues.

    Args:
    ----------
    df (pd.DataFrame): The input DataFrame, which must contain a 'revenue' column.

    Returns:
    ----------
    pd.DataFrame: A DataFrame with added lagged features for revenue data.
    """
    df = add_revenue_yesterday(df)
    df = add_revenue_mean_three_days(df)
    return df


def add_revenue_yesterday(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'yesterday_revenue' to the DataFrame.

    This column represents the revenue from the previous day (shifted by 1 day).
    If it is the first day in the dataset, the function fills the missing value with the revenue of that first day.

    Args:
    ---------
    df (pd.DataFrame): The input DataFrame, which must contain a 'revenue' column.

    Returns:
    ----------
    pd.DataFrame: A DataFrame with an additional 'yesterday_revenue' column.
    """
    df["yesterday_revenue"] = df["revenue"].shift(1)
    # There is no data for the day before the first day, so we use the revenue as a placeholder
    df.yesterday_revenue.iloc[0] = df.revenue.iloc[0]
    return df


def add_revenue_mean_three_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two new columns to the DataFrame: 'three_day_revenue' and 'week_revenue'.

    - 'three_day_revenue': The mean of the revenue from the previous three days (yesterday and two days before).
    - 'week_revenue': The mean of the revenue from the previous seven days (yesterday and six days before).

    Temporary columns ('lag2', 'lag3', ..., 'lag7') are created to help compute these averages and then removed.

    Args:
    ----------
    df (pd.DataFrame): The input DataFrame, which must contain a 'revenue' column.

    Returns:
    ----------
    pd.DataFrame: A DataFrame with additional 'three_day_revenue' and 'week_revenue' columns.
    """
    df["lag2"] = df["revenue"].shift(2)
    df["lag3"] = df["revenue"].shift(3)
    df["lag4"] = df["revenue"].shift(4)
    df["lag5"] = df["revenue"].shift(5)
    df["lag6"] = df["revenue"].shift(6)
    df["lag7"] = df["revenue"].shift(7)

    df["three_day_revenue"] = df[["yesterday_revenue", "lag2", "lag3"]].mean(axis=1)
    df["week_revenue"] = df[
        ["yesterday_revenue", "lag2", "lag3", "lag4", "lag5", "lag6", "lag7"]
    ].mean(axis=1)

    # Remove temporary lag columns used to calculate the rolling averages
    df = df.drop(["lag" + str(x) for x in range(2, 8)], axis=1)

    return df


def plot_data(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2)
    df["tmax"].plot(ax=axes[0, 0])
    axes[0, 0].set_title("Max temperature")
    df["revenue"].plot(ax=axes[0, 1])
    axes[0, 1].set_title("Revenue")
    df["prcp"].plot(ax=axes[1, 0])
    axes[1, 0].set_title("PRCP")
    df["tavg"].plot(ax=axes[1, 1])
    axes[1, 1].set_title("Average Temperature")
    plt.tight_layout()
    plt.show()


def plot_avg_revenue_per_weekday(df: pd.DataFrame) -> None:
    # Sort values by day of the week
    revenue_per_day = df.sort_values(by=["DOW"])

    # Create a bar plot using seaborn catplot
    g = sns.catplot(data=revenue_per_day, kind="bar", x="weekday", y="revenue")

    # Access the underlying axes from the FacetGrid and set the xticklabels font size
    g.ax.set_xticklabels(g.ax.get_xticklabels(), fontsize=7)

    # Adjust layout for better visualization
    # plt.tight_layout()

    # Show the plot
    plt.show()


def plot_revenue(df: pd.DataFrame, predictions) -> None:
    """
    Plots the revenue per weekday using a seaborn swarm plot.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: 'DOW' (day of the week)
        and 'revenue' (revenue values).
    """
    revenue_df = df["revenue"]
    df["predictions"] = predictions
    pred_df = df["predictions"]
    # Create a bar plot using seaborn catplot
    sns.lineplot(data=df, x=df.index, y="revenue", label="Actual Revenue")
    sns.lineplot(data=df, x=df.index, y="predictions", label="Predicted Revenue")

    # Access the underlying axes from the FacetGrid and set the xticklabels font size

    # Adjust layout for better visualization
    # plt.tight_layout()
    plt.legend()

    # Show the plot
    plt.show()


def plot_revenue_per_weekday(df: pd.DataFrame) -> None:
    """
    Plots the revenue per weekday using a seaborn swarm plot.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: 'DOW' (day of the week)
        and 'revenue' (revenue values).
    """
    revenue_per_day = df.sort_values(by=["DOW"])

    # Create a bar plot using seaborn catplot
    g = sns.catplot(data=revenue_per_day, s=5, kind="swarm", x="weekday", y="revenue")

    # Access the underlying axes from the FacetGrid and set the xticklabels font size
    g.ax.set_xticklabels(g.ax.get_xticklabels(), fontsize=7)

    # Adjust layout for better visualization
    # plt.tight_layout()

    # Show the plot
    plt.show()


def split_X_Y(df: pd.DataFrame):
    """
    Splits the DataFrame into feature matrix (X) and target vector (y).

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing features and a target variable 'revenue'.
    """
    X = df.drop("revenue", axis=1)
    X["DOW"] = X["DOW"].astype(int)
    y = df["revenue"]
    if "weekday" in df.columns:
        X = X.drop("weekday", axis=1)
    return X, y
