import matplotlib.pyplot as plt
from numpy import int8
import pandas as pd
import seaborn as sns


import pandas as pd

def prepare_data(weather_data: str, revenue_data: str, rolling_windows: list = [7, 14]) -> pd.DataFrame:
    """
    Reads, cleans, and merges weather data with revenue and holiday data.
    Adds rolling window features for revenue-related columns.

    Parameters:
    -------------
    weather_data (str): Path to the CSV file containing weather data.
    revenue_data (str): Path to the CSV file containing revenue and holiday data.
    rolling_windows (list): List of window sizes for calculating rolling statistics.

    Returns:
    ------------
    df: A merged DataFrame containing both weather and revenue data.
        Includes rolling window features for revenue-related data.
    """
    # Read and prepare weather data
    weather_df = pd.read_csv(weather_data, parse_dates=True, index_col=0)
    weather_df = weather_df.dropna(axis=1)  # Remove columns with missing values

    # Read and prepare revenue and holiday data
    revenue_df = pd.read_csv(revenue_data, sep=";", parse_dates=True, index_col=0)
    revenue_df = revenue_df.fillna(0)

    revenue_df["national_holiday"] = revenue_df["national_holiday"].astype(int)
    revenue_df["holiday_not_bw"] = revenue_df["holiday_not_bw"].astype(int)
    revenue_df["holiday"] = revenue_df["holiday"].astype(int)

    revenue_df["holiday_all_germany"] = revenue_df["holiday_not_bw"] * revenue_df["holiday"]
    revenue_df["national_and_holiday"] = revenue_df["national_holiday"] * revenue_df["holiday"]

    # Add rolling features for revenue columns based on the specified rolling windows
    column = "revenue"
    # Shift the column to exclude the current row from the rolling calculation
    shifted_column = revenue_df[column].shift(1)
    for window in rolling_windows:
        # Compute the rolling mean using the shifted column
        revenue_df[f"{column}_rolling_mean_{window}"] = shifted_column.rolling(window).mean()
        # Fill NaN values with the original column values (optional)
        revenue_df[f"{column}_rolling_mean_{window}"] = revenue_df[f"{column}_rolling_mean_{window}"].fillna(revenue_df[column])



    # Merge weather and revenue data
    df = pd.merge(weather_df, revenue_df, left_index=True, right_index=True)

    # Add additional features
    df = feature_engineering(df)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the input DataFrame.

    Args:
    ----------
    df (pd.DataFrame): The input DataFrame containing weather, revenue, and holiday data.

    Returns:
    ----------
    pd.DataFrame: A DataFrame with additional features engineered from the input data.
    """

    # Add weekday one hot encoding
    df = pd.get_dummies(df, dtype=int8)



    # Add bridge day feature
    df = add_bridge_day(df)

    # Add date components
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["year"] = df.index.year

    df['sunday_and_holiday'] = df['weekday_sunday'] * df['holiday']
    df['saturday_and_holiday'] = df['weekday_saturday'] * df['holiday']
    df['friday_and_holiday'] = df['weekday_friday'] * df['holiday']

    df['sunday_and_national_holiday'] = df['weekday_sunday'] * df['national_holiday']
    df['saturday_and_national_holiday'] = df['weekday_saturday'] * df['national_holiday']
    df['friday_and_national_holiday'] = df['weekday_friday'] * df['national_holiday']

    df["season"] = df["month"].apply(get_season)
    df["spring"] = (df["season"] == "spring").astype(int8)
    df["summer"] = (df["season"] == "summer").astype(int8)
    df["fall"] = (df["season"] == "fall").astype(int8)
    df["winter"] = (df["season"] == "winter").astype(int8)

    df = df.drop("season", axis=1)

    return df

def get_season(month):
    if month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    elif month in [9, 10, 11]:
        return "fall"
    else:
        return "winter"

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
    friday_condition = (df["weekday_friday"] == 1) & (
        df["national_holiday"].shift(1) == 1
    )

    # Condition for Monday with the next day being a national holiday
    monday_condition = (df["weekday_monday"] == 1) & (
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
    """
    Plots the average revenue per weekday using one-hot encoded weekday columns.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing one-hot encoded weekday columns 
        (e.g., 'weekday_monday', 'weekday_tuesday', ...) and 'revenue'.
    """
    # Extract weekday columns
    weekday_columns = [col for col in df.columns if col.startswith("weekday_")]

    # Calculate average revenue for each weekday
    avg_revenue_per_weekday = {
        col.split("_")[1].capitalize(): df.loc[df[col] == 1, "revenue"].mean()
        for col in weekday_columns
    }

    # Convert to a DataFrame for plotting
    avg_revenue_df = pd.DataFrame(
        list(avg_revenue_per_weekday.items()), columns=["Weekday", "Average Revenue"]
    )

    # Sort by weekday order
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    avg_revenue_df["Weekday"] = pd.Categorical(avg_revenue_df["Weekday"], categories=weekday_order, ordered=True)
    avg_revenue_df = avg_revenue_df.sort_values("Weekday")

    # Create a bar plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_revenue_df, x="Weekday", y="Average Revenue", palette="viridis")

    # Add labels and title
    plt.xlabel("Weekday")
    plt.ylabel("Average Revenue")
    plt.title("Average Revenue Per Weekday")

    # Adjust layout for better visualization
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_revenue(df: pd.DataFrame, predictions, title=None) -> None:
    """
    Plots the revenue and predictions over time using a line plot.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: 'DOW' (day of the week)
        and 'revenue' (revenue values).
    predictions : array-like
        The predicted revenue values.
    title : str, optional
        The title of the plot. Default is None.
    """
    # Add predictions to the DataFrame
    df["predictions"] = predictions

    # Create the line plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=df.index, y="revenue", label="Actual Revenue")
    sns.lineplot(data=df, x=df.index, y="predictions", label="Predicted Revenue")

    # Set the title if provided
    if title:
        plt.title(title)

    # Add labels and legend
    plt.xlabel("Index")
    plt.ylabel("Revenue")
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_mse(df: pd.DataFrame, predictions, title=None) -> None:
    """
    Plots the revenue and predictions over time, along with the Mean Squared Error (MSE) for each point.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns: 'DOW' (day of the week)
        and 'revenue' (revenue values).
    predictions : array-like
        The predicted revenue values.
    title : str, optional
        The title of the plot. Default is None.
    """
    # Add predictions to the DataFrame
    df["predictions"] = predictions
    
    # Calculate MSE for each point
    df["mse"] = (df["revenue"] - df["predictions"])**2

    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot the MSE as a secondary line
    sns.lineplot(data=df, x=df.index, y="mse", label="MSE (Per Data Point)", color="red", linestyle="--")

    # Set the title if provided
    if title:
        plt.title(title)

    # Add labels and legend
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_revenue_per_weekday(df: pd.DataFrame) -> None:
    """
    Plots the revenue per weekday using a seaborn swarm plot.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing one-hot encoded weekday columns
        (e.g., 'weekday_monday', 'weekday_tuesday', ...) and 'revenue'.
    """
    # Extract weekday columns
    weekday_columns = [col for col in df.columns if col.startswith("weekday_")]

    # Prepare a long-format DataFrame for plotting
    long_format_data = []
    for col in weekday_columns:
        day_name = col.split("_")[1].capitalize()
        day_revenue = df.loc[df[col] == 1, "revenue"]
        long_format_data.extend([(day_name, revenue) for revenue in day_revenue])

    # Convert to DataFrame
    plot_df = pd.DataFrame(long_format_data, columns=["weekday", "revenue"])

    # Sort by weekday order
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    plot_df["weekday"] = pd.Categorical(plot_df["weekday"], categories=weekday_order, ordered=True)

    # Create a swarm plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.swarmplot(data=plot_df, x="weekday", y="revenue", size=5)

    # Add labels and title
    plt.xlabel("Weekday")
    plt.ylabel("Revenue")
    plt.title("Revenue Distribution Per Weekday")

    # Adjust layout for better visualization
    plt.tight_layout()

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
    y = df["revenue"]

    return X, y
