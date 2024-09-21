from tabnanny import verbose

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from loguru import logger
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm
from xgboost import XGBRegressor

from demos import plot_revenue, split_X_Y


def time_series_cross_validation(
    data: pd.DataFrame,
    params: dict,
    metric=mean_squared_error,
    cv: int = 3,
    plot_data=False,
):
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    if plot_data:
        fig, ax = plt.subplots(ncols=1, nrows=cv, figsize=(7, 3 * cv / 2))
        # plt.tight_layout()
        fig.subplots_adjust(hspace=1)  # Adjust spacing between subplots

    for i, (train_index, test_index) in tqdm(enumerate(tscv.split(data)), total=cv):
        # logger.info(f"Round {i}")
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        X_test, y_test = split_X_Y(test_data)
        X_train, y_train = split_X_Y(train_data)
        # Fit XGBoost
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate Mean Squared Error
        err = metric(y_test, predictions)
        scores.append(err)
        # print(f"Shape of Training set {X_train.shape} and test set {X_test.shape}")
        # print(f"Mean Squared Error for current split: {err}")

        if plot_data:
            data["revenue"].iloc[train_index].plot(ax=ax[i], label="Train")
            data["revenue"].iloc[test_index].plot(ax=ax[i], label="Test")
            if i == 0:
                ax[i].legend()
                ax[i].set_ylabel("revenue")
            if i != cv - 1:
                ax[i].set_xlabel("")
            ax[i].set_title(f"Fold {i+1}")
    # Calculate average Mean Squared Error across all splits

    if plot_data:
        plt.show()
    average_mse = np.mean(scores)
    # print(f"Average Mean Squared Error across all splits: {average_mse}")
    return average_mse


def time_series_cross_validation_lgbm(
    data: pd.DataFrame,
    params: dict,
    metric=mean_squared_error,
    cv: int = 3,
    plot_data=False,
):
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    if plot_data:
        fig, ax = plt.subplots(ncols=1, nrows=cv, figsize=(7, 3 * cv / 2))
        # plt.tight_layout()
        fig.subplots_adjust(hspace=1)  # Adjust spacing between subplots

    try:
        for i, (train_index, test_index) in tqdm(enumerate(tscv.split(data)), total=cv):
            # logger.info(f"Round {i}")
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            X_test, y_test = split_X_Y(test_data)
            X_train, y_train = split_X_Y(train_data)
            # Fit XGBoost
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate Mean Squared Error
            err = metric(y_test, predictions)
            scores.append(err)
            # print(f"Shape of Training set {X_train.shape} and test set {X_test.shape}")
            # print(f"Mean Squared Error for current split: {err}")

            if plot_data:
                data["revenue"].iloc[train_index].plot(ax=ax[i], label="Train")
                data["revenue"].iloc[test_index].plot(ax=ax[i], label="Test")
                if i == 0:
                    ax[i].legend()
                    ax[i].set_ylabel("revenue")
                if i != cv - 1:
                    ax[i].set_xlabel("")
                ax[i].set_title(f"Fold {i+1}")
        # Calculate average Mean Squared Error across all splits

        if plot_data:
            plt.show()
        average_mse = np.mean(scores)
    except:
        return np.Inf
    # print(f"Average Mean Squared Error across all splits: {average_mse}")
    return average_mse


def time_series_cross_validation_arima(
    data: pd.DataFrame,
    params: dict,
    metric=mean_squared_error,
    cv: int = 3,
    plot_data=False,
):
    tscv = TimeSeriesSplit(n_splits=cv)
    scores = []
    if plot_data:
        fig, ax = plt.subplots(ncols=1, nrows=cv, figsize=(7, 3 * cv / 2))
        # plt.tight_layout()
        fig.subplots_adjust(hspace=1)  # Adjust spacing between subplots
    try:
        for i, (train_index, test_index) in tqdm(enumerate(tscv.split(data)), total=cv):
            # logger.info(f"Round {i}")
            train_data, test_data = data.iloc[train_index], data.iloc[test_index]
            X_test, y_test = split_X_Y(test_data)
            X_train, y_train = split_X_Y(train_data)
            # Fit XGBoost
            p, d, q = params["p"], params["d"], params["q"]
            model = ARIMA(y_train, order=(p, d, q), trend=params["trend"])
            model = model.fit()
            # Make predictions
            predictions = model.forecast(X_test.shape[0] + 1)
            # Calculate Mean Squared Error
            err = metric(y_test, predictions[1:])
            scores.append(err)

            if plot_data:
                data["revenue"].iloc[train_index].plot(ax=ax[i], label="Train")
                data["revenue"].iloc[test_index].plot(ax=ax[i], label="Test")
                if i == 0:
                    ax[i].legend()
                    ax[i].set_ylabel("revenue")
                if i != cv - 1:
                    ax[i].set_xlabel("")
                ax[i].set_title(f"Fold {i+1}")
                plot_revenue(y_train, predictions)
        # Calculate average Mean Squared Error across all splits

        if plot_data:
            plt.show()
        average_mse = np.mean(scores)
        return average_mse
    except:
        return np.Inf
        # print(f"Average Mean Squared Error across all splits: {average_mse}")


def recursive_feature_elimination(data, model, params):
    raise NotImplementedError("Muss noch implementiert werden")
