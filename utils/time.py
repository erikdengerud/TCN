# time.py

import numpy as np
import pandas as pd


def MinuteOfHour(time_index):
    minutes = time_index.minute.to_numpy()
    minutes = minutes / 59.0 - 0.5
    return minutes


def HourOfDay(time_index):
    hours = time_index.hour.to_numpy()
    hours = hours / 23.0 - 0.5
    return hours


def DayOfWeek(time_index):
    days = time_index.dayofweek.to_numpy()
    days = days / 6.0 - 0.5
    return days


def DayOfMonth(time_index):
    days = time_index.day.to_numpy() / 30.0 - 0.5
    return days


def DayOfYear(time_index):
    return time_index.dayofyear.to_numpy() / 364.0 - 0.5


def MonthOfYear(time_index):
    return time_index.month.to_numpy() / 11.0 - 0.5


def WeekOfYear(time_index):
    return time_index.weekofyear.to_numpy() / 51.0 - 0.5


if __name__ == "__main__":
    import torch

    dates_index = pd.date_range(start="2012/01/01", periods=362 * 12, freq="H")

    time_index = pd.DatetimeIndex(dates_index)
    time_index = pd.DatetimeIndex(time_index)
    Z = np.matrix(
        [
            HourOfDay(time_index),
            DayOfWeek(time_index),
            DayOfMonth(time_index),
            DayOfYear(time_index),
            MonthOfYear(time_index),
            WeekOfYear(time_index),
        ]
    )
    Z = torch.from_numpy(Z)
    print(Z)
