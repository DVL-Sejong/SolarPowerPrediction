from src.loader import *
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from matplotlib import pyplot
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.constant import *


def plot_day_power(plant_number, date, duration=1):
    day_power = get_power_data(plant_number, date, duration)
    x_value = [date + timedelta(hours=i) for i in range(duration * 24)]
    y_value = [value for value in day_power]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_value, y_value)
    ax.set(xlabel="Time",
           ylabel="Solar Power",
           title="UR00000%s %s" % (str(plant_number), date.strftime("%Y-%m-%d")))

    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    plt.show()


def get_power_data(plant_number, date, duration=1):
    row = pd.Series([0])
    json_data = read_json(plant_number, date)['hrPow']
    json_data = json_data.append(row)
    for i in range(1, duration):
        date = date + timedelta(days=1)
        new_data = read_json(plant_number, date)['hrPow']
        new_data = new_data.append(row)
        json_data = pd.concat([json_data, new_data])
    day_power = normalize(json_data.to_numpy())
    return day_power


def get_weather_data(spot_index, date, feature_type, duration=1):
    csv_data = read_csv(spot_index, date, duration=duration)
    weather_data = interpolate_weather(date, csv_data, feature_type)
    weather_data = normalize(weather_data.to_numpy())
    weather_data = np.nan_to_num(weather_data)

    return weather_data


def plot_day_weather(spot_index, date, feature_type, duration=1):
    weather_data = get_weather_data(spot_index, date, feature_type, duration=duration)

    x_value = [date + timedelta(hours=i) for i in range(duration * 24)]
    unit = feature_type.value
    unit = unit[unit.find("(") + 1:unit.find(")")]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_value, weather_data)
    ax.set(xlabel="Time",
           ylabel=unit,
           title="%s %s" % (date.strftime("%Y-%m-%d"), feature_type.name))

    date_form = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_form)
    fig.autofmt_xdate()
    plt.show()


def insert_row(idx, df, df_insert):
    begin = df.iloc[:idx, ]
    end = df.iloc[idx:, ]
    df_insert = pd.DataFrame(list(df_insert.values()), columns=['일시'])
    return pd.concat([begin, df_insert, end])


def interpolate_weather(date, csv_data, feature_type):
    str_date = date.strftime("%Y-%m-%d %H:%M")

    for i in range(23):
        if csv_data.iloc[i:i+1]['일시'].values[0] != str_date:
            row = {'일시': str_date}
            csv_data = insert_row(i, csv_data, row)
        date = date + timedelta(hours=1)
        str_date = date.strftime("%Y-%m-%d %H:%M")

    weather_data = csv_data[feature_type].interpolate(method='linear')
    return weather_data


def normalize(data):
    normalized = (data-min(data))/(max(data)-min(data))
    return normalized


def get_pearson_correlations(date, plant_number, spot_index, duration=1):
    temp_date = datetime.strptime("20190820", "%Y%m%d")
    power_data = get_power_data(plant_number, temp_date, duration=duration)
    attributes = [attribute.value for attribute in FeatureType]

    correlations = []
    weather_data_list = []
    for attribute in attributes:
        weather_data = get_weather_data(spot_index, date, attribute, duration=duration)
        all_zeros = not np.any(weather_data)
        if all_zeros:
            corr = -10
        else:
            corr, _ = pearsonr(power_data, weather_data)
        correlations.append(corr)
        weather_data_list.append(weather_data)
    return correlations, power_data, weather_data_list


# have to make multi figures
def plot_correlations(correlations, power_data, weather_data_list):
    pyplot.scatter(power_data, weather_data_list[0])
    pyplot.show()


plant_number = 126
date = datetime.strptime("20190820", "%Y%m%d")
duration = 20
# plot_day_power(plant_number, date)
# power_data = get_power_data(plant_number, date, duration=duration)

spot_index = 90
date = datetime.strptime("20160129", "%Y%m%d")
feature_type = FeatureType.TEMPERATURE
# plot_day_weather(spot_index, date, feature_type, duration=duration)
# weather_data = get_weather_data(spot_index, date, feature_type, duration=duration)

correlations, power_data, weather_data_list = get_pearson_correlations(date, plant_number, spot_index, duration)
plot_correlations(correlations, power_data, weather_data_list)
