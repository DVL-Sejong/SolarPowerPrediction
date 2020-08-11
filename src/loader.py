import os
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


def get_working_path():
    path = Path(os.getcwd()).parent
    return path


def get_child_dirs(path):
    directories = [f.path for f in os.scandir(path) if f.is_dir()]
    return directories


def get_child_files(path):
    files = [f.path for f in os.scandir(path) if f.is_file()]
    return files


def get_power_file(plant_number, date):
    dir_name = "UR00000%d" % plant_number
    file_name = date.strftime("%Y%m%d") + ".json"
    path = os.path.join(get_working_path(),
                              "data", "pow_24", dir_name, file_name)
    return path


def get_weather_file(spot_index, date):
    year = int(date.strftime("%Y"))
    file_name = "SURFACE_ASOS_%d_HR_%d_%d_%d.csv"\
                % (spot_index, year, year, year + 1)
    path = os.path.join(get_working_path(),
                        "data", "weather", file_name)
    return path


def read_json(plant_number, date):
    path = get_power_file(plant_number, date)
    json_data = pd.read_json(path)
    json_data = pd.json_normalize(json_data['result'])

    return json_data


def read_csv(spot_index, date, duration=1):
    path = get_weather_file(spot_index, date)
    dates = [date + timedelta(days=i) for i in range(duration)]
    str_dates = [date.strftime("%Y-%m-%d") for date in dates]

    csv_data = pd.read_csv(path, encoding='cp949')
    csv_data_days = [csv_data[csv_data['일시'].str.contains(str_dates[i])] for i in range(duration)]
    csv_data_days = pd.concat(csv_data_days)

    return csv_data_days
