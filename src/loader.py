import os
import pandas as pd
from datetime import datetime
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


def get_power_file(plant_number, str_date):
    dir_name = "UR00000%d" % plant_number
    file_name = str_date + ".json"
    path = os.path.join(get_working_path(),
                              "data", "pow_24", dir_name, file_name)
    return path


def get_weather_file(spot_index, year):
    file_name = "SURFACE_ASOS_%d_HR_%d_%d_%d.csv"\
                % (spot_index, year, year, year + 1)
    path = os.path.join(get_working_path(),
                        "data", "weather", file_name)
    return path


def read_json(plant_number, str_date):
    path = get_power_file(plant_number, str_date)
    json_data = pd.read_json(path)
    return json_data


def read_csv(spot_index, year):
    path = get_weather_file(spot_index, year)
    csv_data = pd.read_csv(path, encoding='cp949')
    return csv_data

