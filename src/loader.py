import os
import pandas as pd
import numpy as np
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


def insert_json_row(idx, df, df_insert):
    begin = df.iloc[:idx, ]
    end = df.iloc[idx:, ]
    result = pd.concat([begin, df_insert, end], ignore_index=True)
    return result


def check_json_time_series(json_data):
    for i in range(22):
        nan_row = pd.DataFrame.from_dict({"result": [{'hrPow': np.nan, 'logHr': "%02d" % i}]})
        if json_data.size - 1 < i:
            json_data = insert_json_row(i, json_data, nan_row)
        elif int(json_data.loc[i]['result'].get('logHr')) != i:
            json_data = insert_json_row(i, json_data, nan_row)

    new_row = pd.DataFrame.from_dict({"result": [{'hrPow': 0, 'logHr': "23"}]})
    json_data = insert_json_row(23, json_data, new_row)

    return json_data


def read_json(plant_number, date):
    path = get_power_file(plant_number, date)

    if os.path.isfile(path) is False or os.stat(path).st_size == 0:
        row = [pd.DataFrame.from_dict({"result": [{'hrPow': 0, 'logHr': '%02d' % i}]}) for i in range(24)]
        json_data = pd.concat(row, ignore_index=True)
    else:
        json_data = pd.read_json(path)
        json_data = check_json_time_series(json_data)

    json_data = pd.json_normalize(json_data['result'])

    return json_data


def read_csv(spot_index, date, duration=1):
    dates = [date + timedelta(days=i) for i in range(duration)]

    str_dates = []
    years = []
    result = None

    year = int(dates[0].strftime("%Y"))
    years.append(year)
    for i, date in enumerate(dates):
        str_date = date.strftime("%Y-%m-%d")
        new_year = int(date.strftime("%Y"))
        if year != new_year:
            years.append(new_year)
        year = new_year
        str_dates.append(str_date)

    for year in years:
        sub_str_dates = [str_date for str_date in str_dates if str(year) in str_date]
        path = get_weather_file(spot_index, datetime.strptime(sub_str_dates[0], "%Y-%m-%d"))
        csv_data = pd.read_csv(path, encoding='cp949')
        csv_data_days = [csv_data[csv_data['일시'].str.contains(sub_str_dates[i])] for i in range(len(sub_str_dates))]
        result = pd.concat(csv_data_days)

    return result
