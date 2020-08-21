import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from matplotlib.dates import DateFormatter


class Dataset:

    def __init__(self, start_date, end_date):
        self.path = Path(os.getcwd()).parent.parent
        self.start_date = datetime.strptime(start_date, "%Y%m%d")
        self.end_date = datetime.strptime(end_date, "%Y%m%d")
        self.duration = (self.end_date - self.start_date).days + 1

    def get_data(self):
        pass

    def get_file_path(self, date):
        pass

    def read_file(self, date):
        pass

    def insert_row(self, dataframe, new_row, index, columns):
        begin = dataframe.iloc[:index, ]
        end = dataframe.iloc[index:, ]

        if columns is None:
            df_insert = new_row
        else:
            df_insert = pd.DataFrame(list(new_row.values()), columns=columns)

        result = pd.concat([begin, df_insert, end], ignore_index=True)
        return result

    def check_time_series(self, data):
        pass

    @staticmethod
    def interpolate(data):
        interpolated = data.interpolate(method='linear')
        return interpolated

    @staticmethod
    def normalize(data):
        normalized = (data - min(data)) / (max(data) - min(data))
        return normalized

    def plot(self):
        pass


class Power(Dataset):

    def __init__(self, args):
        super(Power, self).__init__(args.start_date, args.end_date)
        self.plant = args.plant

    def get_data(self):
        super().get_data()

        json_data = pd.DataFrame()
        date = self.start_date

        for i in range(self.duration):
            new_data = self.read_file(date + timedelta(days=i))
            new_data = self.check_time_series(new_data)
            new_data = pd.json_normalize(new_data['result'])
            json_data = json_data.append(new_data, ignore_index=True)

        power_data = self.interpolate(json_data)['hrPow']
        # power_data = self.normalize(power_data['hrPow'])

        return power_data

    def get_file_path(self, date):
        super().get_file_path(date)

        dir_name = "UR00000%d" % self.plant
        file_name = date.strftime("%Y%m%d") + ".json"
        path = os.path.join(self.path,
                            "data", "pow_24", dir_name, file_name)
        return path

    def read_file(self, date):
        super().read_file(date)

        file_path = self.get_file_path(date)

        if os.path.isfile(file_path) is False or os.stat(file_path).st_size == 0:
            row = [pd.DataFrame.from_dict({"result": [{'hrPow': 0, 'logHr': '%02d' % i}]}) for i in range(24)]
            json_data = pd.concat(row, ignore_index=True)
        else:
            json_data = pd.read_json(file_path)

        return json_data

    def insert_row(self, dataframe, new_row, index, columns=None):
        return super().insert_row(dataframe, new_row, index, columns)

    def check_time_series(self, json_data):
        super().check_time_series(json_data)

        for i in range(23):
            nan_row = pd.DataFrame.from_dict({"result": [{'hrPow': np.nan, 'logHr': "%02d" % i}]})
            if json_data.size - 1 < i:
                json_data = self.insert_row(json_data, nan_row, i)
            elif int(json_data.loc[i]['result'].get('logHr')) != i:
                json_data = self.insert_row(json_data, nan_row, i)

        new_row = pd.DataFrame.from_dict({"result": [{'hrPow': 0, 'logHr': "23"}]})
        if len(json_data) < 24:
            json_data = self.insert_row(json_data, new_row, 23)

        return json_data

    def plot(self):
        super().plot()

        power_data = self.get_data()
        x_value = [self.start_date + timedelta(hours=i) for i in range(self.duration * 24)]
        y_value = [value for value in power_data]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_value, y_value)
        ax.set(xlabel="Time", ylabel="Solar Power")

        date_form = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(date_form)
        fig.autofmt_xdate()
        plt.show()


class Weather(Dataset):

    def __init__(self, args):
        super(Weather, self).__init__(args.start_date, args.end_date)
        self.spot = args.spot
        self.features = args.features

    def get_data(self):
        super().get_data()

        weather_data = pd.DataFrame()
        csv_data = self.read_file()

        for feature in self.features:
            feature_data = csv_data[feature.value]
            feature_data = self.interpolate(feature_data)
            # feature_data = self.normalize(feature_data.to_numpy())
            feature_data = np.nan_to_num(feature_data)
            weather_data[feature] = feature_data

        return weather_data

    def get_file_path(self, date):
        super().get_file_path(date)

        year = int(date.strftime("%Y"))
        file_name = "SURFACE_ASOS_%d_HR_%d_%d_%d.csv" \
                    % (self.spot, year, year, year + 1)
        path = os.path.join(self.path,
                            "data", "weather", file_name)
        return path

    def read_file(self, date=None):
        super().read_file(date)

        dates = [self.start_date + timedelta(days=i) for i in range(self.duration)]

        str_dates = []
        years = []
        csv_data_days = []

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
            date = datetime.strptime(sub_str_dates[0], "%Y-%m-%d")
            path = self.get_file_path(date)
            csv_data = pd.read_csv(path, encoding='cp949')
            for sub_date in sub_str_dates:
                csv_data_day = csv_data[csv_data['일시'].str.contains(sub_date)]
                csv_data_day = self.check_time_series(csv_data_day)
                csv_data_days.append(csv_data_day)

        csv_df = pd.concat(csv_data_days)
        return csv_df

    def insert_row(self, dataframe, new_row, index, columns=['일시']):
        return super().insert_row(dataframe, new_row, index, columns)

    def check_time_series(self, csv_data):
        super().check_time_series(csv_data)

        str_date = csv_data.iloc[0]['일시']
        date = datetime.strptime(str_date, "%Y-%m-%d %H:%M")

        for i in range(23):
            if csv_data.iloc[i:i + 1]['일시'].values[0] != str_date:
                row = {'일시': str_date}
                csv_data = self.insert_row(csv_data, row, i)
            date = date + timedelta(hours=1)
            str_date = date.strftime("%Y-%m-%d %H:%M")

        return csv_data

    def plot(self):
        super().plot()

        weather_data = self.get_data()

        x_value = [self.start_date + timedelta(hours=i) for i in range(self.duration * 24)]

        for feature_type in self.features:
            unit = feature_type.value
            unit = unit[unit.find("(") + 1:unit.find(")")]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(x_value, weather_data)
            ax.set(xlabel="Time",
                   ylabel=unit,
                   title=feature_type.name)

            date_form = DateFormatter("%H:%M")
            ax.xaxis.set_major_formatter(date_form)
            fig.autofmt_xdate()
            plt.show()


class Loader:
    def __init__(self, args):
        self.start_date = datetime.strptime(args.start_date, "%Y%m%d")
        self.end_date = datetime.strptime(args.end_date, "%Y%m%d")
        self.duration = (self.end_date - self.start_date).days + 1
        self.x_frames = args.x_frames
        self.y_frames = args.y_frames
        self.features = args.features
        self.power_data = Power(args).get_data()
        self.weather_data = Weather(args).get_data()
        self.set_data()
        self.set_durations()
        self.set_dates()

    def set_data(self):
        data = []
        data.append(self.power_data)
        for feature in self.features:
            data.append(self.weather_data[feature])
        self.data = data

    def set_durations(self):
        self.train_duration = math.floor(self.duration * 0.75)
        self.val_duration = math.floor(self.duration * 0.125)
        self.test_duration = math.floor(self.duration * 0.125)

    def set_dates(self):
        self.train_start = self.start_date
        self.train_end = self.train_start + timedelta(days=self.train_duration - 1)
        self.val_start = self.train_end + timedelta(days=1)
        self.val_end = self.val_start + timedelta(days=self.val_duration - 1)
        self.test_start = self.val_end + timedelta(days=1)
        self.test_end = self.test_start + timedelta(days=self.test_duration - 1)

    def get_sample_cnt(self, start, end):
        duration = (end - start).days + 1
        sample_cnt = duration - (self.x_frames + self.y_frames) + 2
        return sample_cnt

    def get_item(self, date):
        index = (date - self.start_date).days * 24
        X = [self.data[i+1][index:index+(self.x_frames * 24)] for i in range(len(self.features))]
        y = self.data[0][index+(self.x_frames * 24):index+((self.x_frames + self.y_frames) * 24)]
        return np.asarray(X), y

    def get_items(self, start, end):
        X = []
        y = []
        sample_cnt = self.get_sample_cnt(start, end)
        for i in range(sample_cnt):
            X_item, y_item = self.get_item(start + timedelta(days=i))
            X.append(X_item)
            y.append(y_item)

        return np.asarray(X), np.asarray(y)

    def get_dataset(self):
        X_train, y_train = self.get_items(self.train_start, self.train_end)
        X_val, y_val = self.get_items(self.val_start, self.val_end)
        X_test, y_test = self.get_items(self.test_start, self.test_end)
        partition = {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [X_test, y_test]}

        return partition
