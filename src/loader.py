import enum
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from matplotlib.dates import DateFormatter
from sklearn import preprocessing


class DataType(enum.Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 1


class Dataset:

    def __init__(self, start_date, end_date, x_frames, y_frames):
        self.path = Path(os.getcwd()).parent.parent
        self.start_date = datetime.strptime(start_date, "%Y%m%d")
        self.end_date = datetime.strptime(end_date, "%Y%m%d") + timedelta(days=-3)
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.sample_cnt = self.get_sample_cnt(self.start_date, self.end_date)
        self.duration = (self.end_date - self.start_date).days + 1 + 3
        self.set_durations()
        self.set_dates()
        self.set_sample_counts()

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

    def set_durations(self):
        self.train_duration = math.floor(self.sample_cnt * 0.75)
        self.val_duration = math.floor(self.sample_cnt * 0.125)
        self.test_duration = math.floor(self.sample_cnt * 0.125)

    def set_dates(self):
        self.train_start = self.start_date
        self.train_end = self.train_start + timedelta(days=self.train_duration - 1)
        self.val_start = self.train_end + timedelta(days=1)
        self.val_end = self.val_start + timedelta(days=self.val_duration - 1)
        self.test_start = self.val_end + timedelta(days=1)
        self.test_end = self.test_start + timedelta(days=self.test_duration - 1)
        print("train start date:", str(self.train_start))
        print("train end date:", str(self.train_end))
        print("val start date:", str(self.val_start))
        print("val end date:", str(self.val_end))
        print("test start date:", str(self.test_start))
        print("test end date:", str(self.test_end))

    def get_day_cnt(self, start, end):
        duration = (end - start).days + 1
        return duration

    def get_sample_cnt(self, start, end):
        duration = (end - start).days + 1
        return duration

    def set_sample_counts(self):
        self.train_cnt = self.get_day_cnt(self.train_start, self.train_end)
        self.val_cnt = self.get_day_cnt(self.val_start, self.val_end)
        self.test_cnt = self.get_day_cnt(self.test_start, self.test_end)

    @staticmethod
    def interpolate(data):
        interpolated = data.interpolate(method='linear')
        return interpolated

    @staticmethod
    def normalize(data, min_value=None, max_value=None):
        if min_value is None and max_value is None:
            min_value = min(data)
            max_value = max(data)
        normalized = (data - min_value) / (max_value - min_value)
        return normalized

    def plot(self):
        pass


class Power(Dataset):

    def __init__(self, args):
        super(Power, self).__init__(args.start_date, args.end_date, args.x_frames, args.y_frames)
        self.plant = args.plant
        self.x_frames = args.x_frames
        self.y_frames = args.y_frames

    def get_data(self):
        super().get_data()

        json_data = pd.DataFrame()
        date = self.start_date

        for i in range(self.duration):
            new_data = self.read_file(date + timedelta(days=i))
            # print(date + timedelta(days=i))
            new_data = self.check_time_series(new_data)
            new_data = pd.json_normalize(new_data['result'])
            json_data = json_data.append(new_data, ignore_index=True)

        power_data = self.interpolate(json_data)['hrPow']
        power_data = np.nan_to_num(power_data)
        # power_data = self.normalize(power_data['hrPow'])

        train_start = self.x_frames * 24
        train_end = train_start + (self.train_cnt * 24)
        val_start = ((self.val_start - self.train_start).days + self.x_frames) * 24
        val_end = val_start + (self.val_cnt * 24)
        test_start = ((self.test_start - self.train_start).days + self.x_frames) * 24
        test_end = test_start + (self.test_cnt * 24)

        train_set = power_data[train_start:train_end]
        val_set = power_data[val_start:val_end]
        test_set = power_data[test_start:test_end]

        # train_set = self.normalize(train_set, min(train_set), max(train_set))
        # val_set = self.normalize(val_set, min(train_set), max(train_set))
        # test_set = self.normalize(test_set, min(train_set), max(train_set))

        scaler = preprocessing.MinMaxScaler()
        train_set = scaler.fit_transform(train_set.reshape(-1, 1))
        val_set = scaler.transform(val_set.reshape(-1, 1))
        test_set = scaler.transform(test_set.reshape(-1, 1))

        train_set = train_set.reshape((train_set.shape[0]))
        val_set = val_set.reshape((val_set.shape[0]))
        test_set = test_set.reshape((test_set.shape[0]))

        return train_set, val_set, test_set, scaler

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

        # print("json_data")
        # print(json_data)

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


class Power2(Dataset):

    def __init__(self, args):
        super(Power2, self).__init__(args.start_date, args.end_date, args.x_frames, args.y_frames)
        self.spot = args.spot
        self.x_frames = args.x_frames
        self.y_frames = args.y_frames

    def get_data(self):
        super().get_data()

        power_values = self.read_file()
        power_values = self.interpolate(pd.DataFrame(power_values, columns=['hrPow']))['hrPow']
        power_values = np.nan_to_num(power_values)

        train_start = self.x_frames * 24
        train_end = train_start + (self.train_cnt * 24)
        val_start = ((self.val_start - self.train_start).days + self.x_frames) * 24
        val_end = val_start + (self.val_cnt * 24)
        test_start = ((self.test_start - self.train_start).days + self.x_frames) * 24
        test_end = test_start + (self.test_cnt * 24)

        train_set = power_values[train_start:train_end]
        val_set = power_values[val_start:val_end]
        test_set = power_values[test_start:test_end]

        # train_set = self.normalize(train_set, min(train_set), max(train_set))
        # val_set = self.normalize(val_set, min(train_set), max(train_set))
        # test_set = self.normalize(test_set, min(train_set), max(train_set))

        scaler = preprocessing.MinMaxScaler()
        train_set = scaler.fit_transform(train_set.reshape(-1, 1))
        val_set = scaler.transform(val_set.reshape(-1, 1))
        test_set = scaler.transform(test_set.reshape(-1, 1))

        train_set = train_set.reshape((train_set.shape[0]))
        val_set = val_set.reshape((val_set.shape[0]))
        test_set = test_set.reshape((test_set.shape[0]))

        return train_set, val_set, test_set, scaler

    def get_file_path(self, year):
        super().get_file_path(year)

        path = os.path.join(self.path, "data", "pow")
        path = os.path.join(path, str(self.spot) + "_" + str(year) + ".csv")

        return path

    def read_file(self, date=None):
        super().read_file(date)

        power_values = list()
        date = self.start_date

        years = []
        old_day = date
        new_day = date
        years.append(old_day.year)
        file_path = self.get_file_path(old_day.year)
        power = pd.read_csv(file_path, encoding='CP949')
        for i in range(self.duration):
            new_day_str = new_day.strftime("%Y.%m.%d")
            power_row = power.loc[power['년월일'] == new_day_str]
            for j in range(24):
                power_values.append(power_row[str(j+1)])

            new_day = date + timedelta(days=1)
            if old_day.year < new_day.year:
                old_day = new_day
                years.append(new_day.year)
                file_path = self.get_file_path(new_day.year)
                power = pd.read_csv(file_path, encoding='CP949')

        return power_values

    def insert_row(self, dataframe, new_row, index, columns=None):
        return super().insert_row(dataframe, new_row, index, columns)

    def check_time_series(self, json_data):
        super().check_time_series(json_data)
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
        super(Weather, self).__init__(args.start_date, args.end_date, args.x_frames, args.y_frames)
        self.spot = args.spot
        self.features = args.features
        self.x_frames = args.x_frames
        self.y_frames = args.y_frames

    def get_data(self):
        super().get_data()

        train_data = pd.DataFrame()
        validation_data = pd.DataFrame()
        test_data = pd.DataFrame()
        csv_data = self.read_file()

        for feature in self.features:
            feature_data = csv_data[feature.value]
            feature_data = self.interpolate(feature_data)
            feature_data = np.nan_to_num(feature_data)
            # feature_data = self.normalize(feature_data)

            train_start = 0
            train_end = train_start + ((self.train_cnt + (self.x_frames - 1)) * 24)
            val_start = (self.val_start - self.train_start).days * 24
            val_end = val_start + ((self.val_cnt + (self.x_frames - 1)) * 24)
            test_start = (self.test_start - self.train_start).days * 24
            test_end = test_start + ((self.test_cnt + (self.x_frames - 1)) * 24)

            train_set = feature_data[train_start:train_end]
            val_set = feature_data[val_start:val_end]
            test_set = feature_data[test_start:test_end]

            # train_set = self.normalize(train_set, min(train_set), max(train_set))
            # val_set = self.normalize(val_set, min(train_set), max(train_set))
            # test_set = self.normalize(test_set, min(train_set), max(train_set))
            scaler = preprocessing.MinMaxScaler()
            train_set = scaler.fit_transform(train_set.reshape(-1, 1))
            val_set = scaler.transform(val_set.reshape(-1, 1))
            test_set = scaler.transform(test_set.reshape(-1, 1))

            train_data[feature] = train_set.reshape((train_set.shape[0]))
            validation_data[feature] = val_set.reshape((val_set.shape[0]))
            test_data[feature] = test_set.reshape((test_set.shape[0]))

            # train_data[feature] = train_set
            # validation_data[feature] = val_set
            # test_data[feature] = test_set

        return train_data, validation_data, test_data

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
        self.features = args.features
        self.x_frames = args.x_frames
        self.y_frames = args.y_frames
        self.weather = Weather(args)
        self.weather_data = self.weather.get_data()
        self.power = Power(args)
        self.power_data = self.power.get_data()

    def get_X_set(self, X_data, frames):
        X_set = []

        for i, feature in enumerate(self.features):
            feature_set = self.split_dataset(X_data[feature], frames)
            X_set.append(feature_set)

        if len(X_set) == 1:
            dataset = np.asarray(X_set[0])
            return dataset

        dataset = np.concatenate(X_set, axis=2)

        return dataset

    def get_y_set(self, y_data, frames):
        dataset = self.split_dataset(y_data, frames)
        return dataset

    def split_dataset(self, data, frames):
        dataset = list()
        sample_cnt = int((len(data) / 24) - frames + 1)
        for i in range(sample_cnt):
            elem = np.asarray(data[i * 24:(i + frames) * 24])
            if frames != 1:
                elem = elem.reshape((frames * 24, 1))
            dataset.append(elem)
        return np.asarray(dataset)

    def get_dataset(self):
        X_train, X_val, X_test = self.weather_data
        y_train, y_val, y_test, scaler = self.power_data

        X_train = self.get_X_set(X_train, self.x_frames)
        X_val = self.get_X_set(X_val, self.x_frames)
        X_test = self.get_X_set(X_test, self.x_frames)

        y_train = self.get_y_set(y_train, self.y_frames)
        y_val = self.get_y_set(y_val, self.y_frames)
        y_test = self.get_y_set(y_test, self.y_frames)

        # X_train = np.zeros((X_train.shape))
        # X_val = np.zeros((X_val.shape))
        # X_test = np.zeros((X_test.shape))

        # print(X_train.shape, X_val.shape, X_test.shape)
        # print(y_train.shape, y_val.shape, y_test.shape)

        partition = {'train': [X_train, y_train], 'val': [X_val, y_val], 'test': [X_test, y_test]}

        return partition, scaler
