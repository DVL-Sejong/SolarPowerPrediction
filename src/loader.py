import math
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from constant import FeatureType
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class Power:
    def __init__(self, args):
        self.args = args
        self.set_duration()

    def set_duration(self):
        self.start = datetime.strptime("%d-01-01 00:00" % self.args.years[0], "%Y-%m-%d %H:%M")
        self.end = datetime.strptime("%d-12-31 23:00" % self.args.years[-1], "%Y-%m-%d %H:%M")
        self.duration = (self.end - self.start).days + 1

        # durations
        self.train_duration = math.floor(self.duration * self.args.ratio[0])
        self.val_duration = math.floor(self.duration * self.args.ratio[1])
        self.test_duration = math.floor(self.duration * self.args.ratio[2])

        # days
        self.train_start = self.start
        self.train_end = self.train_start + timedelta(days=self.train_duration - 1)
        self.val_start = self.train_end + timedelta(days=1)
        self.val_end = self.val_start + timedelta(days=self.val_duration - 1)
        self.test_start = self.val_end + timedelta(days=1)
        self.test_end = self.test_start + timedelta(days=self.test_duration - 1)

        self.train_end += timedelta(hours=23)
        self.val_end += timedelta(hours=23)
        self.test_end += timedelta(hours=23)

        print("train start date:", str(self.train_start))
        print("train end date:", str(self.train_end))
        print("val start date:", str(self.val_start))
        print("val end date:", str(self.val_end))
        print("test start date:", str(self.test_start))
        print("test end date:", str(self.test_end))

    def get_data(self):
        root = Path(os.getcwd()).parent.parent
        power_path = os.path.join(root, "data", "pow")
        extras = [1, 1000, 1]
        zfills = [1, 1, 2]

        power_data_list = []
        for i, year in enumerate(self.args.years):
            power_file_path = os.path.join(power_path, "%s_%d_%d.csv" % (self.args.region, self.args.station, year))
            power_data = pd.read_csv(power_file_path, encoding='euc-kr')

            self.check_missing_dates(year, power_data)
            self.check_midnight_values(power_data)

            power_data_yearly = self.convert_df_to_list(power_data, extra=extras[i], zfill=zfills[i])
            power_data_list.append(power_data_yearly)
        power_data = sum(power_data_list, [])
        power_data = np.asarray(power_data)
        power_data = self.to_dataframe(power_data)

        dataset = self.split_data(power_data)
        power_scaler, y_train, y_val, y_test = self.scale(dataset)
        result = {'scaler': power_scaler, 'train': y_train, 'val': y_val, 'test': y_test}

        return result

    def split_data(self, power_data):
        train_mask = (power_data.index >= self.train_start) & (power_data.index <= self.train_end)
        train_power = power_data.loc[train_mask]['power'].to_numpy()

        val_mask = (power_data.index >= self.val_start) & (power_data.index <= self.val_end)
        val_power = power_data.loc[val_mask]['power'].to_numpy()

        test_mask = (power_data.index >= self.test_start) & (power_data.index <= self.test_end)
        test_power = power_data.loc[test_mask]['power'].to_numpy()

        dataset = {'train': train_power, 'val': val_power, 'test': test_power}

        return dataset

    def check_missing_dates(self, year, power_data):
        date_checker = datetime.strptime("%d.01.01" % year, "%Y.%m.%d")
        dates = power_data['년월일']
        count = 0

        for index, value in dates.items():
            new_date = datetime.strptime(value, "%Y.%m.%d")
            if date_checker != new_date:
                print("standard: %s, file: %s" % (str(date_checker, str(new_date))))
                date_checker = new_date
                count += 1
            date_checker = date_checker + timedelta(days=1)

        print("%d missing dates" % count)

    def check_midnight_values(self, power_data):
        count = 0
        midnights = power_data['24']
        for index, value in midnights.items():
            count = 0
            if value != 0:
                print("index: %d, value: %d" % (index, value))
                count += 0
        print("%d value(s) are not zero" % count)

    def convert_df_to_list(self, power_data, extra=1, zfill=1):
        power_data_list = []
        for index, row in power_data.iterrows():
            for i in range(1, 24):
                value = row[str(i).zfill(zfill)] * extra
                power_data_list.append(int(value))
            if index == 0: print(power_data_list)
            power_data_list.append(0)

        return power_data_list

    def to_dataframe(self, power_data):
        start = datetime.strptime("%d-01-01 00:00" % self.args.years[0], "%Y-%m-%d %H:%M")
        end = datetime.strptime("%d-12-31 23:00" % self.args.years[-1], "%Y-%m-%d %H:%M")
        days = pd.date_range(start, end, freq='H')
        power_data = pd.DataFrame({'일시': days, 'power': power_data})
        power_data = power_data.set_index('일시')
        return power_data

    def scale(self, dataset):
        train = dataset['train']
        val = dataset['val']
        test = dataset['test']

        power_scaler = MinMaxScaler()
        train = power_scaler.fit_transform(train.reshape(-1, 1))
        val = power_scaler.transform(val.reshape(-1, 1))
        test = power_scaler.transform(test.reshape(-1, 1))

        return power_scaler, train, val, test

    def make_y_sample(self, y_data):
        y_list = []
        sample_count = math.floor((y_data.shape[0] - self.args.frame_in ) / self.args.frame_out)
        for i in range(sample_count):
            y = y_data[(i * self.args.frame_out) + self.args.frame_in:
                       (i * self.args.frame_out) + self.args.frame_in + self.args.frame_out]
            y_list.append(y)
        y_list = np.asarray(y_list)
        return y_list


class Weather:
    def __init__(self, args, features):
        self.args = args
        self.features = features
        self.set_duration()

    def set_duration(self):
        self.start = datetime.strptime("%d-01-01 00:00" % self.args.years[0], "%Y-%m-%d %H:%M")
        self.end = datetime.strptime("%d-12-31 23:00" % self.args.years[-1], "%Y-%m-%d %H:%M")
        self.duration = (self.end - self.start).days + 1

        # durations
        self.train_duration = math.floor(self.duration * self.args.ratio[0])
        self.val_duration = math.floor(self.duration * self.args.ratio[1])
        self.test_duration = math.floor(self.duration * self.args.ratio[2])

        # days
        self.train_start = self.start
        self.train_end = self.train_start + timedelta(days=self.train_duration - 1)
        self.val_start = self.train_end + timedelta(days=1)
        self.val_end = self.val_start + timedelta(days=self.val_duration - 1)
        self.test_start = self.val_end + timedelta(days=1)
        self.test_end = self.test_start + timedelta(days=self.test_duration - 1)

        self.train_end += timedelta(hours=23)
        self.val_end += timedelta(hours=23)
        self.test_end += timedelta(hours=23)

        print("train start date:", str(self.train_start))
        print("train end date:", str(self.train_end))
        print("val start date:", str(self.val_start))
        print("val end date:", str(self.val_end))
        print("test start date:", str(self.test_start))
        print("test end date:", str(self.test_end))

    def get_data(self):
        root = Path(os.getcwd()).parent.parent
        weather_path = os.path.join(root, "data", "weather")

        weather_data_list = []
        for i, year in enumerate(self.args.years):
            filename = "SURFACE_ASOS_%d_HR_%d_%d_%d.csv" % (self.args.station, year, year, year + 1)
            weather_file_path = os.path.join(weather_path, filename)
            weather_data = pd.read_csv(weather_file_path, encoding='euc-kr')
            weather_data = self.check_missing_dates(year, weather_data)
            weather_data_yearly = self.interpolate_weather(weather_data, self.features)
            weather_data_yearly = weather_data_yearly.set_index('일시')
            weather_data_list.append(weather_data_yearly)

        weather_data = pd.concat(weather_data_list)
        dataset = self.split_data(weather_data)
        dataset = self.scale_dataset(dataset)

        return dataset

    def split_data(self, weather_data):
        train_mask = (weather_data.index >= self.train_start) & (weather_data.index <= self.train_end)
        train_weather = weather_data.loc[train_mask]

        val_mask = (weather_data.index >= self.val_start) & (weather_data.index <= self.val_end)
        val_weather = weather_data.loc[val_mask]

        test_mask = (weather_data.index >= self.test_start) & (weather_data.index <= self.test_end)
        test_weather = weather_data.loc[test_mask]

        dataset = {'train': train_weather, 'val': val_weather, 'test': test_weather}

        return dataset

    def check_missing_dates(self, year, weather_data):
        weather_data['일시'] = pd.to_datetime(weather_data['일시'], format='%Y-%m-%d %H:%M')
        full_idx = pd.date_range(start=weather_data['일시'].min(), end=weather_data['일시'].max(), freq='60T')

        missing_hour_filled_weather = weather_data.set_index('일시').reindex(full_idx).rename_axis('일시').reset_index()

        start_date = datetime.strptime("%d-01-01 00:00" % year, '%Y-%m-%d %H:%M')
        missing_dates = missing_hour_filled_weather['일시'].isin(weather_data['일시'])
        missing_dates = [(start_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') for i, val in
                         enumerate(missing_dates) if not val]

        print("missing dates:", missing_dates)
        return missing_hour_filled_weather

    def interpolate_weather(self, weather_data, features):
        weather_df = pd.DataFrame()
        weather_df['일시'] = weather_data['일시']
        for feature in features:
            if feature == FeatureType.PRECIPITATION or feature == FeatureType.SUNSHINE:
                weather_feature = weather_data[feature.value].fillna(0)
            else:
                weather_feature = weather_data[feature.value]
            weather_feature = weather_feature.interpolate(mathoed='linear')
            weather_df[feature.value] = weather_feature
        return weather_df

    def scale_dataset(self, dataset):
        train_weather = dataset['train']
        val_weather = dataset['val']
        test_weather = dataset['test']

        weather_scalers = []
        X_train = []
        X_val = []
        X_test = []
        for feature in self.features:
            x_train = train_weather[feature.value].to_numpy()
            x_val = val_weather[feature.value].to_numpy()
            x_test = test_weather[feature.value].to_numpy()
            weather_scaler, x_train, x_val, x_test = self.scale(x_train, x_val, x_test)
            weather_scalers.append(weather_scaler)
            X_train.append(x_train)
            X_val.append(x_val)
            X_test.append(x_test)

        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        X_test = np.asarray(X_test)

        result = {'scaler': weather_scalers, 'train': X_train, 'val': X_val, 'X_test': X_test}
        return result

    def scale(self, train, val, test):
        weather_scaler = MinMaxScaler()
        train = weather_scaler.fit_transform(train.reshape(-1, 1))
        val = weather_scaler.transform(val.reshape(-1, 1))
        test = weather_scaler.transform(test.reshape(-1, 1))

        return weather_scaler, train, val, test

    def make_x_sample(self, x_data, feature_len):
        x_data = x_data[0:feature_len]
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1]))
        x_data = x_data.transpose()

        x_list = []
        sample_count = math.floor((x_data.shape[0] + 24 - 96) / 24)
        for i in range(sample_count):
            x = x_data[(i * 24):(i * 24) + 72]
            x_list.append(x)
        x_list = np.asarray(x_list)
        return x_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    # ====== Data ====== #
    args.years = [2017, 2018, 2019]
    args.region = "Jindo"
    args.station = 192
    args.ratio = [0.6, 0.2, 0.2]

    # ====== Features ====== #
    features = [FeatureType.SUNSHINE,
                FeatureType.GROUND_TEMPERATURE,
                FeatureType.HUMIDITY,
                FeatureType.WIND_SPEED,
                FeatureType.WIND_DIRECTION,
                FeatureType.TEMPERATURE,
                FeatureType.VISIBILITY,
                FeatureType.PRECIPITATION,
                FeatureType.STEAM_PRESSURE,
                FeatureType.DEW_POINT_TEMPERATURE,
                FeatureType.ATMOSPHERIC_PRESSURE]

    # ====== Model ====== #
    args.frame_in = 72
    args.frame_out = 24

    power = Power(args)
    weather = Weather(args, features)

    power_dataset = power.get_data()
    weather_dataset = weather.get_data()

